"""
Distributed Node Management
============================

Manages a peer-to-peer network of Thalos Prime nodes that operate
independently and synchronise knowledge periodically.

Features:
- Node registration and discovery
- Heartbeat / liveness tracking
- Encrypted knowledge sync messages with RSA-PSS digital signatures
- Autonomous failure detection and recovery
"""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from thalos_prime.autonomy.encryption import AsymmetricEncryption, SymmetricEncryption
from thalos_prime.models import NodeInfo


@dataclass
class SyncMessage:
    """Payload exchanged between nodes during a synchronisation round."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_node_id: str = ""
    knowledge_delta: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature: str = ""  # base-64 encoded RSA-PSS signature over the canonical payload


class NodeRegistry:
    """
    In-process registry of known peer nodes.

    In a production deployment this would be backed by a distributed
    hash table (DHT) or a gossip protocol.
    """

    LIVENESS_TIMEOUT = timedelta(minutes=5)

    def __init__(self, local_node: NodeInfo) -> None:
        self._local = local_node
        self._peers: dict[str, NodeInfo] = {}

    def register(self, node: NodeInfo) -> None:
        if node.node_id == self._local.node_id:
            return
        self._peers[node.node_id] = node

    def deregister(self, node_id: str) -> bool:
        if node_id in self._peers:
            del self._peers[node_id]
            return True
        return False

    def heartbeat(self, node_id: str) -> None:
        """Update the last-seen timestamp for a peer node."""
        if node_id in self._peers:
            self._peers[node_id].last_seen = datetime.now(timezone.utc)
            self._peers[node_id].status = "active"

    def prune_stale(self) -> list[str]:
        """Mark nodes that have not sent a heartbeat within the timeout as offline."""
        now = datetime.now(timezone.utc)
        stale = [
            nid
            for nid, node in self._peers.items()
            if (now - node.last_seen) > self.LIVENESS_TIMEOUT
        ]
        for nid in stale:
            self._peers[nid].status = "offline"
        return stale

    @property
    def active_peers(self) -> list[NodeInfo]:
        return [n for n in self._peers.values() if n.status == "active"]

    @property
    def all_peers(self) -> list[NodeInfo]:
        return list(self._peers.values())

    @property
    def local_node(self) -> NodeInfo:
        return self._local

    def peer_count(self) -> int:
        return len(self._peers)

    def get_peer(self, node_id: str) -> NodeInfo | None:
        return self._peers.get(node_id)


class NodeSynchronizer:
    """
    Handles the preparation, encryption, and parsing of synchronisation
    messages between nodes.

    Each outgoing message is:
    1. Serialised to JSON.
    2. Signed with the local node's RSA private key (if a signing_key is
       provided), guaranteeing authenticity and integrity.
    3. Encrypted with AES-256-GCM for confidentiality.

    On receipt the process is reversed:
    1. AES-256-GCM decryption.
    2. JSON parsing.
    3. RSA-PSS signature verification against the sender's public key
       (if available in the supplied public_key_store).  Messages with
       an invalid signature are rejected; unsigned messages are accepted
       with a logged warning.
    """

    def __init__(
        self,
        local_node: NodeInfo,
        shared_key: bytes | None = None,
        signing_key: AsymmetricEncryption | None = None,
        public_key_store: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            local_node:        The identity of this node.
            shared_key:        AES-256 key (32 bytes) for symmetric encryption.
                               Generated randomly if omitted.
            signing_key:       RSA key pair used to sign outgoing messages.
                               If None, messages are sent unsigned.
            public_key_store:  Mapping of node_id → PEM public key used to
                               verify incoming message signatures.
        """
        self._local = local_node
        self._crypto = SymmetricEncryption(key=shared_key)
        self._signing_key: AsymmetricEncryption | None = signing_key
        self._public_key_store: dict[str, str] = public_key_store or {}
        self._sync_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Peer public-key management
    # ------------------------------------------------------------------

    def register_peer_public_key(self, node_id: str, public_pem: str) -> None:
        """Store a peer node's public key for signature verification."""
        self._public_key_store[node_id] = public_pem

    # ------------------------------------------------------------------
    # Message preparation
    # ------------------------------------------------------------------

    def prepare_sync_message(
        self, knowledge_delta: list[dict[str, Any]]
    ) -> bytes:
        """
        Serialise, sign (if a signing key is configured), and encrypt a
        sync message for transmission over an untrusted network.

        Returns encrypted bytes ready for transmission.
        """
        msg_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Canonical payload (deterministic key order for signing)
        canonical = json.dumps(
            {
                "message_id": msg_id,
                "sender_node_id": self._local.node_id,
                "knowledge_delta": knowledge_delta,
                "timestamp": timestamp,
            },
            sort_keys=True,
        ).encode()

        # RSA-PSS signature over the canonical payload
        if self._signing_key is not None:
            sig_bytes = self._signing_key.sign(canonical)
            signature = base64.b64encode(sig_bytes).decode()
        else:
            signature = ""

        # Add signature to the transmitted payload
        full_payload = json.dumps(
            {
                "message_id": msg_id,
                "sender_node_id": self._local.node_id,
                "knowledge_delta": knowledge_delta,
                "timestamp": timestamp,
                "signature": signature,
            }
        ).encode()

        encrypted = self._crypto.encrypt(full_payload, aad=self._local.node_id.encode())
        self._sync_log.append(
            {"event": "sent", "message_id": msg_id, "at": timestamp, "signed": bool(signature)}
        )
        return encrypted

    def parse_sync_message(self, encrypted: bytes) -> SyncMessage | None:
        """
        Decrypt and parse a received sync message.

        1. AES-GCM decryption (using the local node id as AAD – the sender
           must have used the same node id when encrypting, which is
           guaranteed when both sides share the same symmetric key).
        2. JSON parsing.
        3. Signature verification: if the sender's public key is in
           ``public_key_store`` and the message carries a non-empty
           signature, verification is required to pass.  If the sender's
           public key is unknown, the message is accepted without
           verification (logged as a warning).

        Returns None on decryption failure, JSON parse error, or failed
        signature verification.
        """
        try:
            payload_bytes = self._crypto.decrypt(encrypted, aad=self._local.node_id.encode())
        except Exception:
            return None

        try:
            data = json.loads(payload_bytes.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        sender_id: str = data.get("sender_node_id", "")
        signature_b64: str = data.get("signature", "")

        # Verify signature when we have the sender's public key
        if signature_b64 and sender_id in self._public_key_store:
            # Re-compute the canonical form (same as signing side)
            canonical = json.dumps(
                {
                    "message_id": data.get("message_id", ""),
                    "sender_node_id": sender_id,
                    "knowledge_delta": data.get("knowledge_delta", []),
                    "timestamp": data.get("timestamp", ""),
                },
                sort_keys=True,
            ).encode()
            sig_bytes = base64.b64decode(signature_b64)
            pub_pem = self._public_key_store[sender_id]
            if not AsymmetricEncryption.verify_with_public_pem(canonical, sig_bytes, pub_pem):
                self._sync_log.append(
                    {"event": "signature_verification_failed", "sender": sender_id}
                )
                return None

        msg = SyncMessage(
            message_id=data.get("message_id", ""),
            sender_node_id=sender_id,
            knowledge_delta=data.get("knowledge_delta", []),
            signature=signature_b64,
        )
        self._sync_log.append(
            {
                "event": "received",
                "message_id": msg.message_id,
                "signed": bool(signature_b64),
                "verified": bool(signature_b64 and sender_id in self._public_key_store),
            }
        )
        return msg

    @property
    def shared_key(self) -> bytes:
        return self._crypto.key

    @property
    def sync_log(self) -> list[dict[str, Any]]:
        return list(self._sync_log)
