"""
Distributed Node Management
============================

Manages a peer-to-peer network of Thalos Prime nodes that operate
independently and synchronise knowledge periodically.

Features:
- Node registration and discovery
- Heartbeat / liveness tracking
- Encrypted knowledge sync messages
- Autonomous failure detection and recovery
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from thalos_prime.autonomy.encryption import SymmetricEncryption
from thalos_prime.models import NodeInfo


@dataclass
class SyncMessage:
    """Payload exchanged between nodes during a synchronisation round."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_node_id: str = ""
    knowledge_delta: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature: str = ""  # placeholder for digital signature


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
        """Remove nodes that have not sent a heartbeat within the timeout."""
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


class NodeSynchronizer:
    """
    Handles the preparation, encryption, and parsing of synchronisation
    messages between nodes.
    """

    def __init__(self, local_node: NodeInfo, shared_key: bytes | None = None) -> None:
        self._local = local_node
        self._crypto = SymmetricEncryption(key=shared_key)
        self._sync_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Message preparation
    # ------------------------------------------------------------------

    def prepare_sync_message(
        self, knowledge_delta: list[dict[str, Any]]
    ) -> bytes:
        """
        Serialise and encrypt a sync message for broadcast.

        Returns encrypted bytes suitable for transmission over an
        untrusted network.
        """
        msg = SyncMessage(
            sender_node_id=self._local.node_id,
            knowledge_delta=knowledge_delta,
        )
        payload = json.dumps(
            {
                "message_id": msg.message_id,
                "sender_node_id": msg.sender_node_id,
                "knowledge_delta": msg.knowledge_delta,
                "timestamp": msg.timestamp.isoformat(),
            }
        ).encode()
        encrypted = self._crypto.encrypt(payload, aad=self._local.node_id.encode())
        self._sync_log.append(
            {"event": "sent", "message_id": msg.message_id, "at": msg.timestamp.isoformat()}
        )
        return encrypted

    def parse_sync_message(self, encrypted: bytes) -> SyncMessage | None:
        """
        Decrypt and parse a received sync message.

        Returns None if decryption or parsing fails.
        """
        try:
            payload = self._crypto.decrypt(encrypted, aad=self._local.node_id.encode())
            data = json.loads(payload.decode())
            msg = SyncMessage(
                message_id=data["message_id"],
                sender_node_id=data["sender_node_id"],
                knowledge_delta=data.get("knowledge_delta", []),
            )
            self._sync_log.append(
                {"event": "received", "message_id": msg.message_id}
            )
            return msg
        except Exception:
            return None

    @property
    def shared_key(self) -> bytes:
        return self._crypto.key

    @property
    def sync_log(self) -> list[dict[str, Any]]:
        return list(self._sync_log)
