"""
Tests for the Autonomy Layer.

Covers:
- SymmetricEncryption (AES-256-GCM)
- AsymmetricEncryption (RSA-2048 OAEP)
- NodeRegistry (registration, heartbeat, prune)
- NodeSynchronizer (prepare/parse sync messages)
- SelfOptimiser (cycle execution, action generation)
"""

import pytest

from thalos_prime.autonomy.encryption import AsymmetricEncryption, SymmetricEncryption
from thalos_prime.autonomy.node import NodeRegistry, NodeSynchronizer
from thalos_prime.autonomy.optimizer import SelfOptimiser
from thalos_prime.models import NodeInfo
from thalos_prime.sbi.organoid import OrganoidNetwork, OrganoidState


# ---------------------------------------------------------------------------
# SymmetricEncryption tests
# ---------------------------------------------------------------------------


class TestSymmetricEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        enc = SymmetricEncryption()
        plaintext = b"hello thalos"
        ciphertext = enc.encrypt(plaintext)
        assert enc.decrypt(ciphertext) == plaintext

    def test_encrypt_str_roundtrip(self):
        enc = SymmetricEncryption()
        original = "autonomous epistemic compiler"
        encrypted = enc.encrypt_str(original)
        assert enc.decrypt_str(encrypted) == original

    def test_different_nonce_each_call(self):
        enc = SymmetricEncryption()
        c1 = enc.encrypt(b"same message")
        c2 = enc.encrypt(b"same message")
        # Nonce is first 12 bytes
        assert c1[:12] != c2[:12]

    def test_ciphertext_differs_from_plaintext(self):
        enc = SymmetricEncryption()
        plaintext = b"secret data"
        ciphertext = enc.encrypt(plaintext)
        assert ciphertext != plaintext

    def test_aad_verification(self):
        from cryptography.exceptions import InvalidTag

        enc = SymmetricEncryption()
        ct = enc.encrypt(b"data", aad=b"valid-aad")
        with pytest.raises(InvalidTag):
            enc.decrypt(ct, aad=b"wrong-aad")

    def test_shared_key_property(self):
        enc = SymmetricEncryption()
        assert len(enc.key) == 32

    def test_provided_key_is_used(self):
        key = b"A" * 32
        enc = SymmetricEncryption(key=key)
        assert enc.key == key


# ---------------------------------------------------------------------------
# AsymmetricEncryption tests
# ---------------------------------------------------------------------------


class TestAsymmetricEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        enc = AsymmetricEncryption()
        plaintext = b"top secret message"
        ciphertext = enc.encrypt(plaintext)
        assert ciphertext != plaintext  # ciphertext is different from plaintext
        assert enc.decrypt(ciphertext) == plaintext

    def test_public_key_pem_format(self):
        enc = AsymmetricEncryption()
        pem = enc.public_key_pem()
        assert pem.startswith("-----BEGIN PUBLIC KEY-----")

    def test_private_key_pem_roundtrip(self):
        enc = AsymmetricEncryption()
        pem = enc.private_key_pem()
        restored = AsymmetricEncryption.from_private_pem(pem)
        # Encrypting with original and decrypting with restored should work
        ct = enc.encrypt(b"test")
        assert restored.decrypt(ct) == b"test"

    def test_encrypt_with_public_pem(self):
        enc = AsymmetricEncryption()
        pub_pem = enc.public_key_pem()
        ct = AsymmetricEncryption.encrypt_with_public_pem(b"hello", pub_pem)
        assert enc.decrypt(ct) == b"hello"


# ---------------------------------------------------------------------------
# NodeRegistry tests
# ---------------------------------------------------------------------------


def make_node(host: str = "localhost", port: int = 9000) -> NodeInfo:
    return NodeInfo(host=host, port=port, public_key_pem="", status="active")


class TestNodeRegistry:
    def test_register_and_list(self):
        local = make_node(port=8080)
        registry = NodeRegistry(local_node=local)
        peer = make_node(port=9001)
        registry.register(peer)
        assert registry.peer_count() == 1
        assert peer in registry.all_peers

    def test_register_local_node_ignored(self):
        local = make_node(port=8080)
        registry = NodeRegistry(local_node=local)
        registry.register(local)  # should be ignored
        assert registry.peer_count() == 0

    def test_deregister(self):
        local = make_node(port=8080)
        registry = NodeRegistry(local_node=local)
        peer = make_node(port=9002)
        registry.register(peer)
        result = registry.deregister(peer.node_id)
        assert result is True
        assert registry.peer_count() == 0

    def test_deregister_unknown_returns_false(self):
        local = make_node()
        registry = NodeRegistry(local_node=local)
        assert registry.deregister("nonexistent") is False

    def test_heartbeat_updates_last_seen(self):
        local = make_node()
        registry = NodeRegistry(local_node=local)
        peer = make_node(port=9003)
        registry.register(peer)
        before = peer.last_seen
        registry.heartbeat(peer.node_id)
        assert peer.last_seen >= before

    def test_active_peers_excludes_offline(self):
        local = make_node()
        registry = NodeRegistry(local_node=local)
        peer = make_node(port=9004)
        peer.status = "offline"
        registry.register(peer)
        assert len(registry.active_peers) == 0


# ---------------------------------------------------------------------------
# NodeSynchronizer tests
# ---------------------------------------------------------------------------


class TestNodeSynchronizer:
    def test_prepare_and_parse_roundtrip(self):
        local = make_node()
        sync = NodeSynchronizer(local_node=local)
        delta = [{"object_id": "obj-1", "content": "water is H2O", "confidence": 0.9}]
        encrypted = sync.prepare_sync_message(delta)
        msg = sync.parse_sync_message(encrypted)
        assert msg is not None
        assert msg.sender_node_id == local.node_id
        assert msg.knowledge_delta == delta

    def test_parse_invalid_bytes_returns_none(self):
        local = make_node()
        sync = NodeSynchronizer(local_node=local)
        result = sync.parse_sync_message(b"not-valid-encrypted-data")
        assert result is None

    def test_sync_log_tracks_events(self):
        local = make_node()
        sync = NodeSynchronizer(local_node=local)
        delta = [{"object_id": "x", "content": "test", "confidence": 0.5}]
        encrypted = sync.prepare_sync_message(delta)
        sync.parse_sync_message(encrypted)
        log = sync.sync_log
        events = [entry["event"] for entry in log]
        assert "sent" in events
        assert "received" in events

    def test_shared_key_is_bytes(self):
        local = make_node()
        sync = NodeSynchronizer(local_node=local)
        assert isinstance(sync.shared_key, bytes)
        assert len(sync.shared_key) == 32


# ---------------------------------------------------------------------------
# SelfOptimiser tests
# ---------------------------------------------------------------------------


class TestSelfOptimiser:
    def test_run_cycle_returns_list(self):
        network = OrganoidNetwork(initial_clusters=2)
        optimiser = SelfOptimiser(network=network)
        actions = optimiser.run_cycle()
        assert isinstance(actions, list)

    def test_high_utilisation_triggers_growth(self):
        network = OrganoidNetwork(initial_clusters=2)
        # Saturate all clusters
        for cluster in network.clusters:
            cluster.activate()
            while cluster.accept_task():
                pass

        optimiser = SelfOptimiser(network=network)
        optimiser.HIGH_UTILISATION = 0.0  # always trigger growth
        actions = optimiser.run_cycle()
        growth_actions = [a for a in actions if a.action_type == "grow_cluster"]
        assert len(growth_actions) >= 1

    def test_degraded_cluster_is_isolated(self):
        network = OrganoidNetwork(initial_clusters=2)
        # Force one cluster into degraded state
        cluster = network.clusters[0]
        cluster.health = 0.0
        cluster.state = OrganoidState.DEGRADED

        optimiser = SelfOptimiser(network=network)
        actions = optimiser.run_cycle()
        isolate_actions = [a for a in actions if a.action_type == "isolate"]
        assert len(isolate_actions) >= 1

    def test_report_structure(self):
        network = OrganoidNetwork(initial_clusters=2)
        optimiser = SelfOptimiser(network=network)
        report = optimiser.report()
        assert "network" in report
        assert "actions_taken" in report
        assert "last_actions" in report


# ---------------------------------------------------------------------------
# RSA-PSS signing tests (new)
# ---------------------------------------------------------------------------


class TestAsymmetricSigning:
    def test_sign_and_verify_roundtrip(self):
        key = AsymmetricEncryption()
        msg = b"Thalos Prime sync payload"
        sig = key.sign(msg)
        assert key.verify(msg, sig) is True

    def test_verify_tampered_message_fails(self):
        key = AsymmetricEncryption()
        msg = b"original message"
        sig = key.sign(msg)
        assert key.verify(b"tampered message", sig) is False

    def test_verify_with_public_pem(self):
        key = AsymmetricEncryption()
        msg = b"inter-node payload"
        sig = key.sign(msg)
        pub_pem = key.public_key_pem()
        assert AsymmetricEncryption.verify_with_public_pem(msg, sig, pub_pem) is True

    def test_verify_with_wrong_key_fails(self):
        key1 = AsymmetricEncryption()
        key2 = AsymmetricEncryption()
        msg = b"signed with key1"
        sig = key1.sign(msg)
        # Verify with key2's public key must fail
        assert AsymmetricEncryption.verify_with_public_pem(msg, sig, key2.public_key_pem()) is False

    def test_signature_is_bytes(self):
        key = AsymmetricEncryption()
        sig = key.sign(b"data")
        assert isinstance(sig, bytes)
        assert len(sig) == 256  # RSA-2048 signature = 256 bytes


class TestNodeSynchronizerWithSigning:
    def test_signed_message_carries_signature(self):
        key = AsymmetricEncryption()
        local = make_node()
        sync = NodeSynchronizer(local_node=local, signing_key=key)
        encrypted = sync.prepare_sync_message([{"object_id": "x", "content": "test"}])
        msg = sync.parse_sync_message(encrypted)
        assert msg is not None
        assert msg.signature != ""

    def test_unsigned_message_is_accepted(self):
        """Nodes without a signing key still produce valid (unsigned) messages."""
        local = make_node()
        sync = NodeSynchronizer(local_node=local, signing_key=None)
        encrypted = sync.prepare_sync_message([{"object_id": "y", "content": "data"}])
        msg = sync.parse_sync_message(encrypted)
        assert msg is not None
        assert msg.signature == ""

    def test_valid_signature_verified_when_public_key_known(self):
        signing_key = AsymmetricEncryption()
        local = make_node()
        # Sender signs
        sender = NodeSynchronizer(local_node=local, signing_key=signing_key)
        encrypted = sender.prepare_sync_message([{"object_id": "z", "content": "fact"}])
        # Receiver knows the sender's public key
        receiver = NodeSynchronizer(
            local_node=local,
            shared_key=sender.shared_key,
            public_key_store={local.node_id: signing_key.public_key_pem()},
        )
        msg = receiver.parse_sync_message(encrypted)
        assert msg is not None

    def test_tampered_signature_rejected(self):
        import base64
        import json

        signing_key = AsymmetricEncryption()
        local = make_node()
        sender = NodeSynchronizer(local_node=local, signing_key=signing_key)
        encrypted = sender.prepare_sync_message([{"object_id": "q", "content": "claim"}])

        # Decrypt, swap signature to garbage, re-encrypt
        from thalos_prime.autonomy.encryption import SymmetricEncryption
        sym = SymmetricEncryption(key=sender.shared_key)
        raw = sym.decrypt(encrypted, aad=local.node_id.encode())
        data = json.loads(raw)
        data["signature"] = base64.b64encode(b"fakesig" * 37).decode()
        tampered_raw = json.dumps(data).encode()
        tampered_encrypted = sym.encrypt(tampered_raw, aad=local.node_id.encode())

        receiver = NodeSynchronizer(
            local_node=local,
            shared_key=sender.shared_key,
            public_key_store={local.node_id: signing_key.public_key_pem()},
        )
        msg = receiver.parse_sync_message(tampered_encrypted)
        assert msg is None  # must be rejected

    def test_sync_log_records_signed_flag(self):
        signing_key = AsymmetricEncryption()
        local = make_node()
        sync = NodeSynchronizer(local_node=local, signing_key=signing_key)
        sync.prepare_sync_message([])
        log = sync.sync_log
        assert log[0]["signed"] is True


class TestNodeRegistryGetPeer:
    def test_get_peer_returns_node(self):
        local = make_node()
        registry = NodeRegistry(local_node=local)
        peer = make_node(port=9099)
        registry.register(peer)
        found = registry.get_peer(peer.node_id)
        assert found is peer

    def test_get_peer_unknown_returns_none(self):
        local = make_node()
        registry = NodeRegistry(local_node=local)
        assert registry.get_peer("nonexistent-id") is None
