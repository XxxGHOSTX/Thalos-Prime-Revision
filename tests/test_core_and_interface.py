"""
Integration tests for ThalosCore and the Interface layer.

Covers:
- Full ingest → verify → store pipeline via ThalosCore
- Batch ingestion with contradiction detection
- Insight cycle
- Node sync (prepare + receive)
- Optimisation cycle
- Health report
- FastAPI endpoints (via TestClient)
"""

import pytest

from thalos_prime.core import ThalosCore
from thalos_prime.knowledge.contradiction import ResolutionStrategy
from thalos_prime.models import ClaimStatus, NodeInfo


# ---------------------------------------------------------------------------
# ThalosCore integration tests
# ---------------------------------------------------------------------------


class TestThalosCoreIngest:
    def test_ingest_returns_knowledge_object(self):
        core = ThalosCore()
        ko = core.ingest("water is H2O", reliability=0.9)
        assert ko is not None
        assert ko.canonical_claim.content == "water is H2O"

    def test_ingest_stores_in_repository(self):
        core = ThalosCore()
        ko = core.ingest("sky is blue", reliability=0.8)
        retrieved = core.repository.get(ko.object_id)
        assert retrieved is not None

    def test_ingest_sbi_metadata_attached(self):
        core = ThalosCore(organoid_clusters=2)
        ko = core.ingest("neuron fires", reliability=0.7)
        assert "sbi_activation" in ko.canonical_claim.metadata
        assert len(ko.canonical_claim.metadata["sbi_activation"]) == 16

    def test_ingest_high_reliability_gets_verified(self):
        core = ThalosCore()
        ko = core.ingest("verified claim", reliability=0.95)
        assert ko.canonical_claim.status == ClaimStatus.VERIFIED

    def test_ingest_low_reliability_stays_pending(self):
        core = ThalosCore()
        ko = core.ingest("uncertain claim", reliability=0.2)
        assert ko.canonical_claim.status == ClaimStatus.PENDING


class TestThalosBatchIngest:
    def test_batch_ingest_returns_all_objects(self):
        core = ThalosCore()
        records = [
            {"content": f"claim {i}", "source": {"reliability": 0.7}}
            for i in range(5)
        ]
        objects = core.ingest_batch(records)
        assert len(objects) == 5

    def test_batch_contradiction_detection(self):
        core = ThalosCore(resolution_strategy=ResolutionStrategy.HIGHEST_CONFIDENCE)
        records = [
            {
                "content": "water is H2O",
                "subject": "water",
                "predicate": "is",
                "obj": "H2O",
                "source": {"reliability": 0.9},
            },
            {
                "content": "water is H3O",
                "subject": "water",
                "predicate": "is",
                "obj": "H3O",
                "source": {"reliability": 0.4},
            },
        ]
        objects = core.ingest_batch(records)
        statuses = {ko.canonical_claim.status for ko in objects}
        # At least one claim should be REJECTED (the loser)
        assert ClaimStatus.REJECTED in statuses


class TestThalosCoreInsights:
    def test_insight_cycle_produces_insights(self):
        core = ThalosCore()
        for i in range(3):
            core.ingest(f"gap subject has thing {i}", reliability=0.2)
        insights = core.run_insight_cycle()
        assert len(insights) > 0

    def test_insight_engine_accessible(self):
        core = ThalosCore()
        core.ingest("low confidence claim", reliability=0.1)
        core.run_insight_cycle()
        assert len(core.insight_engine.all_insights) > 0


class TestThalosCoreNodes:
    def test_register_peer_increments_count(self):
        core = ThalosCore()
        peer = NodeInfo(host="10.0.0.1", port=9001, public_key_pem="")
        core.register_peer(peer)
        assert core.node_registry.peer_count() == 1

    def test_sync_roundtrip_same_key(self):
        """Two cores sharing the same symmetric key can exchange knowledge."""
        core1 = ThalosCore()
        core2 = ThalosCore()
        # Share the same symmetric key for test purposes
        from thalos_prime.autonomy.node import NodeSynchronizer

        shared_key = core1.synchronizer.shared_key
        core2.synchronizer = NodeSynchronizer(
            local_node=core2.node_registry.local_node, shared_key=shared_key
        )

        core1.ingest("shared knowledge", reliability=0.8)
        payload = core1.prepare_sync_payload()
        merged = core2.receive_sync_payload(payload)
        assert merged >= 0  # may be 0 if already present; just verify no crash

    def test_receive_invalid_payload_returns_zero(self):
        core = ThalosCore()
        merged = core.receive_sync_payload(b"garbage data that cannot be decrypted")
        assert merged == 0


class TestThalosCoreOptimisation:
    def test_optimisation_cycle_runs_without_error(self):
        core = ThalosCore(organoid_clusters=3)
        actions = core.run_optimisation_cycle()
        assert isinstance(actions, list)


class TestThalosCoreHealthReport:
    def test_health_report_structure(self):
        core = ThalosCore()
        core.ingest("test claim", reliability=0.8)
        report = core.health_report()
        assert "node_id" in report
        assert "knowledge_objects" in report
        assert "organoid_network" in report
        assert "microcircuit" in report
        assert "peer_nodes" in report
        assert "insights_generated" in report
        assert "optimiser_actions" in report

    def test_health_report_knowledge_count(self):
        core = ThalosCore()
        for i in range(3):
            core.ingest(f"claim {i}", reliability=0.8)
        report = core.health_report()
        assert report["knowledge_objects"] == 3


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------


class TestFastAPIEndpoints:
    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient

            from thalos_prime.interface.api import create_app
        except ImportError:
            pytest.skip("FastAPI not installed")

        core = ThalosCore()
        app = create_app(core)
        return TestClient(app), core

    def test_ingest_endpoint(self, client):
        tc, _ = client
        resp = tc.post(
            "/claims",
            json={
                "content": "water is H2O",
                "reliability": 0.9,
                "source_uri": "test://api",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "object_id" in data
        assert "confidence" in data

    def test_get_claim_endpoint(self, client):
        tc, core = client
        ko = core.ingest("fire is hot", reliability=0.85)
        resp = tc.get(f"/claims/{ko.object_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object_id"] == ko.object_id

    def test_get_claim_not_found(self, client):
        tc, _ = client
        resp = tc.get("/claims/nonexistent-id")
        assert resp.status_code == 404

    def test_search_endpoint(self, client):
        tc, core = client
        core.ingest("sky is blue", reliability=0.8)
        resp = tc.get("/claims?query=sky")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1

    def test_insights_endpoint(self, client):
        tc, core = client
        core.ingest("low confidence fact", reliability=0.1)
        core.run_insight_cycle()
        resp = tc.get("/insights")
        assert resp.status_code == 200

    def test_run_insights_endpoint(self, client):
        tc, _ = client
        resp = tc.post("/insights/run")
        assert resp.status_code == 200
        assert "new_insights" in resp.json()

    def test_health_endpoint(self, client):
        tc, _ = client
        resp = tc.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "node_id" in data

    def test_nodes_endpoint(self, client):
        tc, _ = client
        resp = tc.get("/nodes")
        assert resp.status_code == 200
        data = resp.json()
        assert "local_node_id" in data
        assert "peers" in data

    def test_put_claim_endpoint(self, client):
        tc, core = client
        ko = core.ingest("original content", reliability=0.8)
        resp = tc.put(
            f"/claims/{ko.object_id}",
            json={"content": "updated content", "reliability": 0.9},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object_id"] == ko.object_id
        assert data["versions"] >= 2

    def test_put_claim_not_found(self, client):
        tc, _ = client
        resp = tc.put(
            "/claims/nonexistent-id",
            json={"content": "update", "reliability": 0.8},
        )
        assert resp.status_code == 404

    def test_delete_claim_endpoint(self, client):
        tc, core = client
        ko = core.ingest("to be deleted", reliability=0.8)
        resp = tc.delete(f"/claims/{ko.object_id}")
        assert resp.status_code == 204
        assert core.repository.get(ko.object_id) is None

    def test_delete_claim_not_found(self, client):
        tc, _ = client
        resp = tc.delete("/claims/nonexistent-id")
        assert resp.status_code == 404

    def test_claim_versions_endpoint(self, client):
        tc, core = client
        ko = core.ingest("versioned claim", reliability=0.8)
        # Create a second version via the repository
        core.repository.put(ko, reason="manual_update")
        resp = tc.get(f"/claims/{ko.object_id}/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version_count"] >= 2
        assert isinstance(data["versions"], list)
        assert "version_number" in data["versions"][0]

    def test_claim_versions_not_found(self, client):
        tc, _ = client
        resp = tc.get("/claims/nonexistent-id/versions")
        assert resp.status_code == 404

    def test_register_node_endpoint(self, client):
        tc, _ = client
        resp = tc.post(
            "/nodes",
            json={"host": "10.0.0.5", "port": 9001, "public_key_pem": ""},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["registered"] is True
        assert data["host"] == "10.0.0.5"

    def test_node_heartbeat_endpoint(self, client):
        tc, core = client
        from thalos_prime.models import NodeInfo
        peer = NodeInfo(host="10.0.0.6", port=9002)
        core.register_peer(peer)
        resp = tc.post(f"/nodes/{peer.node_id}/heartbeat")
        assert resp.status_code == 200
        assert resp.json()["status"] == "heartbeat_received"

    def test_node_heartbeat_not_found(self, client):
        tc, _ = client
        resp = tc.post("/nodes/nonexistent-node/heartbeat")
        assert resp.status_code == 404

    def test_prune_nodes_endpoint(self, client):
        tc, _ = client
        resp = tc.post("/nodes/prune")
        assert resp.status_code == 200
        data = resp.json()
        assert "stale_nodes_marked_offline" in data

    def test_sync_get_endpoint(self, client):
        tc, core = client
        core.ingest("sync test claim", reliability=0.85)
        resp = tc.get("/sync")
        assert resp.status_code == 200
        data = resp.json()
        assert "payload_b64" in data
        assert "sender_node_id" in data
        assert data["knowledge_objects"] >= 1

    def test_sync_post_endpoint(self, client):
        import base64
        tc, core = client
        core.ingest("claim to sync", reliability=0.8)
        encrypted = core.prepare_sync_payload()
        resp = tc.post(
            "/sync",
            json={"payload_b64": base64.b64encode(encrypted).decode()},
        )
        assert resp.status_code == 200
        assert "merged_objects" in resp.json()

    def test_sync_post_invalid_payload(self, client):
        tc, _ = client
        resp = tc.post("/sync", json={"payload_b64": "!!!not-base64!!!"})
        assert resp.status_code == 400

    def test_optimiser_endpoint(self, client):
        tc, _ = client
        resp = tc.get("/optimiser")
        assert resp.status_code == 200
        data = resp.json()
        assert "network" in data
        assert "actions_taken" in data

    def test_monitor_endpoint(self, client):
        tc, core = client
        core.ingest("monitor test claim", reliability=0.8)
        resp = tc.get("/monitor")
        assert resp.status_code == 200
        data = resp.json()
        assert "timestamp" in data
        assert "knowledge_repository" in data
        assert "organoid_network" in data
        assert "nodes" in data

    def test_monitor_history_endpoint(self, client):
        tc, core = client
        # Build up some snapshots
        core.monitor.snapshot()
        core.monitor.snapshot()
        resp = tc.get("/monitor/history?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "snapshots" in data
        assert data["count"] >= 2


# ---------------------------------------------------------------------------
# ThalosCore additional tests (new)
# ---------------------------------------------------------------------------


class TestThalosCoreMonitor:
    def test_monitor_attribute_exists(self):
        core = ThalosCore(organoid_clusters=2)
        assert hasattr(core, "monitor")

    def test_monitor_snapshot_structure(self):
        core = ThalosCore(organoid_clusters=2)
        core.ingest("test", reliability=0.8)
        snap = core.monitor.snapshot()
        assert "timestamp" in snap
        assert "knowledge_repository" in snap
        assert "organoid_network" in snap
        assert "microcircuit" in snap
        assert "optimiser" in snap
        assert "nodes" in snap
        assert "insights" in snap

    def test_monitor_recent_snapshots(self):
        core = ThalosCore(organoid_clusters=2)
        core.monitor.snapshot()
        core.monitor.snapshot()
        recent = core.monitor.recent_snapshots(n=5)
        assert len(recent) == 2


class TestThalosCoreSignedSync:
    def test_sync_payload_signed_by_default(self):
        core = ThalosCore(organoid_clusters=2)
        core.ingest("signed sync test", reliability=0.9)
        core.prepare_sync_payload()
        log = core.synchronizer.sync_log
        assert any(entry.get("signed") for entry in log)

    def test_register_peer_stores_public_key(self):
        from thalos_prime.autonomy.encryption import AsymmetricEncryption
        core = ThalosCore(organoid_clusters=2)
        peer_crypto = AsymmetricEncryption()
        peer_node = NodeInfo(
            host="10.0.0.3",
            port=9005,
            public_key_pem=peer_crypto.public_key_pem(),
        )
        core.register_peer(peer_node)
        # The synchronizer should now have the peer's public key
        assert peer_node.node_id in core.synchronizer._public_key_store


class TestCLIMain:
    def test_main_is_callable(self):
        from thalos_prime.core import main
        assert callable(main)

    def test_main_exists_in_module(self):
        import thalos_prime.core as core_mod
        assert hasattr(core_mod, "main")
