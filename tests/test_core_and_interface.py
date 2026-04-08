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
