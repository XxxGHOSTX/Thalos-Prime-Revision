"""
Tests for the Knowledge Processing Layer.

Covers:
- Claim models
- Ingestion pipeline
- Verification engine
- Contradiction resolution
- Knowledge repository (CRUD + versioning + search)
- Insight generation
"""

import pytest

from thalos_prime.knowledge.contradiction import ContradictionResolver, ResolutionStrategy
from thalos_prime.knowledge.ingestion import ClaimNormalizer, IngestionPipeline
from thalos_prime.knowledge.insight import InsightEngine
from thalos_prime.knowledge.verification import VerificationEngine
from thalos_prime.models import (
    Claim,
    ClaimStatus,
    KnowledgeObject,
    SourceRecord,
    SourceType,
)
from thalos_prime.storage.repository import KnowledgeRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_source(reliability: float = 0.8, uri: str = "test://source") -> SourceRecord:
    return SourceRecord(
        source_id="src-1",
        source_type=SourceType.TEXT,
        uri=uri,
        reliability_score=reliability,
    )


def make_claim(
    content: str = "water is H2O",
    subject: str = "water",
    predicate: str = "is",
    obj: str = "H2O",
    confidence: float = 0.8,
    status: ClaimStatus = ClaimStatus.PENDING,
) -> Claim:
    return Claim(
        content=content,
        subject=subject,
        predicate=predicate,
        obj=obj,
        confidence=confidence,
        status=status,
        sources=[make_source()],
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestClaimModel:
    def test_default_status_is_pending(self):
        claim = Claim(content="test")
        assert claim.status == ClaimStatus.PENDING

    def test_confidence_validation_lower(self):
        with pytest.raises(ValueError):
            Claim(confidence=-0.1)

    def test_confidence_validation_upper(self):
        with pytest.raises(ValueError):
            Claim(confidence=1.1)

    def test_avg_source_reliability_empty(self):
        claim = Claim(content="test")
        assert claim.avg_source_reliability == 0.0

    def test_avg_source_reliability_single(self):
        claim = make_claim()
        assert claim.avg_source_reliability == pytest.approx(0.8)

    def test_avg_source_reliability_multiple(self):
        claim = Claim(
            content="test",
            sources=[make_source(0.6), make_source(0.4)],
        )
        assert claim.avg_source_reliability == pytest.approx(0.5)

    def test_touch_updates_timestamp(self):
        claim = make_claim()
        before = claim.updated_at
        claim.touch()
        assert claim.updated_at >= before

    def test_unique_claim_ids(self):
        ids = {Claim().claim_id for _ in range(100)}
        assert len(ids) == 100


class TestSourceRecord:
    def test_invalid_reliability_low(self):
        with pytest.raises(ValueError):
            SourceRecord(
                source_id="x",
                source_type=SourceType.TEXT,
                uri="x",
                reliability_score=-0.1,
            )

    def test_invalid_reliability_high(self):
        with pytest.raises(ValueError):
            SourceRecord(
                source_id="x",
                source_type=SourceType.TEXT,
                uri="x",
                reliability_score=1.5,
            )


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------


class TestClaimNormalizer:
    def test_spo_extraction(self):
        normalizer = ClaimNormalizer()
        source = make_source()
        claim = normalizer.normalise("water is H2O", source)
        assert claim.subject == "water"
        assert claim.predicate == "is"
        assert claim.obj == "H2O"

    def test_fallback_no_spo(self):
        normalizer = ClaimNormalizer()
        source = make_source()
        claim = normalizer.normalise("something complicated without pattern", source)
        assert claim.subject != ""
        assert claim.predicate == ""
        assert claim.obj == ""

    def test_normalise_dict(self):
        normalizer = ClaimNormalizer()
        source = make_source()
        claim = normalizer.normalise_dict(
            {"content": "sky is blue", "subject": "sky", "predicate": "is", "obj": "blue"},
            source,
        )
        assert claim.subject == "sky"
        assert claim.obj == "blue"


class TestIngestionPipeline:
    def test_ingest_text_queues_claim(self):
        pipeline = IngestionPipeline()
        claim = pipeline.ingest_text("water is H2O", "test://src")
        assert pipeline.queue_size() == 1
        assert claim.content == "water is H2O"

    def test_drain_clears_queue(self):
        pipeline = IngestionPipeline()
        pipeline.ingest_text("claim 1", "src://a")
        pipeline.ingest_text("claim 2", "src://b")
        drained = pipeline.drain()
        assert len(drained) == 2
        assert pipeline.queue_size() == 0

    def test_ingest_batch(self):
        pipeline = IngestionPipeline()
        records = [
            {"content": "A is B", "source": {"uri": "batch://1", "reliability": 0.7}},
            {"content": "C has D", "source": {"uri": "batch://2", "reliability": 0.8}},
        ]
        claims = pipeline.ingest_batch(records)
        assert len(claims) == 2

    def test_claim_inherits_source_reliability(self):
        pipeline = IngestionPipeline()
        claim = pipeline.ingest_text("test", "src://x", reliability=0.9)
        assert claim.confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Verification tests
# ---------------------------------------------------------------------------


class TestVerificationEngine:
    def test_high_reliability_source_gets_verified(self):
        engine = VerificationEngine()
        claim = make_claim(confidence=0.9)
        claim.sources[0].reliability_score = 0.9
        result = engine.verify(claim)
        assert result.status == ClaimStatus.VERIFIED

    def test_low_reliability_stays_pending(self):
        engine = VerificationEngine()
        claim = make_claim(confidence=0.3)
        claim.sources[0].reliability_score = 0.3
        result = engine.verify(claim)
        assert result.status == ClaimStatus.PENDING

    def test_corroboration_boosts_confidence(self):
        engine = VerificationEngine()
        # Pre-register a highly confident matching KO
        existing_claim = make_claim(content="water is H2O", confidence=0.95)
        existing_claim.status = ClaimStatus.VERIFIED
        ko = KnowledgeObject(canonical_claim=existing_claim)
        engine.register_knowledge_object(ko)

        new_claim = make_claim(content="water is H2O", confidence=0.5)
        new_claim.sources[0].reliability_score = 0.5
        result = engine.verify(new_claim)
        assert result.confidence > 0.5

    def test_verify_batch(self):
        engine = VerificationEngine()
        claims = [make_claim(confidence=0.8) for _ in range(5)]
        results = engine.verify_batch(claims)
        assert len(results) == 5
        assert all(r.status == ClaimStatus.VERIFIED for r in results)


# ---------------------------------------------------------------------------
# Contradiction resolution tests
# ---------------------------------------------------------------------------


class TestContradictionResolver:
    def test_detects_contradiction(self):
        resolver = ContradictionResolver()
        claims = [
            make_claim("water is H2O", subject="water", predicate="is", obj="H2O"),
            make_claim("water is H3O", subject="water", predicate="is", obj="H3O"),
        ]
        records = resolver.detect(claims)
        assert len(records) == 1
        assert records[0].resolved is False

    def test_no_contradiction_same_obj(self):
        resolver = ContradictionResolver()
        claims = [
            make_claim("water is H2O", subject="water", predicate="is", obj="H2O"),
            make_claim("water is H2O", subject="water", predicate="is", obj="H2O"),
        ]
        records = resolver.detect(claims)
        assert len(records) == 0

    def test_resolution_by_confidence(self):
        resolver = ContradictionResolver(strategy=ResolutionStrategy.HIGHEST_CONFIDENCE)
        claim_a = make_claim(confidence=0.9, obj="H2O")
        claim_b = make_claim(confidence=0.4, obj="H3O")
        record = resolver.detect([claim_a, claim_b])[0]
        winner, resolved_record = resolver.resolve(claim_a, claim_b, record)
        assert winner.claim_id == claim_a.claim_id
        assert resolved_record.resolved is True
        assert claim_b.status == ClaimStatus.REJECTED

    def test_resolution_by_source_weight(self):
        resolver = ContradictionResolver(strategy=ResolutionStrategy.SOURCE_WEIGHTED)
        claim_a = make_claim(confidence=0.5, obj="H2O")
        claim_a.sources[0].reliability_score = 0.9
        claim_b = make_claim(confidence=0.9, obj="H3O")
        claim_b.sources[0].reliability_score = 0.2
        record = resolver.detect([claim_a, claim_b])[0]
        winner, _ = resolver.resolve(claim_a, claim_b, record)
        # claim_a: 0.9 * 0.5 = 0.45, claim_b: 0.2 * 0.9 = 0.18 → claim_a wins
        assert winner.claim_id == claim_a.claim_id

    def test_retain_all_strategy(self):
        resolver = ContradictionResolver(strategy=ResolutionStrategy.RETAIN_ALL)
        claim_a = make_claim(obj="H2O")
        claim_b = make_claim(obj="H3O")
        record = resolver.detect([claim_a, claim_b])[0]
        winner, _ = resolver.resolve(claim_a, claim_b, record)
        # Both claims should NOT be rejected under RETAIN_ALL
        assert claim_b.status != ClaimStatus.REJECTED


# ---------------------------------------------------------------------------
# Repository tests
# ---------------------------------------------------------------------------


class TestKnowledgeRepository:
    def test_put_and_get(self):
        repo = KnowledgeRepository()
        ko = KnowledgeObject(canonical_claim=make_claim())
        repo.put(ko)
        retrieved = repo.get(ko.object_id)
        assert retrieved is not None
        assert retrieved.object_id == ko.object_id

    def test_versions_increment_on_update(self):
        repo = KnowledgeRepository()
        ko = KnowledgeObject(canonical_claim=make_claim())
        repo.put(ko, reason="v1")
        ko.canonical_claim.content = "updated content"
        repo.put(ko, reason="v2")
        assert len(ko.versions) == 2

    def test_delete(self):
        repo = KnowledgeRepository()
        ko = KnowledgeObject(canonical_claim=make_claim())
        repo.put(ko)
        assert repo.delete(ko.object_id) is True
        assert repo.get(ko.object_id) is None

    def test_search_returns_relevant(self):
        repo = KnowledgeRepository()
        ko1 = KnowledgeObject(canonical_claim=make_claim(content="water is H2O"))
        ko2 = KnowledgeObject(canonical_claim=make_claim(content="fire is hot"))
        repo.put(ko1)
        repo.put(ko2)
        results = repo.search("water")
        assert len(results) == 1
        assert results[0].object_id == ko1.object_id

    def test_search_empty_query_returns_all(self):
        repo = KnowledgeRepository()
        for i in range(5):
            ko = KnowledgeObject(canonical_claim=make_claim(content=f"claim {i}"))
            repo.put(ko)
        results = repo.search("")
        assert len(results) == 5

    def test_low_confidence_filter(self):
        repo = KnowledgeRepository()
        ko_low = KnowledgeObject(canonical_claim=make_claim(confidence=0.3))
        ko_high = KnowledgeObject(canonical_claim=make_claim(confidence=0.9))
        repo.put(ko_low)
        repo.put(ko_high)
        low = repo.low_confidence(threshold=0.5)
        assert ko_low.object_id in [k.object_id for k in low]
        assert ko_high.object_id not in [k.object_id for k in low]

    def test_get_version(self):
        repo = KnowledgeRepository()
        ko = KnowledgeObject(canonical_claim=make_claim())
        repo.put(ko, reason="first")
        v = repo.get_version(ko.object_id, 1)
        assert v is not None
        assert v.version_number == 1

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "repo.json")
        repo = KnowledgeRepository(persist_path=path)
        ko = KnowledgeObject(canonical_claim=make_claim(content="persistent claim"))
        repo.put(ko)

        # Reload
        repo2 = KnowledgeRepository(persist_path=path)
        retrieved = repo2.get(ko.object_id)
        assert retrieved is not None
        assert retrieved.canonical_claim.content == "persistent claim"


# ---------------------------------------------------------------------------
# Insight engine tests
# ---------------------------------------------------------------------------


class TestInsightEngine:
    def test_low_confidence_insight(self):
        engine = InsightEngine()
        ko = KnowledgeObject(canonical_claim=make_claim(confidence=0.2))
        insights = engine.analyse([ko])
        types = [i.insight_type for i in insights]
        assert "low_confidence" in types

    def test_gap_insight_triggered_for_many_pending(self):
        engine = InsightEngine()
        claims = [
            make_claim(subject="unknown_subject", status=ClaimStatus.PENDING)
            for _ in range(3)
        ]
        objects = [KnowledgeObject(canonical_claim=c) for c in claims]
        insights = engine.analyse(objects)
        types = [i.insight_type for i in insights]
        assert "gap" in types

    def test_top_insights_sorted_by_priority(self):
        engine = InsightEngine()
        objects = [
            KnowledgeObject(canonical_claim=make_claim(confidence=0.1)),
            KnowledgeObject(canonical_claim=make_claim(confidence=0.4)),
        ]
        engine.analyse(objects)
        top = engine.top_insights(n=2)
        if len(top) >= 2:
            assert top[0].priority >= top[1].priority

    def test_clear_resets_insights(self):
        engine = InsightEngine()
        ko = KnowledgeObject(canonical_claim=make_claim(confidence=0.2))
        engine.analyse([ko])
        engine.clear()
        assert len(engine.all_insights) == 0
