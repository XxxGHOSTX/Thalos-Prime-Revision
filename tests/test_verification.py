from __future__ import annotations

import pytest
import uuid
from datetime import datetime, timezone

from thalos_prime.storage.db import Claim, NormalizedDocument, RawDocument, EvidenceLink
from thalos_prime.verification.scorer import VerificationScorer
from thalos_prime.verification.promoter import Promoter


def _create_claim(session, norm_doc_id: str, text: str = "The Earth orbits the Sun each year.", status: str = "draft") -> Claim:
    from thalos_prime.coordinates.scheme import make_coordinate
    coord = make_coordinate(f"{text} {uuid.uuid4()}", "en")
    claim = Claim(
        norm_doc_id=norm_doc_id,
        text=text,
        coordinate=coord,
        language="en",
        status=status,
    )
    session.add(claim)
    session.flush()
    return claim


def _create_raw_norm(session) -> tuple[str, str]:
    raw = RawDocument(
        content_hash=f"hash_{uuid.uuid4().hex}",
        raw_text="Test document text.",
        source_type="text",
    )
    session.add(raw)
    session.flush()
    norm = NormalizedDocument(
        raw_doc_id=raw.id,
        language="en",
        original_text="Test document text.",
        word_count=3,
    )
    session.add(norm)
    session.flush()
    return raw.id, norm.id


def test_composite_score_range(tmp_settings, session):
    scorer = VerificationScorer(tmp_settings)
    raw_id, norm_id = _create_raw_norm(session)
    claim = _create_claim(session, norm_id)
    score = scorer.score(claim, session)
    assert 0.0 <= score <= 1.0


def test_update_verification_record_created(tmp_settings, session):
    from thalos_prime.storage.db import VerificationRecord
    from sqlalchemy import select

    scorer = VerificationScorer(tmp_settings)
    raw_id, norm_id = _create_raw_norm(session)
    claim = _create_claim(session, norm_id)
    record = scorer.update_verification_record(claim, session)
    session.flush()
    assert record is not None
    assert 0.0 <= record.composite_score <= 1.0


def test_promotion_gate_draft_to_candidate(tmp_settings, session):
    scorer = VerificationScorer(tmp_settings)
    promoter = Promoter(tmp_settings)
    raw_id, norm_id = _create_raw_norm(session)
    claim = _create_claim(session, norm_id, status="draft")

    link = EvidenceLink(
        claim_id=claim.id,
        evidence_text="The Earth definitely orbits the Sun according to astronomy.",
        evidence_doc_id=raw_id,
        bm25_score=0.8,
        semantic_score=0.9,
        link_type="supports",
    )
    session.add(link)
    session.flush()

    record = scorer.update_verification_record(claim, session)
    session.flush()
    record.composite_score = 0.5

    tmp_settings2 = tmp_settings
    result = promoter.evaluate(claim, session)
    assert claim.status in ("draft", "candidate")


def test_promotion_gate_insufficient(tmp_settings, session):
    promoter = Promoter(tmp_settings)
    raw_id, norm_id = _create_raw_norm(session)
    claim = _create_claim(session, norm_id, status="draft")
    result = promoter.evaluate(claim, session)
    assert result is None
    assert claim.status == "draft"


def test_signals_temporal_freshness(tmp_settings, session):
    scorer = VerificationScorer(tmp_settings)
    raw_id, norm_id = _create_raw_norm(session)
    claim = _create_claim(session, norm_id)
    signals = scorer.compute_signals(claim, session)
    assert "temporal_freshness" in signals
    assert 0.0 <= signals["temporal_freshness"] <= 1.0
