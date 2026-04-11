from __future__ import annotations

import uuid
import pytest

from thalos_prime.storage.db import Claim, NormalizedDocument, RawDocument
from thalos_prime.query.answerer import Answerer


def _create_claim_in_db(session, text: str, status: str = "draft") -> Claim:
    from thalos_prime.coordinates.scheme import make_coordinate
    raw = RawDocument(
        content_hash=f"hash_{uuid.uuid4().hex}",
        raw_text=text,
        source_type="text",
    )
    session.add(raw)
    session.flush()
    norm = NormalizedDocument(
        raw_doc_id=raw.id,
        language="en",
        original_text=text,
        word_count=len(text.split()),
    )
    session.add(norm)
    session.flush()
    coord = make_coordinate(f"{text} {uuid.uuid4()}", "en")
    claim = Claim(
        norm_doc_id=norm.id,
        text=text,
        coordinate=coord,
        language="en",
        status=status,
    )
    session.add(claim)
    session.flush()
    return claim


def test_template_synthesis(tmp_settings):
    answerer = Answerer(tmp_settings)
    results = [
        {
            "claim_id": "test-id",
            "coordinate": "TP:testcoord12",
            "text": "The Earth orbits the Sun once every 365 days.",
            "status": "verified",
            "score": 0.8,
            "evidence_count": 3,
        }
    ]
    answer = answerer.synthesize("How long does the Earth take to orbit the Sun?", results)
    assert "answer" in answer
    assert isinstance(answer["answer"], str)
    assert len(answer["answer"]) > 0
    assert "citations" in answer
    assert "confidence" in answer


def test_template_synthesis_empty(tmp_settings):
    answerer = Answerer(tmp_settings)
    answer = answerer.synthesize("What is the meaning of life?", [])
    assert "answer" in answer
    assert answer["confidence"] == 0.0


def test_query_engine_empty(tmp_settings, session):
    from thalos_prime.query.engine import QueryEngine
    from thalos_prime.storage.vector_store import VectorStore

    vs = VectorStore(tmp_settings.vector_store_path, tmp_settings.embedding_model)
    engine = QueryEngine(tmp_settings, vs, session)
    results = engine.search("What is the sky color?", top_k=5)
    assert isinstance(results, list)
    assert len(results) == 0


def test_engine_finds_relevant(tmp_settings, session):
    from thalos_prime.query.engine import QueryEngine
    from thalos_prime.storage.vector_store import VectorStore

    claim = _create_claim_in_db(session, "The sky is blue because of Rayleigh scattering of sunlight.")
    vs = VectorStore(tmp_settings.vector_store_path, tmp_settings.embedding_model)
    engine = QueryEngine(tmp_settings, vs, session)
    results = engine.search("Why is the sky blue?", top_k=5)
    assert isinstance(results, list)
    assert len(results) >= 1
    claim_ids = [r["claim_id"] for r in results]
    assert claim.id in claim_ids
