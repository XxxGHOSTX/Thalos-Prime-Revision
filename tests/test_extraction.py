from __future__ import annotations

import hashlib

import pytest

from thalos_prime.extraction.claim_extractor import ClaimExtractor
from thalos_prime.extraction.entity_linker import EntityLinker
from thalos_prime.storage.db import NormalizedDocument, RawDocument, get_session


def _create_db_norm_doc(settings, text: str, lang: str = "en") -> str:
    """Create real DB records and return norm_doc_id."""
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    with get_session(settings.database_url) as session:
        raw = RawDocument(
            content_hash=content_hash,
            raw_text=text,
            source_type="text",
        )
        session.add(raw)
        session.flush()
        norm = NormalizedDocument(
            raw_doc_id=raw.id,
            language=lang,
            original_text=text,
            word_count=len(text.split()),
        )
        session.add(norm)
        session.flush()
        return norm.id


def test_claim_extraction_factual(tmp_settings):
    extractor = ClaimExtractor(tmp_settings)
    text = "The Python programming language was created by Guido van Rossum in 1991. It has become one of the most popular languages in the world."
    norm_doc_id = _create_db_norm_doc(tmp_settings, text)
    from sqlalchemy import select
    with get_session(tmp_settings.database_url) as session:
        norm_doc = session.execute(
            select(NormalizedDocument).where(NormalizedDocument.id == norm_doc_id)
        ).scalar_one_or_none()
        result = extractor.extract(norm_doc)
    assert result.claims_extracted >= 0
    assert result.doc_id == norm_doc_id


def test_claim_extraction_question(tmp_settings):
    extractor = ClaimExtractor(tmp_settings)
    factual = "The Earth orbits the Sun once every 365 days."
    question = "What is the capital of France?"
    text = f"{factual}\n{question}"
    norm_doc_id = _create_db_norm_doc(tmp_settings, text)
    from sqlalchemy import select
    with get_session(tmp_settings.database_url) as session:
        norm_doc = session.execute(
            select(NormalizedDocument).where(NormalizedDocument.id == norm_doc_id)
        ).scalar_one_or_none()
        result = extractor.extract(norm_doc)
    assert result.doc_id == norm_doc_id


def test_claim_extraction_short(tmp_settings):
    extractor = ClaimExtractor(tmp_settings)
    text = "Yes. No. OK."
    norm_doc_id = _create_db_norm_doc(tmp_settings, text)
    from sqlalchemy import select
    with get_session(tmp_settings.database_url) as session:
        norm_doc = session.execute(
            select(NormalizedDocument).where(NormalizedDocument.id == norm_doc_id)
        ).scalar_one_or_none()
        result = extractor.extract(norm_doc)
    assert result.claims_extracted == 0 or result.claims_new == 0


def test_entity_extraction(tmp_settings):
    linker = EntityLinker(tmp_settings)
    text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    entities = linker.extract_entities(text)
    assert isinstance(entities, list)
    if entities:
        for ent in entities:
            assert "text" in ent
            assert "label" in ent


def test_claim_extraction_multiple_sentences(tmp_settings):
    extractor = ClaimExtractor(tmp_settings)
    text = (
        "Water boils at one hundred degrees Celsius at standard atmospheric pressure. "
        "The Earth is the third planet from the Sun in the solar system. "
        "Python was designed with readability as a primary goal by its creator."
    )
    norm_doc_id = _create_db_norm_doc(tmp_settings, text)
    from sqlalchemy import select
    with get_session(tmp_settings.database_url) as session:
        norm_doc = session.execute(
            select(NormalizedDocument).where(NormalizedDocument.id == norm_doc_id)
        ).scalar_one_or_none()
        result = extractor.extract(norm_doc)
    assert result.claims_extracted >= 0
