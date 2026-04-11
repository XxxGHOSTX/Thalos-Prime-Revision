from __future__ import annotations

import pytest
from pathlib import Path


def test_core_init(tmp_settings):
    from thalos_prime.core import ThalosCore
    core = ThalosCore(tmp_settings)
    assert core is not None
    assert core._initialized is True


def test_ingest_file_end_to_end(tmp_settings, tmp_path):
    from thalos_prime.core import ThalosCore
    core = ThalosCore(tmp_settings)
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text(
        "Python is a high-level programming language known for its simple and readable syntax. "
        "It was created by Guido van Rossum and first released in 1991. "
        "Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming."
    )
    result = core.ingest_file(str(txt_file))
    assert result.status == "ingested"
    assert result.doc_id is not None
    assert result.word_count > 0


def test_query_after_ingest(tmp_settings, tmp_path):
    from thalos_prime.core import ThalosCore
    core = ThalosCore(tmp_settings)
    txt_file = tmp_path / "knowledge.txt"
    txt_file.write_text(
        "The Pacific Ocean is the largest ocean on Earth, covering more than 165 million square kilometers. "
        "It stretches from the Arctic Ocean in the north to the Southern Ocean in the south. "
        "The Pacific Ocean contains more than half of the world's oceanic water."
    )
    ingest_result = core.ingest_file(str(txt_file))
    assert ingest_result.status == "ingested"

    query_result = core.query("What is the largest ocean?", top_k=5)
    assert "answer" in query_result
    assert isinstance(query_result["answer"], str)


def test_pipeline_run(tmp_settings, tmp_path):
    from thalos_prime.core import ThalosCore
    from thalos_prime.ingestion.file_loader import FileLoader

    core = ThalosCore(tmp_settings)
    loader = FileLoader(tmp_settings)
    txt_file = tmp_path / "pipeline_test.txt"
    txt_file.write_text(
        "Mount Everest is the highest mountain on Earth above sea level, standing at 8,849 meters. "
        "It is located in the Himalayas on the border between Nepal and Tibet. "
        "The mountain was first summited on May 29, 1953 by Edmund Hillary and Tenzing Norgay."
    )
    ingest_result = loader.load(txt_file)
    assert ingest_result.status == "ingested"

    pipeline_result = core.run_pipeline(ingest_result.doc_id)
    assert "doc_id" in pipeline_result
    assert pipeline_result["doc_id"] == ingest_result.doc_id
    assert "claims_extracted" in pipeline_result


def test_get_claim_not_found(tmp_settings):
    from thalos_prime.core import ThalosCore
    core = ThalosCore(tmp_settings)
    result = core.get_claim("TP:nonexistent1")
    assert result is None


def test_retract_claim(tmp_settings, tmp_path):
    from thalos_prime.core import ThalosCore
    core = ThalosCore(tmp_settings)
    txt_file = tmp_path / "retract_test.txt"
    txt_file.write_text(
        "The Sun is a star at the center of the solar system. "
        "It provides light and heat to all the planets orbiting around it."
    )
    result = core.ingest_file(str(txt_file))
    assert result.status == "ingested"

    from thalos_prime.storage.db import Claim, get_session
    from sqlalchemy import select
    claim_coordinate = None
    with get_session(tmp_settings.database_url) as session:
        claim = session.execute(select(Claim).limit(1)).scalar_one_or_none()
        if claim:
            claim_coordinate = claim.coordinate

    if claim_coordinate:
        retracted = core.retract_claim(claim_coordinate)
        assert retracted is True

        retracted_again = core.retract_claim("TP:notexists1x")
        assert retracted_again is False
