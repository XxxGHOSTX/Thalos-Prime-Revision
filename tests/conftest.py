from __future__ import annotations

import pytest
from pathlib import Path

from thalos_prime.config import Settings
from thalos_prime.storage.db import init_db, get_session


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    db_path = tmp_path / "test.db"
    vector_path = tmp_path / "vectors"
    vector_path.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        database_url=f"sqlite:///{db_path}",
        data_dir=tmp_path,
        vector_store_path=vector_path,
        log_level="WARNING",
        spacy_model="en_core_web_sm",
        embedding_model="all-MiniLM-L6-v2",
        min_claim_words=6,
        max_claim_words=150,
        evidence_top_k=5,
        claim_score_threshold_candidate=0.40,
        claim_score_threshold_verified=0.65,
        claim_score_threshold_authoritative=0.85,
        respect_robots=False,
        use_cloud_llm=False,
        use_cloud_translation=False,
    )
    init_db(settings.database_url)
    return settings


@pytest.fixture
def session(tmp_settings: Settings):
    with get_session(tmp_settings.database_url) as s:
        yield s


@pytest.fixture
def core(tmp_settings: Settings):
    from thalos_prime.core import ThalosCore
    return ThalosCore(tmp_settings)
