from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="THALOS_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///./thalos.db"
    data_dir: Path = Path("./thalos_data")
    log_level: str = "INFO"

    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-3.5-turbo"

    deepl_api_key: Optional[str] = None
    use_cloud_translation: bool = False
    use_cloud_llm: bool = False

    max_crawl_depth: int = 1
    respect_robots: bool = True

    min_claim_words: int = 6
    max_claim_words: int = 150

    evidence_top_k: int = 10

    claim_score_threshold_candidate: float = 0.40
    claim_score_threshold_verified: float = 0.65
    claim_score_threshold_authoritative: float = 0.85

    autonomy_cron: str = "0 * * * *"
    autonomy_max_age_days: int = 7

    spacy_model: str = "en_core_web_sm"
    embedding_model: str = "all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-MiniLM-L6-v2"

    vector_store_path: Path = Path("./thalos_data/vectors")

    api_host: str = "0.0.0.0"
    api_port: int = 8000


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
