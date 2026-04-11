from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IngestionResult:
    doc_id: Optional[str]
    url: Optional[str]
    status: str  # ingested / duplicate / error
    word_count: int = 0
    message: str = ""
