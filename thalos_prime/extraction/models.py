from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedClaim:
    text: str
    coordinate: str
    language: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    doc_id: str
    claims_extracted: int = 0
    claims_new: int = 0
    claims_duplicate: int = 0
