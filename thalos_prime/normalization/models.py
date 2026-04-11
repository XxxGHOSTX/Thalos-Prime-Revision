from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NormalizationResult:
    doc_id: str
    language: str
    translated: bool
    word_count: int
