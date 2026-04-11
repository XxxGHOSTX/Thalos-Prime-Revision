from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvidenceResult:
    claim_id: str
    links_found: int = 0
    links_new: int = 0
