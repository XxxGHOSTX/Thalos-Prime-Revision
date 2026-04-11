from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VerificationSignals:
    evidence_coverage: float = 0.0
    cross_source_corroboration: float = 0.0
    nli_support_ratio: float = 0.0
    source_authority: float = 0.0
    temporal_freshness: float = 0.0


@dataclass
class PromotionResult:
    claim_id: str
    from_status: str
    to_status: str
    score: float
