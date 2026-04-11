from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContradictionResult:
    pairs_found: int = 0
    pairs_new: int = 0
