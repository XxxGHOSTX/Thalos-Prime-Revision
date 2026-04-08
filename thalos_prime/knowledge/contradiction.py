"""
Contradiction Detection and Resolution
=======================================

Detects conflicting claims and resolves them using configurable
strategies: highest-confidence wins, source-weighted merge, or
manual escalation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from thalos_prime.models import Claim, ClaimStatus, ContradictionRecord


class ResolutionStrategy(str, Enum):
    HIGHEST_CONFIDENCE = "highest_confidence"
    SOURCE_WEIGHTED = "source_weighted"
    RETAIN_ALL = "retain_all"


class ContradictionResolver:
    """
    Identifies contradictions between claims on the same subject-predicate
    pair and resolves them according to the chosen strategy.
    """

    def __init__(
        self,
        strategy: ResolutionStrategy = ResolutionStrategy.HIGHEST_CONFIDENCE,
    ) -> None:
        self.strategy = strategy
        self._records: list[ContradictionRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, claims: list[Claim]) -> list[ContradictionRecord]:
        """
        Scan a list of claims for contradictions.

        Two claims contradict if they share the same subject and predicate
        but assert different objects.
        """
        new_records: list[ContradictionRecord] = []
        indexed: dict[tuple[str, str], list[Claim]] = {}

        for claim in claims:
            if not claim.subject or not claim.predicate:
                continue
            key = (claim.subject.lower(), claim.predicate.lower())
            indexed.setdefault(key, []).append(claim)

        for (subject, predicate), group in indexed.items():
            distinct_objects = {c.obj.lower() for c in group if c.obj}
            if len(distinct_objects) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        if group[i].obj.lower() != group[j].obj.lower():
                            record = ContradictionRecord(
                                claim_a_id=group[i].claim_id,
                                claim_b_id=group[j].claim_id,
                                conflict_description=(
                                    f"'{subject}' {predicate} '{group[i].obj}' "
                                    f"vs '{group[j].obj}'"
                                ),
                            )
                            new_records.append(record)
                            self._records.append(record)

        return new_records

    def resolve(
        self,
        claim_a: Claim,
        claim_b: Claim,
        record: ContradictionRecord,
    ) -> tuple[Claim, ContradictionRecord]:
        """
        Resolve a contradiction between two claims.

        Returns the winning claim and the updated ContradictionRecord.
        """
        if self.strategy == ResolutionStrategy.HIGHEST_CONFIDENCE:
            winner = self._resolve_by_confidence(claim_a, claim_b)
        elif self.strategy == ResolutionStrategy.SOURCE_WEIGHTED:
            winner = self._resolve_by_source_weight(claim_a, claim_b)
        else:
            # RETAIN_ALL: keep claim_a as canonical but don't mark the other rejected
            winner = claim_a

        loser = claim_b if winner.claim_id == claim_a.claim_id else claim_a

        if self.strategy != ResolutionStrategy.RETAIN_ALL:
            loser.status = ClaimStatus.REJECTED
            loser.touch()

        winner.status = ClaimStatus.RESOLVED
        winner.touch()

        record.resolved = True
        record.winning_claim_id = winner.claim_id
        record.resolution_strategy = self.strategy.value
        record.resolved_at = datetime.now(timezone.utc)

        return winner, record

    @property
    def contradiction_records(self) -> list[ContradictionRecord]:
        return list(self._records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_by_confidence(self, a: Claim, b: Claim) -> Claim:
        return a if a.confidence >= b.confidence else b

    def _resolve_by_source_weight(self, a: Claim, b: Claim) -> Claim:
        score_a = a.avg_source_reliability * a.confidence
        score_b = b.avg_source_reliability * b.confidence
        return a if score_a >= score_b else b
