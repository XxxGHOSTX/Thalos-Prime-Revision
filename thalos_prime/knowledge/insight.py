"""
Insight Generation Engine
==========================

Analyses the knowledge repository for:
- Low-confidence areas requiring verification
- Gaps (subjects with no resolved claims)
- Contradiction clusters needing investigation
- Suggested next queries / research directions
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone

from thalos_prime.models import ClaimStatus, InsightRecord, KnowledgeObject


class InsightEngine:
    """
    Proactively scans the knowledge store and generates actionable insights.
    """

    LOW_CONFIDENCE_THRESHOLD = 0.5
    GAP_MIN_PENDING = 3  # number of pending claims before flagging a gap

    def __init__(self) -> None:
        self._insights: list[InsightRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, knowledge_objects: list[KnowledgeObject]) -> list[InsightRecord]:
        """Run all insight detectors and return new insights."""
        new_insights: list[InsightRecord] = []
        new_insights.extend(self._detect_low_confidence(knowledge_objects))
        new_insights.extend(self._detect_gaps(knowledge_objects))
        new_insights.extend(self._detect_contradiction_clusters(knowledge_objects))
        self._insights.extend(new_insights)
        return new_insights

    def top_insights(self, n: int = 10) -> list[InsightRecord]:
        """Return the top-N insights sorted by priority."""
        return sorted(self._insights, key=lambda i: i.priority, reverse=True)[:n]

    def clear(self) -> None:
        self._insights.clear()

    @property
    def all_insights(self) -> list[InsightRecord]:
        return list(self._insights)

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def _detect_low_confidence(
        self, objects: list[KnowledgeObject]
    ) -> list[InsightRecord]:
        insights: list[InsightRecord] = []
        low = [
            ko
            for ko in objects
            if ko.canonical_claim.confidence < self.LOW_CONFIDENCE_THRESHOLD
        ]
        for ko in low:
            insight = InsightRecord(
                insight_type="low_confidence",
                description=(
                    f"Claim '{ko.canonical_claim.content[:80]}' has low confidence "
                    f"({ko.canonical_claim.confidence:.2f}). Consider additional "
                    "source verification."
                ),
                related_claim_ids=[ko.canonical_claim.claim_id],
                priority=1.0 - ko.canonical_claim.confidence,
                created_at=datetime.now(timezone.utc),
            )
            insights.append(insight)
        return insights

    def _detect_gaps(self, objects: list[KnowledgeObject]) -> list[InsightRecord]:
        """Flag subjects that appear frequently in PENDING state."""
        insights: list[InsightRecord] = []
        pending_subjects: Counter[str] = Counter()
        pending_ids: dict[str, list[str]] = {}

        for ko in objects:
            claim = ko.canonical_claim
            if claim.status == ClaimStatus.PENDING and claim.subject:
                pending_subjects[claim.subject] += 1
                pending_ids.setdefault(claim.subject, []).append(claim.claim_id)

        for subject, count in pending_subjects.items():
            if count >= self.GAP_MIN_PENDING:
                insight = InsightRecord(
                    insight_type="gap",
                    description=(
                        f"Subject '{subject}' has {count} unresolved pending claims. "
                        "A focused investigation may close this knowledge gap."
                    ),
                    related_claim_ids=pending_ids[subject],
                    priority=min(1.0, count / 10.0),
                    created_at=datetime.now(timezone.utc),
                )
                insights.append(insight)
        return insights

    def _detect_contradiction_clusters(
        self, objects: list[KnowledgeObject]
    ) -> list[InsightRecord]:
        """Flag subjects where contradictions cluster."""
        insights: list[InsightRecord] = []
        contradicted: dict[str, list[str]] = {}

        for ko in objects:
            claim = ko.canonical_claim
            if claim.status == ClaimStatus.CONTRADICTED and claim.subject:
                contradicted.setdefault(claim.subject, []).append(claim.claim_id)

        for subject, ids in contradicted.items():
            if len(ids) >= 2:
                insight = InsightRecord(
                    insight_type="contradiction_cluster",
                    description=(
                        f"Subject '{subject}' has {len(ids)} contradicted claims. "
                        "Resolution may require additional high-reliability sources."
                    ),
                    related_claim_ids=ids,
                    priority=min(1.0, len(ids) / 5.0),
                    created_at=datetime.now(timezone.utc),
                )
                insights.append(insight)
        return insights
