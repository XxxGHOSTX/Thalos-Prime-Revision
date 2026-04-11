"""
Verification Engine
===================

Assigns confidence scores to claims based on:
- Source reliability
- Cross-referencing with existing knowledge
- Consensus across independent sources
"""

from __future__ import annotations

from thalos_prime.models import Claim, ClaimStatus, KnowledgeObject


class VerificationEngine:
    """
    Evaluates incoming claims against existing knowledge objects and
    updates confidence scores.
    """

    # Weight given to existing knowledge vs new source
    _EXISTING_WEIGHT = 0.6
    _NEW_SOURCE_WEIGHT = 0.4

    # Minimum confidence to consider a claim verified
    VERIFIED_THRESHOLD = 0.65

    def __init__(self, knowledge_store: dict[str, KnowledgeObject] | None = None) -> None:
        # Keyed by canonical content for fast lookup
        self._store: dict[str, KnowledgeObject] = knowledge_store or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, claim: Claim) -> Claim:
        """
        Compute a refined confidence score for a claim and update its status.

        Strategy:
        1. Base score from source reliability.
        2. Boost if corroborated by existing knowledge objects.
        3. Mark as VERIFIED if confidence ≥ threshold, else leave PENDING.
        """
        base_score = claim.avg_source_reliability

        # Cross-reference boost
        corroboration_boost = self._compute_corroboration(claim)

        refined = (
            base_score * self._NEW_SOURCE_WEIGHT
            + (base_score + corroboration_boost) * self._EXISTING_WEIGHT
        )
        refined = max(0.0, min(1.0, refined))

        claim.confidence = refined
        claim.status = (
            ClaimStatus.VERIFIED
            if refined >= self.VERIFIED_THRESHOLD
            else ClaimStatus.PENDING
        )
        claim.touch()
        return claim

    def verify_batch(self, claims: list[Claim]) -> list[Claim]:
        """Verify a list of claims in order."""
        return [self.verify(c) for c in claims]

    def register_knowledge_object(self, ko: KnowledgeObject) -> None:
        """Register a resolved KnowledgeObject for cross-referencing."""
        content_key = ko.canonical_claim.content.lower().strip()
        self._store[content_key] = ko

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_corroboration(self, claim: Claim) -> float:
        """
        Return a boost (0.0–0.3) if similar content already exists in the store.
        Simple lexical overlap; production systems would use semantic similarity.
        """
        claim_words = set(claim.content.lower().split())
        best_overlap = 0.0
        best_confidence = 0.0

        for key, ko in self._store.items():
            existing_words = set(key.split())
            if not claim_words or not existing_words:
                continue
            overlap = len(claim_words & existing_words) / len(claim_words | existing_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_confidence = ko.canonical_claim.confidence

        if best_overlap > 0.5:
            # Corroborated by existing knowledge
            return 0.3 * best_confidence
        return 0.0
