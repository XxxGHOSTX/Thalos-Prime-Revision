from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from thalos_prime.config import Settings
from thalos_prime.verification.models import PromotionResult
from thalos_prime.verification.scorer import VerificationScorer

logger = logging.getLogger(__name__)

STATUS_ORDER = ["draft", "candidate", "verified", "authoritative", "retracted"]


class Promoter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.scorer = VerificationScorer(settings)

    def _get_active_contradictions(self, claim_id: str, session) -> int:
        from sqlalchemy import select
        from thalos_prime.storage.db import ContradictionPair

        count = session.execute(
            select(ContradictionPair).where(
                (ContradictionPair.claim_a_id == claim_id) | (ContradictionPair.claim_b_id == claim_id),
                ContradictionPair.label == "contradiction",
            )
        ).scalars().all()
        return len(count)

    def _get_distinct_sources(self, claim_id: str, session) -> int:
        from sqlalchemy import select
        from thalos_prime.storage.db import EvidenceLink, RawDocument
        from urllib.parse import urlparse

        links = session.execute(
            select(EvidenceLink).where(EvidenceLink.claim_id == claim_id)
        ).scalars().all()

        domains: set[str] = set()
        for link in links:
            if link.evidence_doc_id:
                doc = session.execute(
                    select(RawDocument).where(RawDocument.id == link.evidence_doc_id)
                ).scalar_one_or_none()
                if doc and doc.url:
                    try:
                        host = urlparse(doc.url).hostname or doc.url
                        domains.add(host)
                    except Exception:
                        domains.add(doc.url)
        return len(domains)

    def _get_supporting_count(self, claim_id: str, session) -> int:
        from sqlalchemy import select
        from thalos_prime.storage.db import EvidenceLink

        links = session.execute(
            select(EvidenceLink).where(
                EvidenceLink.claim_id == claim_id,
                EvidenceLink.link_type == "supports",
            )
        ).scalars().all()
        return len(links)

    def evaluate(self, claim, session) -> Optional[PromotionResult]:
        from sqlalchemy import select
        from thalos_prime.storage.db import VerificationRecord

        record = self.scorer.update_verification_record(claim, session)
        session.flush()

        score = record.composite_score
        old_status = claim.status

        if old_status == "retracted":
            return None

        new_status = None

        if old_status == "draft":
            supporting = self._get_supporting_count(claim.id, session)
            if score >= self.settings.claim_score_threshold_candidate and supporting >= 1:
                new_status = "candidate"

        elif old_status == "candidate":
            sources = self._get_distinct_sources(claim.id, session)
            contradictions = self._get_active_contradictions(claim.id, session)
            if (
                score >= self.settings.claim_score_threshold_verified
                and sources >= 2
                and contradictions == 0
            ):
                new_status = "verified"

        elif old_status == "verified":
            sources = self._get_distinct_sources(claim.id, session)
            if score >= self.settings.claim_score_threshold_authoritative and sources >= 5:
                new_status = "authoritative"

        if new_status is not None:
            claim.status = new_status
            claim.updated_at = datetime.now(timezone.utc)
            record.promoted_to = new_status
            record.promoted_at = datetime.now(timezone.utc)
            record.status = "promoted"
            logger.info("Promoted claim %s: %s -> %s (score=%.3f)", claim.id, old_status, new_status, score)
            return PromotionResult(
                claim_id=claim.id,
                from_status=old_status,
                to_status=new_status,
                score=score,
            )
        return None

    def promote_all(self, session) -> list[PromotionResult]:
        from sqlalchemy import select
        from thalos_prime.storage.db import Claim

        claims = session.execute(
            select(Claim).where(Claim.status.in_(["draft", "candidate", "verified"]))
        ).scalars().all()

        results = []
        for claim in claims:
            pr = self.evaluate(claim, session)
            if pr:
                results.append(pr)
        return results
