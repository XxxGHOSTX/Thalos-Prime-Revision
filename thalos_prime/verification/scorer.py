from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from thalos_prime.config import Settings
from thalos_prime.storage.db import Claim, EvidenceLink, VerificationRecord, get_session

logger = logging.getLogger(__name__)

SIGNAL_WEIGHTS = {
    "evidence_coverage": 0.30,
    "cross_source_corroboration": 0.25,
    "nli_support_ratio": 0.20,
    "source_authority": 0.15,
    "temporal_freshness": 0.10,
}

TRUSTED_DOMAINS = {
    "wikipedia.org",
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "nature.com",
    "science.org",
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "who.int",
    "cdc.gov",
    "nih.gov",
    "nasa.gov",
}


def _is_trusted_domain(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        host = urlparse(url).hostname or ""
        if host.endswith(".gov") or host.endswith(".edu"):
            return True
        for td in TRUSTED_DOMAINS:
            if host == td or host.endswith("." + td):
                return True
    except Exception:
        pass
    return False


class VerificationScorer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def compute_signals(self, claim: Claim, session) -> dict:
        from sqlalchemy import select
        from thalos_prime.storage.db import RawDocument

        links = session.execute(
            select(EvidenceLink).where(EvidenceLink.claim_id == claim.id)
        ).scalars().all()

        supporting = [l for l in links if l.link_type == "supports"]
        refuting = [l for l in links if l.link_type == "refutes"]

        avg_score = (
            sum(l.semantic_score for l in supporting) / len(supporting)
            if supporting else 0.0
        )
        evidence_coverage = min(1.0, len(supporting) / 5.0) * avg_score

        domains: set[str] = set()
        urls: list[Optional[str]] = []
        for link in links:
            if link.evidence_doc_id:
                doc = session.execute(
                    select(RawDocument).where(RawDocument.id == link.evidence_doc_id)
                ).scalar_one_or_none()
                if doc and doc.url:
                    urls.append(doc.url)
                    try:
                        host = urlparse(doc.url).hostname or ""
                        domains.add(host)
                    except Exception:
                        pass

        cross_source_corroboration = min(1.0, len(domains) / 3.0)

        supports_count = len(supporting)
        refutes_count = len(refuting)
        nli_support_ratio = supports_count / max(1, supports_count + refutes_count)

        trusted_count = sum(1 for url in urls if _is_trusted_domain(url))
        total_sources = max(1, len(urls))
        source_authority = min(1.0, trusted_count / total_sources)

        now = datetime.now(timezone.utc)
        created = claim.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        days_old = (now - created).days
        temporal_freshness = math.exp(-days_old / 30.0)

        return {
            "evidence_coverage": evidence_coverage,
            "cross_source_corroboration": cross_source_corroboration,
            "nli_support_ratio": nli_support_ratio,
            "source_authority": source_authority,
            "temporal_freshness": temporal_freshness,
        }

    def score(self, claim: Claim, session) -> float:
        signals = self.compute_signals(claim, session)
        total = sum(SIGNAL_WEIGHTS[k] * signals[k] for k in SIGNAL_WEIGHTS)
        return round(min(1.0, max(0.0, total)), 6)

    def update_verification_record(self, claim: Claim, session) -> VerificationRecord:
        from sqlalchemy import select

        signals = self.compute_signals(claim, session)
        composite = sum(SIGNAL_WEIGHTS[k] * signals[k] for k in SIGNAL_WEIGHTS)
        composite = round(min(1.0, max(0.0, composite)), 6)

        record = session.execute(
            select(VerificationRecord).where(VerificationRecord.claim_id == claim.id)
        ).scalar_one_or_none()

        if record is None:
            record = VerificationRecord(
                claim_id=claim.id,
                composite_score=composite,
                signals_json=json.dumps(signals),
                status="pending",
            )
            session.add(record)
        else:
            record.composite_score = composite
            record.signals_json = json.dumps(signals)
            record.updated_at = datetime.now(timezone.utc)

        return record
