"""
Knowledge Ingestion – Input Aggregation Layer
=============================================

Responsible for collecting raw data from multiple source types,
normalising it into Claim objects, and queuing it for verification.
"""

from __future__ import annotations

import re
from typing import Any

from thalos_prime.models import Claim, ClaimStatus, SourceRecord, SourceType


class ClaimNormalizer:
    """Normalises raw text or dict payloads into structured Claim objects."""

    # Simple subject-predicate-object extraction heuristic
    _SPO_PATTERN = re.compile(
        r"^(?P<subject>.+?)\s+(?P<predicate>is|has|contains|causes|inhibits|produces|requires)\s+(?P<obj>.+)$",
        re.IGNORECASE,
    )

    def normalise(
        self,
        content: str,
        source: SourceRecord,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Claim:
        """Convert a raw content string into a Claim."""
        content = content.strip()
        subject, predicate, obj = self._extract_spo(content)

        claim = Claim(
            content=content,
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=source.reliability_score,
            status=ClaimStatus.PENDING,
            sources=[source],
            tags=tags or [],
            metadata=metadata or {},
        )
        return claim

    def normalise_dict(
        self,
        data: dict[str, Any],
        source: SourceRecord,
    ) -> Claim:
        """Convert a structured dict payload into a Claim."""
        content = data.get("content", "")
        tags = data.get("tags", [])
        metadata = {k: v for k, v in data.items() if k not in ("content", "tags")}

        claim = Claim(
            content=content,
            subject=data.get("subject", ""),
            predicate=data.get("predicate", ""),
            obj=data.get("obj", data.get("object", "")),
            confidence=data.get("confidence", source.reliability_score),
            status=ClaimStatus.PENDING,
            sources=[source],
            tags=tags,
            metadata=metadata,
        )
        return claim

    def _extract_spo(self, text: str) -> tuple[str, str, str]:
        match = self._SPO_PATTERN.match(text)
        if match:
            return match.group("subject"), match.group("predicate"), match.group("obj")
        # Fallback: whole sentence is the subject, no predicate/object
        return text, "", ""


class IngestionPipeline:
    """
    Aggregates claims from heterogeneous sources and feeds them into
    the verification queue.
    """

    def __init__(self) -> None:
        self._normalizer = ClaimNormalizer()
        self._queue: list[Claim] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_text(
        self,
        content: str,
        source_uri: str,
        source_type: SourceType = SourceType.TEXT,
        reliability: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Claim:
        """Ingest a single plain-text claim."""
        import uuid

        source = SourceRecord(
            source_id=str(uuid.uuid4()),
            source_type=source_type,
            uri=source_uri,
            reliability_score=reliability,
            metadata=metadata or {},
        )
        claim = self._normalizer.normalise(content, source, tags=tags, metadata=metadata)
        self._queue.append(claim)
        return claim

    def ingest_batch(
        self,
        records: list[dict[str, Any]],
        default_source_uri: str = "batch://unknown",
        default_reliability: float = 0.5,
    ) -> list[Claim]:
        """Ingest a batch of dict-encoded claims."""
        import uuid

        claims: list[Claim] = []
        for record in records:
            source_info = record.pop("source", {})
            source = SourceRecord(
                source_id=str(uuid.uuid4()),
                source_type=SourceType(
                    source_info.get("type", SourceType.TEXT.value)
                ),
                uri=source_info.get("uri", default_source_uri),
                reliability_score=source_info.get("reliability", default_reliability),
                metadata=source_info.get("metadata", {}),
            )
            claim = self._normalizer.normalise_dict(record, source)
            self._queue.append(claim)
            claims.append(claim)
        return claims

    def drain(self) -> list[Claim]:
        """Return and clear the current ingestion queue."""
        pending = list(self._queue)
        self._queue.clear()
        return pending

    def queue_size(self) -> int:
        return len(self._queue)
