"""
Versioned Knowledge Repository
================================

Provides an in-memory (with optional JSON persistence) store for
KnowledgeObjects that records full version history.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from thalos_prime.models import (
    Claim,
    ClaimStatus,
    KnowledgeObject,
    KnowledgeVersion,
    SourceRecord,
    SourceType,
)


def _serialize(obj: Any) -> Any:
    """Recursively make an object JSON-serializable."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


class KnowledgeRepository:
    """
    Stores and retrieves KnowledgeObjects with full version history.

    Thread safety is handled at the application level; this class is
    intentionally single-threaded to keep the implementation simple.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self._objects: dict[str, KnowledgeObject] = {}
        self._persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self._load(persist_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def put(self, ko: KnowledgeObject, reason: str = "initial") -> KnowledgeObject:
        """Store or update a KnowledgeObject, creating a new version snapshot."""
        if ko.object_id in self._objects:
            reason = reason or "update"
        ko.add_version(reason=reason)
        self._objects[ko.object_id] = ko
        if self._persist_path:
            self._save()
        return ko

    def get(self, object_id: str) -> KnowledgeObject | None:
        return self._objects.get(object_id)

    def delete(self, object_id: str) -> bool:
        if object_id in self._objects:
            del self._objects[object_id]
            if self._persist_path:
                self._save()
            return True
        return False

    def all(self) -> list[KnowledgeObject]:
        return list(self._objects.values())

    def count(self) -> int:
        return len(self._objects)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[KnowledgeObject]:
        """Naive keyword search across canonical claim content."""
        q = query.lower()
        results = [
            ko
            for ko in self._objects.values()
            if q in ko.canonical_claim.content.lower()
        ]
        results.sort(key=lambda ko: ko.canonical_claim.confidence, reverse=True)
        return results[:limit]

    def by_status(self, status: ClaimStatus) -> list[KnowledgeObject]:
        return [
            ko
            for ko in self._objects.values()
            if ko.canonical_claim.status == status
        ]

    def low_confidence(self, threshold: float = 0.5) -> list[KnowledgeObject]:
        return [
            ko
            for ko in self._objects.values()
            if ko.canonical_claim.confidence < threshold
        ]

    # ------------------------------------------------------------------
    # Version history
    # ------------------------------------------------------------------

    def get_version(
        self, object_id: str, version_number: int
    ) -> KnowledgeVersion | None:
        ko = self._objects.get(object_id)
        if not ko:
            return None
        for v in ko.versions:
            if v.version_number == version_number:
                return v
        return None

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        data = {oid: _serialize(asdict(ko)) for oid, ko in self._objects.items()}
        assert self._persist_path is not None
        Path(self._persist_path).write_text(json.dumps(data, indent=2))

    def _load(self, path: str) -> None:
        raw = json.loads(Path(path).read_text())
        for oid, ko_dict in raw.items():
            ko = self._dict_to_ko(ko_dict)
            self._objects[oid] = ko

    @staticmethod
    def _dict_to_ko(d: dict[str, Any]) -> KnowledgeObject:
        """Reconstruct a KnowledgeObject from a serialised dict."""
        claim_d = d.get("canonical_claim", {})
        sources = [
            SourceRecord(
                source_id=s.get("source_id", ""),
                source_type=SourceType(s.get("source_type", "text")),
                uri=s.get("uri", ""),
                reliability_score=s.get("reliability_score", 0.5),
            )
            for s in claim_d.get("sources", [])
        ]
        claim = Claim(
            claim_id=claim_d.get("claim_id", ""),
            content=claim_d.get("content", ""),
            subject=claim_d.get("subject", ""),
            predicate=claim_d.get("predicate", ""),
            obj=claim_d.get("obj", ""),
            confidence=claim_d.get("confidence", 0.5),
            status=ClaimStatus(claim_d.get("status", "pending")),
            sources=sources,
            tags=claim_d.get("tags", []),
            metadata=claim_d.get("metadata", {}),
        )
        versions = [
            KnowledgeVersion(
                version_id=v.get("version_id", ""),
                claim_snapshot=v.get("claim_snapshot", {}),
                version_number=v.get("version_number", 1),
                reason=v.get("reason", ""),
            )
            for v in d.get("versions", [])
        ]
        ko = KnowledgeObject(
            object_id=d.get("object_id", ""),
            canonical_claim=claim,
            versions=versions,
            related_ids=d.get("related_ids", []),
        )
        return ko
