"""
Core data models for Thalos Prime.

Defines the fundamental structures used across all layers:
- Claim: a single unit of knowledge with provenance metadata
- KnowledgeObject: versioned, resolved knowledge
- SourceRecord: provenance and lineage information
- NodeInfo: distributed node registration
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ClaimStatus(str, Enum):
    """Lifecycle state of a claim."""

    PENDING = "pending"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    RESOLVED = "resolved"
    REJECTED = "rejected"


class SourceType(str, Enum):
    """Category of a data source."""

    TEXT = "text"
    DATABASE = "database"
    EXPERIMENTAL = "experimental"
    SENSOR = "sensor"
    SYNTHETIC = "synthetic"
    PEER_NODE = "peer_node"


@dataclass
class SourceRecord:
    """Provenance information for a single knowledge contribution."""

    source_id: str
    source_type: SourceType
    uri: str
    reliability_score: float  # 0.0 – 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability_score <= 1.0:
            raise ValueError(
                f"reliability_score must be in [0, 1], got {self.reliability_score}"
            )


@dataclass
class Claim:
    """
    A single unit of knowledge ingested into Thalos Prime.

    Each claim carries full provenance so the lineage of every
    piece of information can be traced.
    """

    claim_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    subject: str = ""
    predicate: str = ""
    obj: str = ""
    confidence: float = 0.5
    status: ClaimStatus = ClaimStatus.PENDING
    sources: list[SourceRecord] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )

    def touch(self) -> None:
        """Update the last-modified timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    @property
    def avg_source_reliability(self) -> float:
        """Weighted average source reliability, or 0 when no sources present."""
        if not self.sources:
            return 0.0
        return sum(s.reliability_score for s in self.sources) / len(self.sources)


@dataclass
class KnowledgeVersion:
    """An immutable snapshot of a claim at a point in time."""

    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claim_snapshot: dict[str, Any] = field(default_factory=dict)
    version_number: int = 1
    reason: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class KnowledgeObject:
    """
    A fully verified, versioned knowledge object.

    Wraps one or more claims that have been reconciled into a
    single canonical statement.
    """

    object_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    canonical_claim: Claim = field(default_factory=Claim)
    versions: list[KnowledgeVersion] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_version(self, reason: str = "") -> KnowledgeVersion:
        """Snapshot the current state of the canonical claim."""
        import dataclasses

        snapshot = dataclasses.asdict(self.canonical_claim)
        # Serialize enums and datetimes for storage
        for key, val in snapshot.items():
            if isinstance(val, Enum):
                snapshot[key] = val.value
            elif isinstance(val, datetime):
                snapshot[key] = val.isoformat()

        version = KnowledgeVersion(
            claim_snapshot=snapshot,
            version_number=len(self.versions) + 1,
            reason=reason,
        )
        self.versions.append(version)
        self.updated_at = datetime.now(timezone.utc)
        return version


@dataclass
class ContradictionRecord:
    """Records a detected contradiction between two claims."""

    contradiction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claim_a_id: str = ""
    claim_b_id: str = ""
    conflict_description: str = ""
    resolution_strategy: str = ""
    resolved: bool = False
    winning_claim_id: str | None = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None


@dataclass
class InsightRecord:
    """A proactive insight or gap identified by the insight engine."""

    insight_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    insight_type: str = ""  # "gap", "low_confidence", "contradiction_cluster", etc.
    description: str = ""
    related_claim_ids: list[str] = field(default_factory=list)
    priority: float = 0.5  # 0.0 – 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeInfo:
    """Registration information for a distributed Thalos Prime node."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    host: str = "localhost"
    port: int = 8080
    public_key_pem: str = ""
    status: str = "active"  # "active", "syncing", "offline"
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
