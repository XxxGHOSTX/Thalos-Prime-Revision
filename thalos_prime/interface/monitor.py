"""
System Monitor
===============

Tracks system-wide health metrics: organoid network, knowledge
repository statistics, node synchronisation, and optimiser actions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class SystemMonitor:
    """
    Aggregates health metrics from all Thalos Prime subsystems.
    """

    def __init__(self, core: Any) -> None:
        self._core = core
        self._snapshots: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Capture and return a full system health snapshot."""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "knowledge_repository": self._knowledge_stats(),
            "organoid_network": self._core.organoid_network.health_report(),
            "microcircuit": self._core.microcircuit.status(),
            "optimiser": self._core.optimiser.report(),
            "nodes": {
                "local_id": self._core.node_registry.local_node.node_id,
                "peer_count": self._core.node_registry.peer_count(),
                "active_peers": len(self._core.node_registry.active_peers),
            },
            "insights": {
                "total": len(self._core.insight_engine.all_insights),
                "top_priority": (
                    self._core.insight_engine.top_insights(1)[0].priority
                    if self._core.insight_engine.all_insights
                    else 0.0
                ),
            },
        }
        self._snapshots.append(report)
        return report

    def recent_snapshots(self, n: int = 10) -> list[dict[str, Any]]:
        return self._snapshots[-n:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _knowledge_stats(self) -> dict[str, Any]:
        repo = self._core.repository
        all_objects = repo.all()
        if not all_objects:
            return {"total": 0, "mean_confidence": 0.0, "by_status": {}}

        status_counts: dict[str, int] = {}
        total_confidence = 0.0
        for ko in all_objects:
            s = ko.canonical_claim.status.value
            status_counts[s] = status_counts.get(s, 0) + 1
            total_confidence += ko.canonical_claim.confidence

        return {
            "total": len(all_objects),
            "mean_confidence": total_confidence / len(all_objects),
            "by_status": status_counts,
        }
