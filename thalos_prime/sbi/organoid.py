"""
Synthetic Biological Intelligence – Organoid Network Simulation
===============================================================

Models a network of neural organoid clusters that perform
distributed, parallel evaluation of knowledge claims.

Note: This is a computational *simulation* of organoid behaviour.
Each organoid is represented as a stateful processing unit with
capacity, activation, and health metrics.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class OrganoidState(str, Enum):
    DORMANT = "dormant"
    ACTIVE = "active"
    OVERLOADED = "overloaded"
    GROWING = "growing"
    DEGRADED = "degraded"


@dataclass
class OrganoidCluster:
    """
    A simulated neural organoid cluster acting as a processing unit.

    Attributes:
        cluster_id:  unique identifier
        capacity:    maximum concurrent tasks (analogous to neuron density)
        health:      0.0 (dead) – 1.0 (optimal)
        state:       current operational state
        load:        current number of active tasks
    """

    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    capacity: int = 10
    health: float = 1.0
    state: OrganoidState = OrganoidState.DORMANT
    load: int = 0
    activation_threshold: float = 0.3  # health minimum to activate
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> bool:
        """Attempt to activate the cluster. Returns True on success."""
        if self.health < self.activation_threshold:
            self.state = OrganoidState.DEGRADED
            return False
        self.state = OrganoidState.ACTIVE
        return True

    def deactivate(self) -> None:
        self.state = OrganoidState.DORMANT
        self.load = 0

    def grow(self, capacity_delta: int = 5) -> None:
        """Simulate biological growth by increasing capacity."""
        self.state = OrganoidState.GROWING
        self.capacity += capacity_delta
        self.health = min(1.0, self.health + 0.05)
        self.state = OrganoidState.ACTIVE

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def accept_task(self) -> bool:
        """
        Accept a processing task. Returns False if at capacity or unhealthy.
        """
        if self.state not in (OrganoidState.ACTIVE, OrganoidState.DORMANT):
            return False
        if self.load >= self.capacity:
            self.state = OrganoidState.OVERLOADED
            return False
        if self.health < self.activation_threshold:
            self.state = OrganoidState.DEGRADED
            return False
        self.state = OrganoidState.ACTIVE
        self.load += 1
        return True

    def release_task(self) -> None:
        """Mark one task as complete."""
        self.load = max(0, self.load - 1)
        if self.load < self.capacity:
            self.state = OrganoidState.ACTIVE

    def tick_health(self, degradation: float = 0.001) -> None:
        """Simulate natural health degradation over time."""
        self.health = max(0.0, self.health - degradation)
        if self.health < self.activation_threshold:
            self.state = OrganoidState.DEGRADED

    @property
    def utilisation(self) -> float:
        """Current load as a fraction of capacity."""
        if self.capacity == 0:
            return 1.0
        return self.load / self.capacity


class OrganoidNetwork:
    """
    Manages a collection of OrganoidCluster instances and distributes
    processing tasks across them.
    """

    # Trigger growth when mean utilisation exceeds this threshold
    GROWTH_TRIGGER_UTILISATION = 0.80
    MAX_CLUSTERS = 64

    def __init__(self, initial_clusters: int = 4) -> None:
        self._clusters: dict[str, OrganoidCluster] = {}
        for _ in range(initial_clusters):
            self._add_cluster()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Route a payload to an available cluster for evaluation.

        Returns a result dict with the cluster_id used and a simulated
        activation vector.
        """
        cluster = self._select_cluster()
        if cluster is None:
            # Spawn a new cluster if headroom allows
            if len(self._clusters) < self.MAX_CLUSTERS:
                cluster = self._add_cluster()
                cluster.activate()
            else:
                return {"error": "all clusters at capacity", "result": None}

        accepted = cluster.accept_task()
        if not accepted:
            return {"error": f"cluster {cluster.cluster_id} rejected task", "result": None}

        # Simulate processing: generate a pseudo-activation vector
        activation = self._simulate_activation(payload, cluster)
        cluster.release_task()

        # Check if growth is needed
        self._maybe_grow()

        return {
            "cluster_id": cluster.cluster_id,
            "activation": activation,
            "health": cluster.health,
            "utilisation": cluster.utilisation,
        }

    def health_report(self) -> dict[str, Any]:
        """Return a summary of the network's health and capacity."""
        clusters = list(self._clusters.values())
        if not clusters:
            return {"total": 0, "active": 0, "mean_health": 0.0, "mean_utilisation": 0.0}
        return {
            "total": len(clusters),
            "active": sum(1 for c in clusters if c.state == OrganoidState.ACTIVE),
            "overloaded": sum(1 for c in clusters if c.state == OrganoidState.OVERLOADED),
            "degraded": sum(1 for c in clusters if c.state == OrganoidState.DEGRADED),
            "mean_health": sum(c.health for c in clusters) / len(clusters),
            "mean_utilisation": sum(c.utilisation for c in clusters) / len(clusters),
            "total_capacity": sum(c.capacity for c in clusters),
            "total_load": sum(c.load for c in clusters),
        }

    def tick(self) -> None:
        """Advance the simulation by one time step (health degradation)."""
        for cluster in self._clusters.values():
            cluster.tick_health()

    @property
    def cluster_count(self) -> int:
        return len(self._clusters)

    def trigger_growth(self) -> bool:
        """
        Publicly trigger a growth check.

        Returns True if a new cluster was spawned, False otherwise.
        """
        before = len(self._clusters)
        self._maybe_grow()
        return len(self._clusters) > before

    def clusters_by_state(self, state: OrganoidState) -> list[OrganoidCluster]:
        """Return all clusters in the given state."""
        return [c for c in self._clusters.values() if c.state == state]

    @property
    def clusters(self) -> list[OrganoidCluster]:
        """Return a snapshot of all clusters (read-only list)."""
        return list(self._clusters.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_cluster(self, capacity: int = 10) -> OrganoidCluster:
        cluster = OrganoidCluster(capacity=capacity)
        cluster.activate()
        self._clusters[cluster.cluster_id] = cluster
        return cluster

    def _select_cluster(self) -> OrganoidCluster | None:
        """Select the least-loaded active cluster."""
        candidates = [
            c
            for c in self._clusters.values()
            if c.state == OrganoidState.ACTIVE and c.load < c.capacity
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda c: c.utilisation)

    def _maybe_grow(self) -> None:
        """Trigger a new cluster if mean utilisation is high."""
        report = self.health_report()
        if (
            report["mean_utilisation"] > self.GROWTH_TRIGGER_UTILISATION
            and len(self._clusters) < self.MAX_CLUSTERS
        ):
            new_cluster = self._add_cluster()
            new_cluster.grow()

    @staticmethod
    def _simulate_activation(
        payload: dict[str, Any], cluster: OrganoidCluster
    ) -> list[float]:
        """
        Generate a simulated activation vector.

        In a real SBI system this would be replaced by actual organoid
        electrophysiology measurements or decoded spike trains.
        """
        seed = hash(str(payload)) % (2**32)
        rng = random.Random(seed + hash(cluster.cluster_id))
        dimension = 16
        raw = [rng.gauss(0, 1) for _ in range(dimension)]
        # Apply a sigmoid-like normalisation
        return [1.0 / (1.0 + math.exp(-x)) for x in raw]
