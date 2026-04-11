"""
Synthetic Microcircuit Integration
====================================

Models the interface layer between the organoid network and the
digital knowledge processing pipeline.

Microcircuits:
- Route signals between organoid clusters
- Handle analogue-to-digital conversion (simulated)
- Detect bottlenecks and trigger growth events via callback
- Provide long-term memory via a synthetic storage buffer
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SynapticBuffer:
    """
    Simulates a microfluidic / synthetic long-term memory buffer.

    Stores encoded signal traces that can be replayed to reinforce
    organoid learning.
    """

    buffer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    capacity: int = 1000
    _store: dict[str, Any] = field(default_factory=dict, repr=False)

    def write(self, key: str, value: Any) -> bool:
        if len(self._store) >= self.capacity:
            # Evict oldest entry (FIFO)
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = value
        return True

    def read(self, key: str) -> Any | None:
        return self._store.get(key)

    def flush(self) -> dict[str, Any]:
        data = dict(self._store)
        self._store.clear()
        return data

    @property
    def utilisation(self) -> float:
        return len(self._store) / self.capacity if self.capacity else 1.0


@dataclass
class MicrocircuitChannel:
    """
    A single bidirectional communication channel between two organoid clusters.
    """

    channel_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_cluster_id: str = ""
    target_cluster_id: str = ""
    bandwidth: int = 100  # arbitrary signal units per tick
    active: bool = True


class MicrocircuitController:
    """
    Manages the routing of signals through the synthetic microcircuit network.

    Responsibilities:
    - Register and manage channels between organoid clusters
    - Route knowledge payloads to the appropriate cluster
    - Detect bottlenecks (high queue depth) and invoke a growth callback
    - Maintain a SynapticBuffer for long-term memory
    """

    BOTTLENECK_QUEUE_DEPTH = 50

    def __init__(
        self,
        growth_callback: Callable[[], Any] | None = None,
    ) -> None:
        """
        Args:
            growth_callback: Called whenever a queue-depth bottleneck is
                detected.  In a live deployment this would be wired to
                ``OrganoidNetwork.trigger_growth()`` so the network
                automatically expands its processing capacity.
        """
        self._channels: dict[str, MicrocircuitChannel] = {}
        self._queue: list[dict[str, Any]] = []
        self._memory = SynapticBuffer()
        self._signal_log: list[dict[str, Any]] = []
        self._growth_callback: Callable[[], Any] | None = growth_callback
        self._bottleneck_count: int = 0

    # ------------------------------------------------------------------
    # Channel management
    # ------------------------------------------------------------------

    def add_channel(
        self, source_cluster_id: str, target_cluster_id: str, bandwidth: int = 100
    ) -> MicrocircuitChannel:
        channel = MicrocircuitChannel(
            source_cluster_id=source_cluster_id,
            target_cluster_id=target_cluster_id,
            bandwidth=bandwidth,
        )
        self._channels[channel.channel_id] = channel
        return channel

    def remove_channel(self, channel_id: str) -> bool:
        if channel_id in self._channels:
            del self._channels[channel_id]
            return True
        return False

    # ------------------------------------------------------------------
    # Signal routing
    # ------------------------------------------------------------------

    def route_signal(
        self, payload: dict[str, Any], target_cluster_id: str
    ) -> dict[str, Any]:
        """
        Encode and route a signal to the target organoid cluster.

        Returns a routing receipt with timing and checksum.
        """
        encoded = self._encode(payload)
        receipt = {
            "signal_id": str(uuid.uuid4()),
            "target_cluster_id": target_cluster_id,
            "checksum": hashlib.sha256(encoded.encode()).hexdigest()[:16],
            "routed_at": datetime.now(timezone.utc).isoformat(),
            "payload_size": len(encoded),
        }
        self._signal_log.append(receipt)

        # Persist to long-term memory
        self._memory.write(receipt["signal_id"], payload)

        return receipt

    def enqueue(self, payload: dict[str, Any]) -> None:
        """Queue a payload for asynchronous processing."""
        self._queue.append(payload)
        if len(self._queue) >= self.BOTTLENECK_QUEUE_DEPTH:
            self._on_bottleneck()

    def drain_queue(self) -> list[dict[str, Any]]:
        """Return and clear the current signal queue."""
        items = list(self._queue)
        self._queue.clear()
        return items

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "channels": len(self._channels),
            "queue_depth": len(self._queue),
            "memory_utilisation": self._memory.utilisation,
            "signals_routed": len(self._signal_log),
            "bottleneck_events": self._bottleneck_count,
        }

    @property
    def memory(self) -> SynapticBuffer:
        return self._memory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, payload: dict[str, Any]) -> str:
        """Simulate analogue-to-digital encoding of a signal payload."""
        return str(sorted(payload.items()))

    def _on_bottleneck(self) -> None:
        """
        Called when queue depth exceeds threshold.

        Logs the event *and* invokes the growth callback (if registered)
        so that the organoid network can autonomously provision new
        processing capacity.
        """
        self._bottleneck_count += 1
        event: dict[str, Any] = {
            "event": "bottleneck_detected",
            "queue_depth": len(self._queue),
            "bottleneck_number": self._bottleneck_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self._growth_callback is not None:
            try:
                growth_result = self._growth_callback()
                event["growth_triggered"] = True
                event["new_cluster_spawned"] = bool(growth_result)
            except Exception as exc:
                event["growth_triggered"] = False
                event["growth_error"] = str(exc)
        else:
            event["growth_triggered"] = False
        self._signal_log.append(event)
