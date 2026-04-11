"""
Tests for the SBI (Synthetic Biological Intelligence) Simulation Layer.

Covers:
- OrganoidCluster lifecycle
- OrganoidNetwork load distribution and growth
- MicrocircuitController routing and memory
"""

import pytest

from thalos_prime.sbi.microcircuit import MicrocircuitController, SynapticBuffer
from thalos_prime.sbi.organoid import (
    OrganoidCluster,
    OrganoidNetwork,
    OrganoidState,
)


# ---------------------------------------------------------------------------
# OrganoidCluster tests
# ---------------------------------------------------------------------------


class TestOrganoidCluster:
    def test_initial_state_is_dormant(self):
        cluster = OrganoidCluster()
        assert cluster.state == OrganoidState.DORMANT

    def test_activate_sets_active(self):
        cluster = OrganoidCluster(health=1.0)
        result = cluster.activate()
        assert result is True
        assert cluster.state == OrganoidState.ACTIVE

    def test_activate_fails_below_threshold(self):
        cluster = OrganoidCluster(health=0.1, activation_threshold=0.3)
        result = cluster.activate()
        assert result is False
        assert cluster.state == OrganoidState.DEGRADED

    def test_accept_and_release_task(self):
        cluster = OrganoidCluster(capacity=5)
        cluster.activate()
        assert cluster.accept_task() is True
        assert cluster.load == 1
        cluster.release_task()
        assert cluster.load == 0

    def test_overloaded_when_at_capacity(self):
        cluster = OrganoidCluster(capacity=2)
        cluster.activate()
        cluster.accept_task()
        cluster.accept_task()
        # Now at capacity
        result = cluster.accept_task()
        assert result is False
        assert cluster.state == OrganoidState.OVERLOADED

    def test_grow_increases_capacity(self):
        cluster = OrganoidCluster(capacity=10)
        cluster.activate()
        before = cluster.capacity
        cluster.grow(capacity_delta=5)
        assert cluster.capacity == before + 5

    def test_tick_health_degrades(self):
        cluster = OrganoidCluster(health=1.0)
        for _ in range(100):
            cluster.tick_health(degradation=0.01)
        assert cluster.health < 1.0

    def test_utilisation_zero_when_idle(self):
        cluster = OrganoidCluster(capacity=10)
        assert cluster.utilisation == 0.0

    def test_utilisation_full_when_at_capacity(self):
        cluster = OrganoidCluster(capacity=2)
        cluster.activate()
        cluster.accept_task()
        cluster.accept_task()
        assert cluster.utilisation == pytest.approx(1.0)

    def test_deactivate_clears_load(self):
        cluster = OrganoidCluster(capacity=5)
        cluster.activate()
        cluster.accept_task()
        cluster.deactivate()
        assert cluster.load == 0
        assert cluster.state == OrganoidState.DORMANT


# ---------------------------------------------------------------------------
# OrganoidNetwork tests
# ---------------------------------------------------------------------------


class TestOrganoidNetwork:
    def test_initial_cluster_count(self):
        network = OrganoidNetwork(initial_clusters=4)
        assert network.cluster_count == 4

    def test_evaluate_returns_result(self):
        network = OrganoidNetwork(initial_clusters=2)
        result = network.evaluate({"content": "test claim"})
        assert "cluster_id" in result
        assert "activation" in result
        assert len(result["activation"]) == 16

    def test_activation_is_bounded(self):
        network = OrganoidNetwork(initial_clusters=2)
        result = network.evaluate({"content": "water is H2O"})
        for v in result["activation"]:
            assert 0.0 <= v <= 1.0

    def test_health_report_structure(self):
        network = OrganoidNetwork(initial_clusters=3)
        report = network.health_report()
        assert "total" in report
        assert "active" in report
        assert "mean_health" in report
        assert "mean_utilisation" in report

    def test_tick_degrades_health(self):
        network = OrganoidNetwork(initial_clusters=2)
        initial_health = network.health_report()["mean_health"]
        for _ in range(50):
            network.tick()
        assert network.health_report()["mean_health"] < initial_health

    def test_evaluate_deterministic_for_same_payload(self):
        network = OrganoidNetwork(initial_clusters=2)
        r1 = network.evaluate({"content": "same content"})
        r2 = network.evaluate({"content": "same content"})
        # Activation vectors should be the same (deterministic hash-based seed)
        assert r1["activation"] == r2["activation"]

    def test_evaluate_different_for_different_payloads(self):
        network = OrganoidNetwork(initial_clusters=2)
        r1 = network.evaluate({"content": "claim A"})
        r2 = network.evaluate({"content": "claim B"})
        assert r1["activation"] != r2["activation"]


# ---------------------------------------------------------------------------
# MicrocircuitController tests
# ---------------------------------------------------------------------------


class TestSynapticBuffer:
    def test_write_and_read(self):
        buf = SynapticBuffer(capacity=10)
        buf.write("k1", "value1")
        assert buf.read("k1") == "value1"

    def test_read_missing_returns_none(self):
        buf = SynapticBuffer()
        assert buf.read("nonexistent") is None

    def test_evicts_oldest_when_full(self):
        buf = SynapticBuffer(capacity=3)
        buf.write("a", 1)
        buf.write("b", 2)
        buf.write("c", 3)
        # Writing a 4th entry should evict "a"
        buf.write("d", 4)
        assert buf.read("a") is None
        assert buf.read("d") == 4

    def test_flush_clears_buffer(self):
        buf = SynapticBuffer()
        buf.write("x", 42)
        data = buf.flush()
        assert "x" in data
        assert buf.read("x") is None

    def test_utilisation(self):
        buf = SynapticBuffer(capacity=10)
        buf.write("a", 1)
        assert buf.utilisation == pytest.approx(0.1)


class TestMicrocircuitController:
    def test_add_and_remove_channel(self):
        ctrl = MicrocircuitController()
        ch = ctrl.add_channel("src-1", "tgt-1")
        assert ctrl.status()["channels"] == 1
        ctrl.remove_channel(ch.channel_id)
        assert ctrl.status()["channels"] == 0

    def test_route_signal_returns_receipt(self):
        ctrl = MicrocircuitController()
        receipt = ctrl.route_signal({"data": "test"}, "cluster-1")
        assert "signal_id" in receipt
        assert "checksum" in receipt

    def test_enqueue_and_drain(self):
        ctrl = MicrocircuitController()
        ctrl.enqueue({"msg": 1})
        ctrl.enqueue({"msg": 2})
        items = ctrl.drain_queue()
        assert len(items) == 2
        assert ctrl.status()["queue_depth"] == 0

    def test_memory_persists_routed_signal(self):
        ctrl = MicrocircuitController()
        payload = {"key": "value"}
        receipt = ctrl.route_signal(payload, "cluster-1")
        retrieved = ctrl.memory.read(receipt["signal_id"])
        assert retrieved == payload

    def test_status_structure(self):
        ctrl = MicrocircuitController()
        s = ctrl.status()
        assert "channels" in s
        assert "queue_depth" in s
        assert "memory_utilisation" in s
        assert "signals_routed" in s


# ---------------------------------------------------------------------------
# MicrocircuitController bottleneck callback tests (new)
# ---------------------------------------------------------------------------


class TestMicrocircuitBottleneckCallback:
    def test_bottleneck_increments_counter(self):
        from thalos_prime.sbi.microcircuit import MicrocircuitController
        ctrl = MicrocircuitController()
        for i in range(MicrocircuitController.BOTTLENECK_QUEUE_DEPTH + 5):
            ctrl.enqueue({"data": i})
        assert ctrl._bottleneck_count >= 1

    def test_bottleneck_triggers_growth_callback(self):
        from thalos_prime.sbi.microcircuit import MicrocircuitController
        from thalos_prime.sbi.organoid import OrganoidNetwork

        net = OrganoidNetwork(initial_clusters=2)
        # Saturate clusters so growth can happen
        for c in net.clusters:
            c.load = c.capacity
        before = net.cluster_count
        ctrl = MicrocircuitController(growth_callback=net.trigger_growth)
        for i in range(MicrocircuitController.BOTTLENECK_QUEUE_DEPTH + 1):
            ctrl.enqueue({"data": i})
        # At least one bottleneck event should have been processed
        assert ctrl._bottleneck_count >= 1
        log_entry = next(
            (e for e in ctrl._signal_log if e.get("event") == "bottleneck_detected"),
            None,
        )
        assert log_entry is not None
        assert log_entry["growth_triggered"] is True

    def test_no_callback_still_logs_event(self):
        from thalos_prime.sbi.microcircuit import MicrocircuitController
        ctrl = MicrocircuitController(growth_callback=None)
        for i in range(MicrocircuitController.BOTTLENECK_QUEUE_DEPTH + 1):
            ctrl.enqueue({"data": i})
        log_entry = next(
            (e for e in ctrl._signal_log if e.get("event") == "bottleneck_detected"),
            None,
        )
        assert log_entry is not None
        assert log_entry["growth_triggered"] is False

    def test_bottleneck_count_in_status(self):
        from thalos_prime.sbi.microcircuit import MicrocircuitController
        ctrl = MicrocircuitController()
        for i in range(MicrocircuitController.BOTTLENECK_QUEUE_DEPTH + 1):
            ctrl.enqueue({"data": i})
        status = ctrl.status()
        assert "bottleneck_events" in status
        assert status["bottleneck_events"] >= 1
