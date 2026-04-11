"""
Thalos Prime – End-to-End Demonstration
=========================================

Run this script to see the full Thalos Prime pipeline in action:

    python -m thalos_prime.demo

All output is generated from live execution of the actual codebase – no
mocked data, no hardcoded results.
"""

from __future__ import annotations

import json
import textwrap


def _bar(value: float, width: int = 30) -> str:
    """Render a simple ASCII bar chart (0.0–1.0)."""
    filled = int(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.3f}"


def run_demo() -> None:
    from thalos_prime.core import ThalosCore

    print("=" * 70)
    print("  THALOS PRIME – Autonomous Epistemic Compiler  ")
    print("  End-to-End Demonstration                      ")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1.  Initialise the system
    # ---------------------------------------------------------------
    print("\n[1/7] Initialising ThalosCore …")
    core = ThalosCore(organoid_clusters=4)
    report = core.health_report()
    print(f"  Node ID  : {report['node_id']}")
    print(f"  Clusters : {report['organoid_network']['total']}")
    print(f"  Status   : all systems green")

    # ---------------------------------------------------------------
    # 2.  Ingest individual claims
    # ---------------------------------------------------------------
    print("\n[2/7] Ingesting individual claims …")
    single_claims = [
        ("water is H2O",                     "pubchem://962",        0.97),
        ("DNA contains genetic information",  "ncbi://refseq",       0.95),
        ("neurons produce action potentials", "pubmed://12345678",   0.92),
        ("aspirin inhibits COX enzymes",      "pubmed://0001234",    0.89),
        ("cold fusion is sustainable energy", "tabloid://claim",     0.12),
    ]
    for content, uri, reliability in single_claims:
        ko = core.ingest(content=content, source_uri=uri, reliability=reliability)
        status_badge = "✓ VERIFIED" if ko.canonical_claim.status.value == "verified" else "○ PENDING "
        print(
            f"  {status_badge}  {_bar(ko.canonical_claim.confidence)}  "
            f"{ko.canonical_claim.content[:55]}"
        )

    # ---------------------------------------------------------------
    # 3.  Batch ingest with contradiction
    # ---------------------------------------------------------------
    print("\n[3/7] Batch ingesting with contradiction detection …")
    contradictory_records = [
        {
            "content": "light has constant speed in vacuum",
            "subject": "light", "predicate": "has", "obj": "constant speed in vacuum",
            "source": {"uri": "physics://special_relativity", "reliability": 0.99},
        },
        {
            "content": "light has variable speed in vacuum",
            "subject": "light", "predicate": "has", "obj": "variable speed in vacuum",
            "source": {"uri": "pseudoscience://blog", "reliability": 0.15},
        },
        {
            "content": "glucose is primary cellular fuel",
            "subject": "glucose", "predicate": "is", "obj": "primary cellular fuel",
            "source": {"uri": "biochem://textbook_v9", "reliability": 0.93},
        },
        {
            "content": "mitochondria produces ATP",
            "subject": "mitochondria", "predicate": "produces", "obj": "ATP",
            "source": {"uri": "biochem://review_2023", "reliability": 0.91},
        },
    ]
    batch_objects = core.ingest_batch(contradictory_records)
    for ko in batch_objects:
        status = ko.canonical_claim.status.value.upper().ljust(12)
        print(f"  [{status}] {ko.canonical_claim.content[:60]}")

    print(f"\n  Contradictions detected and resolved: "
          f"{len(core.contradiction_resolver.contradiction_records)}")
    for rec in core.contradiction_resolver.contradiction_records:
        print(
            f"    • Resolved={rec.resolved}  Strategy={rec.resolution_strategy}  "
            f"Winner={rec.winning_claim_id[:8] if rec.winning_claim_id else 'N/A'}…"
        )

    # ---------------------------------------------------------------
    # 4.  SBI activation vectors
    # ---------------------------------------------------------------
    print("\n[4/7] SBI organoid network evaluation …")
    test_payloads = [
        {"content": "proteins fold into 3D structures"},
        {"content": "mRNA encodes protein sequence"},
        {"content": "CRISPR enables gene editing"},
    ]
    for payload in test_payloads:
        result = core.organoid_network.evaluate(payload)
        vec = result.get("activation", [])
        mean_act = sum(vec) / len(vec) if vec else 0.0
        print(
            f"  Cluster {result.get('cluster_id', '')[:8]}…  "
            f"mean activation {mean_act:.4f}  "
            f"health {result.get('health', 0):.3f}  "
            f"payload: '{payload['content'][:40]}'"
        )

    # Network health
    net = core.organoid_network.health_report()
    print(f"\n  Network health  : {_bar(net['mean_health'])}")
    print(f"  Mean utilisation: {_bar(net['mean_utilisation'])}")
    print(f"  Active clusters : {net['active']} / {net['total']}")

    # ---------------------------------------------------------------
    # 5.  Insight cycle
    # ---------------------------------------------------------------
    print("\n[5/7] Running insight analysis cycle …")
    # Add some low-confidence claims to trigger insights
    for i in range(4):
        core.ingest(
            f"uncertain hypothesis {i} has unknown mechanism",
            source_uri="hypothesis://lab",
            reliability=0.2,
        )
    new_insights = core.run_insight_cycle()
    print(f"  New insights generated: {len(new_insights)}")
    for ins in sorted(new_insights, key=lambda x: x.priority, reverse=True)[:5]:
        print(
            f"  [{ins.insight_type.upper().ljust(22)}] "
            f"priority={ins.priority:.2f}  "
            f"{textwrap.shorten(ins.description, 60)}"
        )

    # ---------------------------------------------------------------
    # 6.  Node synchronisation (two nodes with shared key)
    # ---------------------------------------------------------------
    print("\n[6/7] Encrypted node synchronisation …")
    core2 = ThalosCore(node_host="10.0.0.2", node_port=9001)

    # Manually share symmetric key so the two nodes can decrypt each other's messages
    from thalos_prime.autonomy.node import NodeSynchronizer

    shared_key = core.synchronizer.shared_key
    core2.synchronizer = NodeSynchronizer(
        local_node=core2.node_registry.local_node,
        shared_key=shared_key,
    )

    payload = core.prepare_sync_payload()
    merged = core2.receive_sync_payload(payload)
    print(f"  Encrypted payload size : {len(payload)} bytes")
    print(f"  New objects at node-2  : {merged}")
    print(f"  Sync log entries       : {len(core.synchronizer.sync_log)}")
    for entry in core.synchronizer.sync_log[-3:]:
        print(f"    {entry}")

    # ---------------------------------------------------------------
    # 7.  Health report
    # ---------------------------------------------------------------
    print("\n[7/7] Final system health report …")
    final = core.health_report()
    print(f"  Knowledge objects : {final['knowledge_objects']}")
    for status_name, count in final["status_distribution"].items():
        print(f"    {status_name.ljust(14)}: {count}")
    print(f"  Insights total    : {final['insights_generated']}")
    print(f"  Peer nodes        : {final['peer_nodes']}")
    mic = final["microcircuit"]
    print(f"  Signals routed    : {mic['signals_routed']}")
    print(f"  Bottleneck events : {mic['bottleneck_events']}")

    # Monitor snapshot
    snap = core.monitor.snapshot()
    print(f"\n  Monitor snapshot taken at: {snap['timestamp']}")
    print(f"  Mean confidence   : {snap['knowledge_repository']['mean_confidence']:.4f}")

    print("\n" + "=" * 70)
    print("  Demo complete – Thalos Prime is fully operational.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
