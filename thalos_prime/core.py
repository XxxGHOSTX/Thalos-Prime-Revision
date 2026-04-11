"""
Thalos Prime Core Orchestrator
================================

ThalosCore is the top-level entry point that wires all subsystems
together into a single, autonomous knowledge-processing pipeline:

  Ingestion → Verification → Contradiction Resolution → Repository
      ↓                                                      ↓
  SBI Evaluation ←────────────────────────── Insight Engine
      ↓
  Node Synchronisation (AES-256-GCM encrypted + RSA-PSS signed)
      ↓
  Self-Optimisation
"""

from __future__ import annotations

import uuid
from typing import Any

from thalos_prime.autonomy.encryption import AsymmetricEncryption
from thalos_prime.autonomy.node import NodeRegistry, NodeSynchronizer
from thalos_prime.autonomy.optimizer import SelfOptimiser
from thalos_prime.interface.monitor import SystemMonitor
from thalos_prime.knowledge.contradiction import ContradictionResolver, ResolutionStrategy
from thalos_prime.knowledge.ingestion import IngestionPipeline
from thalos_prime.knowledge.insight import InsightEngine
from thalos_prime.knowledge.verification import VerificationEngine
from thalos_prime.models import Claim, KnowledgeObject, NodeInfo, SourceType
from thalos_prime.sbi.microcircuit import MicrocircuitController
from thalos_prime.sbi.organoid import OrganoidNetwork
from thalos_prime.storage.repository import KnowledgeRepository


class ThalosCore:
    """
    Top-level orchestrator for Thalos Prime.

    All subsystems are created and owned by this class.  External
    code should interact exclusively through this interface.
    """

    def __init__(
        self,
        node_host: str = "localhost",
        node_port: int = 8080,
        persist_path: str | None = None,
        organoid_clusters: int = 4,
        resolution_strategy: ResolutionStrategy = ResolutionStrategy.HIGHEST_CONFIDENCE,
    ) -> None:
        # --- Identity & encryption ---
        self._crypto = AsymmetricEncryption()
        local_node = NodeInfo(
            node_id=str(uuid.uuid4()),
            host=node_host,
            port=node_port,
            public_key_pem=self._crypto.public_key_pem(),
        )

        # --- Knowledge pipeline ---
        self.ingestion = IngestionPipeline()
        self.verification = VerificationEngine()
        self.contradiction_resolver = ContradictionResolver(strategy=resolution_strategy)
        self.repository = KnowledgeRepository(persist_path=persist_path)
        self.insight_engine = InsightEngine()

        # --- SBI layer ---
        self.organoid_network = OrganoidNetwork(initial_clusters=organoid_clusters)
        # Wire bottleneck detection → organoid growth
        self.microcircuit = MicrocircuitController(
            growth_callback=self.organoid_network.trigger_growth
        )

        # --- Autonomy layer ---
        self.node_registry = NodeRegistry(local_node=local_node)
        # Sign outgoing sync messages with this node's RSA private key
        self.synchronizer = NodeSynchronizer(
            local_node=local_node,
            signing_key=self._crypto,
        )
        self.optimiser = SelfOptimiser(network=self.organoid_network)

        # --- Monitoring ---
        self.monitor = SystemMonitor(core=self)

    # ------------------------------------------------------------------
    # Primary ingestion API
    # ------------------------------------------------------------------

    def ingest(
        self,
        content: str,
        source_uri: str = "api://manual",
        source_type: str = "text",
        reliability: float = 0.7,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeObject:
        """
        Ingest a single text claim through the full processing pipeline.

        Returns the resulting KnowledgeObject.
        """
        # 1. Ingest
        claim = self.ingestion.ingest_text(
            content=content,
            source_uri=source_uri,
            source_type=SourceType(source_type),
            reliability=reliability,
            tags=tags or [],
            metadata=metadata or {},
        )

        # 2. SBI evaluation
        sbi_result = self.organoid_network.evaluate(
            {"content": content, "tags": tags or []}
        )
        claim.metadata["sbi_activation"] = sbi_result.get("activation", [])
        claim.metadata["sbi_cluster"] = sbi_result.get("cluster_id", "")

        # 3. Verify
        claim = self.verification.verify(claim)

        # 4. Wrap in KnowledgeObject and persist
        ko = KnowledgeObject(canonical_claim=claim)
        self.repository.put(ko, reason="initial_ingest")
        self.verification.register_knowledge_object(ko)

        return ko

    def ingest_batch(
        self,
        records: list[dict[str, Any]],
        default_source_uri: str = "batch://upload",
        default_reliability: float = 0.6,
    ) -> list[KnowledgeObject]:
        """Ingest multiple claims, run contradiction detection across the batch."""
        claims = self.ingestion.ingest_batch(
            records,
            default_source_uri=default_source_uri,
            default_reliability=default_reliability,
        )

        # Verify all
        claims = self.verification.verify_batch(claims)

        # Detect contradictions within the batch
        contradiction_records = self.contradiction_resolver.detect(claims)

        # Build a claim id → claim map for resolution
        claim_map: dict[str, Claim] = {c.claim_id: c for c in claims}
        for record in contradiction_records:
            a = claim_map.get(record.claim_a_id)
            b = claim_map.get(record.claim_b_id)
            if a and b:
                winner, record = self.contradiction_resolver.resolve(a, b, record)

        # Persist all as KnowledgeObjects
        knowledge_objects: list[KnowledgeObject] = []
        for claim in claims:
            ko = KnowledgeObject(canonical_claim=claim)
            self.repository.put(ko, reason="batch_ingest")
            self.verification.register_knowledge_object(ko)
            knowledge_objects.append(ko)

        return knowledge_objects

    # ------------------------------------------------------------------
    # Insight API
    # ------------------------------------------------------------------

    def run_insight_cycle(self) -> list[Any]:
        """Analyse the repository and return newly generated insights."""
        return self.insight_engine.analyse(self.repository.all())

    # ------------------------------------------------------------------
    # Node API
    # ------------------------------------------------------------------

    def register_peer(self, node: NodeInfo) -> None:
        self.node_registry.register(node)
        # Register the peer's public key so we can verify their signatures
        if node.public_key_pem:
            self.synchronizer.register_peer_public_key(node.node_id, node.public_key_pem)

    def prepare_sync_payload(self) -> bytes:
        """Produce an encrypted, signed sync message from recent knowledge."""
        delta = [
            {
                "object_id": ko.object_id,
                "content": ko.canonical_claim.content,
                "confidence": ko.canonical_claim.confidence,
                "status": ko.canonical_claim.status.value,
            }
            for ko in self.repository.all()[-50:]  # last 50 objects
        ]
        return self.synchronizer.prepare_sync_message(delta)

    def receive_sync_payload(self, encrypted: bytes) -> int:
        """
        Receive and merge a sync payload from a peer.

        Returns the number of new knowledge objects merged.
        """
        msg = self.synchronizer.parse_sync_message(encrypted)
        if not msg:
            return 0
        merged = 0
        for item in msg.knowledge_delta:
            if not self.repository.get(item.get("object_id", "")):
                self.ingest(
                    content=item.get("content", ""),
                    source_uri=f"node://{msg.sender_node_id}",
                    source_type="peer_node",
                    reliability=0.6,
                )
                merged += 1
        return merged

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def run_optimisation_cycle(self) -> list[Any]:
        """Execute one self-optimisation cycle."""
        self.organoid_network.tick()
        return self.optimiser.run_cycle()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_report(self) -> dict[str, Any]:
        """Return a combined health report for all subsystems."""
        repo_objects = self.repository.all()
        status_counts: dict[str, int] = {}
        for ko in repo_objects:
            s = ko.canonical_claim.status.value
            status_counts[s] = status_counts.get(s, 0) + 1

        return {
            "node_id": self.node_registry.local_node.node_id,
            "knowledge_objects": len(repo_objects),
            "status_distribution": status_counts,
            "organoid_network": self.organoid_network.health_report(),
            "microcircuit": self.microcircuit.status(),
            "peer_nodes": self.node_registry.peer_count(),
            "insights_generated": len(self.insight_engine.all_insights),
            "optimiser_actions": len(self.optimiser.action_history),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Start the Thalos Prime API server.

    Usage::

        thalos-prime [--host HOST] [--port PORT] [--persist PATH]
                     [--clusters N]
    """
    import argparse

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required to run the server. Install it with: pip install uvicorn")
        raise SystemExit(1)

    from thalos_prime.interface.api import create_app

    parser = argparse.ArgumentParser(
        description="Thalos Prime – Autonomous Epistemic Compiler"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument(
        "--persist", default=None, metavar="PATH",
        help="JSON file for persistent knowledge storage",
    )
    parser.add_argument(
        "--clusters", type=int, default=4,
        help="Initial organoid cluster count (default: 4)",
    )
    args = parser.parse_args()

    core = ThalosCore(
        node_host=args.host,
        node_port=args.port,
        persist_path=args.persist,
        organoid_clusters=args.clusters,
    )
    app = create_app(core)
    print(f"Thalos Prime starting on http://{args.host}:{args.port}")
    print(f"  Node ID  : {core.node_registry.local_node.node_id}")
    print(f"  Clusters : {core.organoid_network.cluster_count}")
    print(f"  Persist  : {args.persist or 'in-memory (no persistence)'}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
