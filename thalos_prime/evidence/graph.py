from __future__ import annotations

import logging
from typing import Optional

from thalos_prime.storage.db import Claim, EvidenceLink

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    logger.warning("networkx not installed; ClaimEvidenceGraph unavailable")


class ClaimEvidenceGraph:
    def __init__(self):
        if not HAS_NX:
            raise ImportError("networkx is required for ClaimEvidenceGraph")

    def build_from_db(self, session) -> "nx.DiGraph":
        from sqlalchemy import select
        G = nx.DiGraph()
        claims = session.execute(select(Claim)).scalars().all()
        for c in claims:
            G.add_node(c.id, coordinate=c.coordinate, status=c.status, text=c.text[:100])
        links = session.execute(select(EvidenceLink)).scalars().all()
        for link in links:
            G.add_edge(
                link.evidence_doc_id or "unknown",
                link.claim_id,
                link_type=link.link_type,
                score=link.semantic_score,
            )
        return G

    def get_claim_subgraph(self, claim_id: str) -> "nx.DiGraph":
        raise NotImplementedError("Call build_from_db first and use G.subgraph()")

    def get_support_count(self, claim_id: str, G: Optional["nx.DiGraph"] = None) -> int:
        if G is None:
            return 0
        count = 0
        for _, _, data in G.in_edges(claim_id, data=True):
            if data.get("link_type") == "supports":
                count += 1
        return count

    def get_refute_count(self, claim_id: str, G: Optional["nx.DiGraph"] = None) -> int:
        if G is None:
            return 0
        count = 0
        for _, _, data in G.in_edges(claim_id, data=True):
            if data.get("link_type") == "refutes":
                count += 1
        return count

    def get_connected_claims(self, claim_id: str, G: Optional["nx.DiGraph"] = None) -> list[str]:
        if G is None or claim_id not in G:
            return []
        neighbors = set(G.predecessors(claim_id)) | set(G.successors(claim_id))
        return [n for n in neighbors if n != claim_id]
