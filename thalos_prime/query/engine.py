from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from thalos_prime.config import Settings
from thalos_prime.storage.db import Claim, EvidenceLink
from thalos_prime.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

STATUS_RANK = {"draft": 0, "candidate": 1, "verified": 2, "authoritative": 3, "retracted": -1}


class QueryEngine:
    def __init__(self, settings: Settings, vector_store: VectorStore, session):
        self.settings = settings
        self.vector_store = vector_store
        self.session = session

    def search(self, query: str, top_k: int = 10, min_status: str = "draft") -> list[dict]:
        from sqlalchemy import select

        min_rank = STATUS_RANK.get(min_status, 0)
        all_claims = self.session.execute(select(Claim)).scalars().all()
        claims = [
            c for c in all_claims
            if STATUS_RANK.get(c.status, -1) >= min_rank and c.status != "retracted"
        ]

        if not claims:
            return []

        claim_texts = [c.text for c in claims]

        bm25_scores = self._bm25_scores(query, claim_texts)
        semantic_scores = self._semantic_scores(query, claim_texts)

        bm25_ranked = self._rank(bm25_scores)
        semantic_ranked = self._rank(semantic_scores)

        k_rrf = 60
        rrf_scores = {}
        for claim_idx, rank in enumerate(bm25_ranked):
            rrf_scores[claim_idx] = rrf_scores.get(claim_idx, 0.0) + 1.0 / (k_rrf + rank + 1)
        for claim_idx, rank in enumerate(semantic_ranked):
            rrf_scores[claim_idx] = rrf_scores.get(claim_idx, 0.0) + 1.0 / (k_rrf + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        top_indices = sorted_indices[:top_k]

        results = []
        for idx in top_indices:
            claim = claims[idx]
            evidence_count = self.session.execute(
                select(EvidenceLink).where(EvidenceLink.claim_id == claim.id)
            ).scalars().all()
            results.append({
                "claim_id": claim.id,
                "coordinate": claim.coordinate,
                "text": claim.text,
                "status": claim.status,
                "language": claim.language,
                "score": rrf_scores[idx],
                "evidence_count": len(evidence_count),
            })

        return results

    def _bm25_scores(self, query: str, corpus: list[str]) -> list[float]:
        if not corpus:
            return []
        try:
            from rank_bm25 import BM25Okapi

            def tokenize(text: str) -> list[str]:
                return text.lower().split()

            bm25 = BM25Okapi([tokenize(t) for t in corpus])
            scores = bm25.get_scores(tokenize(query))
            return [float(s) for s in scores]
        except Exception as e:
            logger.warning("BM25 query failed: %s", e)
            return [0.0] * len(corpus)

    def _semantic_scores(self, query: str, corpus: list[str]) -> list[float]:
        if not corpus:
            return []
        try:
            all_texts = [query] + corpus
            vecs = self.vector_store.encode(all_texts)
            query_vec = vecs[0:1]
            corpus_vecs = vecs[1:]
            norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            corpus_vecs = corpus_vecs / norms
            q_norm = query_vec / max(np.linalg.norm(query_vec), 1e-9)
            scores = (corpus_vecs @ q_norm.T).flatten()
            return [float(s) for s in scores]
        except Exception as e:
            logger.warning("Semantic query failed: %s", e)
            return [0.0] * len(corpus)

    def _rank(self, scores: list[float]) -> list[int]:
        indexed = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for rank_pos, orig_idx in enumerate(indexed):
            ranks[orig_idx] = rank_pos
        return ranks
