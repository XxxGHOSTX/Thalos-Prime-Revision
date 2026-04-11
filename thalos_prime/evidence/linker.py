from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from thalos_prime.config import Settings
from thalos_prime.evidence.models import EvidenceResult
from thalos_prime.storage.db import Claim, EvidenceLink, get_session
from thalos_prime.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EvidenceLinker:
    def __init__(self, settings: Settings, vector_store: VectorStore):
        self.settings = settings
        self.vector_store = vector_store

    def _bm25_scores(self, claim_text: str, corpus: list[str]) -> list[float]:
        if not corpus:
            return []
        try:
            from rank_bm25 import BM25Okapi

            def tokenize(text: str) -> list[str]:
                return text.lower().split()

            tokenized_corpus = [tokenize(s) for s in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenize(claim_text))
            max_score = float(max(scores)) if max(scores) > 0 else 1.0
            return [float(s) / max_score for s in scores]
        except Exception as e:
            logger.warning("BM25 failed: %s", e)
            return [0.0] * len(corpus)

    def _semantic_scores(self, claim_text: str, corpus: list[str]) -> list[float]:
        if not corpus:
            return []
        try:
            all_texts = [claim_text] + corpus
            vecs = self.vector_store.encode(all_texts)
            claim_vec = vecs[0:1]
            corpus_vecs = vecs[1:]
            norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            corpus_vecs_norm = corpus_vecs / norms
            claim_norm = claim_vec / max(np.linalg.norm(claim_vec), 1e-9)
            scores = (corpus_vecs_norm @ claim_norm.T).flatten()
            return [float(s) for s in scores]
        except Exception as e:
            logger.warning("Semantic scoring failed: %s", e)
            return [0.0] * len(corpus)

    def link(self, claim: Claim, corpus_sentences: list[tuple[str, str]]) -> EvidenceResult:
        result = EvidenceResult(claim_id=claim.id)
        if not corpus_sentences:
            return result

        doc_ids = [s[0] for s in corpus_sentences]
        texts = [s[1] for s in corpus_sentences]

        bm25_scores = self._bm25_scores(claim.text, texts)
        semantic_scores = self._semantic_scores(claim.text, texts)

        combined = []
        for i in range(len(texts)):
            score = 0.4 * bm25_scores[i] + 0.6 * semantic_scores[i]
            combined.append((i, score, bm25_scores[i], semantic_scores[i]))

        combined.sort(key=lambda x: x[1], reverse=True)
        top_k = combined[: self.settings.evidence_top_k]

        with get_session(self.settings.database_url) as session:
            for i, score, bm25_s, sem_s in top_k:
                if score < 0.1:
                    continue
                result.links_found += 1
                if sem_s > 0.7:
                    link_type = "supports"
                elif sem_s < 0.2:
                    link_type = "refutes"
                else:
                    link_type = "neutral"

                link = EvidenceLink(
                    claim_id=claim.id,
                    evidence_text=texts[i][:2000],
                    evidence_doc_id=doc_ids[i] if doc_ids[i] != "unknown" else None,
                    bm25_score=bm25_s,
                    semantic_score=sem_s,
                    link_type=link_type,
                )
                session.add(link)
                result.links_new += 1

        return result
