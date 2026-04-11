from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from thalos_prime.config import Settings
from thalos_prime.contradiction.models import ContradictionResult
from thalos_prime.storage.db import Claim, ContradictionPair, get_session
from thalos_prime.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ContradictionDetector:
    def __init__(self, settings: Settings, vector_store: Optional[VectorStore] = None):
        self.settings = settings
        self.vector_store = vector_store
        self._model = None
        self._model_failed = False

    def _load_model(self):
        if self._model is not None or self._model_failed:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.settings.nli_model)
            logger.info("Loaded NLI model: %s", self.settings.nli_model)
        except Exception as e:
            logger.warning("Failed to load NLI model: %s. Using cosine similarity fallback.", e)
            self._model_failed = True
            self._model = None
        return self._model

    def classify_pair(self, text_a: str, text_b: str) -> tuple[str, float]:
        model = self._load_model()
        if model is not None:
            try:
                scores = model.predict([(text_a, text_b)])
                score_arr = scores[0] if hasattr(scores[0], "__len__") else scores
                if len(score_arr) == 3:
                    labels = ["contradiction", "entailment", "neutral"]
                    idx = int(np.argmax(score_arr))
                    return labels[idx], float(score_arr[idx])
                else:
                    score_val = float(score_arr[0]) if hasattr(score_arr, "__len__") else float(score_arr)
                    label = "entailment" if score_val > 0.5 else "contradiction"
                    return label, score_val
            except Exception as e:
                logger.warning("NLI classification failed: %s", e)

        if self.vector_store is not None:
            try:
                vecs = self.vector_store.encode([text_a, text_b])
                cos_sim = float(np.dot(vecs[0], vecs[1]))
                if cos_sim > 0.85:
                    return "entailment", cos_sim
                elif cos_sim < 0.2:
                    return "contradiction", 1.0 - cos_sim
                else:
                    return "neutral", cos_sim
            except Exception as e:
                logger.warning("Cosine similarity fallback failed: %s", e)

        return "neutral", 0.5

    def detect(self, claims: list[Claim], session) -> ContradictionResult:
        result = ContradictionResult()
        if len(claims) < 2:
            return result

        from sqlalchemy import select

        texts = [c.text for c in claims]
        try:
            if self.vector_store is not None:
                vecs = self.vector_store.encode(texts)
                sims = vecs @ vecs.T
            else:
                sims = None
        except Exception:
            sims = None

        candidates: list[tuple[int, int]] = []
        n = len(claims)

        if sims is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(sims[i, j])
                    if 0.3 <= sim <= 0.85:
                        candidates.append((i, j))
        else:
            candidates = [(i, j) for i in range(n) for j in range(i + 1, min(n, i + 10))]

        for i, j in candidates[:500]:
            claim_a = claims[i]
            claim_b = claims[j]
            pair_ids = sorted([claim_a.id, claim_b.id])
            existing = session.execute(
                select(ContradictionPair).where(
                    ContradictionPair.claim_a_id == pair_ids[0],
                    ContradictionPair.claim_b_id == pair_ids[1],
                )
            ).scalar_one_or_none()

            if existing:
                result.pairs_found += 1
                continue

            label, score = self.classify_pair(claim_a.text, claim_b.text)
            result.pairs_found += 1

            if label in ("contradiction", "entailment"):
                pair = ContradictionPair(
                    claim_a_id=pair_ids[0],
                    claim_b_id=pair_ids[1],
                    nli_score=score,
                    label=label,
                )
                session.add(pair)
                result.pairs_new += 1

        return result
