from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, store_path: Path, model_name: str = "all-MiniLM-L6-v2"):
        self.store_path = Path(store_path)
        self.model_name = model_name
        self._model = None
        self._doc_ids: list[str] = []
        self._vectors: Optional[np.ndarray] = None
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._index_file = self.store_path / "index.npz"
        self._ids_file = self.store_path / "doc_ids.pkl"
        self.load()

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("Loaded embedding model: %s", self.model_name)
            except Exception as e:
                logger.warning("Failed to load SentenceTransformer: %s", e)
                self._model = None
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        if model is None:
            return np.zeros((len(texts), 384), dtype=np.float32)
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._doc_ids:
            return
        vec = self.encode([text])
        if self._vectors is None:
            self._vectors = vec
        else:
            self._vectors = np.vstack([self._vectors, vec])
        self._doc_ids.append(doc_id)
        self.save()

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self._vectors is None or len(self._doc_ids) == 0:
            return []
        query_vec = self.encode([query])
        sims = (self._vectors @ query_vec.T).flatten()
        k = min(top_k, len(self._doc_ids))
        top_indices = np.argsort(sims)[::-1][:k]
        return [(self._doc_ids[i], float(sims[i])) for i in top_indices]

    def save(self) -> None:
        if self._vectors is not None:
            np.savez(self._index_file, vectors=self._vectors)
        with open(self._ids_file, "wb") as f:
            pickle.dump(self._doc_ids, f)

    def load(self) -> None:
        try:
            if self._index_file.exists() and self._ids_file.exists():
                data = np.load(self._index_file)
                self._vectors = data["vectors"]
                with open(self._ids_file, "rb") as f:
                    self._doc_ids = pickle.load(f)
                logger.debug("Loaded vector store: %d vectors", len(self._doc_ids))
        except Exception as e:
            logger.warning("Failed to load vector store: %s", e)
            self._doc_ids = []
            self._vectors = None

    def clear(self) -> None:
        self._doc_ids = []
        self._vectors = None
        if self._index_file.exists():
            self._index_file.unlink()
        if self._ids_file.exists():
            self._ids_file.unlink()
