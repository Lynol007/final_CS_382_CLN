"""
Vector store: turn chunks into vectors and support similarity search over them.

Design decisions (see README):
- Embeddings use sentence-transformers "all-MiniLM-L6-v2": a real semantic
  embedding model that runs locally, free, with no API key. Unlike the TF-IDF
  baseline from the starter, it matches by meaning ("stop procrastinating"
  finds passages about starting small) instead of exact word overlap.
- Vectors are L2-normalised and searched with a dot product (= cosine
  similarity). At ~600 chunks an in-memory matrix is faster than a vector
  database; the build/query interface stays identical if FAISS/Chroma is
  swapped in later.
- Embeddings are cached on disk keyed by a fingerprint of the corpus + model,
  so the app only pays the encoding cost when documents actually change.
- If sentence-transformers is not installed, the store falls back to TF-IDF
  so the app still runs; the UI shows which backend is active.
"""

import hashlib
import json
import os
from typing import List, Tuple

import numpy as np

from .ingest import Chunk

_INDEX_DIR = os.path.join("data", "index")
_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, index_dir: str = _INDEX_DIR):
        self.index_dir = index_dir
        self.chunks: List[Chunk] = []
        self.matrix = None  # (n_chunks, dim) L2-normalised float32
        self._model = None
        self._tfidf = None
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            self.backend = "embeddings"
            # Cosine floor: below this, treat as "nothing found". Tuned on the
            # eval set: valid niche queries score ~0.23+, off-topic ones ~0.17.
            self.min_relevance = 0.20
        except ImportError:
            self.backend = "tfidf"
            self.min_relevance = 0.05

    # ---------- build ----------

    def build(self, chunks: List[Chunk]) -> None:
        """Vectorize all chunks, loading from the disk cache when unchanged."""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(stop_words="english")
            self.matrix = self._tfidf.fit_transform(texts)
            return

        fingerprint = self._fingerprint(texts)
        cached = self._load_cache(fingerprint, expected_rows=len(texts))
        if cached is not None:
            self.matrix = cached
            return

        model = self._get_model()
        vectors = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        self.matrix = np.asarray(vectors, dtype=np.float32)
        self._save_cache(fingerprint)

    # ---------- query ----------

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a query string."""
        if self.matrix is None:
            raise RuntimeError("VectorStore.build() must be called before query().")

        if self.backend == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity

            query_vec = self._tfidf.transform([query_text])
            scores = cosine_similarity(query_vec, self.matrix).flatten()
        else:
            query_vec = self._get_model().encode([query_text], normalize_embeddings=True)
            scores = (self.matrix @ np.asarray(query_vec, dtype=np.float32).T).flatten()

        top_k = min(top_k, len(self.chunks))
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_idx]

    # ---------- internals ----------

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(_MODEL_NAME)
        return self._model

    def _fingerprint(self, texts: List[str]) -> str:
        h = hashlib.sha256(_MODEL_NAME.encode())
        for t in texts:
            h.update(t.encode("utf-8"))
        return h.hexdigest()

    def _cache_paths(self):
        return (
            os.path.join(self.index_dir, "embeddings.npy"),
            os.path.join(self.index_dir, "meta.json"),
        )

    def _load_cache(self, fingerprint: str, expected_rows: int):
        npy_path, meta_path = self._cache_paths()
        if not (os.path.exists(npy_path) and os.path.exists(meta_path)):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") != fingerprint:
                return None
            matrix = np.load(npy_path)
            if matrix.shape[0] != expected_rows:
                return None
            return matrix
        except (OSError, ValueError, json.JSONDecodeError):
            return None  # corrupt cache: rebuild from scratch

    def _save_cache(self, fingerprint: str) -> None:
        npy_path, meta_path = self._cache_paths()
        os.makedirs(self.index_dir, exist_ok=True)
        np.save(npy_path, self.matrix)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fingerprint, "model": _MODEL_NAME, "rows": int(self.matrix.shape[0])}, f)
