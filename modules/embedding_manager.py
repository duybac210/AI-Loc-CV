"""
modules/embedding_manager.py
Loads the sentence-transformers model once (cached) and provides
helpers for encoding text and computing cosine similarity.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Module-level singleton – loaded once per Streamlit session
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return the cached SentenceTransformer model, loading it if necessary."""
    global _model  # noqa: PLW0603
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings into L2-normalised embedding vectors.

    Returns
    -------
    np.ndarray  shape (N, D)
    """
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # L2-normalise so that dot product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.
    Both vectors must already be L2-normalised (as returned by `encode`).
    """
    return float(np.dot(vec_a, vec_b))


def top_k_chunks(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: list[str],
    k: int = 3,
) -> list[tuple[str, float]]:
    """
    Return the *k* chunks most similar to the query embedding.

    Parameters
    ----------
    query_embedding  : np.ndarray  shape (D,)
    chunk_embeddings : np.ndarray  shape (N, D)
    chunks           : list[str]   the original text chunks
    k                : int

    Returns
    -------
    list of (chunk_text, score) sorted descending by score
    """
    scores = chunk_embeddings @ query_embedding  # shape (N,)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top_indices]
