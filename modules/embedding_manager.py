"""
modules/embedding_manager.py
Loads the sentence-transformers model once (cached) and provides
helpers for encoding text and computing cosine similarity.
Optionally uses a cross-encoder to re-rank evidence chunks for higher
accuracy (requires no extra packages — sentence-transformers already
ships cross-encoder support).
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, CROSS_ENCODER_MODEL

# ---------------------------------------------------------------------------
# Module-level singletons – loaded once per Streamlit session
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None
_cross_encoder = None  # sentence_transformers.CrossEncoder | None


def get_model() -> SentenceTransformer:
    """Return the cached SentenceTransformer model, loading it if necessary."""
    global _model  # noqa: PLW0603
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_cross_encoder():
    """
    Return the cached CrossEncoder model, loading it lazily.

    Returns None if CROSS_ENCODER_MODEL is empty or if the model
    cannot be loaded (e.g. no internet on first run).
    """
    global _cross_encoder  # noqa: PLW0603
    if _cross_encoder is not None:
        return _cross_encoder
    if not CROSS_ENCODER_MODEL:
        return None
    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        return _cross_encoder
    except Exception:
        return None


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
    query_text: str = "",
    use_cross_encoder: bool = True,
) -> list[tuple[str, float]]:
    """
    Return the *k* chunks most similar to the query embedding.

    If *use_cross_encoder* is True and a CrossEncoder model is available,
    the top-2k bi-encoder candidates are re-scored by the cross-encoder for
    higher accuracy. Falls back silently to bi-encoder-only ranking.

    Parameters
    ----------
    query_embedding  : np.ndarray  shape (D,)
    chunk_embeddings : np.ndarray  shape (N, D)
    chunks           : list[str]   the original text chunks
    k                : int
    query_text       : str         original query text, required for cross-encoder
    use_cross_encoder: bool        whether to attempt cross-encoder re-ranking

    Returns
    -------
    list of (chunk_text, score) sorted descending by score
    """
    scores = chunk_embeddings @ query_embedding  # shape (N,)
    top_indices = np.argsort(scores)[::-1]

    # Cross-encoder re-ranking: fetch top-2k candidates then re-score
    if use_cross_encoder and query_text:
        ce = get_cross_encoder()
        if ce is not None:
            candidate_k = min(k * 2, len(chunks))
            candidate_indices = top_indices[:candidate_k]
            pairs = [[query_text, chunks[i]] for i in candidate_indices]
            try:
                ce_scores = ce.predict(pairs)
                ranked = sorted(
                    zip(candidate_indices, ce_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                return [(chunks[i], float(s)) for i, s in ranked[:k]]
            except Exception:
                pass  # fall through to bi-encoder result

    return [(chunks[i], float(scores[i])) for i in top_indices[:k]]
