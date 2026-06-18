"""Embedding layer backed by sentence-transformers.

The model (nomic-ai/nomic-embed-text-v1) is loaded exactly once, lazily, the
first time an embedding is requested. nomic requires trust_remote_code=True
because it ships custom modeling code.
"""
from __future__ import annotations

import threading

from config import settings

_model = None
_lock = threading.Lock()

# nomic-embed-text-v1 is trained with task-instruction prefixes. We use
# search_document for stored chunks and search_query for queries.
_DOCUMENT_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


def _load_model():
    """Load and cache the SentenceTransformer model (thread-safe singleton)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(
                    settings.EMBED_MODEL,
                    trust_remote_code=True,
                )
    return _model


def warmup() -> None:
    """Force the model to load now (called at startup)."""
    _load_model()


def get_embedding(text: str, *, is_query: bool = False) -> list[float]:
    """Embed a single string and return a plain list of floats."""
    model = _load_model()
    prefix = _QUERY_PREFIX if is_query else _DOCUMENT_PREFIX
    vector = model.encode(prefix + text, normalize_embeddings=True)
    return vector.tolist()


def get_embeddings_batch(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    """Embed many strings at once for efficient bulk ingestion."""
    if not texts:
        return []
    model = _load_model()
    prefix = _QUERY_PREFIX if is_query else _DOCUMENT_PREFIX
    prefixed = [prefix + t for t in texts]
    vectors = model.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]
