"""Bedrock-backed text embedding for the RAG pipeline.

Public surface is intentionally tiny: the :class:`BedrockEmbedder` class
for callers that want metrics and reuse, plus the
:func:`embed_chunk_graph` and :func:`embed_texts` helpers for one-shot
use.

See :mod:`datasheet_rag.embedding.embedder` for implementation details
(request shape, concurrency, retry semantics, cost reference).
"""

from __future__ import annotations

from datasheet_rag.embedding.embedder import (
    BedrockEmbedder,
    Embedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    embed_chunk_graph,
    embed_texts,
    get_embedder,
)

__all__ = [
    "BedrockEmbedder",
    "Embedder",
    "OllamaEmbedder",
    "SentenceTransformerEmbedder",
    "embed_chunk_graph",
    "embed_texts",
    "get_embedder",
]
