"""SQLite-backed vector + keyword store for the RAG pipeline.

This package owns the local persistence layer:

* `schema`   – connection factory and DDL (chunks, chunk_vecs, chunk_fts,
               doc_metadata, schema_version).
* `sqlite`   – CRUD helpers for chunks and vectors.
* `search`   – vector / keyword / hybrid retrieval.
* `metadata` – document-level metadata sidecar (independent of chunks).

The store is intentionally small: module-level functions rather than
classes, mirroring the style of `aws_rag.storage` and `aws_rag.aws`.
"""

from __future__ import annotations

from aws_rag.store.metadata import (
    DocMetadata,
    apply_metadata_to_chunks,
    get_metadata,
    list_docs,
    set_metadata,
)
from aws_rag.store.schema import connect, init_schema
from aws_rag.store.search import (
    SearchFilters,
    SearchResult,
    hybrid_search,
    keyword_search,
    vector_search,
)
from aws_rag.store.sqlite import (
    count_chunks,
    delete_doc,
    get_chunk,
    insert_chunk_graph,
    insert_chunks,
    list_figure_chunks,
    update_figure_description,
)

__all__ = [
    "DocMetadata",
    "SearchFilters",
    "SearchResult",
    "apply_metadata_to_chunks",
    "connect",
    "count_chunks",
    "delete_doc",
    "get_chunk",
    "get_metadata",
    "hybrid_search",
    "init_schema",
    "insert_chunk_graph",
    "insert_chunks",
    "keyword_search",
    "list_docs",
    "list_figure_chunks",
    "set_metadata",
    "update_figure_description",
    "vector_search",
]
