"""SQLite-backed vector + keyword store for the RAG pipeline.

This package owns the local persistence layer:

* `schema`   – connection factory and DDL (chunks, chunk_vecs, chunk_fts,
               doc_metadata, schema_version).
* `sqlite`   – CRUD helpers for chunks and vectors.
* `search`   – vector / keyword / hybrid retrieval.
* `metadata` – document-level metadata sidecar (independent of chunks).

The store is intentionally small: module-level functions rather than
classes, mirroring the style of `datasheet_rag.storage` and `datasheet_rag.aws`.
"""

from __future__ import annotations

from datasheet_rag.store.control import (
    ApiKeyRecord,
    count_api_keys,
    create_api_key,
    hash_token,
    list_api_keys,
    list_audit,
    lookup_api_key,
    record_audit,
    revoke_api_key,
)
from datasheet_rag.store.metadata import (
    DocMetadata,
    apply_metadata_to_chunks,
    delete_metadata,
    get_metadata,
    list_docs,
    set_metadata,
)
from datasheet_rag.store.schema import connect, init_schema
from datasheet_rag.store.search import (
    SearchFilters,
    SearchResult,
    hybrid_search,
    keyword_search,
    vector_search,
)
from datasheet_rag.store.sqlite import (
    InsertStats,
    count_chunks,
    delete_doc,
    figure_source_available,
    get_chunk,
    get_doc_titles,
    get_ingested_docs,
    insert_chunk_graph,
    insert_chunks,
    insert_chunks_stats,
    list_figure_chunks,
    resolve_doc_id,
    resolve_figure_path,
    set_doc_title,
    set_figure_source,
    to_relative_figure_path,
    update_figure_description,
)

__all__ = [
    "ApiKeyRecord",
    "DocMetadata",
    "SearchFilters",
    "SearchResult",
    "apply_metadata_to_chunks",
    "connect",
    "count_api_keys",
    "count_chunks",
    "create_api_key",
    "delete_doc",
    "delete_metadata",
    "figure_source_available",
    "get_chunk",
    "get_doc_titles",
    "get_ingested_docs",
    "get_metadata",
    "hash_token",
    "hybrid_search",
    "init_schema",
    "InsertStats",
    "insert_chunk_graph",
    "insert_chunks",
    "insert_chunks_stats",
    "keyword_search",
    "list_api_keys",
    "list_audit",
    "list_docs",
    "list_figure_chunks",
    "lookup_api_key",
    "record_audit",
    "resolve_doc_id",
    "resolve_figure_path",
    "revoke_api_key",
    "set_doc_title",
    "set_figure_source",
    "set_metadata",
    "to_relative_figure_path",
    "update_figure_description",
    "vector_search",
]
