"""SQLite-backed vector + keyword store for the RAG pipeline.

This package owns the local persistence layer:

* `schema`   – connection factory and DDL (chunks, chunk_vecs, chunk_fts,
               revisions, schema_version).
* `sqlite`   – CRUD helpers for chunks and vectors.
* `search`   – vector / keyword / hybrid retrieval.
* `documents` – documents (`D12`) and the order of their revisions (PDFs).
* `metadata` – document metadata and the per-revision sidecar.
* `fields`   – registry of typed, filterable metadata fields and their values.
* `identity` – the optional rule that links a new PDF to a stored document.

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
from datasheet_rag.store.fields import (
    FieldSpec,
    MetadataFieldError,
    list_fields,
    register_field,
    remove_field,
)
from datasheet_rag.store.metadata import (
    DocMetadata,
    TitleSource,
    delete_metadata,
    get_metadata,
    list_docs,
    set_metadata,
)
from datasheet_rag.store.schema import (
    FtsOutOfSyncError,
    FtsStatus,
    connect,
    fts_status,
    init_schema,
    rebuild_fts,
    require_fts_in_sync,
    stored_embedding_dim,
)
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
    resolve_figure_path,
    resolve_rev_id,
    set_doc_title,
    set_figure_source,
    to_relative_figure_path,
    update_figure_description,
)

__all__ = [
    "ApiKeyRecord",
    "DocMetadata",
    "FieldSpec",
    "MetadataFieldError",
    "list_fields",
    "register_field",
    "remove_field",
    "SearchFilters",
    "SearchResult",
    "TitleSource",
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
    "FtsOutOfSyncError",
    "FtsStatus",
    "fts_status",
    "init_schema",
    "rebuild_fts",
    "require_fts_in_sync",
    "stored_embedding_dim",
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
    "resolve_rev_id",
    "resolve_figure_path",
    "revoke_api_key",
    "set_doc_title",
    "set_figure_source",
    "set_metadata",
    "to_relative_figure_path",
    "update_figure_description",
    "vector_search",
]
