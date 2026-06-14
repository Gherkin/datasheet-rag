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

from aws_rag.store.control import (
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
    get_doc_titles,
    get_ingested_docs,
    insert_chunk_graph,
    insert_chunks,
    list_figure_chunks,
    resolve_doc_id,
    resolve_figure_path,
    set_doc_title,
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
    "get_chunk",
    "get_doc_titles",
    "get_ingested_docs",
    "get_metadata",
    "hash_token",
    "hybrid_search",
    "init_schema",
    "insert_chunk_graph",
    "insert_chunks",
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
    "set_metadata",
    "to_relative_figure_path",
    "update_figure_description",
    "vector_search",
]
