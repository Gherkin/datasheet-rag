"""MCP stdio server exposing the local RAG store as tools.

Tools surfaced to the LLM agent:

* ``search``                — hybrid / vector / keyword retrieval over chunks.
* ``get_chunk``             — fetch one chunk by ID (with optional neighbours).
* ``navigate``              — generic graph step: parent / children / prev / next / chapter_root.
* ``zoom_in`` / ``zoom_out``— sugar over ``navigate`` for the common cases.
* ``list_documents``        — what's in the current project (or filtered).
* ``get_document_metadata`` — the sidecar row for a document.
* ``stats``                 — chunk counts per level / doc, for sanity checking.

Project scoping
---------------
The server is intended to be launched **once per project** by Claude Code via
a ``.mcp.json`` entry. The default project is read from the
``RAG_DEFAULT_PROJECT_ID`` env var (or ``settings.default_project_id``); every
tool accepts an optional ``project_id`` override.

Connection / embedder caching
-----------------------------
The SQLite connection and Bedrock embedder are built lazily on first use and
cached for the lifetime of the process. SQLite is opened with
``check_same_thread=False`` so concurrent tool calls share one connection.

Pure-vs-transport split
-----------------------
Each ``@mcp.tool()`` is a thin wrapper around a ``_impl`` function. Tests
exercise the ``_impl`` layer directly without needing the MCP transport.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from aws_rag.config import get_settings
from aws_rag.models.chunk import ChunkLevel, LayoutType
from aws_rag.store import (
    SearchFilters,
    SearchResult,
    apply_metadata_to_chunks,  # noqa: F401  (re-exported convenience)
    connect,
    count_chunks,
    get_chunk as store_get_chunk,
    get_metadata,
    hybrid_search,
    keyword_search,
    list_docs,
    vector_search,
)

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_conn_lock = Lock()
_conn: sqlite3.Connection | None = None
_conn_path: str | None = None

_embedder_lock = Lock()
_embedder: Any | None = None  # BedrockEmbedder, imported lazily to keep import cost low


def _get_conn(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Return a process-wide cached SQLite connection.

    Re-opens if the requested path differs from the cached one (rare —
    practically every call uses the default).
    """
    global _conn, _conn_path
    settings = get_settings()
    target = str(db_path or settings.sqlite_db_path)
    with _conn_lock:
        if _conn is None or _conn_path != target:
            if _conn is not None:
                _conn.close()
            _conn = connect(target)
            _conn_path = target
        return _conn


def _get_embedder() -> Any:
    """Return a process-wide cached BedrockEmbedder. Imported lazily."""
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            from aws_rag.embedding import BedrockEmbedder

            _embedder = BedrockEmbedder()
        return _embedder


def _resolve_project(project_id: str | None) -> str | None:
    """Resolve the effective project_id for a tool call.

    Priority: explicit arg → settings.default_project_id (RAG_DEFAULT_PROJECT_ID env).
    Returns None if neither set, meaning "search globally".
    """
    if project_id is not None:
        return project_id
    return get_settings().default_project_id


def _resolve_level(level: str | None) -> ChunkLevel | None:
    if level is None:
        return None
    mapping = {
        "macro": ChunkLevel.MACRO,
        "meso": ChunkLevel.MESO,
        "micro": ChunkLevel.MICRO,
    }
    if level.lower() not in mapping:
        raise ValueError(f"Unknown level '{level}'. Use one of: macro, meso, micro.")
    return mapping[level.lower()]


def _resolve_layout_types(layout_types: list[str] | None) -> list[LayoutType] | None:
    if not layout_types:
        return None
    out: list[LayoutType] = []
    for lt in layout_types:
        try:
            out.append(LayoutType(lt.lower()))
        except ValueError as e:
            raise ValueError(
                f"Unknown layout_type '{lt}'. Valid: "
                f"{[m.value for m in LayoutType]}"
            ) from e
    return out


# ---------------------------------------------------------------------------
# Result shaping — lean dicts the LLM can read efficiently
# ---------------------------------------------------------------------------


def _shape_chunk(result_or_chunk: Any, *, score: float | None = None) -> dict[str, Any]:
    """Convert a Chunk or SearchResult into a compact dict for the agent."""
    if isinstance(result_or_chunk, SearchResult):
        chunk = result_or_chunk.chunk
        score = result_or_chunk.score
    else:
        chunk = result_or_chunk

    page = ""
    pages = chunk.metadata.page_numbers
    if pages:
        page = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"

    is_figure = chunk.metadata.layout_type == LayoutType.FIGURE
    has_figure = is_figure and bool(chunk.figure_image_path or chunk.figure_s3_key)

    out: dict[str, Any] = {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "level": chunk.level.name,
        "page": page,
        "section": chunk.metadata.section_title or "",
        "chapter": chunk.metadata.chapter_title or "",
        "text": chunk.text,
        "parent_id": chunk.parent_id,
        "prev_id": chunk.prev_id,
        "next_id": chunk.next_id,
        "chapter_root_id": chunk.chapter_root_id,
    }
    if has_figure:
        # The agent uses this flag to decide whether to call get_figure.
        out["has_figure"] = True
        out["figure_caption"] = chunk.figure_caption or ""
        if chunk.figure_description:
            out["figure_description"] = chunk.figure_description
        # Stable URI the agent can show / link to without fetching bytes.
        out["figure_uri"] = f"rag://figure/{chunk.id}"
    if score is not None:
        out["score"] = round(float(score), 4)
    return out


# ---------------------------------------------------------------------------
# Tool impls — pure functions, tested directly
# ---------------------------------------------------------------------------

SearchMode = Literal["hybrid", "vector", "keyword"]


def _search_impl(
    query: str,
    *,
    mode: SearchMode = "hybrid",
    k: int = 5,
    project_id: str | None = None,
    doc_id: str | None = None,
    level: str | None = None,
    layout_types: list[str] | None = None,
    conn: sqlite3.Connection | None = None,
    embedder: Any | None = None,
) -> list[dict[str, Any]]:
    """Run a search and return a list of result dicts."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    conn = conn or _get_conn()

    filters = SearchFilters(
        doc_ids=[doc_id] if doc_id else None,
        project_id=_resolve_project(project_id),
        level=_resolve_level(level),
        layout_types=_resolve_layout_types(layout_types),
    )

    if mode in ("vector", "hybrid"):
        emb = embedder or _get_embedder()
        query_vec = emb.embed_one(query)
    else:
        query_vec = None

    if mode == "vector":
        results = vector_search(conn, query_vec, k=k, filters=filters)  # type: ignore[arg-type]
    elif mode == "keyword":
        results = keyword_search(conn, query, k=k, filters=filters)
    else:
        results = hybrid_search(conn, query_vec, query, k=k, filters=filters)  # type: ignore[arg-type]

    return [_shape_chunk(r) for r in results]


def _get_chunk_impl(
    chunk_id: str,
    *,
    include_neighbors: bool = False,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Fetch a chunk by ID. If include_neighbors, also embed prev/next/parent text."""
    conn = conn or _get_conn()
    chunk = store_get_chunk(conn, chunk_id)
    if chunk is None:
        return None
    out = _shape_chunk(chunk)
    if include_neighbors:
        neighbors: dict[str, dict[str, Any] | None] = {}
        for key, nid in [
            ("parent", chunk.parent_id),
            ("prev", chunk.prev_id),
            ("next", chunk.next_id),
        ]:
            if nid:
                nchunk = store_get_chunk(conn, nid)
                neighbors[key] = _shape_chunk(nchunk) if nchunk else None
            else:
                neighbors[key] = None
        out["neighbors"] = neighbors
    return out


Direction = Literal["parent", "children", "prev", "next", "chapter_root"]


def _navigate_impl(
    chunk_id: str,
    direction: Direction,
    *,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Step through the chunk graph.

    Returns a list because ``children`` is one-to-many; the other directions
    return at most one element.
    """
    conn = conn or _get_conn()
    chunk = store_get_chunk(conn, chunk_id)
    if chunk is None:
        return []

    if direction == "children":
        # Children aren't denormalised onto the chunk row — query by parent_id.
        rows = conn.execute(
            "SELECT id FROM chunks WHERE parent_id = ? ORDER BY rowid",
            (chunk_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            child = store_get_chunk(conn, row["id"])
            if child is not None:
                out.append(_shape_chunk(child))
        return out

    target_id: str | None
    if direction == "parent":
        target_id = chunk.parent_id
    elif direction == "prev":
        target_id = chunk.prev_id
    elif direction == "next":
        target_id = chunk.next_id
    elif direction == "chapter_root":
        target_id = chunk.chapter_root_id
    else:
        raise ValueError(
            f"Unknown direction '{direction}'. Use: parent, children, prev, next, chapter_root."
        )

    if target_id is None:
        return []
    target = store_get_chunk(conn, target_id)
    return [_shape_chunk(target)] if target else []


def _list_documents_impl(
    *,
    project_id: str | None = None,
    group: str | None = None,
    mpn: str | None = None,
    manufacturer: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """List documents in the metadata sidecar, optionally filtered."""
    conn = conn or _get_conn()
    docs = list_docs(
        conn,
        project_id=_resolve_project(project_id),
        group_name=group,
        mpn=mpn,
    )
    # list_docs doesn't filter by manufacturer; do it client-side.
    if manufacturer is not None:
        docs = [d for d in docs if d.manufacturer == manufacturer]

    return [
        {
            "doc_id": d.doc_id,
            "project_id": d.project_id,
            "group_name": d.group_name,
            "mpn": d.mpn,
            "manufacturer": d.manufacturer,
            "subsystem": d.subsystem,
            "doc_type": d.doc_type,
            "tags": d.tags,
        }
        for d in docs
    ]


def _get_document_metadata_impl(
    doc_id: str,
    *,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    conn = conn or _get_conn()
    meta = get_metadata(conn, doc_id)
    if meta is None:
        return None
    return meta.model_dump(exclude_none=False)


def _figure_image_bytes(chunk: Any) -> tuple[bytes, str]:
    """Read a figure chunk's image, returning ``(bytes, format)``.

    Prefers the local cropped file (``figure_image_path``). Falls back to
    S3 (``figure_s3_key``) using the configured bucket. Raises if neither
    is available or the file is missing.
    """
    if chunk.figure_image_path:
        path = Path(chunk.figure_image_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"figure_image_path on chunk {chunk.id} points to a missing "
                f"file: {path}"
            )
        fmt = path.suffix.lstrip(".").lower() or "png"
        return path.read_bytes(), fmt

    if chunk.figure_s3_key:
        from aws_rag.aws import s3_client

        settings = get_settings()
        client = s3_client()
        resp = client.get_object(Bucket=settings.s3_bucket, Key=chunk.figure_s3_key)
        data = resp["Body"].read()
        ext = Path(chunk.figure_s3_key).suffix.lstrip(".").lower() or "png"
        return data, ext

    raise ValueError(
        f"chunk {chunk.id} has no figure_image_path or figure_s3_key — "
        f"nothing to fetch."
    )


def _get_figure_impl(
    chunk_id: str,
    *,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Return everything needed to render and cite a figure.

    Includes:
    - ``image_bytes`` — raw file bytes (base64 when serialized over MCP)
    - ``format`` — 'png' / 'jpg' / 'webp' for MIME detection
    - ``caption``, ``description`` — text the agent can use for reasoning
    - ``citation`` — page / section / doc for showing the user where it came from

    Tests exercise this directly; the MCP tool wrapper repackages the
    bytes into an MCP ``Image`` content block.
    """
    conn = conn or _get_conn()
    chunk = store_get_chunk(conn, chunk_id)
    if chunk is None:
        raise ValueError(f"unknown chunk_id: {chunk_id}")
    if chunk.metadata.layout_type != LayoutType.FIGURE:
        raise ValueError(
            f"chunk {chunk_id} is not a figure "
            f"(layout_type={chunk.metadata.layout_type.value})"
        )

    image_bytes, fmt = _figure_image_bytes(chunk)

    pages = chunk.metadata.page_numbers
    page = (str(pages[0]) if len(pages) == 1
            else f"{pages[0]}-{pages[-1]}" if pages else "")

    return {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "image_bytes": image_bytes,
        "format": fmt,
        "local_path": str(chunk.figure_image_path) if chunk.figure_image_path else None,
        "caption": chunk.figure_caption or "",
        "description": chunk.figure_description or "",
        "citation": {
            "doc_id": chunk.doc_id,
            "page": page,
            "section": chunk.metadata.section_title or "",
            "chapter": chunk.metadata.chapter_title or "",
        },
    }


def _stats_impl(
    *,
    project_id: str | None = None,
    doc_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Return chunk counts. Useful for the agent to sanity-check the corpus size."""
    conn = conn or _get_conn()
    pid = _resolve_project(project_id)

    total = count_chunks(conn, doc_id=doc_id, project_id=pid)

    by_level: dict[str, int] = {}
    where: list[str] = []
    params: list[Any] = []
    if doc_id:
        where.append("doc_id = ?")
        params.append(doc_id)
    if pid:
        where.append("project_id = ?")
        params.append(pid)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    rows = conn.execute(
        f"SELECT level, COUNT(*) AS c FROM chunks{where_sql} GROUP BY level",
        params,
    ).fetchall()
    for row in rows:
        try:
            name = ChunkLevel(int(row["level"])).name
        except ValueError:
            name = str(row["level"])
        by_level[name] = int(row["c"])

    return {
        "total_chunks": total,
        "by_level": by_level,
        "project_id": pid,
        "doc_id": doc_id,
    }


# ---------------------------------------------------------------------------
# MCP server construction
# ---------------------------------------------------------------------------


_FIGURE_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


def build_server() -> Any:
    """Construct and return the FastMCP server with all tools registered.

    Imported lazily so the module is testable without the ``mcp`` SDK
    installed in the sandbox.
    """
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ImageContent, TextContent

    settings = get_settings()
    project_hint = settings.default_project_id or "(unscoped)"
    mcp = FastMCP(
        name=f"aws-rag[{project_hint}]",
        instructions=(
            "Tools for searching and navigating a project-scoped RAG database "
            "of electronics datasheets and reference manuals. Prefer "
            "`search` with mode='hybrid' for most queries — it combines "
            "BM25 keyword matching (good for part numbers, register names, "
            "exact signal names) with dense vector similarity (good for "
            "conceptual questions). Use mode='keyword' for exact-match "
            "lookups of identifiers, mode='vector' for conceptual queries. "
            "After finding a relevant chunk, use `zoom_out` to read the "
            "wider section summary, `zoom_in` for finer detail, or "
            "`navigate` with direction='next' to read sequentially. "
            "\n\n"
            "Figures: search and get_chunk results include `has_figure: "
            "true` and a `figure_uri` when the chunk represents a "
            "diagram, schematic, or block-diagram. Read the "
            "`figure_description` and `figure_caption` to reason about "
            "the figure without fetching it. Call `get_figure(chunk_id)` "
            "only when the user explicitly asks to see the image, or "
            "when the description is insufficient to answer."
        ),
    )

    @mcp.tool()
    def search(
        query: str,
        mode: SearchMode = "hybrid",
        k: int = 5,
        project_id: str | None = None,
        doc_id: str | None = None,
        level: str | None = None,
        layout_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the RAG database for chunks relevant to a query.

        Hybrid mode (default) combines BM25 keyword + dense vector with
        Reciprocal Rank Fusion. Use `keyword` for exact part numbers /
        register names, `vector` for conceptual questions.

        Optional filters:
        - `project_id`: override the server's default project.
        - `doc_id`: restrict to a single document.
        - `level`: 'macro' (chapter summaries), 'meso' (subsections),
                   'micro' (paragraphs / tables).
        - `layout_types`: restrict to text/table/figure/key_value/list.
        """
        return _search_impl(
            query, mode=mode, k=k, project_id=project_id,
            doc_id=doc_id, level=level, layout_types=layout_types,
        )

    @mcp.tool()
    def get_chunk(chunk_id: str, include_neighbors: bool = False) -> dict[str, Any] | None:
        """Fetch a chunk by ID. If include_neighbors, also returns parent/prev/next."""
        return _get_chunk_impl(chunk_id, include_neighbors=include_neighbors)

    @mcp.tool()
    def navigate(chunk_id: str, direction: Direction) -> list[dict[str, Any]]:
        """Step through the chunk graph: parent | children | prev | next | chapter_root."""
        return _navigate_impl(chunk_id, direction)

    @mcp.tool()
    def zoom_in(chunk_id: str) -> list[dict[str, Any]]:
        """Get finer-grained child chunks (e.g. macro -> meso, meso -> micro)."""
        return _navigate_impl(chunk_id, "children")

    @mcp.tool()
    def zoom_out(chunk_id: str) -> list[dict[str, Any]]:
        """Get the parent chunk (broader context, e.g. micro -> meso, meso -> macro)."""
        return _navigate_impl(chunk_id, "parent")

    @mcp.tool()
    def list_documents(
        project_id: str | None = None,
        group: str | None = None,
        mpn: str | None = None,
        manufacturer: str | None = None,
    ) -> list[dict[str, Any]]:
        """List documents in the current (or specified) project."""
        return _list_documents_impl(
            project_id=project_id, group=group, mpn=mpn, manufacturer=manufacturer,
        )

    @mcp.tool()
    def get_document_metadata(doc_id: str) -> dict[str, Any] | None:
        """Fetch the metadata sidecar row for a document (mpn, manufacturer, tags, …)."""
        return _get_document_metadata_impl(doc_id)

    @mcp.tool()
    def stats(
        project_id: str | None = None,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """Return chunk counts (total + by level) for the current scope."""
        return _stats_impl(project_id=project_id, doc_id=doc_id)

    @mcp.tool()
    def get_figure(chunk_id: str):
        """Fetch the image for a figure chunk and return it for the user to see.

        Returns a list of MCP content blocks: an Image block (the actual
        figure, rendered inline by clients that support it) followed by a
        text block with caption, description, and citation. Raises
        ``ValueError`` if the chunk_id is unknown or not a figure chunk.

        Use ``search`` (or ``get_chunk``) first — figure chunks have
        ``has_figure: true`` and a ``figure_uri`` field. Call this tool
        when the user asks to see the figure, or when reasoning from the
        description alone is insufficient.
        """
        import base64
        result = _get_figure_impl(chunk_id)
        mime = _FIGURE_MIME.get(result["format"], "image/png")
        image_b64 = base64.b64encode(result["image_bytes"]).decode()
        caption = result["caption"]
        description = result["description"]
        citation = result["citation"]
        path_line = (
            f"\n  file: {result['local_path']}"
            if result.get("local_path") else ""
        )
        text_summary = (
            f"Figure {result['chunk_id']}\n"
            f"  caption: {caption or '(none)'}\n"
            f"  description: {description or '(none generated yet)'}\n"
            f"  source: doc {citation['doc_id'][:10]} · "
            f"page {citation['page']} · section {citation['section']!r}"
            f"{path_line}"
        )
        return [
            ImageContent(type="image", data=image_b64, mimeType=mime),
            TextContent(type="text", text=text_summary),
        ]

    @mcp.resource("rag://figure/{chunk_id}")
    def figure_resource(chunk_id: str) -> bytes:
        """Stable URI for a figure. Clients can dereference this on demand.

        Returned as raw bytes; the MCP SDK negotiates the content type
        from the resource registration. See ``get_figure`` for the
        in-tool-result rendering path.
        """
        result = _get_figure_impl(chunk_id)
        return result["image_bytes"]

    return mcp


def main() -> None:
    """Console-script entry point. Runs the MCP server on stdio."""
    # Honor RAG_DEFAULT_PROJECT_ID set in the .mcp.json env block.
    if "RAG_DEFAULT_PROJECT_ID" in os.environ:
        # Bust the settings cache so the env var is picked up.
        get_settings.cache_clear()  # type: ignore[attr-defined]

    server = build_server()
    # Helpful diagnostic on startup (goes to stderr — stdout is the MCP transport).
    settings = get_settings()
    print(
        json.dumps({
            "event": "rag-mcp.start",
            "db_path": str(settings.sqlite_db_path),
            "default_project_id": settings.default_project_id,
            "embedding_model_id": settings.embedding_model_id,
        }),
        file=sys.stderr,
    )
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
