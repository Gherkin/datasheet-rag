"""MCP server exposing the RAG store as tools, over stdio or HTTP.

Tools surfaced to the LLM agent:

* ``search``                — hybrid / vector / keyword retrieval over chunks.
* ``get_chunk``             — fetch one chunk by ID (with optional neighbours).
* ``navigate``              — generic graph step: parent / children / prev / next / chapter_root.
* ``zoom_in`` / ``zoom_out``— sugar over ``navigate`` for the common cases.
* ``list_documents``        — what's in the current project (or filtered).
* ``get_document_metadata`` — the sidecar row for a document.
* ``stats``                 — chunk counts per level / doc, for sanity checking.

Transports
----------
``main()`` runs this on stdio, for a ``rag-mcp`` process launched by the
client. :mod:`datasheet_rag.server.mcp_mount` serves the same tools from the
RAG server's FastAPI app at ``/mcp``, so a client can point at the server
directly with nothing installed locally (GH #39). ``build_server`` takes the
two arguments that separate those worlds: the backend every call goes through,
and whether the client shares this machine.

Project scoping
---------------
The default project is resolved per call by ``_resolve_project``: an explicit
``project_id`` argument wins, then the project carried by the current request
(the ``/mcp/<project_id>`` path segment or an ``X-RAG-Project`` header, HTTP
only), then a ``.rag.toml`` discovered by walking up from the server's working
directory (same discovery as the CLI, stdio only), then the
``RAG_DEFAULT_PROJECT_ID`` env var (or ``settings.default_project_id``).
Running Claude Code inside a checkout with a ``.rag.toml`` therefore scopes
every stdio tool call to that project automatically, without per-project
``.mcp.json``/env configuration; over HTTP the URL carries the same
information, since the server's own working directory means nothing to a
remote caller.

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
import sys
from contextvars import ContextVar
from threading import Thread
from typing import Any, Literal

from datasheet_rag import pdf_viewer
from datasheet_rag.backend import (
    FigureUnavailableError,
    RagBackend,
    backend_mode,
    emit_local_notice,
    get_backend,
)
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import ChunkLevel, LayoutType
from datasheet_rag.store import DocMetadata, SearchFilters, SearchResult

# ---------------------------------------------------------------------------
# Backend access — local sqlite or remote HTTP, chosen from config.
# ---------------------------------------------------------------------------


def _backend(
    backend: RagBackend | None,
    conn: Any | None = None,
    embedder: Any | None = None,
) -> RagBackend:
    """Resolve the backend for an impl call.

    Prefers an explicit ``backend``; otherwise, if a legacy ``conn`` (and/or
    ``embedder``) is supplied — as the test-suite does — wrap it in a
    ``LocalBackend``; otherwise fall back to the configured backend.
    """
    if backend is not None:
        return backend
    if conn is not None or embedder is not None:
        from datasheet_rag.backend import LocalBackend

        return LocalBackend(conn=conn, embedder=embedder)
    return get_backend()


# ---------------------------------------------------------------------------
# Request-scoped call environment.
#
# Both of these are per-request rather than process-wide. Over HTTP a single
# process serves every project and every client, so neither "which project is
# this" nor "can the caller open a loopback URL" is a property of the process.
# The HTTP mount sets them per request; stdio leaves them at their defaults,
# which describe a client sharing this machine.
# ---------------------------------------------------------------------------

#: The project a call is scoped to, from the ``/mcp/<project_id>`` path
#: segment or the ``X-RAG-Project`` header. Unset over stdio.
request_project: ContextVar[str | None] = ContextVar("rag_request_project", default=None)

#: Whether the MCP client shares this machine with the server — true for
#: stdio, false for the HTTP mount. See ``build_server``'s ``local_client``.
request_local_client: ContextVar[bool] = ContextVar("rag_request_local_client", default=True)


def _source_page_tool() -> str:
    """The tool to point users at for a page they cannot otherwise see.

    ``show_pdf`` hands back a loopback URL, which only means something when
    the client and the server share a machine; ``show_page`` renders inline
    and works anywhere.
    """
    return "show_pdf" if request_local_client.get() else "show_page"


def _resolve_project(project_id: str | None) -> str | None:
    """Resolve the effective project_id for a tool call.

    Priority: explicit arg → the current request's project (HTTP only) →
    ``.rag.toml`` discovered from the server's cwd (stdio only) →
    settings.default_project_id (RAG_DEFAULT_PROJECT_ID env). Returns None if
    none are set, meaning "search globally". The ``.rag.toml`` lookup mirrors
    ``resolve_cli_project_id`` so the MCP server scopes itself the same way
    the CLI does when launched inside a project checkout.
    """
    if project_id:
        return project_id

    scoped = request_project.get()
    if scoped:
        return scoped

    if request_local_client.get():
        from datasheet_rag.project_config import get_project_config

        config = get_project_config()
        if config is not None and config.project_id:
            return config.project_id

    return get_settings().default_project_id or None


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
                f"Unknown layout_type '{lt}'. Valid: {[m.value for m in LayoutType]}"
            ) from e
    return out


# ---------------------------------------------------------------------------
# Result shaping — lean dicts the LLM can read efficiently
# ---------------------------------------------------------------------------


def _figure_is_retrievable(chunk: Any) -> bool:
    """Whether ``show_figure``/``get_figure`` would actually return an image.

    ``figure_available`` is computed by the host that owns the figure store
    (see ``store.sqlite._row_to_chunk``), so it is authoritative in local mode
    and travels over HTTP in remote mode. ``None`` means nobody checked — a
    server older than GH #41 — so fall back to the historical
    is-the-column-set guess rather than hiding every figure it serves.
    """
    if chunk.figure_available is None:
        return bool(chunk.figure_image_path or chunk.figure_s3_key)
    return bool(chunk.figure_available)


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
    has_figure = is_figure and _figure_is_retrievable(chunk)

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
        out["has_figure"] = True
        out["figure_caption"] = chunk.figure_caption or ""
        if chunk.figure_description:
            out["figure_description"] = chunk.figure_description
        out["figure_uri"] = f"rag://figure/{chunk.id}"
        # Directive embedded in result data — more reliable than server instructions
        # because the model reads it at the moment it is deciding what to do next.
        out["DISPLAY_INSTRUCTION"] = (
            f"This chunk has a figure. If it is relevant to the user's question, "
            f"you MUST call show_figure('{chunk.id}') — do not skip it. "
            f"If relevance is uncertain, offer it to the user instead of silently skipping."
        )
    elif is_figure:
        # A figure-typed chunk with no image: caption text only. Say so
        # explicitly — silence here is what sent agents into a retry loop of
        # show_figure calls that could never succeed (GH #41).
        out["has_figure"] = False
        out["figure_caption"] = chunk.figure_caption or ""
        if chunk.figure_description:
            out["figure_description"] = chunk.figure_description
        out["DISPLAY_INSTRUCTION"] = (
            "This chunk describes a figure but no image is stored for it. "
            "Do NOT call show_figure or get_figure on it — they will fail. "
            f"To let the user see it, call {_source_page_tool()}('{chunk.doc_id}'"
            + (f", {pages[0]})" if pages else ")")
            + " instead."
        )
    if chunk.metadata.layout_type == LayoutType.TABLE and pages:
        warning = chunk.metadata.table_structure_warning
        if warning:
            out["table_structure_warning"] = warning
            out["DISPLAY_INSTRUCTION"] = (
                f"This table's structure could not be fully verified ({warning}). "
                f"Before citing specific values from it, call "
                f"show_page('{chunk.doc_id}', {pages[0]}) to visually check "
                f"the source page against the text above."
            )
        else:
            out["DISPLAY_INSTRUCTION"] = (
                "Docling's table extraction can mislabel headers or merge stray text into "
                "cells even when no warning is raised. If you're about to cite a specific "
                f"value from this table, consider calling show_page('{chunk.doc_id}', {pages[0]}) "
                "to visually confirm it against the source page."
            )
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
    backend: RagBackend | None = None,
    conn: Any | None = None,
    embedder: Any | None = None,
) -> list[dict[str, Any]]:
    """Run a search and return a list of result dicts."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    be = _backend(backend, conn, embedder)

    filters = SearchFilters(
        doc_ids=[doc_id] if doc_id else None,
        project_id=_resolve_project(project_id),
        level=_resolve_level(level),
        layout_types=_resolve_layout_types(layout_types),
    )

    results = be.search(query, mode=mode, k=k, filters=filters)
    return [_shape_chunk(r) for r in results]


def _get_chunk_impl(
    chunk_id: str,
    *,
    include_neighbors: bool = False,
    backend: RagBackend | None = None,
    conn: Any | None = None,
) -> dict[str, Any] | None:
    """Fetch a chunk by ID. If include_neighbors, also embed prev/next/parent text."""
    be = _backend(backend, conn)
    chunk = be.get_chunk(chunk_id)
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
                nchunk = be.get_chunk(nid)
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
    backend: RagBackend | None = None,
    conn: Any | None = None,
) -> list[dict[str, Any]]:
    """Step through the chunk graph.

    Returns a list because ``children`` is one-to-many; the other directions
    return at most one element.
    """
    be = _backend(backend, conn)
    chunk = be.get_chunk(chunk_id)
    if chunk is None:
        return []

    if direction == "children":
        return [_shape_chunk(c) for c in be.get_children(chunk_id)]

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
    target = be.get_chunk(target_id)
    return [_shape_chunk(target)] if target else []


def _list_documents_impl(
    *,
    project_id: str | None = None,
    group: str | None = None,
    mpn: str | None = None,
    manufacturer: str | None = None,
    backend: RagBackend | None = None,
    conn: Any | None = None,
) -> list[dict[str, Any]]:
    """List the store's documents, optionally filtered by sidecar metadata."""
    be = _backend(backend, conn)
    docs = be.list_documents(
        project_id=_resolve_project(project_id),
        group_name=group,
        mpn=mpn,
        manufacturer=manufacturer,
    )
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
            **({"doc_title": d.doc_title} if d.doc_title else {}),
            **({"page_count": d.page_count} if d.page_count is not None else {}),
        }
        for d in docs
    ]


def _get_document_metadata_impl(
    doc_id: str,
    *,
    backend: RagBackend | None = None,
    conn: Any | None = None,
) -> dict[str, Any] | None:
    be = _backend(backend, conn)
    meta = be.get_metadata(doc_id)
    summary = next((d for d in be.list_documents() if d.doc_id == doc_id), None)
    if meta is None and summary is None:
        return None

    # An ingested document need not have a sidecar row. Answering with the
    # empty row says "this document exists, nothing is tagged on it", which
    # is the truth; ``None`` reads as "no such document" and sends the caller
    # hunting for a doc_id that search and stats just handed them.
    result = (meta or DocMetadata(doc_id=doc_id)).model_dump(exclude_none=False)
    if summary is not None:
        if summary.doc_title:
            result["doc_title"] = summary.doc_title
        if summary.page_count is not None:
            result["page_count"] = summary.page_count
    return result


_MCP_IMAGE_BYTE_LIMIT = 700_000  # base64 of this fits safely under the 1 MB MCP limit


def _compress_for_mcp(image_bytes: bytes, fmt: str) -> tuple[bytes, str]:
    """Re-encode image as JPEG if it would exceed the 1 MB MCP tool-result limit.

    Base64 encoding inflates bytes by ~33%, so the safe ceiling for raw bytes
    is ~750 KB.  We use 700 KB to leave room for JSON envelope overhead.
    Returns (bytes, format) unchanged when already small enough.
    """
    if len(image_bytes) <= _MCP_IMAGE_BYTE_LIMIT:
        return image_bytes, fmt

    import io

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue(), "jpg"


def _figure_unavailable_text(chunk_id: str, exc: Exception) -> str:
    """The text a figure tool returns when there is no image to return.

    A hard error here reads like a transient fault and invites a retry; the
    agent then burns turns re-calling a tool that can never succeed (GH #41).
    So the tools answer normally, state that the image does not exist, and
    point at the one thing that does work — the source page.
    """
    doc_id = chunk_id.split(":", 1)[0]
    return (
        f"No image is stored for chunk {chunk_id} ({exc}). This is permanent, "
        f"not a transient failure — do not retry show_figure or get_figure on "
        f"this chunk, and do not try other figure chunks from the same "
        f"document expecting a different result. The chunk's own text and "
        f"caption are all there is. If the user needs to see the figure, call "
        f"{_source_page_tool()}('{doc_id}', <page>) with the chunk's page and let "
        f"them read it in the source document."
    )


def _get_figure_impl(
    chunk_id: str,
    *,
    backend: RagBackend | None = None,
    conn: Any | None = None,
) -> dict[str, Any]:
    """Return everything needed to render and cite a figure.

    Includes:
    - ``image_bytes`` — raw file bytes (base64 when serialized over MCP)
    - ``format`` — 'png' / 'jpg' / 'webp' for MIME detection
    - ``caption``, ``description`` — text the agent can use for reasoning
    - ``citation`` — page / section / doc for showing the user where it came from

    The figure bytes come from the backend, which reads them from the local
    figure store (local mode) or fetches them over HTTP from the server
    (remote mode). Tests exercise this directly; the MCP tool wrapper
    repackages the bytes into an MCP ``Image`` content block.
    """
    be = _backend(backend, conn)
    fig = be.get_figure_bytes(chunk_id)
    return {
        "chunk_id": fig.chunk_id,
        "doc_id": fig.doc_id,
        "image_bytes": fig.image_bytes(),
        "format": fig.format,
        "local_path": fig.local_path,
        "caption": fig.caption,
        "description": fig.description,
        "citation": fig.citation.model_dump(),
    }


def _stats_impl(
    *,
    project_id: str | None = None,
    doc_id: str | None = None,
    backend: RagBackend | None = None,
    conn: Any | None = None,
) -> dict[str, Any]:
    """Return chunk counts. Useful for the agent to sanity-check the corpus size."""
    be = _backend(backend, conn)
    pid = _resolve_project(project_id)
    result = be.stats(project_id=pid, doc_id=doc_id)
    return result.model_dump()


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


def build_server(
    backend: RagBackend | None = None,
    *,
    local_client: bool = True,
    **fastmcp_kwargs: Any,
) -> Any:
    """Construct and return the FastMCP server with all tools registered.

    Imported lazily so the module is testable without the ``mcp`` SDK
    installed in the sandbox.

    Args:
        backend: the backend every tool call goes through. ``None`` means
            "resolve from config" (stdio's behaviour). The HTTP mount passes
            its own ``LocalBackend`` so the server cannot recurse into a
            ``RemoteBackend`` pointing at itself.
        local_client: whether the MCP client runs on this machine — true for
            stdio, false for the HTTP mount. Two things hang off it. The
            loopback PDF viewer behind ``show_pdf`` is only reachable when
            client and server share a host, so over HTTP that tool is not
            registered at all and the agent is pointed at ``show_page``
            instead (GH #45 tracks serving PDFs from the server properly).
            And ``.rag.toml`` discovery from the working directory only makes
            sense when the client chose that directory; a remote caller scopes
            itself through the URL instead.

            This decides which tools get registered and how the instructions
            read — both fixed at build time. The same fact reaches the
            per-call text through ``request_local_client``, which the HTTP
            mount sets on every request.
        **fastmcp_kwargs: passed through to ``FastMCP`` (transport settings
            for the HTTP mount).
    """
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ImageContent, TextContent

    source_page_tool = "show_pdf" if local_client else "show_page"

    settings = get_settings()
    project_hint = settings.default_project_id or "(unscoped)"
    mcp = FastMCP(
        name=f"datasheet-rag[{project_hint}]",
        **fastmcp_kwargs,
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
            "true` when the chunk is a diagram, schematic, or block-diagram "
            "AND its image can actually be served. Use `figure_description` "
            "and `figure_caption` to reason about the content without "
            "fetching it. A chunk whose text reads like a figure but which "
            "carries `has_figure: false` has no image behind it — its "
            "document was ingested without figures. Never call `show_figure` "
            f"or `get_figure` on those; offer `{source_page_tool}` instead. "
            "\n\n"
            "Showing figures — decision rule: For every figure chunk found "
            "(has_figure: true), decide: "
            "(1) Clearly relevant — the figure directly illustrates what "
            "the user asked about (e.g. a timing diagram for a propagation "
            "delay question, a characteristic curve for an electrical spec "
            "question). You MUST call `show_figure`. This is not optional; "
            "do not skip it even if the text already answers the question. "
            "Visualisation paired with text is always more useful than text "
            "alone. "
            "(2) Possibly relevant — you are unsure whether the figure adds "
            "value. Do NOT silently skip it. Instead, mention it in your "
            "reply and offer it: e.g. 'I also found a figure showing X — "
            "it covers [brief aspect] rather than [what was asked]; want me "
            "to show it?' "
            "(3) Clearly unrelated — only then may you silently skip it. "
            "\n\n"
            "Showing figures — composition: Once you have decided to show a "
            "figure, treat its placement as a question of communication "
            "clarity. Ask yourself: where in my response does a reader most "
            "need to see this? Write your explanation up to that natural "
            "handoff point, call `show_figure` there, then continue. The "
            "goal is a cohesive response where each figure appears exactly "
            "where it illuminates the text around it. "
            "\n\n"
            "Showing figures — citation: Immediately after each `show_figure` "
            "call write one short line: caption (or brief description if "
            "none), page, and section. The widget shows only the image. "
            "\n\n"
            "Use `get_figure` only on non-Desktop hosts or when you need "
            "the raw bytes for your own visual analysis."
            "\n\n"
            "Source pages: Every chunk result includes a `doc_id` and a "
            "`page` field. When the user wants to see the original document "
            "— to read surrounding text, check the exact layout, or browse "
            f"adjacent pages — call `{source_page_tool}(doc_id, page)`. "
            + (
                "This opens a full interactive PDF viewer in the browser. "
                if local_client
                else "This renders the page inline. "
            )
            + "Good triggers: the user "
            "asks 'can I see the datasheet?', 'show me that page', or asks "
            "a detailed question about layout/formatting/context not captured "
            "in the chunk text. After an in-depth answer drawn from a "
            "specific chunk, proactively offer it: e.g. 'Want to see page "
            "12 in the original datasheet?'"
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
            query,
            mode=mode,
            k=k,
            project_id=project_id,
            doc_id=doc_id,
            level=level,
            layout_types=layout_types,
            backend=backend,
        )

    @mcp.tool()
    def get_chunk(chunk_id: str, include_neighbors: bool = False) -> dict[str, Any] | None:
        """Fetch a chunk by ID. If include_neighbors, also returns parent/prev/next."""
        return _get_chunk_impl(chunk_id, include_neighbors=include_neighbors, backend=backend)

    @mcp.tool()
    def navigate(chunk_id: str, direction: Direction) -> list[dict[str, Any]]:
        """Step through the chunk graph: parent | children | prev | next | chapter_root."""
        return _navigate_impl(chunk_id, direction, backend=backend)

    @mcp.tool()
    def zoom_in(chunk_id: str) -> list[dict[str, Any]]:
        """Get finer-grained child chunks (e.g. macro -> meso, meso -> micro)."""
        return _navigate_impl(chunk_id, "children", backend=backend)

    @mcp.tool()
    def zoom_out(chunk_id: str) -> list[dict[str, Any]]:
        """Get the parent chunk (broader context, e.g. micro -> meso, meso -> macro)."""
        return _navigate_impl(chunk_id, "parent", backend=backend)

    @mcp.tool()
    def list_documents(
        project_id: str | None = None,
        group: str | None = None,
        mpn: str | None = None,
        manufacturer: str | None = None,
    ) -> list[dict[str, Any]]:
        """List documents in the current (or specified) project.

        Every ingested document appears, whether or not metadata has been
        assigned to it; untagged ones come back with null sidecar fields.
        Filtering by group, mpn or manufacturer narrows this to documents
        that do have metadata, since that is the only place those live.
        """
        return _list_documents_impl(
            project_id=project_id,
            group=group,
            mpn=mpn,
            manufacturer=manufacturer,
            backend=backend,
        )

    @mcp.tool()
    def get_document_metadata(doc_id: str) -> dict[str, Any] | None:
        """Fetch the metadata sidecar row for a document (mpn, manufacturer, tags, …).

        An ingested document with no metadata assigned returns a row of null
        fields. ``None`` means no such document is in the store.
        """
        return _get_document_metadata_impl(doc_id, backend=backend)

    @mcp.tool()
    def stats(
        project_id: str | None = None,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """Return chunk counts (total + by level) for the current scope."""
        return _stats_impl(project_id=project_id, doc_id=doc_id, backend=backend)

    @mcp.tool()
    def get_figure(chunk_id: str):
        """Fetch raw figure bytes for further reasoning (fallback for non-Desktop hosts).

        Prefer ``show_figure`` in Claude Desktop — it renders the image inline
        as an interactive widget. Use this only on hosts that do not support
        MCP Apps, or when you need the image bytes for your own visual analysis.

        Returns an Image content block followed by a text block with caption,
        description, and citation. Do not emit markdown image links — the Image
        block is what the client renders.
        """
        import base64

        try:
            result = _get_figure_impl(chunk_id, backend=backend)
        except FigureUnavailableError as exc:
            return [TextContent(type="text", text=_figure_unavailable_text(chunk_id, exc))]
        image_bytes, fmt = _compress_for_mcp(result["image_bytes"], result["format"])
        mime = _FIGURE_MIME.get(fmt, "image/png")
        image_b64 = base64.b64encode(image_bytes).decode()
        caption = result["caption"]
        description = result["description"]
        citation = result["citation"]
        text_summary = (
            f"Figure {result['chunk_id']}\n"
            f"  caption: {caption or '(none)'}\n"
            f"  description: {description or '(none generated yet)'}\n"
            f"  source: doc {citation['doc_id'][:10]} · "
            f"page {citation['page']} · section {citation['section']!r}"
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
        result = _get_figure_impl(chunk_id, backend=backend)
        return result["image_bytes"]

    # ------------------------------------------------------------------
    # MCP Apps experiment — Goose-style inline UI for figures.
    # See https://modelcontextprotocol.io/extensions/apps/overview
    # ------------------------------------------------------------------

    # MCP Apps signal lives on the TOOL DEFINITION (`_meta` in tools/list),
    # not on the call result. Verified against Claude Desktop 1.8555 — the
    # built-in "imagine" server (show_widget) declares
    # ``_meta: {ui: {resourceUri: csA}}`` on the tool itself. Goose's
    # tutorial showed _meta on the call result, but Claude Desktop ignores
    # that placement entirely.
    @mcp.tool(meta={"ui": {"resourceUri": "ui://datasheet-rag/figure-app"}})
    def show_figure(chunk_id: str):
        """Display a figure as an inline rendered image widget (preferred in Claude Desktop).

        Calling this is MANDATORY when the figure is clearly relevant to the
        user's question — do not skip it even if the text already answers the
        question. Visualisation paired with text is always more useful.

        Place this call at the natural handoff point in your response where the
        reader most needs to see the figure, then continue writing after it.

        Immediately after this call, write one short line: caption (or brief
        description if none), page number, and section.
        """
        import base64

        try:
            result = _get_figure_impl(chunk_id, backend=backend)
        except FigureUnavailableError as exc:
            return [TextContent(type="text", text=_figure_unavailable_text(chunk_id, exc))]
        image_bytes, fmt = _compress_for_mcp(result["image_bytes"], result["format"])
        mime = _FIGURE_MIME.get(fmt, "image/png")
        image_b64 = base64.b64encode(image_bytes).decode()
        caption = result["caption"]
        citation = result["citation"]
        return [
            ImageContent(type="image", data=image_b64, mimeType=mime),
            TextContent(
                type="text",
                text=(
                    f"caption: {caption or '(none)'} | "
                    f"page {citation['page']} · {citation['section']}"
                ),
            ),
        ]

    @mcp.resource(
        "ui://datasheet-rag/figure-app",
        mime_type="text/html;profile=mcp-app",
    )
    def figure_app_html() -> str:
        """Figure renderer MCP App.

        The iframe receives:
        - chunk_id via ``ui/notifications/tool-input`` params.arguments
        - image base64 via ``ui/notifications/tool-result`` content[0].data
        - caption/citation text via content[1].text

        No extra tool calls needed — show_figure embeds the image directly
        in its return value.
        """
        return """<!doctype html>
<html><head><meta charset="utf-8"><style>
  html, body { margin: 0; padding: 0; width: 100%; }
  body { padding: 12px; font-family: system-ui, sans-serif; font-size: 13px;
         background: var(--color-background-primary, #fff);
         color: var(--color-text-primary, #000); box-sizing: border-box; }
  img { max-width: 100%; height: auto; border-radius: 4px; display: block; }
  .caption { margin-top: 8px; font-size: 0.85em; opacity: 0.7; }
  .err { color: #c44; font-size: 0.85em; }
  #loading { opacity: 0.5; }
</style></head>
<body>
  <div id="loading">Loading figure…</div>
  <img id="fig" style="display:none">
  <div id="caption" class="caption"></div>
  <div id="err" class="err"></div>
  <script>
    (function () {
      const loadingEl = document.getElementById('loading');
      const figEl = document.getElementById('fig');
      const captionEl = document.getElementById('caption');
      const errEl = document.getElementById('err');
      const pending = new Map();
      let nextId = 0;
      let toolResult = null;

      function sendSize() {
        window.parent.postMessage({
          jsonrpc: '2.0',
          method: 'ui/notifications/size-changed',
          params: { width: document.documentElement.scrollWidth, height: document.body.scrollHeight + 16 }
        }, '*');
      }

      function request(method, params) {
        return new Promise((resolve, reject) => {
          const id = ++nextId;
          pending.set(id, {resolve, reject});
          window.parent.postMessage({jsonrpc: '2.0', id, method, params}, '*');
          setTimeout(() => {
            if (pending.has(id)) { pending.delete(id); reject(new Error('timeout')); }
          }, 10000);
        });
      }

      function renderFigure() {
        if (!toolResult) return;
        const imgBlock = toolResult.find(c => c.type === 'image');
        const textBlock = toolResult.find(c => c.type === 'text');
        // No image block means the chunk has no stored figure — the text
        // block explains why, so show that rather than a bare 'no image'.
        if (!imgBlock) { errEl.textContent = textBlock ? textBlock.text : 'No image in tool-result.'; loadingEl.style.display = 'none'; sendSize(); return; }
        figEl.onload = () => { loadingEl.style.display = 'none'; sendSize(); };
        figEl.src = 'data:' + imgBlock.mimeType + ';base64,' + imgBlock.data;
        figEl.style.display = 'block';
        if (textBlock) captionEl.textContent = textBlock.text;
        sendSize();
      }

      window.addEventListener('message', (e) => {
        const msg = e.data;
        if (!msg) return;
        if (msg.method === 'ui/notifications/tool-result' && msg.params && msg.params.content) {
          toolResult = msg.params.content;
          renderFigure();
        }
        if (msg.method === 'ui/notifications/host-context-changed' && msg.params && msg.params.styles) {
          const vars = msg.params.styles.variables || {};
          const root = document.documentElement;
          Object.entries(vars).forEach(([k, v]) => root.style.setProperty(k, v));
        }
        if (msg.id != null && pending.has(msg.id)) {
          const {resolve, reject} = pending.get(msg.id);
          pending.delete(msg.id);
          if (msg.error) reject(msg.error); else resolve(msg.result);
        }
      });

      sendSize();
      window.parent.postMessage({jsonrpc: '2.0', method: 'ui/notifications/initialized', params: {}}, '*');

      request('ui/initialize', {
        protocolVersion: '2026-01-26',
        appInfo: { name: 'datasheet-rag-figure', version: '1.0.0' },
        appCapabilities: {}
      }).then(
        (res) => { sendSize(); },
        (err) => { errEl.textContent = 'init error: ' + JSON.stringify(err); sendSize(); }
      );
    })();
  </script>
</body></html>"""

    # ------------------------------------------------------------------
    # PDF viewer — loopback browser viewer + show_page inline rendering
    # ------------------------------------------------------------------

    # show_pdf hands back a 127.0.0.1 URL served by this process, which is
    # only openable when the client shares the machine. Over HTTP the tool
    # is therefore left unregistered rather than offered and then
    # disappointing; show_page covers the remote case by rendering inline.
    # GH #45 tracks serving PDFs from the server so this can come back.
    if local_client:

        @mcp.tool()
        def show_pdf(doc_id: str, page: int = 1):
            """Open the source PDF in a browser-based interactive viewer.

            Starts a local HTTP server (if not already running) and returns a
            URL the user can open in their browser for a full scrollable, zoomable
            PDF.js viewer.  Use show_page to render a single page inline without
            leaving the chat.

            Use when:
            - The user asks to see the full datasheet / original document.
            - They want to browse pages freely beyond a single screenshot.

            Args:
                doc_id: From chunk.doc_id or list_documents results.
                page:   1-based starting page number (passed as URL hash).
            """
            try:
                # Fetch via the backend (HTTP in remote mode) and prime the
                # viewer cache so the loopback server can serve it locally.
                pdf_viewer.prime_pdf_cache(doc_id, _backend(backend).get_pdf_bytes(doc_id))
                url = pdf_viewer.viewer_url(doc_id, page=page)
            except FileNotFoundError as exc:
                return [TextContent(type="text", text=f"Error: {exc}")]
            except Exception as exc:
                return [TextContent(type="text", text=f"Failed to load PDF: {exc}")]
            meta = _get_document_metadata_impl(doc_id, backend=backend)
            label = ""
            if meta:
                parts = [meta.get("mpn") or "", meta.get("manufacturer") or ""]
                label = " — ".join(p for p in parts if p)
            desc = f" ({label})" if label else ""
            return [
                TextContent(
                    type="text",
                    text=(
                        f"PDF viewer{desc}: {url}\n\n"
                        "Open this URL in your browser to view the full document."
                    ),
                )
            ]

    @mcp.tool(meta={"ui": {"resourceUri": "ui://datasheet-rag/figure-app"}})
    def show_page(doc_id: str, page: int = 1):
        """Render a single PDF page as an inline image widget in Claude Desktop.

        Converts the page to PNG server-side (via poppler/pdf2image) and
        returns it as an ImageContent block — displayed inline just like
        show_figure, with no browser tab needed.

        Use when:
        - The user wants to see a specific page without switching to a browser.
        - A chunk's meaning depends on a diagram, table, or layout on that page.
        - Alongside an answer, to show the source page for context.

        Args:
            doc_id: From chunk.doc_id or list_documents results.
            page:   1-based page number (use the chunk's ``page`` field).
        """
        import base64
        import io

        from pdf2image import convert_from_bytes

        try:
            pdf_bytes = _backend(backend).get_pdf_bytes(doc_id)
        except FileNotFoundError as exc:
            return [TextContent(type="text", text=f"Error: {exc}")]
        except Exception as exc:
            return [TextContent(type="text", text=f"Failed to load PDF: {exc}")]
        try:
            images = convert_from_bytes(pdf_bytes, first_page=page, last_page=page, dpi=150)
        except Exception as exc:
            return [TextContent(type="text", text=f"Failed to render page {page}: {exc}")]
        if not images:
            return [TextContent(type="text", text=f"Page {page} not found in document.")]
        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        meta = _get_document_metadata_impl(doc_id, backend=backend)
        label = ""
        if meta:
            parts = [meta.get("mpn") or "", meta.get("manufacturer") or ""]
            label = " — ".join(p for p in parts if p)
        caption = f"page {page}" + (f" · {label}" if label else "")
        return [
            ImageContent(type="image", data=img_b64, mimeType="image/png"),
            TextContent(type="text", text=caption),
        ]

    # ------------------------------------------------------------------
    # Diagnostic — bare-minimum MCP App to isolate failures.
    # If show_hello renders but show_figure doesn't, the data URI / iframe
    # CSP is the culprit. If show_hello doesn't render either, the host
    # isn't honoring _meta.ui.resourceUri at all.
    # ------------------------------------------------------------------

    @mcp.tool(meta={"ui": {"resourceUri": "ui://datasheet-rag/hello"}})
    def show_hello():
        """Diagnostic: render a trivial 'Hello' MCP App with no images or DB access.

        ``_meta.ui.resourceUri`` is on the tool definition (the placement
        Claude Desktop's built-in apps use). Hosts that support MCP Apps
        load the ``ui://datasheet-rag/hello`` resource into a sandboxed iframe.
        """
        return [
            TextContent(
                type="text",
                text="Hello MCP App rendered above (if host supports it).",
            )
        ]

    @mcp.resource(
        "ui://datasheet-rag/hello",
        mime_type="text/html;profile=mcp-app",
    )
    def hello_app_html() -> str:
        """Bare 'hello world' MCP App with correct handshake."""
        return """<!doctype html>
<html><head><meta charset="utf-8"><style>
  html, body { margin: 0; padding: 0; width: 100%; }
  body { padding: 16px; font-family: var(--font-sans, system-ui, sans-serif);
         background: var(--color-background-primary, #fff);
         color: var(--color-text-primary, #000); box-sizing: border-box; }
  h1 { margin: 0 0 6px; font-size: 1.1em; }
  p { margin: 0; font-size: 0.85em; color: var(--color-text-secondary, #666); }
</style></head>
<body>
  <h1>👋 Hello from MCP App</h1>
  <p>MCP App handshake successful — the host parsed _meta.ui.resourceUri and rendered this iframe.</p>
  <script>
    (function () {
      const pending = new Map();
      let nextId = 0;

      function sendSize() {
        window.parent.postMessage({
          jsonrpc: '2.0',
          method: 'ui/notifications/size-changed',
          params: { width: document.documentElement.scrollWidth, height: document.body.scrollHeight + 16 }
        }, '*');
      }

      function request(method, params) {
        return new Promise((resolve, reject) => {
          const id = ++nextId;
          pending.set(id, {resolve, reject});
          window.parent.postMessage({jsonrpc: '2.0', id, method, params}, '*');
          setTimeout(() => {
            if (pending.has(id)) { pending.delete(id); reject(new Error('timeout')); }
          }, 10000);
        });
      }

      window.addEventListener('message', (e) => {
        const msg = e.data;
        if (!msg) return;
        if (msg.method === 'ui/notifications/host-context-changed' && msg.params && msg.params.styles) {
          const vars = msg.params.styles.variables || {};
          const root = document.documentElement;
          Object.entries(vars).forEach(([k, v]) => root.style.setProperty(k, v));
          sendSize();
        }
        if (msg.id != null && pending.has(msg.id)) {
          const {resolve, reject} = pending.get(msg.id);
          pending.delete(msg.id);
          if (msg.error) reject(msg.error); else resolve(msg.result);
        }
      });

      sendSize();
      window.parent.postMessage({jsonrpc: '2.0', method: 'ui/notifications/initialized', params: {}}, '*');
      request('ui/initialize', {
        protocolVersion: '2026-01-26',
        appInfo: { name: 'datasheet-rag-hello', version: '1.0.0' },
        appCapabilities: {}
      }).then(() => sendSize(), () => {});
    })();
  </script>
</body></html>"""

    return mcp


def main() -> None:
    """Console-script entry point. Runs the MCP server on stdio."""
    # Honor RAG_DEFAULT_PROJECT_ID set in the .mcp.json env block.
    if "RAG_DEFAULT_PROJECT_ID" in os.environ:
        # Bust the settings cache so the env var is picked up.
        get_settings.cache_clear()  # type: ignore[attr-defined]

    server = build_server()
    settings = get_settings()
    mode = backend_mode()
    print(
        json.dumps(
            {
                "event": "rag-mcp.start",
                "mode": mode,
                "server_url": settings.server_url,
                "server_token_set": bool(settings.server_token) if mode == "remote" else None,
                "db_path": str(settings.sqlite_db_path) if mode == "local" else None,
                "default_project_id": settings.default_project_id,
                "embedding_model_id": settings.embedding_model_id,
            }
        ),
        file=sys.stderr,
    )
    # In local mode, emit the one-line notice too (consistent with the CLI).
    emit_local_notice()
    # Pre-fetch PDF.js in the background so the first show_pdf call returns
    # instantly. Errors are silently ignored — pdf_viewer fetches lazily on
    # first /static/ request and surfaces a helpful error if that also fails.
    Thread(target=pdf_viewer.prefetch_pdfjs, daemon=True).start()
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
