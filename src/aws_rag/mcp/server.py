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

import http.server
import json
import os
import socket
import sqlite3
import sys
from pathlib import Path
from threading import Lock, Thread
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

_pdf_server_lock = Lock()
_pdf_server_port: int | None = None

_pdf_cache_lock = Lock()
_pdf_cache: dict[str, bytes] = {}  # doc_id → raw PDF bytes (in-process cache)

_pdfjs_cache_lock = Lock()
_pdfjs_cache: dict[str, bytes] = {}  # filename → JS bytes

_PDFJS_VERSION = "3.11.174"
_PDFJS_CDN = f"https://cdn.jsdelivr.net/npm/pdfjs-dist@{_PDFJS_VERSION}/build/"
_PDFJS_FILES = {"pdf.min.js", "pdf.worker.min.js"}


# ---------------------------------------------------------------------------
# PDF loopback server — serves PDFs from S3 so the MCP App iframe can load
# them via http:// (file:// and direct S3 URLs are blocked by the renderer).
# ---------------------------------------------------------------------------


class _PDFHandler(http.server.BaseHTTPRequestHandler):
    """Minimal handler: GET /pdf/<doc_id>, /viewer/<doc_id>, /static/<file>."""

    def log_message(self, *args: Any) -> None:
        pass  # suppress per-request stderr noise

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].split("#")[0]
        if path.startswith("/pdf/"):
            self._serve_pdf(path[5:])
        elif path.startswith("/viewer/"):
            self._serve_viewer(path[8:])
        elif path.startswith("/static/"):
            self._serve_static(path[8:])
        else:
            self.send_error(404)

    def _serve_pdf(self, doc_id: str) -> None:
        try:
            data = _load_pdf_bytes(doc_id)
        except Exception as exc:
            self.send_error(404, str(exc))
            return
        self._respond(data, "application/pdf")

    def _serve_viewer(self, doc_id: str) -> None:
        html = _build_viewer_html(doc_id).encode("utf-8")
        self._respond(html, "text/html; charset=utf-8")

    def _serve_static(self, filename: str) -> None:
        if filename not in _PDFJS_FILES:
            self.send_error(404)
            return
        try:
            data = _get_pdfjs_bytes(filename)
        except Exception as exc:
            self.send_error(502, str(exc))
            return
        self._respond(data, "application/javascript")

    def _respond(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


def _build_viewer_html(doc_id: str) -> str:
    """Return a standalone PDF.js viewer page for *doc_id*.

    All assets (PDF.js lib + worker, PDF bytes) are loaded from the loopback
    server itself (same origin), so fetch() is unrestricted.
    """
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>PDF Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; font-size: 13px;
          background: #404040; display: flex; flex-direction: column;
          height: 100vh; }}
  #toolbar {{ display: flex; align-items: center; gap: 8px; padding: 6px 12px;
              background: #2a2a2a; color: #eee; flex-shrink: 0; }}
  #toolbar button {{ padding: 3px 10px; border-radius: 4px; border: 1px solid #555;
                     background: #3a3a3a; color: #eee; cursor: pointer; font-size: 13px; }}
  #toolbar button:hover {{ background: #4a4a4a; }}
  #toolbar button:disabled {{ opacity: 0.4; cursor: default; }}
  #pg {{ min-width: 70px; text-align: center; }}
  #scale-select {{ background: #3a3a3a; color: #eee; border: 1px solid #555;
                   border-radius: 4px; padding: 3px 6px; font-size: 13px; }}
  #status {{ color: #aaa; flex: 1; text-align: right; overflow: hidden;
             text-overflow: ellipsis; white-space: nowrap; }}
  #scroller {{ flex: 1; overflow: auto; padding: 20px;
               display: flex; justify-content: center; align-items: flex-start; }}
  canvas {{ display: block; box-shadow: 0 2px 12px rgba(0,0,0,.5); }}
</style>
</head>
<body>
<div id="toolbar">
  <button id="prev" disabled>&#9664; Prev</button>
  <span id="pg">— / —</span>
  <button id="next" disabled>Next &#9654;</button>
  &nbsp;
  <select id="scale-select">
    <option value="0.5">50%</option>
    <option value="0.75">75%</option>
    <option value="1.0">100%</option>
    <option value="1.25">125%</option>
    <option value="1.5" selected>150%</option>
    <option value="2.0">200%</option>
    <option value="2.5">250%</option>
  </select>
  <span id="status">Loading…</span>
</div>
<div id="scroller"><canvas id="cv"></canvas></div>
<script src="/static/pdf.min.js"></script>
<script>
(function () {{
  pdfjsLib.GlobalWorkerOptions.workerSrc = '/static/pdf.worker.min.js';
  var url = '/pdf/{doc_id}';
  var pdfDoc = null, cur = 1, scale = 1.5, busy = false;
  var cv = document.getElementById('cv');
  var ctx = cv.getContext('2d');
  var status = document.getElementById('status');
  var pg = document.getElementById('pg');
  var prev = document.getElementById('prev');
  var next = document.getElementById('next');
  var scaleSelect = document.getElementById('scale-select');

  function renderPage(n) {{
    if (!pdfDoc || busy) return;
    busy = true;
    status.textContent = 'Rendering page ' + n + '…';
    pdfDoc.getPage(n).then(function (page) {{
      var vp = page.getViewport({{ scale: scale }});
      cv.width = vp.width; cv.height = vp.height;
      return page.render({{ canvasContext: ctx, viewport: vp }}).promise;
    }}).then(function () {{
      cur = n; busy = false;
      pg.textContent = n + ' / ' + pdfDoc.numPages;
      prev.disabled = n <= 1;
      next.disabled = n >= pdfDoc.numPages;
      status.textContent = '';
    }}).catch(function (e) {{
      busy = false; status.textContent = 'Error: ' + e.message;
    }});
  }}

  prev.addEventListener('click', function () {{ renderPage(cur - 1); }});
  next.addEventListener('click', function () {{ renderPage(cur + 1); }});
  scaleSelect.addEventListener('change', function () {{
    scale = parseFloat(scaleSelect.value); renderPage(cur);
  }});

  status.textContent = 'Fetching PDF…';
  pdfjsLib.getDocument({{ url: url, disableRange: false }}).promise.then(function (doc) {{
    pdfDoc = doc;
    status.textContent = doc.numPages + ' pages';
    renderPage(1);
  }}).catch(function (e) {{
    status.textContent = 'Failed: ' + e.message;
  }});
}})();
</script>
</body></html>"""


def _load_pdf_bytes(doc_id: str) -> bytes:
    """Return the raw PDF bytes for *doc_id*, using a process-level cache.

    Lookup order:
    1. In-process cache (instant on repeat calls).
    2. S3 — ``s3_pdf_prefix/{doc_id}/*.pdf``.
    3. Local filesystem — scans ``_project_root()`` recursively for any
       ``.pdf`` whose SHA-256 content hash equals ``doc_id`` (handles the
       common case where PDFs are indexed locally but not yet uploaded to S3).
    """
    with _pdf_cache_lock:
        cached = _pdf_cache.get(doc_id)
    if cached is not None:
        return cached

    # ── Try S3 ──────────────────────────────────────────────────────────────
    try:
        settings = get_settings()
        from aws_rag.aws import s3_client as _s3_client

        client = _s3_client()
        resp = client.list_objects_v2(
            Bucket=settings.s3_bucket,
            Prefix=f"{settings.s3_pdf_prefix}{doc_id}/",
        )
        for obj in resp.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                body = client.get_object(
                    Bucket=settings.s3_bucket, Key=obj["Key"]
                )["Body"].read()
                with _pdf_cache_lock:
                    _pdf_cache[doc_id] = body
                return body
    except Exception:
        pass  # fall through to local scan

    # ── Local filesystem fallback ────────────────────────────────────────────
    # doc_id is a SHA-256 content hash (see storage.upload_pdf).  Scan the
    # project root (parent of the output/ dir) for any .pdf whose hash matches.
    import hashlib

    try:
        root = _project_root()
        for pdf_path in root.rglob("*.pdf"):
            try:
                h = hashlib.sha256()
                with open(pdf_path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() == doc_id:
                    body = pdf_path.read_bytes()
                    with _pdf_cache_lock:
                        _pdf_cache[doc_id] = body
                    return body
            except OSError:
                continue
    except Exception:
        pass

    raise FileNotFoundError(
        f"PDF not found for doc_id={doc_id!r}. "
        "Check that the document was uploaded to S3 or that the original "
        "PDF file is accessible in the project directory."
    )


def _get_pdfjs_bytes(filename: str) -> bytes:
    """Return PDF.js file bytes, downloading from CDN on first use (server-side).

    The download happens in the MCP server process (Python), not in the iframe,
    so it is not subject to the iframe's CSP. The result is served from the
    loopback HTTP server at /static/<filename> which the iframe can load as a
    regular script tag pointing at 127.0.0.1.
    """
    with _pdfjs_cache_lock:
        cached = _pdfjs_cache.get(filename)
    if cached is not None:
        return cached

    import urllib.request

    url = _PDFJS_CDN + filename
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        data = resp.read()
    with _pdfjs_cache_lock:
        _pdfjs_cache[filename] = data
    return data


def _ensure_pdf_server() -> int:
    """Start the PDF loopback server on first call and return its port."""
    global _pdf_server_port
    with _pdf_server_lock:
        if _pdf_server_port is not None:
            return _pdf_server_port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _PDFHandler)
        Thread(target=srv.serve_forever, daemon=True).start()
        _pdf_server_port = port
        return port


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


def _project_root() -> Path:
    """Project root — parent of the ``output/`` dir that holds rag.sqlite."""
    settings = get_settings()
    return Path(settings.sqlite_db_path).resolve().parent.parent


def _resolve_project(project_id: str | None) -> str | None:
    """Resolve the effective project_id for a tool call.

    Priority: explicit arg → settings.default_project_id (RAG_DEFAULT_PROJECT_ID env).
    Returns None if neither set, meaning "search globally".
    """
    if project_id:
        return project_id
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


def _derived_doc_fields(conn: sqlite3.Connection, doc_id: str) -> dict[str, Any]:
    """Query chunks to derive fields not stored in the metadata sidecar."""
    row = conn.execute(
        "SELECT doc_title FROM chunks WHERE doc_id = ? AND doc_title != '' LIMIT 1",
        (doc_id,),
    ).fetchone()
    title = row["doc_title"] if row else None

    page_row = conn.execute(
        """
        SELECT page_numbers FROM chunks
         WHERE doc_id = ? AND page_numbers IS NOT NULL AND page_numbers != '[]'
         ORDER BY rowid DESC
         LIMIT 1
        """,
        (doc_id,),
    ).fetchone()
    page_count: int | None = None
    if page_row:
        try:
            pages = json.loads(page_row["page_numbers"])
            if pages:
                page_count = max(pages)
        except (ValueError, TypeError):
            pass

    out: dict[str, Any] = {}
    if title:
        out["doc_title"] = title
    if page_count is not None:
        out["page_count"] = page_count
    return out


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
            **_derived_doc_fields(conn, d.doc_id),
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
    result = meta.model_dump(exclude_none=False)
    result.update(_derived_doc_fields(conn, doc_id))
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


def _figure_image_bytes(chunk: Any) -> tuple[bytes, str, Path | None]:
    """Read a figure chunk's image, returning ``(bytes, format, resolved_path)``.

    Prefers the local cropped file (``figure_image_path``). Falls back to
    S3 (``figure_s3_key``) using the configured bucket. Raises if neither
    is available or the file is missing. ``resolved_path`` is the absolute
    filesystem path when read from disk, ``None`` for S3-sourced figures.
    """
    if chunk.figure_image_path:
        path = Path(chunk.figure_image_path)
        if not path.is_absolute():
            # Stored as relative path (old ingestion runs). Resolve relative to
            # the project root, which is two directories above the sqlite db
            # (e.g. <root>/output/rag.sqlite → <root>/).
            settings = get_settings()
            project_root = Path(settings.sqlite_db_path).resolve().parent.parent
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"figure_image_path on chunk {chunk.id} points to a missing "
                f"file: {path}"
            )
        fmt = path.suffix.lstrip(".").lower() or "png"
        return path.read_bytes(), fmt, path

    if chunk.figure_s3_key:
        from aws_rag.aws import s3_client

        settings = get_settings()
        client = s3_client()
        resp = client.get_object(Bucket=settings.s3_bucket, Key=chunk.figure_s3_key)
        data = resp["Body"].read()
        ext = Path(chunk.figure_s3_key).suffix.lstrip(".").lower() or "png"
        return data, ext, None

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

    image_bytes, fmt, resolved_path = _figure_image_bytes(chunk)

    pages = chunk.metadata.page_numbers
    page = (str(pages[0]) if len(pages) == 1
            else f"{pages[0]}-{pages[-1]}" if pages else "")

    return {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "image_bytes": image_bytes,
        "format": fmt,
        "local_path": str(resolved_path) if resolved_path else None,
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
            "true` when the chunk is a diagram, schematic, or block-diagram. "
            "Use `figure_description` and `figure_caption` to reason about "
            "the content without fetching it. "
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
            "adjacent pages — call `show_pdf(doc_id, page)`. This opens a "
            "full interactive PDF viewer inline. Good triggers: the user "
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
        """Fetch raw figure bytes for further reasoning (fallback for non-Desktop hosts).

        Prefer ``show_figure`` in Claude Desktop — it renders the image inline
        as an interactive widget. Use this only on hosts that do not support
        MCP Apps, or when you need the image bytes for your own visual analysis.

        Returns an Image content block followed by a text block with caption,
        description, and citation. Do not emit markdown image links — the Image
        block is what the client renders.
        """
        import base64
        result = _get_figure_impl(chunk_id)
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
        result = _get_figure_impl(chunk_id)
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
    @mcp.tool(meta={"ui": {"resourceUri": "ui://aws-rag/figure-app"}})
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
        result = _get_figure_impl(chunk_id)
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
        "ui://aws-rag/figure-app",
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
        if (!imgBlock) { errEl.textContent = 'No image in tool-result.'; loadingEl.style.display = 'none'; sendSize(); return; }
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
        appInfo: { name: 'aws-rag-figure', version: '1.0.0' },
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
            _load_pdf_bytes(doc_id)  # validate + warm cache
        except FileNotFoundError as exc:
            return [TextContent(type="text", text=f"Error: {exc}")]
        except Exception as exc:
            return [TextContent(type="text", text=f"Failed to load PDF: {exc}")]
        port = _ensure_pdf_server()
        url = f"http://127.0.0.1:{port}/viewer/{doc_id}#page={page}"
        meta = _get_document_metadata_impl(doc_id)
        label = ""
        if meta:
            parts = [meta.get("mpn") or "", meta.get("manufacturer") or ""]
            label = " — ".join(p for p in parts if p)
        desc = f" ({label})" if label else ""
        return [TextContent(
            type="text",
            text=f"PDF viewer{desc}: {url}\n\nOpen this URL in your browser to view the full document.",
        )]

    @mcp.tool(meta={"ui": {"resourceUri": "ui://aws-rag/figure-app"}})
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
            pdf_bytes = _load_pdf_bytes(doc_id)
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
        meta = _get_document_metadata_impl(doc_id)
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

    @mcp.tool(meta={"ui": {"resourceUri": "ui://aws-rag/hello"}})
    def show_hello():
        """Diagnostic: render a trivial 'Hello' MCP App with no images or DB access.

        ``_meta.ui.resourceUri`` is on the tool definition (the placement
        Claude Desktop's built-in apps use). Hosts that support MCP Apps
        load the ``ui://aws-rag/hello`` resource into a sandboxed iframe.
        """
        return [TextContent(
            type="text",
            text="Hello MCP App rendered above (if host supports it).",
        )]

    @mcp.resource(
        "ui://aws-rag/hello",
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
        appInfo: { name: 'aws-rag-hello', version: '1.0.0' },
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
    print(
        json.dumps({
            "event": "rag-mcp.start",
            "db_path": str(settings.sqlite_db_path),
            "default_project_id": settings.default_project_id,
            "embedding_model_id": settings.embedding_model_id,
        }),
        file=sys.stderr,
    )
    # Pre-fetch PDF.js in the background so pdf_app_html() returns instantly
    # when the first show_pdf call comes in.  Errors are silently ignored —
    # pdf_app_html() will try again and show a helpful error message if needed.
    def _prefetch_pdfjs() -> None:
        for fname in _PDFJS_FILES:
            try:
                _get_pdfjs_bytes(fname)
            except Exception:
                pass

    Thread(target=_prefetch_pdfjs, daemon=True).start()
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
