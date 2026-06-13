"""Loopback HTTP server that serves source PDFs as a browser-based viewer.

Used by both the MCP server (``show_pdf`` tool, for the agent) and the CLI
(``rag open``, for the human) — file:// and direct S3 URLs are blocked by
most renderers, so we serve PDF bytes + a PDF.js viewer page over
``http://127.0.0.1:<port>`` instead. One server per process; the port is
ephemeral and only reachable while that process is alive.
"""

from __future__ import annotations

import http.server
import socket
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from aws_rag.config import get_settings

_pdf_server_lock = Lock()
_pdf_server_port: int | None = None

_pdf_cache_lock = Lock()
_pdf_cache: dict[str, bytes] = {}  # doc_id → raw PDF bytes (in-process cache)

_pdfjs_cache_lock = Lock()
_pdfjs_cache: dict[str, bytes] = {}  # filename → JS bytes

PDFJS_VERSION = "3.11.174"
_PDFJS_CDN = f"https://cdn.jsdelivr.net/npm/pdfjs-dist@{PDFJS_VERSION}/build/"
PDFJS_FILES = {"pdf.min.js", "pdf.worker.min.js"}


# ---------------------------------------------------------------------------
# HTTP handler
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
            data = load_pdf_bytes(doc_id)
        except Exception as exc:
            self.send_error(404, str(exc))
            return
        self._respond(data, "application/pdf")

    def _serve_viewer(self, doc_id: str) -> None:
        html = _build_viewer_html(doc_id).encode("utf-8")
        self._respond(html, "text/html; charset=utf-8")

    def _serve_static(self, filename: str) -> None:
        if filename not in PDFJS_FILES:
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


# ---------------------------------------------------------------------------
# PDF + PDF.js asset loading (cached)
# ---------------------------------------------------------------------------


def prime_pdf_cache(doc_id: str, body: bytes) -> None:
    """Seed the process PDF cache so ``load_pdf_bytes``/``viewer_url`` resolve
    a doc whose bytes live on a remote server rather than the local store.

    Used by the MCP server in remote mode: it fetches the PDF over HTTP via
    the backend, then primes the cache so the loopback viewer can serve it
    without a local file.
    """
    with _pdf_cache_lock:
        _pdf_cache[doc_id] = body


def load_pdf_bytes(doc_id: str) -> bytes:
    """Return the raw PDF bytes for *doc_id*, using a process-level cache.

    Lookup order:
    1. In-process cache (instant on repeat calls).
    2. Local PDF store — ``<pdf_dir>/<doc_id>.pdf`` (doc_id is a content
       hash, so the filename doubles as the lookup key — direct path join,
       no scanning or hashing).
    3. S3 — ``s3_pdf_prefix/{doc_id}/*.pdf``, for stores that were
       explicitly uploaded remotely.
    """
    with _pdf_cache_lock:
        cached = _pdf_cache.get(doc_id)
    if cached is not None:
        return cached

    settings = get_settings()

    local_path = settings.pdf_dir / f"{doc_id}.pdf"
    if local_path.is_file():
        body = local_path.read_bytes()
        with _pdf_cache_lock:
            _pdf_cache[doc_id] = body
        return body

    if settings.s3_bucket:
        try:
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
            pass

    raise FileNotFoundError(
        f"PDF not found for doc_id={doc_id!r}. Expected it at {local_path} "
        "(or in S3 if RAG_S3_BUCKET is configured)."
    )


def _get_pdfjs_bytes(filename: str) -> bytes:
    """Return PDF.js file bytes, downloading from CDN on first use (server-side).

    The download happens in the server process (Python), not in the iframe,
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


def prefetch_pdfjs() -> None:
    """Warm the PDF.js asset cache. Errors are silently ignored — assets are
    fetched lazily on first /static/ request if this fails."""
    for fname in PDFJS_FILES:
        try:
            _get_pdfjs_bytes(fname)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def ensure_pdf_server() -> int:
    """Start the PDF server on first call and return its port.

    Idempotent and thread-safe — repeat calls within the same process return
    the same port. The server runs in a daemon thread for the life of the
    process; there is no explicit shutdown.

    Binds ``0.0.0.0`` (reachable from the LAN / over SSH, not just
    localhost) — this is a personal dev tool for reading your own ingested
    PDFs, not a hardened public service. ``rag open`` prints every local IP
    so you can pick the one reachable from your browser; the MCP `show_pdf`
    tool always hands the agent a ``127.0.0.1`` URL since Claude Desktop
    runs on the same machine.
    """
    global _pdf_server_port
    with _pdf_server_lock:
        if _pdf_server_port is not None:
            return _pdf_server_port
        with socket.socket() as s:
            s.bind(("0.0.0.0", 0))
            port = s.getsockname()[1]
        srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), _PDFHandler)
        Thread(target=srv.serve_forever, daemon=True).start()
        _pdf_server_port = port
        return port


def viewer_url(doc_id: str, *, page: int = 1) -> str:
    """Ensure the loopback server is running and return a viewer URL for *doc_id*.

    Raises ``FileNotFoundError`` if the PDF can't be located (S3 or local).
    """
    load_pdf_bytes(doc_id)  # validate + warm cache before handing out a URL
    port = ensure_pdf_server()
    return f"http://127.0.0.1:{port}/viewer/{doc_id}#page={page}"
