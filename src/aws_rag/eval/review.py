"""Local web app for hand-reviewing the golden set.

The reviewer's job is visual and page-centric: read the question, look at
the rendered PDF page, and confirm *which page(s)* answer it. Because the
metrics credit a hit by page overlap (see :func:`aws_rag.eval.metrics.is_hit`),
``gold_pages`` is the primary human label — you rarely need to pick exact
chunk ids by hand. The tool derives the chunk mapping for you: it lists the
indexed chunks living on the chosen page(s) so you can verify the text was
captured and optionally tighten ``gold_chunk_ids``, and it shows what hybrid
retrieval currently returns so data-quality gaps are obvious.

Implementation is a stdlib ``http.server`` (no new deps), mirroring the
loopback PDF viewer pattern already used by the MCP server. PDF pages are
rendered to PNG via :mod:`aws_rag.pdf_render`.
"""

from __future__ import annotations

import http.server
import json
import socket
import sqlite3
import webbrowser
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from urllib.parse import parse_qs, urlparse

from rich.console import Console

from aws_rag.eval.dataset import CATEGORIES, EvalSet, GoldenItem
from aws_rag.eval.metrics import is_hit
from aws_rag.models.chunk import Chunk
from aws_rag.store.search import hybrid_search
from aws_rag.store.sqlite import _row_to_chunk

console = Console()


def _preview(chunk: Chunk, limit: int = 180) -> str:
    body = chunk.text or chunk.figure_caption or chunk.figure_description or ""
    body = " ".join(body.split())
    return body[:limit] + ("…" if len(body) > limit else "")


def chunks_on_pages(
    conn: sqlite3.Connection, doc_id: str, pages: list[int]
) -> list[dict[str, Any]]:
    """Indexed chunks whose page span overlaps any of ``pages``."""
    if not pages:
        return []
    page_set = set(pages)
    rows = conn.execute(
        "SELECT * FROM chunks WHERE doc_id = ? ORDER BY level, id", (doc_id,)
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        chunk = _row_to_chunk(row)
        if set(chunk.metadata.page_numbers) & page_set:
            out.append(
                {
                    "chunk_id": chunk.id,
                    "level": chunk.level.name,
                    "layout_type": chunk.metadata.layout_type.value,
                    "pages": chunk.metadata.page_numbers,
                    "preview": _preview(chunk),
                }
            )
    return out


def retrieval_preview(
    conn: sqlite3.Connection,
    embedder: Any,
    item: GoldenItem,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Top-k hybrid results for the item's question, flagged as hit/miss
    against the item's *current* gold labels."""
    query_vec = embedder.embed_one(item.question)
    results = hybrid_search(conn, query_vec, item.question, k=k)
    return [
        {
            "rank": i,
            "chunk_id": r.chunk_id,
            "score": round(r.score, 4),
            "pages": r.chunk.metadata.page_numbers,
            "level": r.chunk.level.name,
            "layout_type": r.chunk.metadata.layout_type.value,
            "preview": _preview(r.chunk),
            "hit": is_hit(r.chunk, item),
        }
        for i, r in enumerate(results, start=1)
    ]


def item_to_dict(item: GoldenItem, index: int) -> dict[str, Any]:
    d: dict[str, Any] = item.model_dump()
    d["index"] = index
    return d


def apply_update(item: GoldenItem, payload: dict[str, Any]) -> GoldenItem:
    """Return a copy of ``item`` with reviewer-editable fields applied."""
    data = item.model_dump()
    for field in ("question", "category", "answer_notes", "source"):
        if field in payload and payload[field] is not None:
            data[field] = payload[field]
    if "gold_pages" in payload and payload["gold_pages"] is not None:
        data["gold_pages"] = [int(p) for p in payload["gold_pages"]]
    if "gold_chunk_ids" in payload and payload["gold_chunk_ids"] is not None:
        data["gold_chunk_ids"] = list(payload["gold_chunk_ids"])
    return GoldenItem(**data)


class ReviewState:
    """In-memory review session backed by the JSONL file on disk."""

    def __init__(
        self,
        set_path: Path,
        conn: sqlite3.Connection,
        *,
        k: int = 5,
    ) -> None:
        self.set_path = set_path
        self.conn = conn
        self.k = k
        self.lock = Lock()
        self.eval_set = EvalSet.load(set_path)
        self._embedder: Any | None = None

    def embedder(self) -> Any:
        if self._embedder is None:
            from aws_rag.embedding import BedrockEmbedder

            self._embedder = BedrockEmbedder()
        return self._embedder

    def save(self) -> None:
        self.eval_set.save(self.set_path)

    def summary(self) -> dict[str, int]:
        total = len(self.eval_set)
        reviewed = sum(1 for it in self.eval_set.items if it.source == "human")
        return {"total": total, "reviewed": reviewed}


def _make_handler(state: ReviewState) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        # -- helpers ----------------------------------------------------
        def _json(self, obj: Any, status: int = 200) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            parsed: dict[str, Any] = json.loads(self.rfile.read(length) or b"{}")
            return parsed

        # -- routing ----------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            try:
                if path == "/":
                    self._bytes(_INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/api/items":
                    with state.lock:
                        items = [
                            item_to_dict(it, i)
                            for i, it in enumerate(state.eval_set.items)
                        ]
                        self._json({
                            "items": items,
                            "categories": list(CATEGORIES),
                            "summary": state.summary(),
                        })
                elif path == "/api/chunks":
                    doc_id = qs.get("doc_id", [""])[0]
                    pages = [int(p) for p in qs.get("pages", [""])[0].split(",") if p.strip()]
                    with state.lock:
                        self._json(chunks_on_pages(state.conn, doc_id, pages))
                elif path == "/api/retrieval":
                    idx = int(qs.get("i", ["-1"])[0])
                    with state.lock:
                        item = state.eval_set.items[idx]
                    try:
                        results = retrieval_preview(
                            state.conn, state.embedder(), item, k=state.k
                        )
                        self._json({"results": results})
                    except Exception as exc:  # noqa: BLE001
                        self._json({"error": str(exc)}, status=200)
                elif path.startswith("/api/page/"):
                    self._serve_page(path[len("/api/page/"):])
                else:
                    self.send_error(404)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, status=500)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/item/"):
                self.send_error(404)
                return
            idx = int(parsed.path[len("/api/item/"):])
            payload = self._body()
            try:
                with state.lock:
                    item = state.eval_set.items[idx]
                    state.eval_set.items[idx] = apply_update(item, payload)
                    state.save()
                    self._json({"item": item_to_dict(state.eval_set.items[idx], idx),
                                "summary": state.summary()})
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, status=500)

        def do_DELETE(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/item/"):
                self.send_error(404)
                return
            idx = int(parsed.path[len("/api/item/"):])
            try:
                with state.lock:
                    del state.eval_set.items[idx]
                    state.save()
                    self._json({"summary": state.summary(), "total": len(state.eval_set)})
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, status=500)

        def _serve_page(self, rest: str) -> None:
            # rest = "<doc_id>/<page>.png"
            try:
                doc_id, page_part = rest.rsplit("/", 1)
                page = int(page_part.removesuffix(".png"))
            except ValueError:
                self.send_error(400)
                return
            from aws_rag.pdf_render import render_page_png

            try:
                png = render_page_png(doc_id, page)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, status=404)
                return
            self._bytes(png, "image/png")

    return Handler


def serve(
    set_path: Path | str,
    *,
    db_path: Path | str | None = None,
    port: int = 0,
    k: int = 5,
    open_browser: bool = True,
) -> None:
    """Start the review server and block until interrupted."""
    from aws_rag.config import get_settings
    from aws_rag.store import connect

    settings = get_settings()
    conn = connect(db_path or settings.sqlite_db_path)
    state = ReviewState(Path(set_path), conn, k=k)

    if port == 0:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

    handler = _make_handler(state)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    console.print(
        f"[green]Review server[/] → {url}  "
        f"({state.summary()['total']} items, {state.summary()['reviewed']} reviewed)"
    )
    console.print("[dim]Edits save to the JSONL on every action. Ctrl-C to stop.[/]")
    if open_browser:
        Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/]")
    finally:
        httpd.shutdown()
        conn.close()


_INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Golden set review</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; font-size: 13px; height: 100vh;
         display: flex; flex-direction: column; background: #1e1e1e; color: #ddd; }
  #top { display: flex; align-items: center; gap: 10px; padding: 8px 12px;
         background: #2a2a2a; border-bottom: 1px solid #000; flex-shrink: 0; }
  #top button { padding: 4px 10px; border-radius: 4px; border: 1px solid #555;
                background: #3a3a3a; color: #eee; cursor: pointer; }
  #top button:hover { background: #4a4a4a; }
  #prog { color: #8c8; }
  #main { flex: 1; display: flex; overflow: hidden; }
  #left { width: 46%; overflow: auto; padding: 14px; border-right: 1px solid #000; }
  #right { flex: 1; overflow: auto; background: #404040; display: flex;
           flex-direction: column; align-items: center; }
  #pdfbar { display: flex; gap: 8px; align-items: center; padding: 6px;
            background: #2a2a2a; width: 100%; justify-content: center; flex-shrink: 0; }
  #pdfbar button { padding: 3px 9px; border-radius: 4px; border: 1px solid #555;
                   background: #3a3a3a; color: #eee; cursor: pointer; }
  #pageimg { max-width: 98%; margin: 12px 0; box-shadow: 0 2px 12px rgba(0,0,0,.6); }
  .cat { display: inline-block; padding: 1px 7px; border-radius: 10px;
         background: #355; color: #aef; font-size: 11px; }
  h2 { font-size: 15px; margin: 8px 0; line-height: 1.35; }
  .row { margin: 10px 0; }
  label { display: block; color: #9ab; margin-bottom: 3px; font-size: 11px;
          text-transform: uppercase; letter-spacing: .04em; }
  input[type=text], textarea, select { width: 100%; background: #2a2a2a; color: #eee;
        border: 1px solid #555; border-radius: 4px; padding: 5px 7px; font-size: 13px; }
  textarea { resize: vertical; min-height: 48px; }
  .chunk, .res { border: 1px solid #444; border-radius: 5px; padding: 6px 8px;
                 margin: 5px 0; cursor: pointer; background: #262626; }
  .chunk:hover, .res:hover { border-color: #888; }
  .chunk.sel { border-color: #5b5; background: #1f2a1f; }
  .res.hit { border-left: 3px solid #5b5; }
  .res.miss { border-left: 3px solid #b55; }
  .meta { color: #89a; font-size: 11px; }
  .preview { color: #bbb; margin-top: 3px; }
  .pages-now { color: #fc8; }
  #notes { color: #aaa; font-style: italic; margin-top: 4px; }
  .btnrow { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
  .btnrow button { padding: 6px 12px; border-radius: 5px; border: 1px solid #555;
                   background: #3a3a3a; color: #eee; cursor: pointer; }
  #accept { background: #2e4a2e; border-color: #5b5; }
  #reject { background: #4a2e2e; border-color: #b55; }
  .hint { color: #777; font-size: 11px; }
</style></head>
<body>
<div id="top">
  <button id="prev">◀ Prev</button>
  <span id="counter">—</span>
  <button id="next">Next ▶</button>
  <span id="prog"></span>
  <span class="hint" style="margin-left:auto">edits autosave to JSONL</span>
</div>
<div id="main">
  <div id="left"></div>
  <div id="right">
    <div id="pdfbar">
      <button id="pgprev">◀ page</button>
      <span id="pglabel">—</span>
      <button id="pgnext">page ▶</button>
      <button id="setpage">⤓ use this page as gold</button>
    </div>
    <img id="pageimg" />
  </div>
</div>
<script>
let ITEMS = [], CATS = [], cur = 0, viewPage = 1, doc = "";

async function api(path, opts) {
  const r = await fetch(path, opts);
  return await r.json();
}
function esc(s){ return (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

async function load() {
  const d = await api("/api/items");
  ITEMS = d.items; CATS = d.categories;
  document.getElementById("prog").textContent =
    `${d.summary.reviewed}/${d.summary.total} reviewed`;
  if (cur >= ITEMS.length) cur = Math.max(0, ITEMS.length - 1);
  render();
}

function item(){ return ITEMS[cur]; }

async function render() {
  const it = item();
  if (!it) { document.getElementById("left").innerHTML = "<h2>No items.</h2>"; return; }
  doc = it.doc_id;
  document.getElementById("counter").textContent = `${cur+1} / ${ITEMS.length}`;
  const pages = (it.gold_pages||[]).join(", ");
  const left = document.getElementById("left");
  left.innerHTML = `
    <span class="cat">${esc(it.category)}</span>
    <span class="meta"> source=${esc(it.source)}</span>
    <h2>${esc(it.question)}</h2>
    <div id="notes">${esc(it.answer_notes)}</div>
    <div class="row"><label>question</label>
      <textarea id="q">${esc(it.question)}</textarea></div>
    <div class="row"><label>category</label>
      <select id="cat">${CATS.map(c=>`<option ${c===it.category?'selected':''}>${c}</option>`).join("")}</select></div>
    <div class="row"><label>gold pages <span class="pages-now">(current: ${pages||'none'})</span></label>
      <input type="text" id="pages" value="${pages}" placeholder="e.g. 14, 15"></div>
    <div class="row"><label>chunks on those pages — click to toggle as gold chunk</label>
      <div id="chunks"></div></div>
    <div class="row"><label>retrieval now (hybrid, top ${5}) — green=hit, red=miss</label>
      <div id="results"><span class="hint">loading…</span></div></div>
    <div class="btnrow">
      <button id="save">Save</button>
      <button id="accept">✓ Accept (mark reviewed)</button>
      <button id="reject">✗ Reject (delete)</button>
    </div>`;
  // default the PDF view to the first gold page
  viewPage = (it.gold_pages && it.gold_pages[0]) || 1;
  showPage();
  loadChunks();
  loadResults();
  wire();
}

function showPage(){
  document.getElementById("pglabel").textContent = "page " + viewPage;
  document.getElementById("pageimg").src = `/api/page/${doc}/${viewPage}.png?t=${Date.now()}`;
}

async function loadChunks(){
  const pages = currentPages();
  const box = document.getElementById("chunks");
  if (!pages.length){ box.innerHTML = "<span class='hint'>set gold pages to see chunks</span>"; return; }
  const list = await api(`/api/chunks?doc_id=${encodeURIComponent(doc)}&pages=${pages.join(",")}`);
  const gold = new Set(item().gold_chunk_ids||[]);
  box.innerHTML = list.map(c => `
    <div class="chunk ${gold.has(c.chunk_id)?'sel':''}" data-id="${esc(c.chunk_id)}">
      <div class="meta">${esc(c.level)} · ${esc(c.layout_type)} · p${c.pages.join(',')} · ${esc(c.chunk_id.split(':').slice(-2).join(':'))}</div>
      <div class="preview">${esc(c.preview)}</div></div>`).join("") || "<span class='hint'>no indexed chunks on these pages — possible ingest gap</span>";
  box.querySelectorAll(".chunk").forEach(el => el.onclick = () => {
    const id = el.dataset.id;
    let g = item().gold_chunk_ids || [];
    if (g.includes(id)) g = g.filter(x=>x!==id); else g = g.concat([id]);
    item().gold_chunk_ids = g; el.classList.toggle("sel");
  });
}

async function loadResults(){
  const box = document.getElementById("results");
  const d = await api(`/api/retrieval?i=${cur}`);
  if (d.error){ box.innerHTML = `<span class="hint">retrieval unavailable: ${esc(d.error)}</span>`; return; }
  box.innerHTML = d.results.map(r => `
    <div class="res ${r.hit?'hit':'miss'}" data-page="${r.pages[0]||1}">
      <div class="meta">#${r.rank} · score ${r.score} · ${esc(r.level)} · p${r.pages.join(',')} · ${esc(r.chunk_id.split(':').slice(-2).join(':'))} ${r.hit?'· HIT':''}</div>
      <div class="preview">${esc(r.preview)}</div></div>`).join("");
  box.querySelectorAll(".res").forEach(el => el.onclick = () => {
    viewPage = parseInt(el.dataset.page)||viewPage; showPage();
  });
}

function currentPages(){
  return (document.getElementById("pages").value||"")
    .split(",").map(s=>parseInt(s.trim())).filter(n=>!isNaN(n));
}

function collect(){
  const it = item();
  return {
    question: document.getElementById("q").value,
    category: document.getElementById("cat").value,
    gold_pages: currentPages(),
    gold_chunk_ids: it.gold_chunk_ids || [],
    answer_notes: it.answer_notes,
  };
}

async function save(extra){
  const payload = Object.assign(collect(), extra||{});
  const d = await api(`/api/item/${cur}`, {method:"POST", body: JSON.stringify(payload)});
  if (d.summary) document.getElementById("prog").textContent =
    `${d.summary.reviewed}/${d.summary.total} reviewed`;
  if (d.item) ITEMS[cur] = d.item;
}

function wire(){
  document.getElementById("pages").onchange = () => loadChunks();
  document.getElementById("save").onclick = () => save();
  document.getElementById("accept").onclick = async () => { await save({source:"human"}); next(); };
  document.getElementById("reject").onclick = async () => {
    if (!confirm("Delete this item from the golden set?")) return;
    await api(`/api/item/${cur}`, {method:"DELETE"}); await load();
  };
}

function prev(){ if (cur>0){ cur--; render(); } }
function next(){ if (cur<ITEMS.length-1){ cur++; render(); } else render(); }

document.getElementById("prev").onclick = prev;
document.getElementById("next").onclick = next;
document.getElementById("pgprev").onclick = () => { if(viewPage>1){viewPage--; showPage();} };
document.getElementById("pgnext").onclick = () => { viewPage++; showPage(); };
document.getElementById("setpage").onclick = () => {
  const inp = document.getElementById("pages");
  const set = new Set(currentPages()); set.add(viewPage);
  inp.value = [...set].sort((a,b)=>a-b).join(", "); loadChunks();
};
window.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if (e.key === "ArrowLeft") prev();
  if (e.key === "ArrowRight") next();
});
load();
</script>
</body></html>
"""
