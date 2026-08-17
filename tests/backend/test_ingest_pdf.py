"""Raw-PDF ingest (GH #16): SSE wire format, RemoteBackend streaming client,
and the LocalBackend.ingest_pdf seam.

These avoid the heavy Docling/torch stack entirely: the parse step is
monkeypatched, and the remote client is driven by an httpx MockTransport that
plays back a Server-Sent-Events stream. The FastAPI ``/ingest-pdf`` route
itself is covered in ``test_backend_and_server.py`` (needs the server extra).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator

import httpx
import pytest

from datasheet_rag.backend.base import RagServerError
from datasheet_rag.backend.local import LocalBackend
from datasheet_rag.backend.models import IngestResult
from datasheet_rag.backend.remote import RemoteBackend, _iter_sse
from datasheet_rag.ingest_pipeline import ParseResult, ProgressEvent
from datasheet_rag.models.chunk import ChunkGraph
from datasheet_rag.store.schema import connect

EMBED_DIM = 4


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---- SSE wire format -----------------------------------------------------


def test_progress_event_roundtrip() -> None:
    ev = ProgressEvent(kind="step", text="Docling layout analysis", step=2)
    assert ProgressEvent.from_dict(ev.to_dict()) == ev


def test_iter_sse_parses_events() -> None:
    body = _sse("progress", {"kind": "step", "text": "A", "step": 1})
    body += _sse("progress", {"kind": "detail", "text": "42 chunks", "step": 1})
    body += _sse("result", {"doc_id": "x", "inserted": 7})
    # A keepalive comment line should be ignored.
    lines = (":keepalive\n" + body).split("\n")

    events = list(_iter_sse(iter(lines)))
    assert [e[0] for e in events] == ["progress", "progress", "result"]
    assert events[0][1] == {"kind": "step", "text": "A", "step": 1}
    assert events[2][1] == {"doc_id": "x", "inserted": 7}


# ---- RemoteBackend.ingest_pdf (SSE client) -------------------------------


def _remote_with_handler(handler) -> RemoteBackend:
    rb = RemoteBackend("http://test")
    rb._client = httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    return rb


def _pdf(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


def test_remote_ingest_pdf_streams_progress_and_result(tmp_path) -> None:
    did = "a" * 64
    body = _sse("progress", {"kind": "step", "text": "Docling layout analysis", "step": 1})
    body += _sse("progress", {"kind": "detail", "text": "12 chunks", "step": 1})
    body += _sse("progress", {"kind": "step", "text": "Embed & store", "step": 2})
    body += _sse("result", {"doc_id": did, "inserted": 12, "described": 3})

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/ingest-pdf"
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    rb = _remote_with_handler(handler)
    seen: list[ProgressEvent] = []
    result = rb.ingest_pdf(
        _pdf(tmp_path), project_id="p", progress=seen.append
    )

    assert result == IngestResult(doc_id=did, inserted=12, described=3)
    assert [e.text for e in seen] == [
        "Docling layout analysis",
        "12 chunks",
        "Embed & store",
    ]


def test_remote_ingest_pdf_error_event_raises(tmp_path) -> None:
    body = _sse("error", {"detail": "boom in docling"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    rb = _remote_with_handler(handler)
    with pytest.raises(RagServerError) as exc:
        rb.ingest_pdf(_pdf(tmp_path))
    assert "boom in docling" in exc.value.detail


def test_remote_ingest_pdf_http_error_raises(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "forbidden"})

    rb = _remote_with_handler(handler)
    with pytest.raises(RagServerError) as exc:
        rb.ingest_pdf(_pdf(tmp_path))
    assert exc.value.status_code == 403


def test_remote_ingest_pdf_no_result_raises(tmp_path) -> None:
    # Stream ends after progress but never sends a result.
    body = _sse("progress", {"kind": "step", "text": "A", "step": 1})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    rb = _remote_with_handler(handler)
    with pytest.raises(RagServerError, match="without a result"):
        rb.ingest_pdf(_pdf(tmp_path))


# ---- LocalBackend.ingest_pdf (parse + store seam) ------------------------


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    try:
        yield c
    finally:
        c.close()


def test_local_ingest_pdf_runs_pipeline_then_stores(conn, tmp_path, monkeypatch) -> None:
    import datasheet_rag.ingest_pipeline as ip

    did = "b" * 64
    graph = ChunkGraph(doc_id=did)

    def fake_parse(pdf_path, *, progress=None, **kw):
        if progress:
            progress(ProgressEvent(kind="step", text="Docling layout analysis", step=1))
        return ParseResult(
            graph=graph,
            doc_id=did,
            resolved_backend="docling",
            title_hints={"running_header": "Datasheet"},
            figure_count=0,
        )

    monkeypatch.setattr(ip, "parse_pdf_to_graph", fake_parse)

    captured: dict = {}

    def spy_ingest(self, g, **kw):
        captured["graph"] = g
        captured.update(kw)
        return IngestResult(doc_id=g.doc_id, inserted=5)

    monkeypatch.setattr(LocalBackend, "ingest_chunk_graph", spy_ingest)

    be = LocalBackend(conn=conn)
    seen: list[ProgressEvent] = []
    result = be.ingest_pdf(
        tmp_path / "x.pdf",
        project_id="proj",
        skip_describe=True,
        progress=seen.append,
    )

    assert result.inserted == 5
    assert captured["graph"] is graph
    assert captured["project_id"] == "proj"
    assert captured["describe_figures"] is False  # skip_describe=True
    assert captured["title_hints"] == {"running_header": "Datasheet"}
    assert [e.text for e in seen] == ["Docling layout analysis"]


def test_local_ingest_pdf_describes_figures_by_default(conn, tmp_path, monkeypatch) -> None:
    import datasheet_rag.ingest_pipeline as ip

    did = "c" * 64
    monkeypatch.setattr(
        ip,
        "parse_pdf_to_graph",
        lambda pdf_path, **kw: ParseResult(
            graph=ChunkGraph(doc_id=did), doc_id=did, resolved_backend="docling"
        ),
    )
    captured: dict = {}
    monkeypatch.setattr(
        LocalBackend,
        "ingest_chunk_graph",
        lambda self, g, **kw: captured.update(kw) or IngestResult(doc_id=did),
    )

    LocalBackend(conn=conn).ingest_pdf(tmp_path / "x.pdf")
    # Neither skip_figures nor skip_describe → figure description on.
    assert captured["describe_figures"] is True
