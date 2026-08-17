"""Backend boundary + FastAPI server round-trips.

These exercise the local backend directly and the HTTP server via TestClient,
including the trickier paths: figure upload + server-side path rewriting,
metadata writes, and the embed=False ingest split (no model needed).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from datasheet_rag.backend.local import LocalBackend
from datasheet_rag.backend.models import MetadataPatch
from datasheet_rag.models.chunk import (
    Chunk,
    ChunkGraph,
    ChunkLevel,
    ChunkMetadata,
    LayoutType,
)
from datasheet_rag.store.schema import connect

EMBED_DIM = 4


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture()
def backend(conn: sqlite3.Connection) -> LocalBackend:
    # Inject the in-memory connection; embed=False everywhere so no model loads.
    return LocalBackend(conn=conn)


def _graph_with_figure(did: str) -> ChunkGraph:
    g = ChunkGraph(doc_id=did)
    g.add(
        Chunk(
            id=f"{did}:L0:0",
            doc_id=did,
            level=ChunkLevel.MACRO,
            text="Power supply overview section",
            metadata=ChunkMetadata(doc_id=did, doc_title="Test Doc", page_numbers=[1]),
        )
    )
    g.add(
        Chunk(
            id=f"{did}:L2:0",
            doc_id=did,
            level=ChunkLevel.MICRO,
            text="Figure: functional block diagram of the regulator",
            parent_id=f"{did}:L0:0",
            figure_image_path="/client/only/path/fig.png",
            metadata=ChunkMetadata(
                doc_id=did,
                doc_title="Test Doc",
                page_numbers=[2],
                layout_type=LayoutType.FIGURE,
            ),
        )
    )
    return g


# ---- LocalBackend --------------------------------------------------------


def test_local_ingest_no_figures_trusts_existing_paths(backend: LocalBackend) -> None:
    did = "a" * 64
    g = ChunkGraph(doc_id=did)
    g.add(
        Chunk(
            id=f"{did}:L2:0",
            doc_id=did,
            level=ChunkLevel.MICRO,
            text="SCL clock stretching by the slave device",
            metadata=ChunkMetadata(doc_id=did, page_numbers=[1]),
        )
    )
    res = backend.ingest_chunk_graph(
        g, project_id="p1", metadata=MetadataPatch(mpn="X1"), embed=False
    )
    assert res.inserted == 1
    assert backend.stats(project_id="p1").total_chunks == 1
    assert backend.get_metadata(did).mpn == "X1"
    hits = backend.search("clock stretching", mode="keyword", k=5)
    assert hits and hits[0].chunk_id == f"{did}:L2:0"


def test_local_ingest_rewrites_uploaded_figure_path(
    backend: LocalBackend, tmp_path, monkeypatch
) -> None:
    # Point figures_dir at a temp dir via settings override.
    from datasheet_rag.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "figures_dir", tmp_path / "figs")

    did = "b" * 64
    g = _graph_with_figure(did)
    png = b"\x89PNG\r\n\x1a\nfake"
    res = backend.ingest_chunk_graph(
        g, figures={f"{did}:L2:0": (png, "png")}, embed=False
    )
    assert res.inserted == 2
    ch = backend.get_chunk(f"{did}:L2:0")
    # Path is stored RELATIVE to figures_dir (portable) — not the client path,
    # and not absolute.
    assert "/client/only/path" not in (ch.figure_image_path or "")
    assert not Path(ch.figure_image_path).is_absolute()
    assert ch.figure_image_path.startswith(did)  # <doc_id>/<chunk>.png
    # Resolution joins figures_dir, so the bytes still come back.
    fig = backend.get_figure_bytes(f"{did}:L2:0")
    assert fig.image_bytes() == png


def test_local_delete_doc_purges_local_files(
    backend: LocalBackend, tmp_path, monkeypatch
) -> None:
    from datasheet_rag.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "pdf_dir", tmp_path / "pdfs")
    monkeypatch.setattr(settings, "figures_dir", tmp_path / "figs")
    monkeypatch.setattr(settings, "output_dir", tmp_path / "cache")
    monkeypatch.setattr(settings, "rag_home", tmp_path)
    for d in (settings.pdf_dir, settings.figures_dir, settings.output_dir):
        d.mkdir(parents=True, exist_ok=True)

    did = "c" * 64
    g = _graph_with_figure(did)
    png = b"\x89PNG\r\n\x1a\nfake"
    backend.ingest_chunk_graph(
        g,
        figures={f"{did}:L2:0": (png, "png")},
        metadata=MetadataPatch(mpn="M1"),
        embed=False,
    )

    # Artifacts a real ingest run would also leave behind: the source PDF
    # and pipeline caches, both keyed by doc_id alone (no DB lookup needed).
    pdf_path = settings.pdf_dir / f"{did}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    cache_path = settings.output_dir / f"{did}_chunks.json"
    cache_path.write_text("{}")
    render_cache_dir = settings.rag_home / "page_render_cache" / did
    render_cache_dir.mkdir(parents=True)
    (render_cache_dir / "page-1.png").write_bytes(b"x")
    figures_dir = settings.figures_dir / did
    assert figures_dir.is_dir()  # written by ingest_chunk_graph above

    assert backend.get_metadata(did) is not None

    deleted = backend.delete_doc(did)

    assert deleted == 2
    assert backend.count_chunks(doc_id=did) == 0
    assert backend.get_metadata(did) is None
    assert not pdf_path.exists()
    assert not figures_dir.exists()
    assert not cache_path.exists()
    assert not render_cache_dir.exists()


# ---- FastAPI server via TestClient --------------------------------------


@pytest.fixture()
def client(conn: sqlite3.Connection):
    from fastapi.testclient import TestClient

    from datasheet_rag.server import deps
    from datasheet_rag.server.app import build_app

    # Override the server backend to share the in-memory connection.
    deps.get_backend.cache_clear()
    app = build_app()
    app.dependency_overrides[deps.get_backend] = lambda: LocalBackend(conn=conn)
    return TestClient(app)


def test_server_health_and_stats(client) -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/stats").json()["total_chunks"] == 0


def test_server_ingest_and_query_roundtrip(client) -> None:
    did = "c" * 64
    g = _graph_with_figure(did)
    payload = {
        "graph": g.model_dump(mode="json"),
        "project_id": "proj",
        "metadata": MetadataPatch(mpn="M1", tags=["t"]).model_dump(mode="json"),
        "embed": False,
        "describe_figures": False,
    }
    png = b"\x89PNG\r\n\x1a\nXY"
    r = client.post(
        "/ingest",
        files=[
            ("payload", ("payload.json", json.dumps(payload).encode(), "application/json")),
            ("figures", (f"{did}:L2:0.png", png, "image/png")),
        ],
    )
    assert r.status_code == 200, r.text
    assert r.json()["inserted"] == 2

    # search (keyword — no embedder)
    rs = client.post("/search", json={"query": "block diagram", "mode": "keyword", "k": 5})
    assert any(d["chunk_id"] == f"{did}:L2:0" for d in rs.json()["results"])

    # figure bytes rewritten + served
    fb = client.get(f"/figures/{did}:L2:0/bytes").json()
    import base64

    assert base64.b64decode(fb["image_b64"]) == png

    # metadata write visible
    md = client.get(f"/documents/{did}/metadata").json()
    assert md["mpn"] == "M1" and md["tags"] == ["t"]

    # delete
    assert client.delete(f"/documents/{did}").json()["deleted"] == 2


def test_server_ingest_pdf_streams_sse(client, monkeypatch) -> None:
    # Raw-PDF ingest (GH #16): the server runs the parse pipeline and streams
    # progress + a final result as SSE. Stub the parse (no Docling) and the
    # store step (no embedder) — this exercises the route plumbing itself.
    import datasheet_rag.ingest_pipeline as ip
    from datasheet_rag.backend.models import IngestResult
    from datasheet_rag.ingest_pipeline import ParseResult, ProgressEvent

    did = "d" * 64

    def fake_parse(pdf_path, *, progress=None, **kw):
        if progress:
            progress(ProgressEvent(kind="step", text="Docling layout analysis", step=1))
            progress(ProgressEvent(kind="detail", text="10 chunks", step=1))
        return ParseResult(
            graph=ChunkGraph(doc_id=did), doc_id=did, resolved_backend="docling"
        )

    monkeypatch.setattr(ip, "parse_pdf_to_graph", fake_parse)
    monkeypatch.setattr(
        LocalBackend,
        "ingest_chunk_graph",
        lambda self, g, **kw: IngestResult(doc_id=g.doc_id, inserted=10, described=0),
    )

    r = client.post(
        "/ingest-pdf",
        files={"payload": ("x.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"options": json.dumps({"project_id": "proj", "skip_describe": True})},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")

    body = r.text
    assert "event: progress" in body
    assert "Docling layout analysis" in body
    # The injected "Embed & store" step lands after the parse steps.
    assert "Embed & store" in body
    assert "event: result" in body
    # Pull the result event's JSON out of the stream.
    result = None
    for block in body.split("\n\n"):
        if "event: result" in block:
            data_line = [ln for ln in block.splitlines() if ln.startswith("data:")][0]
            result = json.loads(data_line[len("data:"):].strip())
    assert result == {"doc_id": did, "inserted": 10, "described": 0, "title": None}


def test_server_token_required_when_set(conn, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from datasheet_rag.config import get_settings
    from datasheet_rag.server import deps
    from datasheet_rag.server.app import build_app

    monkeypatch.setattr(get_settings(), "server_token", "secret")
    deps.get_backend.cache_clear()
    app = build_app()
    app.dependency_overrides[deps.get_backend] = lambda: LocalBackend(conn=conn)
    c = TestClient(app)
    assert c.get("/stats").status_code == 401
    assert c.get("/stats", headers={"Authorization": "Bearer secret"}).status_code == 200
    # health stays open.
    assert c.get("/health").status_code == 200


# ---- tiered auth + admin key lifecycle + audit --------------------------


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _make_client(conn, monkeypatch, *, read_token: str | None = None):
    from fastapi.testclient import TestClient

    from datasheet_rag.config import get_settings
    from datasheet_rag.server import deps
    from datasheet_rag.server.app import build_app

    s = get_settings()
    monkeypatch.setattr(s, "server_token", None)
    monkeypatch.setattr(s, "server_read_token", read_token)
    monkeypatch.setattr(s, "server_token_file", None)
    monkeypatch.setattr(s, "server_cors_origins", None)
    deps.get_backend.cache_clear()
    app = build_app()
    app.dependency_overrides[deps.get_backend] = lambda: LocalBackend(conn=conn)
    return TestClient(app)


def _ingest_payload(did: str) -> list:
    g = _graph_with_figure(did)
    payload = {
        "graph": g.model_dump(mode="json"),
        "project_id": "proj",
        "embed": False,
        "describe_figures": False,
    }
    png = b"\x89PNG\r\n\x1a\nXY"
    return [
        ("payload", ("payload.json", json.dumps(payload).encode(), "application/json")),
        ("figures", (f"{did}:L2:0.png", png, "image/png")),
    ]


def test_open_mode_allows_everything(conn, monkeypatch) -> None:
    c = _make_client(conn, monkeypatch, read_token=None)
    assert c.get("/stats").status_code == 200
    r = c.post("/ingest", files=_ingest_payload("d" * 64))
    assert r.status_code == 200, r.text


def test_read_token_gates_read_but_not_ingest(conn, monkeypatch) -> None:
    c = _make_client(conn, monkeypatch, read_token="readme")
    # no creds → 401
    assert c.get("/stats").status_code == 401
    # read token → read OK, ingest forbidden (read scope only)
    assert c.get("/stats", headers=_auth("readme")).status_code == 200
    r = c.post("/ingest", files=_ingest_payload("e" * 64), headers=_auth("readme"))
    assert r.status_code == 403, r.text


def test_admin_key_lifecycle_and_ingest(conn, monkeypatch) -> None:
    from datasheet_rag.store import create_api_key, hash_token

    # Bootstrap an admin key directly in the DB (like `rag-server create-key`).
    _, admin_token = create_api_key(conn, label="boot", scopes=["admin"])
    c = _make_client(conn, monkeypatch, read_token="readme")

    # Plaintext is not stored — only its hash.
    rows = conn.execute("SELECT token_sha256 FROM api_keys").fetchall()
    assert all(r["token_sha256"] != admin_token for r in rows)
    assert any(r["token_sha256"] == hash_token(admin_token) for r in rows)

    # Non-admin (read token) is forbidden from admin routes.
    assert c.post("/admin/keys", json={"label": "x"}, headers=_auth("readme")).status_code == 403

    # Mint a per-client ingest key via the admin API.
    r = c.post(
        "/admin/keys",
        json={"label": "alice", "scopes": ["ingest"]},
        headers=_auth(admin_token),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    ingest_token, key_id = body["token"], body["id"]

    # Ingest key can ingest (and read via implied scope).
    assert c.get("/stats", headers=_auth(ingest_token)).status_code == 200
    did = "f" * 64
    assert c.post("/ingest", files=_ingest_payload(did), headers=_auth(ingest_token)).status_code == 200

    # Audit row recorded with the key's label, never the token.
    audit = c.get("/audit", headers=_auth(admin_token)).json()["entries"]
    ing = [e for e in audit if e["action"] == "ingest" and e["status"] == "ok"]
    assert ing and ing[0]["key_label"] == "alice" and ing[0]["doc_id"] == did
    assert all(ingest_token not in json.dumps(e) for e in audit)

    # Revoke → immediately locked out, no restart.
    assert c.delete(f"/admin/keys/{key_id}", headers=_auth(admin_token)).status_code == 200
    assert c.post("/ingest", files=_ingest_payload("0" * 64), headers=_auth(ingest_token)).status_code == 401


def test_cors_preflight(conn, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from datasheet_rag.config import get_settings
    from datasheet_rag.server import deps
    from datasheet_rag.server.app import build_app

    s = get_settings()
    monkeypatch.setattr(s, "server_token", None)
    monkeypatch.setattr(s, "server_read_token", None)
    monkeypatch.setattr(s, "server_token_file", None)
    monkeypatch.setattr(s, "server_cors_origins", "https://ok.example.com")
    deps.get_backend.cache_clear()
    app = build_app()
    app.dependency_overrides[deps.get_backend] = lambda: LocalBackend(conn=conn)
    c = TestClient(app)

    allowed = c.options(
        "/search",
        headers={
            "Origin": "https://ok.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.headers.get("access-control-allow-origin") == "https://ok.example.com"

    denied = c.options(
        "/search",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in denied.headers
