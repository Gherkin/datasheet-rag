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

from aws_rag.backend.local import LocalBackend
from aws_rag.backend.models import MetadataPatch
from aws_rag.models.chunk import (
    Chunk,
    ChunkGraph,
    ChunkLevel,
    ChunkMetadata,
    LayoutType,
)
from aws_rag.store.schema import connect

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
    from aws_rag.config import get_settings

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


# ---- FastAPI server via TestClient --------------------------------------


@pytest.fixture()
def client(conn: sqlite3.Connection):
    from fastapi.testclient import TestClient

    from aws_rag.server import deps
    from aws_rag.server.app import build_app

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


def test_server_token_required_when_set(conn, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from aws_rag.config import get_settings
    from aws_rag.server import deps
    from aws_rag.server.app import build_app

    monkeypatch.setattr(get_settings(), "server_token", "secret")
    deps.get_backend.cache_clear()
    app = build_app()
    app.dependency_overrides[deps.get_backend] = lambda: LocalBackend(conn=conn)
    c = TestClient(app)
    assert c.get("/stats").status_code == 401
    assert c.get("/stats", headers={"Authorization": "Bearer secret"}).status_code == 200
    # health stays open.
    assert c.get("/health").status_code == 200
