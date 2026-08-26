"""The MCP endpoint mounted at /mcp on the RAG server (GH #39).

These drive real JSON-RPC over the mounted ASGI app rather than calling the
tool impls, because everything interesting here lives in the transport: the
routing rewrite, the auth gate, and whether the request-scoped project
survives the SDK's own task machinery on the way to a tool call.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from datasheet_rag.backend.local import LocalBackend
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkGraph, ChunkLevel, ChunkMetadata
from datasheet_rag.store.schema import connect

EMBED_DIM = 4

_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    try:
        yield c
    finally:
        c.close()


def _seed(backend: LocalBackend, doc_char: str, project: str, text: str) -> str:
    did = doc_char * 64
    g = ChunkGraph(doc_id=did)
    g.add(
        Chunk(
            id=f"{did}:L2:0",
            doc_id=did,
            level=ChunkLevel.MICRO,
            text=text,
            metadata=ChunkMetadata(doc_id=did, doc_title="Doc", page_numbers=[1]),
        )
    )
    backend.ingest_chunk_graph(g, project_id=project, embed=False)
    return did


@pytest.fixture()
def app_and_backend(conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch):
    """Build the real app with its MCP mount bound to an in-memory backend.

    ``build_app`` resolves the mount's backend once, at build time — patch the
    name it actually calls rather than ``deps.get_backend``, which app.py has
    already imported by value.
    """
    backend = LocalBackend(conn=conn)
    monkeypatch.setattr("datasheet_rag.server.app.get_backend", lambda: backend)

    from datasheet_rag.server.app import build_app

    return build_app(), backend


@pytest.fixture()
def client(app_and_backend) -> Iterator[TestClient]:
    app, _ = app_and_backend
    # `with` matters: the MCP session manager starts in the app's lifespan.
    with TestClient(app) as c:
        yield c


def _rpc(client: TestClient, method: str, params: dict | None = None, **kw) -> Any:
    path = kw.pop("path", "/mcp/")
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    headers = {**_MCP_HEADERS, **kw.pop("headers", {})}
    return client.post(path, json=body, headers=headers, **kw)


def _tool_call(client: TestClient, name: str, args: dict, **kw) -> Any:
    r = _rpc(client, "tools/call", {"name": name, "arguments": args}, **kw)
    assert r.status_code == 200, r.text
    return r.json()["result"]


# ---- protocol ------------------------------------------------------------


def test_initialize_over_http(client: TestClient) -> None:
    r = _rpc(
        client,
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["result"]["serverInfo"]["name"].startswith("datasheet-rag")


@pytest.mark.parametrize("path", ["/mcp", "/mcp/", "/mcp/proj-a"])
def test_every_url_form_answers_without_a_redirect(client: TestClient, path: str) -> None:
    # A bare /mcp is what most clients are configured with, and a redirect on
    # it would cost every request a 307 and a re-POST of its body — which is
    # exactly what a Starlette mount would do here.
    r = _rpc(client, "tools/list", path=path, follow_redirects=False)
    assert r.status_code == 200, r.text


def test_tools_list_omits_the_loopback_pdf_viewer(client: TestClient) -> None:
    names = {t["name"] for t in _rpc(client, "tools/list").json()["result"]["tools"]}
    assert "search" in names
    # show_pdf hands back a 127.0.0.1 URL belonging to the server host, so it
    # is not offered here; show_page renders inline and is (GH #45).
    assert "show_pdf" not in names
    assert "show_page" in names


# ---- project scoping -----------------------------------------------------


def test_path_segment_scopes_the_search_to_one_project(client: TestClient, app_and_backend) -> None:
    _, backend = app_and_backend
    a = _seed(backend, "a", "proj-a", "thermal shutdown threshold")
    b = _seed(backend, "b", "proj-b", "thermal shutdown threshold")

    def ids(path: str) -> set[str]:
        result = _tool_call(
            client, "search", {"query": "thermal shutdown", "mode": "keyword"}, path=path
        )
        return {row["doc_id"] for row in result["structuredContent"]["result"]}

    assert ids("/mcp/proj-a") == {a}
    assert ids("/mcp/proj-b") == {b}
    # Unscoped: the whole store, since the server has no cwd to infer from.
    assert ids("/mcp/") == {a, b}


def test_project_header_scopes_when_the_path_does_not(client: TestClient, app_and_backend) -> None:
    _, backend = app_and_backend
    a = _seed(backend, "a", "proj-a", "thermal shutdown threshold")
    _seed(backend, "b", "proj-b", "thermal shutdown threshold")
    result = _tool_call(
        client,
        "search",
        {"query": "thermal shutdown", "mode": "keyword"},
        headers={"X-RAG-Project": "proj-a"},
    )
    assert {r["doc_id"] for r in result["structuredContent"]["result"]} == {a}


def test_explicit_project_argument_beats_the_url(client: TestClient, app_and_backend) -> None:
    _, backend = app_and_backend
    _seed(backend, "a", "proj-a", "thermal shutdown threshold")
    b = _seed(backend, "b", "proj-b", "thermal shutdown threshold")
    result = _tool_call(
        client,
        "search",
        {"query": "thermal shutdown", "mode": "keyword", "project_id": "proj-b"},
        path="/mcp/proj-a",
    )
    assert {r["doc_id"] for r in result["structuredContent"]["result"]} == {b}


def test_scoping_does_not_leak_between_requests(client: TestClient, app_and_backend) -> None:
    # The project rides a ContextVar. If it were process-wide instead, the
    # scoped call below would poison every later one.
    _, backend = app_and_backend
    a = _seed(backend, "a", "proj-a", "thermal shutdown threshold")
    b = _seed(backend, "b", "proj-b", "thermal shutdown threshold")
    _tool_call(client, "search", {"query": "thermal", "mode": "keyword"}, path="/mcp/proj-a")
    result = _tool_call(client, "search", {"query": "thermal", "mode": "keyword"})
    assert {r["doc_id"] for r in result["structuredContent"]["result"]} == {a, b}


# ---- auth ----------------------------------------------------------------


@pytest.fixture()
def read_token(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    token = "s3cret-read-token"
    monkeypatch.setattr(get_settings(), "server_read_token", token)
    yield token


def test_open_mode_allows_unauthenticated_mcp(client: TestClient) -> None:
    # No read token and no API keys: the same open posture /search has.
    assert _rpc(client, "tools/list").status_code == 200


def test_read_token_is_required_when_configured(client: TestClient, read_token: str) -> None:
    r = _rpc(client, "tools/list")
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")

    assert _rpc(client, "tools/list", headers={"Authorization": "Bearer wrong"}).status_code == 401

    ok = _rpc(client, "tools/list", headers={"Authorization": f"Bearer {read_token}"})
    assert ok.status_code == 200


def test_ingest_key_also_opens_mcp(client: TestClient, app_and_backend, read_token: str) -> None:
    # Ingest implies read, so a client with an ingest key can search too.
    from datasheet_rag.store import create_api_key

    _, backend = app_and_backend
    _, token = create_api_key(backend.conn, label="c1", scopes=["ingest"])
    r = _rpc(client, "tools/list", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


# ---- opting out ----------------------------------------------------------


def test_mcp_can_be_disabled(conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("datasheet_rag.server.app.get_backend", lambda: LocalBackend(conn=conn))
    monkeypatch.setattr(get_settings(), "server_mcp_enabled", False)

    from datasheet_rag.server.app import build_app

    with TestClient(build_app()) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert _rpc(c, "tools/list").status_code == 404
