"""RAG_COMPUTE=client — the models run here, the server is a vector store (GH #43).

Covers the three pieces that make that topology work: the vector wire format,
``RemoteBackend``'s dispatch (query/chunk embedding, figure description and
title inference done in-process), and the server routes that accept the
precomputed results.

No real model is ever loaded: embedders and describers are stubbed, and the
remote client is driven by an httpx MockTransport.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from datasheet_rag.backend.local import LocalBackend
from datasheet_rag.backend.models import ChunkVectors, TitleContext
from datasheet_rag.backend.remote import RemoteBackend
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


class _StubEmbedder:
    """Deterministic 4-dim embedder — enough to prove vectors travelled."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text) % 7), 1.0, 2.0, 3.0]

    def embed_chunks(self, chunks: Any) -> dict[str, list[float]]:
        return {c.id: self.embed_one(c.context_text or c.text) for c in chunks}


def _client_backend(handler: Any, *, embedder: Any = None) -> RemoteBackend:
    rb = RemoteBackend("http://test", compute="client")
    rb._client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    # Pre-seed the embedder so the /health compatibility probe is not needed
    # for tests that are not about it.
    if embedder is not None:
        rb._embedder = embedder
        rb._embedding_checked = True
    return rb


def _graph(did: str, *, figure: bool = False) -> ChunkGraph:
    g = ChunkGraph(doc_id=did)
    g.add(
        Chunk(
            id=f"{did}:L2:0",
            doc_id=did,
            level=ChunkLevel.MICRO,
            text="The regulator drops 5V to 3V3 at up to 500mA.",
            context_text="Power: The regulator drops 5V to 3V3 at up to 500mA.",
            metadata=ChunkMetadata(doc_id=did, page_numbers=[1]),
        )
    )
    if figure:
        g.add(
            Chunk(
                id=f"{did}:L2:1",
                doc_id=did,
                level=ChunkLevel.MICRO,
                text="Figure 1: functional block diagram",
                context_text="Power: Figure 1: functional block diagram",
                prev_id=f"{did}:L2:0",
                figure_image_path="fig.png",
                figure_caption="Figure 1",
                metadata=ChunkMetadata(doc_id=did, page_numbers=[1], layout_type=LayoutType.FIGURE),
            )
        )
    return g


# ---- vector wire format --------------------------------------------------


def test_chunk_vectors_roundtrip_as_float32() -> None:
    vectors = {"a": [1.5, -2.0, 0.0, 7.25], "b": [0.5, 0.25, 0.125, 1.0]}
    packed = ChunkVectors.from_mapping(vectors)

    assert packed.dim == 4
    # 2 chunks x 4 float32 = 32 bytes, well under the JSON-number spelling.
    assert len(base64.b64decode(packed.data)) == 32
    assert packed.to_mapping() == vectors


def test_chunk_vectors_rejects_truncated_payload() -> None:
    packed = ChunkVectors.from_mapping({"a": [1.0, 2.0, 3.0, 4.0]})
    packed.ids = ["a", "b"]  # claims two chunks, carries one

    with pytest.raises(ValueError, match="expected 8"):
        packed.to_mapping()


# ---- search --------------------------------------------------------------


def test_client_compute_embeds_the_query_itself() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    embedder = _StubEmbedder()
    _client_backend(handler, embedder=embedder).search("dropout voltage", mode="hybrid")

    assert embedder.calls == ["dropout voltage"]
    assert seen["query_vector"] == embedder.embed_one("dropout voltage")
    # The text still travels: hybrid needs it for the keyword half.
    assert seen["query"] == "dropout voltage"


def test_client_compute_skips_the_embedder_for_keyword_search() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["query_vector"] is None
        return httpx.Response(200, json={"results": []})

    embedder = _StubEmbedder()
    _client_backend(handler, embedder=embedder).search("VDD", mode="keyword")

    assert embedder.calls == []


def test_server_compute_sends_no_vector() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["query_vector"] is None
        return httpx.Response(200, json={"results": []})

    rb = RemoteBackend("http://test")  # default compute="server"
    rb._client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    rb.search("dropout voltage")


# ---- embedding compatibility --------------------------------------------


def test_dimension_mismatch_refuses_before_embedding(monkeypatch) -> None:
    from datasheet_rag.backend.base import RagServerError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"status": "ok", "embedding_dimensions": 1024, "embedding_model": "bge-m3"}
        )

    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "768")
    from datasheet_rag.config import get_settings

    get_settings.cache_clear()
    try:
        rb = _client_backend(handler)
        with pytest.raises(RagServerError, match="768 dimensions"):
            rb.search("anything")
    finally:
        get_settings.cache_clear()


def test_model_mismatch_only_warns(monkeypatch, capsys) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "embedding_dimensions": 1024,
                    "embedding_model": "someone-elses/model",
                },
            )
        return httpx.Response(200, json={"results": []})

    import datasheet_rag.embedding as embedding
    from datasheet_rag.config import get_settings

    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setattr(embedding, "get_embedder", lambda **kw: _StubEmbedder())
    get_settings.cache_clear()
    try:
        _client_backend(handler).search("dropout voltage")
    finally:
        get_settings.cache_clear()

    assert "do not match" in capsys.readouterr().err


# ---- ingest --------------------------------------------------------------


def test_client_compute_ships_vectors_and_asks_the_server_not_to_embed() -> None:
    did = "a" * 64
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # Multipart: dig the JSON payload part back out.
        raw = request.content
        start = raw.index(b"{")
        end = raw.rindex(b"}") + 1
        sent.update(json.loads(raw[start:end]))
        return httpx.Response(200, json={"doc_id": did, "inserted": 1})

    embedder = _StubEmbedder()
    rb = _client_backend(handler, embedder=embedder)
    rb.ingest_chunk_graph(_graph(did), project_id="p", embed=True)

    assert sent["embed"] is False
    vectors = ChunkVectors.model_validate(sent["vectors"]).to_mapping()
    assert set(vectors) == {f"{did}:L2:0"}
    assert len(vectors[f"{did}:L2:0"]) == EMBED_DIM
    # The graph itself never carries vectors — they ride the packed buffer.
    assert sent["graph"]["chunks"][f"{did}:L2:0"].get("content_embedding") is None


def test_client_compute_describes_figures_before_embedding(monkeypatch, tmp_path) -> None:
    """The description becomes context_text, so it must land before the embed."""
    did = "b" * 64
    graph = _graph(did, figure=True)

    crop = tmp_path / "fig.png"
    crop.write_bytes(b"\x89PNG\r\n\x1a\nXY")
    graph.chunks[f"{did}:L2:1"].figure_image_path = str(crop)

    import datasheet_rag.description.describer as dd

    class _StubDescriber:
        def __init__(self, *a: Any, **kw: Any) -> None: ...

        def describe_chunks(self, chunks: Any, source: Any) -> dict[str, str]:
            return {c.id: "A buck regulator block diagram." for c in chunks}

        def stats(self) -> dict[str, int]:
            return {}

    monkeypatch.setattr(dd, "FigureDescriber", _StubDescriber)

    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # Nothing described yet in the store, so every figure is still a target.
        if request.url.path == "/figures":
            return httpx.Response(200, json={"chunks": []})
        raw = request.content
        sent.update(json.loads(raw[raw.index(b"{") : raw.rindex(b"}") + 1]))
        return httpx.Response(200, json={"doc_id": did, "inserted": 2})

    embedder = _StubEmbedder()
    rb = _client_backend(handler, embedder=embedder)
    result = rb.ingest_chunk_graph(graph, embed=True, describe_figures=True)

    # The server is told there is nothing left to describe.
    assert sent["describe_figures"] is False
    assert result.described == 1
    # ...because it already happened here, and is folded into the shipped text.
    figure_json = sent["graph"]["chunks"][f"{did}:L2:1"]
    assert figure_json["figure_description"] == "A buck regulator block diagram."
    assert "Description: A buck regulator" in figure_json["context_text"]
    # And the embedding saw that text, not the pre-description version.
    assert any("Description: A buck regulator" in text for text in embedder.calls)


def test_client_compute_infers_the_title_from_the_graph(monkeypatch) -> None:
    did = "c" * 64
    import datasheet_rag.titling as titling

    monkeypatch.setattr(
        titling, "TitleInferer", lambda **kw: type("I", (), {"infer": lambda self, t: "TPS62840"})()
    )

    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/documents/titles":
            return httpx.Response(200, json={"titles": {}})
        raw = request.content
        sent.update(json.loads(raw[raw.index(b"{") : raw.rindex(b"}") + 1]))
        return httpx.Response(200, json={"doc_id": did, "inserted": 1, "title": "TPS62840"})

    rb = _client_backend(handler, embedder=_StubEmbedder())
    rb.ingest_chunk_graph(_graph(did), embed=True, infer_title=True)

    assert sent["infer_title"] is False
    assert sent["inferred_title"] == "TPS62840"


def test_client_compute_does_not_infer_a_title_that_already_exists(monkeypatch) -> None:
    """The server would discard it, so the call is never spent."""
    did = "8" * 64
    import datasheet_rag.titling as titling

    def _boom(**kw: Any) -> Any:
        raise AssertionError("inferred a title for a document that already has one")

    monkeypatch.setattr(titling, "TitleInferer", _boom)

    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        raw = request.content
        sent.update(json.loads(raw[raw.index(b"{") : raw.rindex(b"}") + 1]))
        return httpx.Response(200, json={"doc_id": did, "inserted": 1})

    graph = _graph(did)
    graph.chunks[f"{did}:L2:0"].metadata.doc_title = "TPS62840 Datasheet"

    rb = _client_backend(handler, embedder=_StubEmbedder())
    rb.ingest_chunk_graph(graph, embed=True, infer_title=True)

    assert sent["inferred_title"] is None


# ---- describe / title against a remote store -----------------------------


def test_describe_figures_via_backend_reads_and_writes_over_http(monkeypatch) -> None:
    did = "d" * 64
    png = base64.b64encode(b"\x89PNG\r\n\x1a\nXY").decode()
    written: dict[str, str] = {}

    figure_chunk = _graph(did, figure=True).chunks[f"{did}:L2:1"]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/figures":
            return httpx.Response(200, json={"chunks": [figure_chunk.model_dump(mode="json")]})
        if path.endswith("/bytes"):
            return httpx.Response(
                200,
                json={
                    "chunk_id": figure_chunk.id,
                    "doc_id": did,
                    "image_b64": png,
                    "format": "png",
                    "caption": "Figure 1",
                    "description": "",
                    "citation": {"doc_id": did},
                    "surrounding_text": "The regulator drops 5V to 3V3.",
                },
            )
        if path.endswith("/description"):
            written[path] = json.loads(request.content)["description"]
            return httpx.Response(200, json={"updated": True})
        raise AssertionError(f"unexpected request: {path}")

    import datasheet_rag.description as description_pkg

    seen_prompt: dict[str, Any] = {}

    class _StubDescriber:
        def __init__(self, *a: Any, **kw: Any) -> None: ...

        def describe_chunks(self, chunks: Any, source: Any) -> dict[str, str]:
            # Proves the source really fetched pixels + neighbours over HTTP.
            seen_prompt.update(source.inputs(list(chunks)[0]).__dict__)
            return {c.id: "A block diagram." for c in chunks}

        def stats(self) -> dict[str, int]:
            return {"total_errors": 0}

    monkeypatch.setattr(description_pkg, "FigureDescriber", _StubDescriber)

    rb = _client_backend(handler)
    descriptions, _stats = rb.describe_figures(doc_id=did)

    assert descriptions == {figure_chunk.id: "A block diagram."}
    assert seen_prompt["surrounding_text"] == "The regulator drops 5V to 3V3."
    assert seen_prompt["image_bytes"] == b"\x89PNG\r\n\x1a\nXY"
    assert list(written.values()) == ["A block diagram."]


def test_infer_title_via_backend_respects_a_hand_set_title() -> None:
    did = "e" * 64
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json=TitleContext(
                doc_id=did,
                first_page_text="TPS62840 step-down converter",
                title_source="manual",
            ).model_dump(mode="json"),
        )

    rb = _client_backend(handler)
    assert rb.infer_title(did) is None
    # Only the context fetch — no title was written back.
    assert calls == [f"/documents/{did}/title-context"]


# ---- the server side of it ----------------------------------------------


@pytest.fixture()
def server(conn: sqlite3.Connection):
    from fastapi.testclient import TestClient

    from datasheet_rag.server import deps
    from datasheet_rag.server.app import build_app

    deps.get_backend.cache_clear()
    app = build_app()
    app.dependency_overrides[deps.get_backend] = lambda: LocalBackend(conn=conn)
    return TestClient(app)


def test_health_reports_the_embedding_model(server) -> None:
    body = server.get("/health").json()
    assert body["embedding_model"]
    assert body["embedding_dimensions"]


def test_server_stores_client_vectors_without_an_embedder(server, conn) -> None:
    did = "f" * 64
    graph = _graph(did)
    vectors = {f"{did}:L2:0": [0.1, 0.2, 0.3, 0.4]}
    payload = {
        "graph": graph.model_dump(mode="json"),
        "embed": True,  # deliberately true: the vectors must override it
        "vectors": ChunkVectors.from_mapping(vectors).model_dump(mode="json"),
    }
    r = server.post(
        "/ingest",
        files=[("payload", ("payload.json", json.dumps(payload).encode(), "application/json"))],
    )
    assert r.status_code == 200, r.text
    assert r.json()["inserted"] == 1

    # The vectors landed: a vector search with the same vector finds the chunk,
    # and no embedding model was ever built to get there.
    hits = server.post(
        "/search",
        json={"query": "unused", "mode": "vector", "k": 5, "query_vector": [0.1, 0.2, 0.3, 0.4]},
    ).json()["results"]
    assert [h["chunk_id"] for h in hits] == [f"{did}:L2:0"]


def test_server_rejects_a_wrong_width_query_vector(server) -> None:
    r = server.post("/search", json={"query": "x", "mode": "vector", "query_vector": [1.0, 2.0]})
    assert r.status_code == 400
    assert "different embedding models" in r.json()["detail"]


def test_server_records_a_client_inferred_title(server, conn) -> None:
    did = "1" * 64
    payload = {
        "graph": _graph(did).model_dump(mode="json"),
        "embed": False,
        "inferred_title": "TPS62840 Datasheet",
    }
    r = server.post(
        "/ingest",
        files=[("payload", ("payload.json", json.dumps(payload).encode(), "application/json"))],
    )
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "TPS62840 Datasheet"

    titles = server.get("/documents/titles").json()["titles"]
    assert titles[did] == "TPS62840 Datasheet"
    # Recorded as inferred, so a later manual title still outranks it.
    md = server.get(f"/documents/{did}/metadata").json()
    assert md["attributes"]["title_source"] == "inferred"


def test_title_context_route_serves_the_prompt_inputs(server) -> None:
    did = "2" * 64
    payload = {"graph": _graph(did).model_dump(mode="json"), "embed": False}
    server.post(
        "/ingest",
        files=[("payload", ("payload.json", json.dumps(payload).encode(), "application/json"))],
    )
    ctx = server.get(f"/documents/{did}/title-context").json()
    assert "regulator drops 5V" in ctx["first_page_text"]
    assert ctx["doc_id"] == did


def test_local_backend_refuses_vectors_with_describe(conn) -> None:
    did = "3" * 64
    be = LocalBackend(conn=conn)
    with pytest.raises(ValueError, match="mutually exclusive"):
        be.ingest_chunk_graph(
            _graph(did),
            vectors={f"{did}:L2:0": [0.0, 0.0, 0.0, 0.0]},
            describe_figures=True,
        )


# ---- end to end ----------------------------------------------------------


def test_client_compute_round_trip_against_a_real_server(conn, monkeypatch) -> None:
    """A GPU-less server, driven by a client that embeds and describes for it.

    The client is a real ``RemoteBackend`` speaking to a real FastAPI app over
    ASGI — only the models are stubbed. Nothing on the server side ever builds
    an embedder: ``LocalBackend._get_embedder`` is replaced with a hard failure
    so a regression that re-introduces server-side embedding shows up here.
    """
    from fastapi.testclient import TestClient

    from datasheet_rag.server import deps
    from datasheet_rag.server.app import build_app

    deps.get_backend.cache_clear()
    app = build_app()
    server_backend = LocalBackend(conn=conn)
    app.dependency_overrides[deps.get_backend] = lambda: server_backend

    def _no_models(self: Any) -> Any:
        raise AssertionError("the server built an embedding model — it should not have")

    monkeypatch.setattr(LocalBackend, "_get_embedder", _no_models)

    did = "9" * 64
    rb = RemoteBackend("http://server", compute="client")
    # TestClient is an httpx.Client with a synchronous ASGI transport, so it
    # drops straight into RemoteBackend in place of a network client.
    rb._client = TestClient(app, base_url="http://server")
    rb._embedder = _StubEmbedder()
    rb._embedding_checked = True

    result = rb.ingest_chunk_graph(_graph(did), project_id="p", embed=True)
    assert result.inserted == 1

    # Vector search works, which means the client's vectors really landed in
    # the store's vec0 table — and the client embedded the query too.
    hits = rb.search("regulator dropout", mode="vector", k=5)
    assert [h.chunk_id for h in hits] == [f"{did}:L2:0"]
    assert rb.count_chunks(project_id="p") == 1


def test_dimension_mismatch_keeps_raising_on_a_retry(monkeypatch) -> None:
    """A failed compatibility check must not count as "checked"."""
    from datasheet_rag.backend.base import RagServerError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "embedding_dimensions": 1024})

    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "768")
    from datasheet_rag.config import get_settings

    get_settings.cache_clear()
    try:
        rb = _client_backend(handler)
        for _ in range(2):
            with pytest.raises(RagServerError, match="768 dimensions"):
                rb.search("anything")
    finally:
        get_settings.cache_clear()


# ---- review fixes --------------------------------------------------------


def test_server_rejects_an_unknown_title_source(server, conn) -> None:
    """An unvalidated source ranks like "auto" and would land a partial write."""
    did = "4" * 64
    payload = {"graph": _graph(did).model_dump(mode="json"), "embed": False}
    server.post(
        "/ingest",
        files=[("payload", ("payload.json", json.dumps(payload).encode(), "application/json"))],
    )

    r = server.put(f"/documents/{did}/title", json={"title": "Bogus", "source": "typo"})
    assert r.status_code == 422

    # Nothing was written and nothing is left pending on the shared connection:
    # the title guard used to fire only after the UPDATE had already run.
    assert conn.in_transaction is False
    assert server.get("/documents/titles").json()["titles"].get(did) in (None, "", "—")


def test_set_doc_title_validates_before_touching_the_title(conn) -> None:
    did = "5" * 64
    be = LocalBackend(conn=conn)
    be.ingest_chunk_graph(_graph(did), embed=False, vectors={f"{did}:L2:0": [0.0] * EMBED_DIM})
    be.set_doc_title(did, "Real Title", source="manual")

    with pytest.raises(ValueError, match="unknown title source"):
        be.set_doc_title(did, "Overwritten", source="nonsense")  # type: ignore[arg-type]

    assert be.get_doc_titles()[did] == "Real Title"
    assert conn.in_transaction is False


def test_server_rejects_wrong_width_ingest_vectors(server) -> None:
    """The write side of the check `search` has always done on query vectors."""
    did = "6" * 64
    payload = {
        "graph": _graph(did).model_dump(mode="json"),
        "embed": False,
        "vectors": ChunkVectors.from_mapping({f"{did}:L2:0": [1.0, 2.0]}).model_dump(mode="json"),
    }
    r = server.post(
        "/ingest",
        files=[("payload", ("payload.json", json.dumps(payload).encode(), "application/json"))],
    )
    assert r.status_code == 400
    assert "different embedding models" in r.json()["detail"]
    assert server.get(f"/documents/{did}/title-context").json()["first_page_text"] == ""


def test_health_survives_an_unreadable_store(conn) -> None:
    """/health is a liveness probe first — a broken DB must not 500 it."""
    from fastapi.testclient import TestClient

    from datasheet_rag.server import deps
    from datasheet_rag.server.app import build_app

    class _BrokenBackend(LocalBackend):
        @property
        def conn(self) -> sqlite3.Connection:
            raise sqlite3.OperationalError("database is locked")

    deps.get_backend.cache_clear()
    app = build_app()
    app.dependency_overrides[deps.get_backend] = lambda: _BrokenBackend(conn=conn)
    body = TestClient(app).get("/health").json()
    assert body["status"] == "ok"
    assert body["embedding_dimensions"]


def test_unreachable_health_does_not_count_as_checked() -> None:
    """A failed probe must not disable the compatibility check for the process."""
    probes = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal probes
        if request.url.path == "/health":
            probes += 1
            return httpx.Response(503, json={"detail": "down"})
        return httpx.Response(200, json={"doc_id": "7" * 64, "inserted": 1})

    rb = _client_backend(handler)
    rb._assert_embedding_compatible()
    rb._assert_embedding_compatible()
    assert probes == 2
    assert rb._embedding_checked is False


def test_supplied_vectors_still_trigger_the_compatibility_check(monkeypatch) -> None:
    """`vectors=` bypasses _get_embedder(), so the guard has to live elsewhere."""
    from datasheet_rag.backend.base import RagServerError

    did = "8" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "embedding_dimensions": 1024})
        raise AssertionError("the ingest went out despite a dimension mismatch")

    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "768")
    from datasheet_rag.config import get_settings

    get_settings.cache_clear()
    try:
        rb = _client_backend(handler)
        with pytest.raises(RagServerError, match="768 dimensions"):
            rb.ingest_chunk_graph(_graph(did), vectors={f"{did}:L2:0": [0.0] * 768})
    finally:
        get_settings.cache_clear()


def test_a_supplied_title_turns_off_server_side_inference() -> None:
    did = "a" * 64
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        raw = request.content
        sent.update(json.loads(raw[raw.index(b"{") : raw.rindex(b"}") + 1]))
        return httpx.Response(200, json={"doc_id": did, "inserted": 1})

    rb = _client_backend(handler, embedder=_StubEmbedder())
    rb.ingest_chunk_graph(_graph(did), infer_title=True, inferred_title="TPS62840")

    # Both fields set would have the server infer a title on top of the one we
    # are already sending — the model call client compute exists to avoid.
    assert sent["inferred_title"] == "TPS62840"
    assert sent["infer_title"] is False


def test_client_compute_keeps_the_empty_query_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("an empty query should never reach the server")

    embedder = _StubEmbedder()
    rb = _client_backend(handler, embedder=embedder)
    with pytest.raises(ValueError, match="query must not be empty"):
        rb.search("   ")
    assert embedder.calls == []


def test_reingest_does_not_redescribe_stored_figures(monkeypatch, tmp_path) -> None:
    """A freshly parsed graph has no descriptions; the store's still count."""
    did = "d" * 64
    graph = _graph(did, figure=True)
    crop = tmp_path / "fig.png"
    crop.write_bytes(b"\x89PNG\r\n\x1a\nXY")
    graph.chunks[f"{did}:L2:1"].figure_image_path = str(crop)

    import datasheet_rag.description.describer as dd

    class _ExplodingDescriber:
        def __init__(self, *a: Any, **kw: Any) -> None: ...

        def describe_chunks(self, chunks: Any, source: Any) -> dict[str, str]:
            raise AssertionError("re-described a figure the store already had")

        def stats(self) -> dict[str, int]:
            return {}

    monkeypatch.setattr(dd, "FigureDescriber", _ExplodingDescriber)

    stored = graph.chunks[f"{did}:L2:1"].model_copy(deep=True)
    stored.figure_description = "A buck regulator block diagram."
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/figures":
            return httpx.Response(200, json={"chunks": [stored.model_dump(mode="json")]})
        raw = request.content
        sent.update(json.loads(raw[raw.index(b"{") : raw.rindex(b"}") + 1]))
        return httpx.Response(200, json={"doc_id": did, "inserted": 2})

    rb = _client_backend(handler, embedder=_StubEmbedder())
    result = rb.ingest_chunk_graph(graph, embed=True, describe_figures=True)

    assert result.described == 0
    figure_json = sent["graph"]["chunks"][f"{did}:L2:1"]
    assert figure_json["figure_description"] == "A buck regulator block diagram."
    assert "Description: A buck regulator" in figure_json["context_text"]


def test_client_compute_uploads_the_source_pdf(monkeypatch, tmp_path) -> None:
    """Client-side parse still owes the server the PDF (rag show, show_pdf)."""
    did = "e" * 64
    pdf = tmp_path / "tps62840.pdf"
    pdf.write_bytes(b"%PDF-1.7\nnot really\n")

    import datasheet_rag.ingest_pipeline as ip

    class _Parsed:
        graph = _graph(did)
        title_hints: dict[str, str] = {}

    monkeypatch.setattr(ip, "parse_pdf_to_graph", lambda *a, **kw: _Parsed())
    monkeypatch.setattr(ip, "collect_figure_uploads", lambda g: ({}, []))

    uploaded: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/documents/{did}/pdf":
            uploaded["body"] = request.content
            return httpx.Response(200, json={"stored": True, "doc_id": did})
        return httpx.Response(200, json={"doc_id": did, "inserted": 1})

    rb = _client_backend(handler, embedder=_StubEmbedder())
    result = rb.ingest_pdf(pdf, doc_id=did)

    assert result.doc_id == did
    assert b"%PDF-1.7" in uploaded["body"]


def test_server_stores_an_uploaded_pdf(server, tmp_path, monkeypatch) -> None:
    from datasheet_rag.config import get_settings

    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        did = "0" * 64
        r = server.put(
            f"/documents/{did}/pdf",
            files={"payload": ("x.pdf", b"%PDF-1.7\nbytes\n", "application/pdf")},
        )
        assert r.status_code == 200, r.text
        assert server.get(f"/documents/{did}/pdf").content == b"%PDF-1.7\nbytes\n"
    finally:
        get_settings.cache_clear()


def test_pdf_upload_refuses_a_doc_id_that_escapes_the_store() -> None:
    from datasheet_rag.storage import save_pdf_bytes

    with pytest.raises(ValueError, match="invalid doc_id"):
        save_pdf_bytes(b"%PDF", "../../etc/passwd")
