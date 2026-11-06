"""Revisions across the backend, the HTTP server and the MCP tools (GH #124)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest

from datasheet_rag.backend.local import LocalBackend
from datasheet_rag.backend.models import MetadataPatch
from datasheet_rag.mcp.server import (
    _get_document_metadata_impl,
    _list_documents_impl,
    _list_revisions_impl,
    _search_impl,
)
from datasheet_rag.models.chunk import Chunk, ChunkGraph, ChunkLevel, ChunkMetadata
from datasheet_rag.store.documents import RevisionError
from datasheet_rag.store.fields import register_field
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
    return LocalBackend(conn=conn)


def _graph(did: str, text: str = "output voltage accuracy") -> ChunkGraph:
    g = ChunkGraph(rev_id=did)
    g.add(
        Chunk(
            id=f"{did}:L2:0",
            rev_id=did,
            level=ChunkLevel.MICRO,
            text=text,
            context_text=text,
            metadata=ChunkMetadata(rev_id=did, doc_title=f"Title {did}", page_numbers=[1]),
        )
    )
    return g


def _ingest(be: LocalBackend, did: str, **kw: Any) -> Any:
    return be.ingest_chunk_graph(_graph(did), embed=False, **kw)


# ---- ingest links ----------------------------------------------------------


def test_ingest_without_a_rule_makes_a_new_document(backend: LocalBackend) -> None:
    result = _ingest(backend, "rev_a")
    assert result.link is not None and result.link.action == "new"
    assert result.document_id.startswith("D") and result.link.document_id == result.document_id
    assert result.link.describe() == f"new document {result.document_id}"


def test_new_revision_of_puts_the_pdf_on_top_and_metadata_on_the_document(
    backend: LocalBackend,
) -> None:
    first = _ingest(backend, "rev_a", project_id="board")
    second = _ingest(
        backend,
        "rev_b",
        new_revision_of=first.document_id,
        metadata=MetadataPatch(fields={"manufacturer": "TI"}, revision="Rev B"),
    )
    assert second.link is not None and second.link.action == "explicit"
    assert second.document_id == first.document_id
    revs = backend.list_revisions(first.document_id)
    assert [(r.rev_id, r.revision, r.newest) for r in revs.revisions] == [
        ("rev_b", "Rev B", True),
        ("rev_a", None, False),
    ]
    older = backend.get_metadata("rev_a")
    assert older is not None and older.fields == {"manufacturer": "TI"}
    assert older.projects == ["board"]


def test_new_revision_of_accepts_a_revision_prefix(backend: LocalBackend) -> None:
    first = _ingest(backend, "aaaa1111")
    second = _ingest(backend, "bbbb2222", new_revision_of="aaaa")
    assert second.document_id == first.document_id


def test_reingest_keeps_the_document(backend: LocalBackend) -> None:
    first = _ingest(backend, "rev_a")
    again = _ingest(backend, "rev_a")
    assert again.link is not None and again.link.action == "kept"
    assert again.document_id == first.document_id
    other = _ingest(backend, "other")
    with pytest.raises(RevisionError, match=f"rag revisions add {other.document_id} rev_a"):
        _ingest(backend, "rev_a", new_revision_of=other.document_id)


def test_new_revision_is_checked_against_its_document_before_any_write(
    backend: LocalBackend, conn: sqlite3.Connection
) -> None:
    register_field(conn, "refdes", scope="project")
    first = _ingest(backend, "rev_a", project_id="board")
    # The new PDF is in no project yet; its document is in "board".
    second = _ingest(
        backend,
        "rev_b",
        new_revision_of=first.document_id,
        metadata=MetadataPatch(fields={"refdes": "U3"}, project="board"),
    )
    assert second.document_id == first.document_id
    with pytest.raises(ValueError, match="not in project 'other'"):
        _ingest(
            backend,
            "rev_c",
            new_revision_of=first.document_id,
            metadata=MetadataPatch(fields={"refdes": "U4"}, project="other"),
        )
    assert not conn.in_transaction
    assert [r.rev_id for r in backend.list_revisions(first.document_id).revisions] == [
        "rev_b",
        "rev_a",
    ]


def test_a_failed_embed_leaves_no_revision_behind(
    backend: LocalBackend, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    import datasheet_rag.embedding as embedding

    first = _ingest(backend, "rev_a")

    def boom(*_: Any, **__: Any) -> Any:
        # The write lock is held here: nothing may be pending on the connection.
        assert not conn.in_transaction
        raise RuntimeError("embed failed")

    monkeypatch.setattr(embedding, "embed_chunk_graph", boom)
    monkeypatch.setattr(backend, "_get_embedder", lambda: None)
    with pytest.raises(RuntimeError, match="embed failed"):
        backend.ingest_chunk_graph(_graph("rev_b"), new_revision_of=first.document_id)
    conn.commit()  # what a later, unrelated write on this connection would do
    revs = backend.list_revisions(first.document_id)
    assert [r.rev_id for r in revs.revisions] == ["rev_a"]


def test_the_rule_links_one_match(backend: LocalBackend, conn: sqlite3.Connection) -> None:
    register_field(conn, "intel-id")
    backend.set_identity_rule("intel-id")
    first = _ingest(backend, "rev_a", metadata=MetadataPatch(fields={"intel-id": "123"}))
    assert first.link is not None and first.link.action == "new"
    second = _ingest(backend, "rev_b", metadata=MetadataPatch(fields={"intel-id": "123"}))
    assert second.link is not None and second.link.action == "rule"
    assert second.link.fields == ["intel-id"]
    assert second.document_id == first.document_id
    assert second.link.describe() == (
        f"linked to {first.document_id} as newest revision (rule: intel-id)"
    )
    third = _ingest(backend, "rev_c", metadata=MetadataPatch(fields={"intel-id": "999"}))
    assert third.link is not None and third.link.action == "new"


def test_two_rule_matches_stop_the_ingest_before_any_write(
    backend: LocalBackend, conn: sqlite3.Connection
) -> None:
    register_field(conn, "intel-id")
    register_field(conn, "ti-id")
    backend.set_identity_rule("intel-id OR ti-id")
    _ingest(backend, "a", metadata=MetadataPatch(fields={"intel-id": "1"}))
    _ingest(backend, "b", metadata=MetadataPatch(fields={"ti-id": "2"}))
    patch = MetadataPatch(fields={"intel-id": "1", "ti-id": "2"})

    with pytest.raises(RevisionError, match="matches D1 .* and D2"):
        backend.plan_ingest_link(rev_id="c", metadata=patch)
    with pytest.raises(RevisionError, match="nothing was ingested"):
        _ingest(backend, "c", metadata=patch)
    assert backend.count_chunks(rev_id="c") == 0
    assert backend.get_metadata("c") is None


def test_plan_ingest_link_writes_nothing(backend: LocalBackend) -> None:
    first = _ingest(backend, "rev_a")
    plan = backend.plan_ingest_link(rev_id="rev_b", new_revision_of=first.document_id)
    assert plan.action == "explicit" and plan.document_id == first.document_id
    assert backend.get_metadata("rev_b") is None
    with pytest.raises(RevisionError, match="no document D99"):
        backend.plan_ingest_link(rev_id="rev_b", new_revision_of="D99")


# ---- listing ---------------------------------------------------------------


def test_list_documents_shows_one_row_per_document(backend: LocalBackend) -> None:
    first = _ingest(backend, "rev_a")
    _ingest(backend, "rev_b", new_revision_of=first.document_id)
    _ingest(backend, "other")
    page = backend.list_documents()
    assert [(d.rev_id, len(d.revisions)) for d in page.documents] == [("other", 1), ("rev_b", 2)]
    assert page.total == 2
    every = backend.list_documents(all_revisions=True)
    assert [d.rev_id for d in every.documents] == ["other", "rev_a", "rev_b"]


# ---- HTTP server -----------------------------------------------------------


@pytest.fixture()
def client(conn: sqlite3.Connection):
    from fastapi.testclient import TestClient

    from datasheet_rag.server import deps
    from datasheet_rag.server.app import build_app

    deps.get_backend.cache_clear()
    app = build_app()
    app.dependency_overrides[deps.get_backend] = lambda: LocalBackend(conn=conn)
    return TestClient(app)


def test_server_revision_routes(client: Any, backend: LocalBackend) -> None:
    a = _ingest(backend, "rev_a").document_id
    b = _ingest(backend, "rev_b").document_id

    r = client.post("/revisions/add", json={"document": a, "rev_id": "rev_b"})
    assert r.status_code == 200, r.text
    assert [x["rev_id"] for x in r.json()["revisions"]] == ["rev_b", "rev_a"]
    assert client.get(f"/documents/resolve/{a}").json() == {"rev_id": "rev_b", "document_id": a}
    assert client.get("/documents/rev_a/revisions").json()["document_id"] == a

    r = client.post("/revisions/move", json={"rev_id": "rev_b", "below": "rev_a"})
    assert [x["rev_id"] for x in r.json()["revisions"]] == ["rev_a", "rev_b"]

    r = client.post("/revisions/unlink", json={"rev_id": "rev_b"})
    assert r.status_code == 200, r.text
    assert r.json()["document_id"] not in (a, b)

    listed = client.get("/documents", params={"all_revisions": True}).json()
    assert listed["total"] == 2


def test_server_metadata_by_document_and_per_revision(client: Any, backend: LocalBackend) -> None:
    a = _ingest(backend, "rev_a").document_id
    _ingest(backend, "rev_b", new_revision_of=a)

    r = client.patch(f"/documents/{a}/metadata", json={"tags": ["mcu"]})
    assert r.status_code == 200, r.text
    assert r.json()["document_id"] == a and r.json()["rev_id"] == "rev_b"
    r = client.patch(
        "/documents/rev_a/metadata", json={"this_revision": True, "tags": ["old"], "revision": "A"}
    )
    assert r.json()["tags"] == ["old"] and r.json()["overrides"] == ["tags"]
    assert client.get("/documents/rev_b/metadata").json()["tags"] == ["mcu"]
    # A per-revision write needs a revision, not a document.
    r = client.patch(f"/documents/{a}/metadata", json={"this_revision": True, "tags": ["x"]})
    assert r.status_code == 400
    assert client.get("/documents/D404/metadata").status_code == 204


def test_server_identity_rule_and_link_plan(
    client: Any, backend: LocalBackend, conn: sqlite3.Connection
) -> None:
    register_field(conn, "intel-id")
    assert client.get("/identity-rule").json() == {"rule": None}
    r = client.put("/identity-rule", json={"rule": "intel-id or nope"})
    assert r.status_code == 400
    assert client.put("/identity-rule", json={"rule": "intel-id"}).json() == {"rule": "intel-id"}
    a = _ingest(backend, "rev_a", metadata=MetadataPatch(fields={"intel-id": "7"})).document_id
    r = client.post(
        "/ingest/link-plan",
        json={"rev_id": "rev_b", "metadata": {"fields": {"intel-id": "7"}}},
    )
    assert r.json() == {"action": "rule", "document_id": a, "fields": ["intel-id"]}
    assert client.put("/identity-rule", json={"rule": None}).json() == {"rule": None}


# ---- MCP tools -------------------------------------------------------------


def test_mcp_search_newest_by_default_and_tags_each_hit(backend: LocalBackend) -> None:
    a = _ingest(backend, "rev_a").document_id
    backend.ingest_chunk_graph(
        _graph("rev_b", "output voltage accuracy, revised"), embed=False, new_revision_of=a
    )
    backend.set_metadata("rev_b", MetadataPatch(revision="Rev B"))

    hits = _search_impl("output voltage", mode="keyword", backend=backend)["items"]
    assert {h["rev_id"] for h in hits} == {"rev_b"}
    assert hits[0]["document_id"] == a and hits[0]["revision"] == "Rev B"
    assert "newest_revision" not in hits[0]

    hits = _search_impl("output voltage", mode="keyword", all_revisions=True, backend=backend)
    by_doc = {h["rev_id"]: h for h in hits["items"]}
    assert set(by_doc) == {"rev_a", "rev_b"}
    assert by_doc["rev_a"]["newest_revision"] is False

    # A document id as rev_id: its newest revision, or all of them.
    hits = _search_impl("output voltage", mode="keyword", rev_id=a, backend=backend)["items"]
    assert {h["rev_id"] for h in hits} == {"rev_b"}
    hits = _search_impl(
        "output voltage", mode="keyword", rev_id=a, all_revisions=True, backend=backend
    )["items"]
    assert {h["rev_id"] for h in hits} == {"rev_a", "rev_b"}


def test_mcp_documents_and_revisions(backend: LocalBackend) -> None:
    a = _ingest(backend, "rev_a").document_id
    _ingest(backend, "rev_b", new_revision_of=a)

    items = _list_documents_impl(backend=backend)["items"]
    assert [(i["document_id"], i["rev_id"], i["revision_count"]) for i in items] == [
        (a, "rev_b", 2)
    ]
    revs = _list_revisions_impl(a, backend=backend)
    assert [r["rev_id"] for r in revs["revisions"]] == ["rev_b", "rev_a"]
    meta = _get_document_metadata_impl(a, backend=backend)
    assert meta is not None and meta["document_id"] == a
    assert [r["rev_id"] for r in meta["revisions"]] == ["rev_b", "rev_a"]


def test_remote_backend_round_trips_every_revision_call(
    client: Any, backend: LocalBackend, conn: sqlite3.Connection
) -> None:
    from datasheet_rag.backend.remote import RemoteBackend

    rb = RemoteBackend("http://testserver")
    rb._client = client  # the TestClient is an httpx.Client on the app
    register_field(conn, "intel-id")
    a = _ingest(backend, "rev_a", metadata=MetadataPatch(fields={"intel-id": "7"})).document_id
    _ingest(backend, "rev_b")

    assert rb.resolve_document("rev_a") == a
    assert rb.set_identity_rule("intel-id") == "intel-id"
    assert rb.get_identity_rule() == "intel-id"
    plan = rb.plan_ingest_link(rev_id="new", metadata=MetadataPatch(fields={"intel-id": "7"}))
    assert plan.action == "rule" and plan.document_id == a
    assert [r.rev_id for r in rb.add_revision(a, "rev_b").revisions] == ["rev_b", "rev_a"]
    moved = rb.move_revision("rev_a", above="rev_b")
    assert [r.rev_id for r in moved.revisions] == ["rev_a", "rev_b"]
    assert [r.rev_id for r in rb.list_revisions(a).revisions] == ["rev_a", "rev_b"]
    assert len(rb.list_documents().documents) == 1
    assert len(rb.list_documents(all_revisions=True).documents) == 2
    assert len(rb.list_docs(all_revisions=True)) == 2
    assert rb.unlink_revision("rev_b").document_id != a
    assert rb.set_identity_rule(None) is None
