"""Documents and revisions (GH #124): linking, order, overrides, hiding."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata
from datasheet_rag.store.documents import (
    RevisionError,
    add_revision,
    document_of,
    format_document_id,
    move_revision,
    parse_document_ref,
    resolve_target,
    revisions_of,
    unlink_revision,
)
from datasheet_rag.store.fields import MetadataFieldError, register_field
from datasheet_rag.store.metadata import delete_metadata, get_metadata, list_docs, set_metadata
from datasheet_rag.store.schema import connect
from datasheet_rag.store.search import SearchFilters, keyword_search, vector_search
from datasheet_rag.store.sqlite import delete_doc, insert_chunks, resolve_doc_id

EMBED_DIM = 4


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    try:
        yield c
    finally:
        c.close()


def _ingest(conn: sqlite3.Connection, doc_id: str, text: str = "supply voltage range") -> None:
    chunk = Chunk(
        id=f"{doc_id}:L2:0",
        doc_id=doc_id,
        level=ChunkLevel.MICRO,
        text=text,
        context_text=text,
        metadata=ChunkMetadata(doc_id=doc_id, page_numbers=[1]),
    )
    insert_chunks(conn, [chunk], vectors={chunk.id: [1.0, 0.0, 0.0, 0.0]})


def _doc(conn: sqlite3.Connection, doc_id: str) -> int:
    n = document_of(conn, doc_id)
    assert n is not None
    return n


def _ref(conn: sqlite3.Connection, doc_id: str) -> str:
    return format_document_id(_doc(conn, doc_id))


# ---- ids and creation ----------------------------------------------------


def test_document_ref_is_an_upper_case_d() -> None:
    assert parse_document_ref("D12") == 12
    # Lower case is the start of a hex doc_id, not a document.
    assert parse_document_ref("d12") is None
    assert parse_document_ref("D12a") is None


def test_every_ingested_pdf_is_its_own_document(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a")
    _ingest(conn, "rev_b")
    assert _doc(conn, "rev_a") != _doc(conn, "rev_b")
    assert revisions_of(conn, _doc(conn, "rev_a")) == ["rev_a"]
    md = get_metadata(conn, "rev_a")
    assert md is not None and md.document_id == _ref(conn, "rev_a")
    assert md.revisions == ["rev_a"]


def test_a_document_id_resolves_to_its_newest_revision(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a")
    _ingest(conn, "rev_b")
    add_revision(conn, _doc(conn, "rev_a"), "rev_b")
    assert resolve_doc_id(conn, _ref(conn, "rev_a")) == "rev_b"
    with pytest.raises(ValueError, match="No document D999"):
        resolve_doc_id(conn, "D999")


def test_resolve_target_rejects_an_unknown_document(conn: sqlite3.Connection) -> None:
    with pytest.raises(RevisionError, match="no document D7"):
        resolve_target(conn, "D7")


# ---- add / move / unlink --------------------------------------------------


def test_add_merges_a_lone_document_and_puts_the_pdf_on_top(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a")
    _ingest(conn, "rev_b")
    set_metadata(conn, "rev_a", add_projects=["p1"], tags=["mcu"], fields={"mpn": "X1"})
    set_metadata(conn, "rev_b", add_projects=["p2"], tags=["new"], fields={"manufacturer": "TI"})
    lone = _doc(conn, "rev_b")

    result = add_revision(conn, _doc(conn, "rev_a"), "rev_b")

    assert result.merged
    assert revisions_of(conn, result.document_id) == ["rev_b", "rev_a"]
    assert conn.execute("SELECT 1 FROM documents WHERE id = ?", (lone,)).fetchone() is None
    md = get_metadata(conn, format_document_id(result.document_id))
    assert md is not None
    assert md.doc_id == "rev_b"  # the document view shows its newest revision
    assert md.projects == ["p1", "p2"]
    assert md.tags == ["mcu", "new"]
    assert md.fields == {"mpn": ["X1"], "manufacturer": "TI"}


def test_add_refuses_documents_with_different_values(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a")
    _ingest(conn, "rev_b")
    set_metadata(conn, "rev_a", fields={"manufacturer": "TI"})
    set_metadata(conn, "rev_b", fields={"manufacturer": "ST"})
    with pytest.raises(RevisionError, match="manufacturer"):
        add_revision(conn, _doc(conn, "rev_a"), "rev_b")
    # Nothing moved.
    assert _doc(conn, "rev_a") != _doc(conn, "rev_b")
    assert get_metadata(conn, "rev_b").fields == {"manufacturer": "ST"}  # type: ignore[union-attr]


def test_add_treats_equal_values_as_agreement(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a")
    _ingest(conn, "rev_b")
    set_metadata(conn, "rev_a", fields={"mpn": ["X1", "X2"]})
    set_metadata(conn, "rev_b", fields={"mpn": ["x2", "x1"]})
    add_revision(conn, _doc(conn, "rev_a"), "rev_b")
    assert _doc(conn, "rev_a") == _doc(conn, "rev_b")


def test_add_refuses_a_revision_of_another_family(conn: sqlite3.Connection) -> None:
    for d in ("a1", "a2", "b1"):
        _ingest(conn, d)
    add_revision(conn, _doc(conn, "a1"), "a2")
    with pytest.raises(RevisionError, match="unlink it first"):
        add_revision(conn, _doc(conn, "b1"), "a2")


def test_move_reorders_within_a_document(conn: sqlite3.Connection) -> None:
    for d in ("r1", "r2", "r3"):
        _ingest(conn, d)
    n = _doc(conn, "r1")
    add_revision(conn, n, "r2")
    add_revision(conn, n, "r3")
    assert revisions_of(conn, n) == ["r3", "r2", "r1"]

    move_revision(conn, "r3", below="r1")
    assert revisions_of(conn, n) == ["r2", "r1", "r3"]
    move_revision(conn, "r3", above="r2")
    assert revisions_of(conn, n) == ["r3", "r2", "r1"]
    # Adding one already in the document moves it to the top.
    add_revision(conn, n, "r1")
    assert revisions_of(conn, n) == ["r1", "r3", "r2"]


def test_move_refuses_revisions_of_different_documents(conn: sqlite3.Connection) -> None:
    _ingest(conn, "a")
    _ingest(conn, "b")
    with pytest.raises(RevisionError, match="not revisions of the same document"):
        move_revision(conn, "a", above="b")


def test_unlink_copies_the_metadata_into_a_new_document(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a")
    _ingest(conn, "rev_b")
    n = _doc(conn, "rev_a")
    add_revision(conn, n, "rev_b")
    set_metadata(conn, f"D{n}", add_projects=["p"], tags=["t"], fields={"mpn": "X"})
    set_metadata(conn, "rev_b", fields={"subsystem": "power"}, project="p")

    new = unlink_revision(conn, "rev_b")

    assert new != n
    assert revisions_of(conn, n) == ["rev_a"]
    moved = get_metadata(conn, "rev_b")
    assert moved is not None
    assert moved.document_id == format_document_id(new)
    assert moved.projects == ["p"] and moved.tags == ["t"] and moved.fields == {"mpn": ["X"]}
    assert moved.project_fields == {"p": {"subsystem": "power"}}
    # The two documents' values are separate now.
    set_metadata(conn, "rev_b", fields={"mpn": "Y"})
    assert get_metadata(conn, "rev_a").fields == {"mpn": ["X"]}  # type: ignore[union-attr]


def test_unlink_refuses_the_only_revision(conn: sqlite3.Connection) -> None:
    _ingest(conn, "a")
    with pytest.raises(RevisionError, match="only revision"):
        unlink_revision(conn, "a")


def test_the_document_goes_with_its_last_revision(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a")
    _ingest(conn, "rev_b")
    n = _doc(conn, "rev_a")
    add_revision(conn, n, "rev_b")
    set_metadata(conn, "rev_a", add_projects=["p"])

    delete_doc(conn, "rev_b")
    delete_metadata(conn, "rev_b")
    assert revisions_of(conn, n) == ["rev_a"]
    assert get_metadata(conn, "rev_a").projects == ["p"]  # type: ignore[union-attr]

    delete_doc(conn, "rev_a")
    delete_metadata(conn, "rev_a")
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM document_projects").fetchone()[0] == 0


# ---- metadata targets the document ---------------------------------------


def test_a_write_through_any_revision_lands_on_the_document(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a")
    _ingest(conn, "rev_b")
    add_revision(conn, _doc(conn, "rev_a"), "rev_b")
    set_metadata(conn, "rev_a", fields={"manufacturer": "TI"}, tags=["x"])
    newest = get_metadata(conn, "rev_b")
    assert newest is not None
    assert newest.fields == {"manufacturer": "TI"} and newest.tags == ["x"]
    assert newest.overrides == []


def test_this_revision_overrides_and_clearing_inherits_again(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a")
    _ingest(conn, "rev_b")
    n = _doc(conn, "rev_a")
    add_revision(conn, n, "rev_b")
    set_metadata(conn, f"D{n}", fields={"doc_type": "datasheet"}, tags=["mcu"])

    md = set_metadata(
        conn, "rev_a", this_revision=True, fields={"doc_type": "errata"}, tags=["old"]
    )
    assert md.fields == {"doc_type": "errata"} and md.tags == ["old"]
    assert md.overrides == ["doc_type", "tags"]
    other = get_metadata(conn, "rev_b")
    assert other is not None and other.fields == {"doc_type": "datasheet"}
    document = get_metadata(conn, f"D{n}")
    assert document is not None and document.fields == {"doc_type": "datasheet"}

    md = set_metadata(conn, "rev_a", this_revision=True, fields={"doc_type": None}, tags=[])
    assert md.fields == {"doc_type": "datasheet"} and md.tags == ["mcu"]
    assert md.overrides == []


def test_this_revision_refuses_documents_projects_and_project_fields(
    conn: sqlite3.Connection,
) -> None:
    _ingest(conn, "rev_a")
    set_metadata(conn, "rev_a", add_projects=["p"])
    with pytest.raises(RevisionError, match="whole document"):
        set_metadata(conn, _ref(conn, "rev_a"), this_revision=True, tags=["x"])
    with pytest.raises(RevisionError, match="projects belong"):
        set_metadata(conn, "rev_a", this_revision=True, add_projects=["q"])
    with pytest.raises(MetadataFieldError, match="cannot be set on one revision"):
        set_metadata(conn, "rev_a", this_revision=True, fields={"subsystem": "rf"}, project="p")


def test_the_revision_label_is_per_revision(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a")
    assert set_metadata(conn, "rev_a", revision=" Rev C ").revision == "Rev C"
    assert set_metadata(conn, "rev_a", revision="").revision is None
    with pytest.raises(RevisionError, match="revision label"):
        set_metadata(conn, _ref(conn, "rev_a"), revision="Rev D")


def test_field_filters_use_each_revisions_effective_value(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a")
    _ingest(conn, "rev_b")
    add_revision(conn, _doc(conn, "rev_a"), "rev_b")
    set_metadata(conn, "rev_b", fields={"doc_type": "datasheet"})
    set_metadata(conn, "rev_a", this_revision=True, fields={"doc_type": "errata"})

    hits = list_docs(conn, fields={"doc_type": "errata"}, all_revisions=True)
    assert [m.doc_id for m in hits] == ["rev_a"]
    hits = list_docs(conn, fields={"doc_type": "datasheet"}, all_revisions=True)
    assert [m.doc_id for m in hits] == ["rev_b"]


def test_a_user_field_cannot_take_a_revision_name(conn: sqlite3.Connection) -> None:
    for name in ("revision", "revisions", "document", "document_id", "version"):
        with pytest.raises(MetadataFieldError, match="reserved"):
            register_field(conn, name)


# ---- listing and search show the newest revision ---------------------------


def _two_revisions(conn: sqlite3.Connection) -> None:
    _ingest(conn, "rev_a", "supply voltage range old wording")
    _ingest(conn, "rev_b", "supply voltage range new wording")
    _ingest(conn, "other", "supply voltage range of another part")
    add_revision(conn, _doc(conn, "rev_a"), "rev_b")


def test_list_docs_shows_the_newest_revision_by_default(conn: sqlite3.Connection) -> None:
    _two_revisions(conn)
    assert [m.doc_id for m in list_docs(conn)] == ["other", "rev_b"]
    assert [m.doc_id for m in list_docs(conn, all_revisions=True)] == ["other", "rev_a", "rev_b"]


def test_search_hides_older_revisions_unless_asked(conn: sqlite3.Connection) -> None:
    _two_revisions(conn)
    vec = [1.0, 0.0, 0.0, 0.0]

    def docs(results: list) -> set[str]:  # type: ignore[type-arg]
        return {r.chunk.doc_id for r in results}

    assert docs(keyword_search(conn, "supply voltage")) == {"rev_b", "other"}
    assert docs(vector_search(conn, vec)) == {"rev_b", "other"}
    every = SearchFilters(all_revisions=True)
    assert docs(keyword_search(conn, "supply voltage", filters=every)) == {
        "rev_a",
        "rev_b",
        "other",
    }
    assert docs(vector_search(conn, vec, filters=every)) == {"rev_a", "rev_b", "other"}
    # A revision named outright is searched, newest or not.
    named = SearchFilters(doc_ids=["rev_a"])
    assert docs(keyword_search(conn, "supply voltage", filters=named)) == {"rev_a"}


def test_projects_scope_every_revision(conn: sqlite3.Connection) -> None:
    _two_revisions(conn)
    set_metadata(conn, "rev_a", add_projects=["board"])
    scoped = SearchFilters(project_id="board", all_revisions=True)
    assert {r.chunk.doc_id for r in keyword_search(conn, "supply", filters=scoped)} == {
        "rev_a",
        "rev_b",
    }
