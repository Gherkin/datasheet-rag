"""BOM sync (GH #125): match BOM lines to documents and keep a project in step."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata
from datasheet_rag.store.bom import BomLine, sync_bom
from datasheet_rag.store.documents import (
    add_revision,
    document_of,
    format_document_id,
    unlink_revision,
)
from datasheet_rag.store.fields import MetadataFieldError, get_field, register_field
from datasheet_rag.store.metadata import get_metadata, join_projects, set_metadata
from datasheet_rag.store.schema import SCHEMA_VERSION, connect
from datasheet_rag.store.sqlite import insert_chunks

EMBED_DIM = 4


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    try:
        yield c
    finally:
        c.close()


def _ingest(conn: sqlite3.Connection, rev_id: str, **meta: object) -> str:
    """Store a one-chunk PDF and set its metadata; return its document (``D12``)."""
    chunk = Chunk(
        id=f"{rev_id}:L2:0",
        rev_id=rev_id,
        level=ChunkLevel.MICRO,
        text="text",
        context_text="text",
        metadata=ChunkMetadata(rev_id=rev_id, page_numbers=[1], doc_title=f"title {rev_id}"),
    )
    insert_chunks(conn, [chunk], vectors={chunk.id: [1.0, 0.0, 0.0, 0.0]})
    if meta:
        set_metadata(conn, rev_id, **meta)  # type: ignore[arg-type]
    n = document_of(conn, rev_id)
    assert n is not None
    return format_document_id(n)


def _members(conn: sqlite3.Connection, project: str) -> dict[str, str]:
    return {
        format_document_id(r[0]): r[1]
        for r in conn.execute(
            "SELECT document_id, source FROM document_projects WHERE project_id = ?", (project,)
        )
    }


def _line(ref: str | None = None, **parts: str) -> BomLine:
    return BomLine(ref=ref, parts={k.replace("_", "-"): v for k, v in parts.items()})


# ---- matching -------------------------------------------------------------


def test_exact_match_ignores_case_and_joins_as_bom(conn: sqlite3.Connection) -> None:
    d1 = _ingest(conn, "r1", fields={"mpn": ["TLA2518IRTER"]})
    _ingest(conn, "r2", fields={"mpn": ["TMP112"]})

    result = sync_bom(conn, "netdaq", [_line("U8", mpn="tla2518irter")])

    assert _members(conn, "netdaq") == {d1: "bom"}
    assert [d.document_id for d in result.added] == [d1]
    assert result.added[0].title == "title r1"
    [line] = result.lines
    assert line.ref == "U8"
    assert [(m.document_id, m.fields) for m in line.matches] == [(d1, ["mpn"])]


def test_case_is_folded_for_ascii_only_like_nocase(conn: sqlite3.Connection) -> None:
    # SQLite NOCASE, which `rag list --field` uses, folds ASCII letters only.
    _ingest(conn, "r1", fields={"mpn": ["ÉX-100"]})
    _ingest(conn, "r2", fields={"mpn": ["STRASSE"]})

    result = sync_bom(conn, "p", [_line(mpn="éx-100"), _line(mpn="straße")], dry_run=True)

    assert [line.matches for line in result.lines] == [[], []]
    assert result.added == []


def test_internal_part_number_matches_and_either_identifier_is_enough(
    conn: sqlite3.Connection,
) -> None:
    register_field(conn, "acme-pn", roles=["part"])
    d1 = _ingest(conn, "r1", fields={"acme-pn": "100-2345"})
    d2 = _ingest(conn, "r2", fields={"mpn": ["REF3030"], "acme-pn": "100-9999"})

    result = sync_bom(
        conn,
        "p",
        [_line(acme_pn="100-2345", mpn="NOPE123"), _line(acme_pn="100-9999", mpn="REF3030")],
    )

    assert set(_members(conn, "p")) == {d1, d2}
    assert [m.fields for m in result.lines[1].matches] == [["acme-pn", "mpn"]]


def test_a_line_must_use_document_scoped_part_fields(conn: sqlite3.Connection) -> None:
    register_field(conn, "acme-id")
    register_field(conn, "board-pn", scope="project", roles=["part"])
    with pytest.raises(MetadataFieldError, match="not a part field"):
        sync_bom(conn, "p", [_line(acme_id="X")])
    with pytest.raises(MetadataFieldError, match="not a part field"):
        sync_bom(conn, "p", [_line(unknown="X")])
    with pytest.raises(MetadataFieldError, match="one value per project"):
        sync_bom(conn, "p", [_line(board_pn="X")])
    with pytest.raises(MetadataFieldError, match="no part number"):
        sync_bom(conn, "p", [_line("C1", mpn="  ")])
    with pytest.raises(ValueError, match="empty"):
        sync_bom(conn, " ", [_line(mpn="X")])


def test_a_revision_override_is_that_revisions_value(conn: sqlite3.Connection) -> None:
    d = _ingest(conn, "r1", fields={"mpn": ["OLD1"]})
    _ingest(conn, "r2")
    add_revision(conn, document_of(conn, "r1") or 0, "r2")
    set_metadata(conn, "r2", fields={"mpn": ["NEW1"]}, this_revision=True)

    sync_bom(conn, "a", [_line(mpn="NEW1")])
    sync_bom(conn, "b", [_line(mpn="OLD1")])  # r1 still inherits OLD1
    assert _members(conn, "a") == {d: "bom"}
    assert _members(conn, "b") == {d: "bom"}

    # Once every revision overrides it, the document's own value matches nothing.
    set_metadata(conn, "r1", fields={"mpn": ["NEW1"]}, this_revision=True)
    result = sync_bom(conn, "b", [_line(mpn="OLD1")])
    assert _members(conn, "b") == {}
    assert [r.document_id for r in result.removed] == [d]


# ---- suggestions ----------------------------------------------------------


def test_prefix_is_only_a_suggestion(conn: sqlite3.Connection) -> None:
    d = _ingest(conn, "r1", fields={"mpn": ["TLA2518"]})

    result = sync_bom(conn, "p", [_line("U8", mpn="TLA2518IRTER")])

    assert _members(conn, "p") == {}
    [line] = result.lines
    assert line.matches == []
    assert [(s.document_id, s.field, s.value) for s in line.suggestions] == [(d, "mpn", "TLA2518")]


def test_no_suggestions_from_internal_numbers_short_values_or_matched_lines(
    conn: sqlite3.Connection,
) -> None:
    register_field(conn, "acme-pn", roles=["part"])
    _ingest(conn, "r1", fields={"acme-pn": "100-23", "mpn": ["BAV"]})
    _ingest(conn, "r2", fields={"mpn": ["TMP112", "REF3030"]})

    result = sync_bom(
        conn,
        "p",
        [
            _line(acme_pn="100-2345"),  # part only: never by prefix
            _line(mpn="BAV199"),  # stored value too short
            _line(mpn="REF3030", acme_pn="X"),  # matched: no suggestions
        ],
    )

    assert [ln.suggestions for ln in result.lines] == [[], [], []]


# ---- sync -----------------------------------------------------------------


def test_resync_removes_stale_bom_documents_and_keeps_manual_ones(
    conn: sqlite3.Connection,
) -> None:
    a = _ingest(conn, "r1", fields={"mpn": ["PART-A"]})
    b = _ingest(conn, "r2", fields={"mpn": ["PART-B"]})
    note = _ingest(conn, "r3", add_projects=["p"])  # an app note, added by hand
    sync_bom(conn, "p", [_line(mpn="PART-A"), _line(mpn="PART-B")])
    set_metadata(conn, "r2", fields={"subsystem": "power"}, project="p")

    result = sync_bom(conn, "p", [_line(mpn="PART-A")])

    assert _members(conn, "p") == {a: "bom", note: "manual"}
    assert [d.document_id for d in result.removed] == [b]
    assert get_metadata(conn, "r2").project_fields == {}  # type: ignore[union-attr]


def test_same_bom_twice_changes_nothing(conn: sqlite3.Connection) -> None:
    a = _ingest(conn, "r1", fields={"mpn": ["PART-A"]})
    bom = [_line(mpn="PART-A")]
    sync_bom(conn, "p", bom)
    again = sync_bom(conn, "p", bom)
    assert again.added == [] and again.removed == []
    assert _members(conn, "p") == {a: "bom"}


def test_a_matched_manual_member_stays_manual(conn: sqlite3.Connection) -> None:
    a = _ingest(conn, "r1", add_projects=["p"], fields={"mpn": ["PART-A"]})

    result = sync_bom(conn, "p", [_line(mpn="PART-A")])
    assert [d.document_id for d in result.kept_manual] == [a]
    assert result.added == []

    sync_bom(conn, "p", [])  # the part left the BOM
    assert _members(conn, "p") == {a: "manual"}


def test_a_sync_touches_only_its_project(conn: sqlite3.Connection) -> None:
    a = _ingest(conn, "r1", fields={"mpn": ["PART-A"]})
    sync_bom(conn, "p", [_line(mpn="PART-A")])
    sync_bom(conn, "q", [_line(mpn="PART-A")])
    sync_bom(conn, "q", [])
    assert _members(conn, "p") == {a: "bom"}
    assert _members(conn, "q") == {}


def test_dry_run_writes_nothing(conn: sqlite3.Connection) -> None:
    a = _ingest(conn, "r1", fields={"mpn": ["PART-A"]})
    result = sync_bom(conn, "p", [_line(mpn="PART-A")], dry_run=True)
    assert result.dry_run and [d.document_id for d in result.added] == [a]
    assert _members(conn, "p") == {}
    assert not conn.in_transaction


def test_adding_a_bom_member_by_hand_makes_it_manual(conn: sqlite3.Connection) -> None:
    a = _ingest(conn, "r1", fields={"mpn": ["PART-A"]})
    b = _ingest(conn, "r2", fields={"mpn": ["PART-B"]})
    c = _ingest(conn, "r3", fields={"mpn": ["PART-C"]})
    d = _ingest(conn, "r4", fields={"mpn": ["PART-D"]})
    parts = ["PART-A", "PART-B", "PART-C", "PART-D"]
    sync_bom(conn, "p", [_line(mpn=m) for m in parts])

    set_metadata(conn, "r1", add_projects=["p"])  # explicit: claims
    set_metadata(conn, "r2", default_projects=["p"])  # .rag.toml default: does not
    join_projects(conn, "r3", ["p"], claim=["p"])  # ingest --project-id: claims
    join_projects(conn, "r4", ["p"])  # ingest with the .rag.toml project: does not
    conn.commit()
    assert _members(conn, "p") == {a: "manual", b: "bom", c: "manual", d: "bom"}

    # Every part leaves the BOM: only the BOM-added ones go.
    result = sync_bom(conn, "p", [_line(mpn="OTHER")])
    assert [x.document_id for x in result.removed] == [b, d]
    assert _members(conn, "p") == {a: "manual", c: "manual"}


def test_a_default_project_joins_a_document_not_in_it(conn: sqlite3.Connection) -> None:
    a = _ingest(conn, "r1")
    set_metadata(conn, "r1", default_projects=["p"], fields={"subsystem": "adc"}, project="p")
    assert _members(conn, "p") == {a: "manual"}


# ---- revisions keep the source -----------------------------------------------


def test_merging_documents_keeps_manual_over_bom(conn: sqlite3.Connection) -> None:
    _ingest(conn, "r1", fields={"mpn": ["PART-A"]})
    _ingest(conn, "r2", add_projects=["p"])
    sync_bom(conn, "p", [_line(mpn="PART-A")])
    sync_bom(conn, "q", [_line(mpn="PART-A")])

    add_revision(conn, document_of(conn, "r1") or 0, "r2")  # r2's document merges in

    d = format_document_id(document_of(conn, "r1") or 0)
    assert _members(conn, "p") == {d: "manual"}
    assert _members(conn, "q") == {d: "bom"}


def test_unlinking_a_revision_copies_the_source(conn: sqlite3.Connection) -> None:
    _ingest(conn, "r1", fields={"mpn": ["PART-A"]})
    _ingest(conn, "r2")
    add_revision(conn, document_of(conn, "r1") or 0, "r2")
    sync_bom(conn, "p", [_line(mpn="PART-A")])

    new = format_document_id(unlink_revision(conn, "r2"))

    assert _members(conn, "p")[new] == "bom"


# ---- registry and migration ---------------------------------------------------


def test_mfr_part_needs_part(conn: sqlite3.Connection) -> None:
    with pytest.raises(MetadataFieldError, match="give role part too"):
        register_field(conn, "vendor-pn", roles=["mfr-part"])
    assert register_field(conn, "vendor-pn", roles=["part", "mfr-part"]).roles == [
        "part",
        "mfr-part",
    ]


def test_v6_store_gets_membership_sources(tmp_path: Path) -> None:
    path = tmp_path / "rag.sqlite"
    c = connect(path, embedding_dim=EMBED_DIM)
    a = _ingest(c, "r1", add_projects=["p"])
    # Back to the v6 layout: no source column, mpn with the part role only.
    c.execute("ALTER TABLE document_projects DROP COLUMN source")
    c.execute("UPDATE metadata_fields SET roles = '[\"part\"]' WHERE name = 'mpn'")
    c.execute("UPDATE schema_version SET version = 6")
    c.commit()
    c.close()

    c = connect(path, embedding_dim=EMBED_DIM)
    try:
        assert c.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        assert _members(c, "p") == {a: "manual"}
        spec = get_field(c, "mpn")
        assert spec is not None and spec.roles == ["part", "mfr-part"]
        stored = c.execute("SELECT roles FROM metadata_fields WHERE name = 'mpn'").fetchone()[0]
        assert json.loads(stored) == ["part", "mfr-part"]
    finally:
        c.close()
