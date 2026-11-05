"""Registered metadata fields (GH #120): registry, values, and the v2→v3 migration."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from datasheet_rag.store.fields import (
    BUILTIN_FIELDS,
    MetadataFieldError,
    list_fields,
    register_field,
    remove_field,
)
from datasheet_rag.store.metadata import (
    get_metadata,
    list_docs,
    set_metadata,
    set_needs_reembed,
    set_title_hints,
)
from datasheet_rag.store.schema import SCHEMA_VERSION, MigrationError, _migrate_to_v3, connect

EMBED_DIM = 4


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    try:
        yield c
    finally:
        c.close()


def test_builtin_fields_ship_registered(conn: sqlite3.Connection) -> None:
    fields = {f.name: f for f in list_fields(conn)}
    assert {f.name for f in BUILTIN_FIELDS} <= set(fields)
    assert fields["mpn"].shape == "list" and fields["mpn"].roles == ["part"]
    assert fields["manufacturer"].shape == "single" and fields["manufacturer"].builtin
    assert fields["mpn"].scope == "document"
    assert fields["subsystem"].scope == "project" and fields["subsystem"].builtin


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("Acme", "invalid field name"),
        ("-acme", "invalid field name"),
        ("acme:id", "invalid field name"),
        ("tags", "reserved"),
        ("version", "reserved"),
        ("project", "reserved"),
        ("fields", "reserved"),
        ("title_source", "reserved"),
        ("needs_reembed", "reserved"),
        ("pdf_meta_title", "reserved"),
        ("running_header", "reserved"),
        ("id", "reserved"),  # `rag list --columns` alias of doc_id
        ("chunk_count", "reserved"),
        ("project_fields", "reserved"),
        ("mpn", "already registered"),
        ("subsystem", "already registered"),
    ],
)
def test_register_rejects_bad_names(conn: sqlite3.Connection, name: str, message: str) -> None:
    with pytest.raises(MetadataFieldError, match=message):
        register_field(conn, name)


def test_reopen_of_current_store_takes_no_write_lock(tmp_path: Path) -> None:
    db = tmp_path / "rag.sqlite"
    connect(db, embedding_dim=EMBED_DIM).close()
    writer = sqlite3.connect(db, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    try:
        # Would raise "database is locked" if the open tried to write.
        connect(db, embedding_dim=EMBED_DIM).close()
    finally:
        writer.rollback()
        writer.close()


def test_register_rejects_unknown_shape_and_role(conn: sqlite3.Connection) -> None:
    with pytest.raises(MetadataFieldError, match="unknown shape"):
        register_field(conn, "a", shape="many")  # type: ignore[arg-type]
    with pytest.raises(MetadataFieldError, match="unknown role"):
        register_field(conn, "a", roles=["boss"])
    with pytest.raises(MetadataFieldError, match="unknown scope"):
        register_field(conn, "a", scope="board")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Project-scoped fields (GH #123)
# ---------------------------------------------------------------------------


def test_project_field_holds_one_value_per_project(conn: sqlite3.Connection) -> None:
    set_metadata(conn, "ldo", add_projects=["netdaq", "board-b"])
    set_metadata(conn, "ldo", fields={"subsystem": "power"}, project="netdaq")
    md = set_metadata(conn, "ldo", fields={"subsystem": "aux"}, project="board-b")
    assert md.project_fields == {"netdaq": {"subsystem": "power"}, "board-b": {"subsystem": "aux"}}
    # Not a document-level value.
    assert "subsystem" not in md.fields

    def ids(project: str | None, value: str) -> list[str]:
        return [m.doc_id for m in list_docs(conn, project_id=project, fields={"subsystem": value})]

    assert ids("netdaq", "power") == ["ldo"]
    assert ids("netdaq", "aux") == []
    assert ids("board-b", "AUX") == ["ldo"]  # case-insensitive
    # Unscoped: any project.
    assert ids(None, "aux") == ["ldo"]
    assert ids(None, "power") == ["ldo"]

    # Clearing one project's value keeps the other.
    md = set_metadata(conn, "ldo", fields={"subsystem": ""}, project="board-b")
    assert md.project_fields == {"netdaq": {"subsystem": "power"}}


def test_project_field_needs_a_project(conn: sqlite3.Connection) -> None:
    set_metadata(conn, "d", add_projects=["only"])
    # Never guessed from the document's one project.
    with pytest.raises(MetadataFieldError, match="one value per project"):
        set_metadata(conn, "d", fields={"subsystem": "power"})
    with pytest.raises(MetadataFieldError, match="not in project"):
        set_metadata(conn, "d", fields={"subsystem": "power"}, project="other")
    md = get_metadata(conn, "d")
    assert md is not None and md.project_fields == {}
    # Joining in the same call is enough.
    md = set_metadata(
        conn, "d", add_projects=["other"], fields={"subsystem": "power"}, project="other"
    )
    assert md.project_fields == {"other": {"subsystem": "power"}}
    # Leaving in the same call is not.
    with pytest.raises(MetadataFieldError, match="not in project"):
        set_metadata(
            conn, "d", remove_projects=["other"], fields={"subsystem": "x"}, project="other"
        )


def test_leaving_a_project_drops_its_project_fields(conn: sqlite3.Connection) -> None:
    set_metadata(conn, "d", add_projects=["a", "b"])
    set_metadata(conn, "d", fields={"subsystem": "power"}, project="a")
    set_metadata(conn, "d", fields={"subsystem": "rf"}, project="b")
    md = set_metadata(conn, "d", remove_projects=["a"])
    assert md.project_fields == {"b": {"subsystem": "rf"}}
    # Rejoining starts empty.
    md = set_metadata(conn, "d", add_projects=["a"])
    assert md.project_fields == {"b": {"subsystem": "rf"}}


def test_user_project_field_and_its_removal(conn: sqlite3.Connection) -> None:
    spec = register_field(conn, "board-ref", shape="list", scope="project")
    assert spec.scope == "project"
    set_metadata(conn, "d", add_projects=["p"])
    md = set_metadata(conn, "d", fields={"board-ref": ["U1", "U7"]}, project="p")
    assert md.project_fields == {"p": {"board-ref": ["U1", "U7"]}}
    with pytest.raises(MetadataFieldError, match="clear"):
        remove_field(conn, "board-ref")
    assert remove_field(conn, "board-ref", clear=True) == 1
    md2 = get_metadata(conn, "d")
    assert md2 is not None and md2.project_fields == {}


def test_set_metadata_validates_fields_before_writing(conn: sqlite3.Connection) -> None:
    register_field(conn, "acme-id")
    with pytest.raises(MetadataFieldError, match="unregistered field"):
        set_metadata(conn, "doc", add_projects=["p"], fields={"nope": "x"})
    with pytest.raises(MetadataFieldError, match="expected a string"):
        set_metadata(conn, "doc", add_projects=["p"], fields={"acme-id": ["a", "b"]})
    # Nothing was written by either refused call.
    assert get_metadata(conn, "doc") is None


def test_remove_field(conn: sqlite3.Connection) -> None:
    register_field(conn, "acme-pn", shape="list")
    set_metadata(conn, "d1", fields={"acme-pn": ["P1", "P2"]})

    with pytest.raises(MetadataFieldError, match="built in"):
        remove_field(conn, "mpn")
    with pytest.raises(MetadataFieldError, match="not registered"):
        remove_field(conn, "nope")
    with pytest.raises(MetadataFieldError, match="clear"):
        remove_field(conn, "acme-pn")
    # The refusal changed nothing.
    d1 = get_metadata(conn, "d1")
    assert d1 is not None and d1.fields == {"acme-pn": ["P1", "P2"]}

    assert remove_field(conn, "acme-pn", clear=True) == 1
    assert "acme-pn" not in {f.name for f in list_fields(conn)}
    d1 = get_metadata(conn, "d1")
    assert d1 is not None and d1.fields == {}
    # Registering again starts empty: the values went with the field.
    register_field(conn, "acme-pn", shape="list")
    d1 = get_metadata(conn, "d1")
    assert d1 is not None and d1.fields == {}


def test_remove_field_without_values_needs_no_clear(conn: sqlite3.Connection) -> None:
    register_field(conn, "acme-id")
    assert remove_field(conn, "acme-id") == 0


def test_bookkeeping_is_separate_from_the_metadata_patch(conn: sqlite3.Connection) -> None:
    set_metadata(conn, "d1", add_projects=["p"])
    set_title_hints(conn, "d1", pdf_meta_title="Guide", running_header=None)
    set_needs_reembed(conn, "d1", True)
    # A later metadata write leaves the bookkeeping alone.
    set_metadata(conn, "d1", fields={"subsystem": "power"}, project="p")
    d1 = get_metadata(conn, "d1")
    assert d1 is not None
    assert (d1.projects, d1.project_fields) == (["p"], {"p": {"subsystem": "power"}})
    assert d1.pdf_meta_title == "Guide" and d1.running_header is None
    assert d1.needs_reembed is True
    assert d1.title_source == "auto"
    set_needs_reembed(conn, "d1", False)
    d1 = get_metadata(conn, "d1")
    assert d1 is not None and d1.needs_reembed is False


def test_bookkeeping_creates_the_row_when_missing(conn: sqlite3.Connection) -> None:
    set_needs_reembed(conn, "fresh", True)
    md = get_metadata(conn, "fresh")
    assert md is not None and md.needs_reembed and md.projects == []


def test_title_hints_are_replaced_whole(conn: sqlite3.Connection) -> None:
    set_title_hints(conn, "d1", pdf_meta_title="Guide", running_header="LM317")
    # A re-ingest that finds no hints clears both, not neither.
    set_title_hints(conn, "d1", pdf_meta_title=None, running_header=None)
    d1 = get_metadata(conn, "d1")
    assert d1 is not None and d1.pdf_meta_title is None and d1.running_header is None
    # ...and creates no row for a document that has none.
    set_title_hints(conn, "nohints", pdf_meta_title=None, running_header=None)
    assert get_metadata(conn, "nohints") is None


def test_reserved_names_cover_every_rag_list_column() -> None:
    from datasheet_rag.cli import _DOC_COLUMNS, _LIST_COLUMN_ALIASES, _META_COLUMNS
    from datasheet_rag.store.fields import RESERVED_NAMES

    columns = {*_DOC_COLUMNS, *_META_COLUMNS, *_LIST_COLUMN_ALIASES}
    assert columns <= RESERVED_NAMES, sorted(columns - RESERVED_NAMES)


def test_list_docs_rejects_an_unregistered_filter_field(conn: sqlite3.Connection) -> None:
    with pytest.raises(MetadataFieldError, match="unregistered field"):
        list_docs(conn, fields={"acme_id": "D-17"})


# ---------------------------------------------------------------------------
# v2 → v3 migration
# ---------------------------------------------------------------------------


def _make_v2_store(path: Path, rows: list[tuple[object, ...]]) -> None:
    """A store shaped as v2 left it: mpn/manufacturer/doc_type columns, and
    free-form ``attributes`` JSON holding the pipeline's bookkeeping."""
    c = connect(path, embedding_dim=EMBED_DIM)
    c.execute("DROP TABLE doc_project_field_values")
    c.execute("DROP TABLE doc_field_values")
    c.execute("DROP TABLE metadata_fields")
    c.execute("DROP TABLE doc_metadata")
    c.execute(
        """
        CREATE TABLE doc_metadata (
            doc_id       TEXT PRIMARY KEY,
            project_id   TEXT,
            group_name   TEXT,
            mpn          TEXT,
            manufacturer TEXT,
            subsystem    TEXT,
            doc_type     TEXT,
            tags         TEXT,
            attributes   TEXT,
            updated_at   TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute("CREATE INDEX ix_doc_metadata_mpn ON doc_metadata(mpn)")
    c.executemany(
        "INSERT INTO doc_metadata(doc_id, mpn, manufacturer, doc_type, tags, attributes) "
        "VALUES (?, ?, ?, ?, '[]', ?)",
        rows,
    )
    c.execute("UPDATE schema_version SET version = 2")
    c.commit()
    c.close()


_V2_ROWS: list[tuple[object, ...]] = [
    (
        "d1",
        "A, B,,C",
        "TI",
        "datasheet",
        json.dumps({"title_source": "manual", "pdf_meta_title": "Guide"}),
    ),
    ("d2", "LM317", None, "", json.dumps({"title_inferred": True, "running_header": "LM317"})),
    ("d3", None, None, None, None),
    ("d4", None, None, None, json.dumps({"title_inferred": True, "title_source": "manual"})),
    ("d5", None, None, None, json.dumps({"needs_reembed": True})),
    ("d6", None, None, None, json.dumps({"needs_reembed": False, "pdf_meta_title": ""})),
]


def test_v2_store_migrates_onto_fields_and_bookkeeping_columns(tmp_path: Path) -> None:
    path = tmp_path / "rag.sqlite"
    _make_v2_store(path, _V2_ROWS)

    c = connect(path, embedding_dim=EMBED_DIM)
    try:
        assert c.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        cols = {r["name"] for r in c.execute("PRAGMA table_info(doc_metadata)")}
        assert not cols & {"mpn", "manufacturer", "doc_type", "attributes"}
        assert {"title_source", "needs_reembed", "pdf_meta_title", "running_header"} <= cols
        indexes = {r["name"] for r in c.execute("PRAGMA index_list(doc_metadata)")}
        assert "ix_doc_metadata_mpn" not in indexes

        d1 = get_metadata(c, "d1")
        assert d1 is not None
        assert d1.fields == {"mpn": ["A", "B", "C"], "manufacturer": "TI", "doc_type": "datasheet"}
        assert d1.title_source == "manual" and d1.pdf_meta_title == "Guide"
        d2 = get_metadata(c, "d2")
        assert d2 is not None and d2.fields == {"mpn": ["LM317"]}  # "" doc_type is no value
        assert d2.title_source == "inferred" and d2.running_header == "LM317"
        d3 = get_metadata(c, "d3")
        assert d3 is not None and d3.fields == {} and d3.title_source == "auto"
        d4 = get_metadata(c, "d4")
        assert d4 is not None and d4.title_source == "manual"  # title_source beats the boolean
        d5 = get_metadata(c, "d5")
        assert d5 is not None and d5.needs_reembed is True
        # One encoding of "not stale": NULL, as set_needs_reembed writes it.
        stale = c.execute(
            "SELECT needs_reembed, pdf_meta_title FROM doc_metadata WHERE doc_id = 'd6'"
        ).fetchone()
        assert stale[0] is None and stale[1] is None  # "" becomes NULL too
        assert [m.doc_id for m in list_docs(c, fields={"mpn": "B"})] == ["d1"]

        # A second process that read v2 before this one migrated re-checks
        # under the write lock and does nothing.
        _migrate_to_v3(c)
        d1 = get_metadata(c, "d1")
        assert d1 is not None and d1.fields["mpn"] == ["A", "B", "C"]
    finally:
        c.close()

    # Opening again is a no-op.
    c = connect(path, embedding_dim=EMBED_DIM)
    try:
        d1 = get_metadata(c, "d1")
        assert d1 is not None and d1.fields["mpn"] == ["A", "B", "C"]
    finally:
        c.close()


@pytest.mark.parametrize(
    "attributes",
    [
        {"revision": "B"},  # a free-form attribute
        {"title_source": "guessed"},  # a bookkeeping name with a bad value
        {"pdf_meta_title": 3},
    ],
)
def test_v2_store_with_free_form_attributes_fails_loudly(
    tmp_path: Path, attributes: dict[str, object]
) -> None:
    path = tmp_path / "rag.sqlite"
    _make_v2_store(path, [("d1", "X", None, None, json.dumps(attributes)), *_V2_ROWS[1:]])

    with pytest.raises(MigrationError, match=r"d1: " + next(iter(attributes))):
        connect(path, embedding_dim=EMBED_DIM)

    # Nothing changed: still v2, columns and values in place.
    raw = sqlite3.connect(path)
    try:
        assert raw.execute("SELECT version FROM schema_version").fetchone()[0] == 2
        assert raw.execute(
            "SELECT mpn, attributes FROM doc_metadata WHERE doc_id = 'd1'"
        ).fetchone()
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "metadata_fields" not in tables
    finally:
        raw.close()
