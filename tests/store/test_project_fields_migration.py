"""The v4 → v5 migration (GH #123): subsystem per project, group_name dropped."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata
from datasheet_rag.store.fields import list_fields
from datasheet_rag.store.metadata import get_metadata, list_docs, set_metadata
from datasheet_rag.store.schema import SCHEMA_VERSION, MigrationError, _migrate_to_v5, connect
from datasheet_rag.store.sqlite import insert_chunks

EMBED_DIM = 4


def _chunk(doc_id: str, i: int) -> Chunk:
    return Chunk(
        id=f"{doc_id}:L2:{i}",
        doc_id=doc_id,
        level=ChunkLevel.MICRO,
        text=f"text {i}",
        context_text=f"text {i}",
        metadata=ChunkMetadata(doc_id=doc_id, page_numbers=[1]),
    )


#: (doc_id, projects, doc_metadata.subsystem, doc_metadata.group_name, chunks.group_name)
_Row = tuple[str, list[str], str | None, str | None, str | None]

_GOOD_ROWS: list[_Row] = [
    ("d1", ["a", "b"], "power", None, None),  # one value, two projects
    ("d2", ["a"], None, "psu", "psu"),  # group only: it becomes the subsystem
    ("d3", ["a"], "Power", "power ", "POWER"),  # same value, spelt three ways
    ("d4", [], None, None, None),  # nothing to move
]


def _make_v4_store(path: Path, rows: list[_Row]) -> None:
    """A store shaped as v4 left it: subsystem and group_name on the sidecar,
    group_name on every chunk row, and a field registry without ``scope``."""
    c = connect(path, embedding_dim=EMBED_DIM)
    for doc, projects, *_ in rows:
        insert_chunks(c, [_chunk(doc, 0), _chunk(doc, 1)])
        if projects:
            set_metadata(c, doc, add_projects=projects)
    c.execute("DROP TABLE doc_project_field_values")
    c.execute("DROP TABLE doc_field_values")
    c.execute("DROP TABLE metadata_fields")
    c.execute(
        """
        CREATE TABLE metadata_fields (
            name        TEXT PRIMARY KEY,
            description TEXT NOT NULL DEFAULT '',
            shape       TEXT NOT NULL CHECK (shape IN ('single', 'list')),
            roles       TEXT NOT NULL DEFAULT '[]',
            builtin     INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.executemany(
        "INSERT INTO metadata_fields (name, shape, roles, builtin) VALUES (?, ?, ?, 1)",
        [("mpn", "list", '["part"]'), ("manufacturer", "single", "[]")],
    )
    c.execute(
        """
        CREATE TABLE doc_field_values (
            doc_id   TEXT NOT NULL,
            field    TEXT NOT NULL REFERENCES metadata_fields(name),
            value    TEXT NOT NULL COLLATE NOCASE,
            position INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (doc_id, field, value)
        )
        """
    )
    c.execute("ALTER TABLE chunks ADD COLUMN group_name TEXT")
    c.execute("CREATE INDEX ix_chunks_group_name ON chunks(group_name)")
    c.execute("ALTER TABLE doc_metadata ADD COLUMN group_name TEXT")
    c.execute("ALTER TABLE doc_metadata ADD COLUMN subsystem TEXT")
    c.execute("CREATE INDEX ix_doc_metadata_group ON doc_metadata(group_name)")
    for doc, _projects, subsystem, group, chunk_group in rows:
        c.execute("INSERT OR IGNORE INTO doc_metadata (doc_id, tags) VALUES (?, '[]')", (doc,))
        c.execute(
            "UPDATE doc_metadata SET subsystem = ?, group_name = ? WHERE doc_id = ?",
            (subsystem, group, doc),
        )
        c.execute("UPDATE chunks SET group_name = ? WHERE doc_id = ?", (chunk_group, doc))
    c.execute("UPDATE schema_version SET version = 4")
    c.commit()
    c.close()


def test_v4_store_moves_subsystem_onto_each_project(tmp_path: Path) -> None:
    path = tmp_path / "rag.sqlite"
    _make_v4_store(path, _GOOD_ROWS)

    c = connect(path, embedding_dim=EMBED_DIM)
    try:
        assert c.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        md_cols = {r["name"] for r in c.execute("PRAGMA table_info(doc_metadata)")}
        assert not {"group_name", "subsystem"} & md_cols
        chunk_cols = {r["name"] for r in c.execute("PRAGMA table_info(chunks)")}
        assert "group_name" not in chunk_cols
        indexes = {
            r["name"] for r in c.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
        assert not {"ix_chunks_group_name", "ix_doc_metadata_group"} & indexes

        def per_project(doc: str) -> dict[str, dict[str, str | list[str]]]:
            md = get_metadata(c, doc)
            assert md is not None, doc
            return md.project_fields

        assert per_project("d1") == {"a": {"subsystem": "power"}, "b": {"subsystem": "power"}}
        assert per_project("d2") == {"a": {"subsystem": "psu"}}
        assert per_project("d3") == {"a": {"subsystem": "Power"}}  # subsystem's spelling wins
        assert per_project("d4") == {}

        fields = {f.name: f for f in list_fields(c)}
        assert fields["subsystem"].scope == "project"
        assert fields["mpn"].scope == "document"  # existing rows get the default
        hits = list_docs(c, project_id="a", fields={"subsystem": "power"})
        assert [m.doc_id for m in hits] == ["d1", "d3"]

        # A second process that read v4 before this one migrated does nothing.
        _migrate_to_v5(c)
        assert per_project("d2") == {"a": {"subsystem": "psu"}}
    finally:
        c.close()


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("bad", ["a"], "power", "rf", None), r"bad: different values"),
        (("bad", ["a"], "power", None, "rf"), r"bad: different values"),
        (("bad", [], "power", None, None), r"bad: .* in no project"),
        (("bad", [], None, "psu", None), r"bad: .* in no project"),
    ],
)
def test_v5_migration_fails_loudly_and_changes_nothing(
    tmp_path: Path, row: _Row, message: str
) -> None:
    path = tmp_path / "rag.sqlite"
    _make_v4_store(path, [*_GOOD_ROWS, row])

    with pytest.raises(MigrationError, match=message):
        connect(path, embedding_dim=EMBED_DIM)

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("SELECT version FROM schema_version").fetchone()[0] == 4
        md_cols = {r[1] for r in raw.execute("PRAGMA table_info(doc_metadata)")}
        assert {"group_name", "subsystem"} <= md_cols
        field_cols = {r[1] for r in raw.execute("PRAGMA table_info(metadata_fields)")}
        assert "scope" not in field_cols
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "doc_project_field_values" not in tables
    finally:
        raw.close()


def test_v5_migration_rejects_a_user_field_named_project_fields(tmp_path: Path) -> None:
    path = tmp_path / "rag.sqlite"
    _make_v4_store(path, _GOOD_ROWS)
    raw = sqlite3.connect(path)
    raw.execute("INSERT INTO metadata_fields (name, shape) VALUES ('project_fields', 'single')")
    raw.commit()
    raw.close()

    with pytest.raises(MigrationError, match="'project_fields', which is now reserved"):
        connect(path, embedding_dim=EMBED_DIM)

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("SELECT version FROM schema_version").fetchone()[0] == 4
    finally:
        raw.close()
