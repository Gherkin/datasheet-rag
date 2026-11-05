"""Project membership (GH #121): the v3 → v4 migration into ``doc_projects``."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata
from datasheet_rag.store.metadata import get_metadata, list_docs
from datasheet_rag.store.schema import SCHEMA_VERSION, MigrationError, _migrate_to_v4, connect
from datasheet_rag.store.sqlite import count_chunks, insert_chunks

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


def _make_v3_store(path: Path) -> None:
    """A store shaped as v3 left it: ``project_id`` on chunks and on the sidecar.

    d1: chunks and sidecar agree on "a".
    d2: chunks say "b", the sidecar says "c" — both must survive.
    d3: chunks say " x " (padded), no sidecar row.
    d4: no project anywhere.
    """
    c = connect(path, embedding_dim=EMBED_DIM)
    for doc in ("d1", "d2", "d3", "d4"):
        insert_chunks(c, [_chunk(doc, 0), _chunk(doc, 1)])
    c.execute("DROP TABLE doc_projects")
    c.execute("ALTER TABLE chunks ADD COLUMN project_id TEXT")
    c.execute("CREATE INDEX ix_chunks_project_id ON chunks(project_id)")
    c.execute("ALTER TABLE doc_metadata ADD COLUMN project_id TEXT")
    c.execute("CREATE INDEX ix_doc_metadata_project ON doc_metadata(project_id)")
    for doc, project in (("d1", "a"), ("d2", "b"), ("d3", " x ")):
        c.execute("UPDATE chunks SET project_id = ? WHERE doc_id = ?", (project, doc))
    c.executemany(
        "INSERT INTO doc_metadata (doc_id, project_id, tags) VALUES (?, ?, '[]')",
        [("d1", "a"), ("d2", "c"), ("d4", None)],
    )
    c.execute("UPDATE schema_version SET version = 3")
    c.commit()
    c.close()


def test_v3_store_moves_projects_into_doc_projects(tmp_path: Path) -> None:
    path = tmp_path / "rag.sqlite"
    _make_v3_store(path)

    c = connect(path, embedding_dim=EMBED_DIM)
    try:
        assert c.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        for table in ("chunks", "doc_metadata"):
            cols = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
            assert "project_id" not in cols, table

        def projects(doc: str) -> list[str]:
            md = get_metadata(c, doc)
            assert md is not None, doc
            return md.projects

        assert projects("d1") == ["a"]
        assert projects("d2") == ["b", "c"]
        assert projects("d3") == ["x"]  # stripped; the sidecar row was created
        assert projects("d4") == []

        assert count_chunks(c, project_id="b") == 2
        assert count_chunks(c, project_id="c") == 2
        assert [m.doc_id for m in list_docs(c, project_id="x")] == ["d3"]

        # A second process that read v3 before this one migrated does nothing.
        _migrate_to_v4(c)
        assert projects("d2") == ["b", "c"]
    finally:
        c.close()

    # Opening again is a no-op.
    c = connect(path, embedding_dim=EMBED_DIM)
    try:
        assert count_chunks(c, project_id="a") == 2
    finally:
        c.close()


def test_v4_migration_refuses_a_user_field_named_projects(tmp_path: Path) -> None:
    path = tmp_path / "rag.sqlite"
    _make_v3_store(path)
    raw = sqlite3.connect(path)
    raw.execute("INSERT INTO metadata_fields (name, shape) VALUES ('projects', 'list')")
    raw.commit()
    raw.close()

    with pytest.raises(MigrationError, match="'projects'"):
        connect(path, embedding_dim=EMBED_DIM)

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("SELECT version FROM schema_version").fetchone()[0] == 3
    finally:
        raw.close()


def test_failed_v4_migration_leaves_the_v3_store(tmp_path: Path) -> None:
    path = tmp_path / "rag.sqlite"
    _make_v3_store(path)
    # A view on the column makes DROP COLUMN fail mid-migration.
    raw = sqlite3.connect(path)
    raw.execute("CREATE VIEW v AS SELECT project_id FROM doc_metadata")
    raw.commit()
    raw.close()

    with pytest.raises(sqlite3.OperationalError):
        connect(path, embedding_dim=EMBED_DIM)

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("SELECT version FROM schema_version").fetchone()[0] == 3
        cols = {r[1] for r in raw.execute("PRAGMA table_info(chunks)")}
        assert "project_id" in cols
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "doc_projects" not in tables
    finally:
        raw.close()
