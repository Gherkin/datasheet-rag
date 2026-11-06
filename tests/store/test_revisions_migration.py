"""The v5 → v6 migration (GH #124): metadata moves from each PDF to a document."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata
from datasheet_rag.store.fields import register_field
from datasheet_rag.store.metadata import get_metadata, list_docs, set_metadata
from datasheet_rag.store.schema import SCHEMA_VERSION, MigrationError, _migrate_to_v6, connect
from datasheet_rag.store.sqlite import count_chunks, insert_chunks
from tests.store._legacy import to_v5_layout

EMBED_DIM = 4


def _chunk(doc_id: str) -> Chunk:
    return Chunk(
        id=f"{doc_id}:L2:0",
        doc_id=doc_id,
        level=ChunkLevel.MICRO,
        text="text",
        context_text="text",
        metadata=ChunkMetadata(doc_id=doc_id, page_numbers=[1]),
    )


def _make_v5_store(path: Path) -> None:
    """d1 and d2 with projects, tags, fields and a subsystem; d3 bare."""
    c = connect(path, embedding_dim=EMBED_DIM)
    register_field(c, "acme-id")
    for doc in ("d1", "d2", "d3"):
        insert_chunks(c, [_chunk(doc)])
    set_metadata(
        c,
        "d1",
        add_projects=["a", "b"],
        tags=["mcu"],
        fields={"mpn": ["X1", "X2"], "acme-id": "D-1"},
    )
    set_metadata(c, "d1", fields={"subsystem": "power"}, project="a")
    set_metadata(c, "d2", add_projects=["a"], fields={"manufacturer": "TI"})
    to_v5_layout(c)
    c.close()


def test_v5_store_gets_one_document_per_pdf(tmp_path: Path) -> None:
    path = tmp_path / "rag.sqlite"
    _make_v5_store(path)

    c = connect(path, embedding_dim=EMBED_DIM)
    try:
        assert c.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not {"doc_projects", "doc_project_field_values"} & tables
        assert c.execute("SELECT COUNT(*) FROM doc_field_values").fetchone()[0] == 0

        d1, d2, d3 = (get_metadata(c, d) for d in ("d1", "d2", "d3"))
        assert d1 is not None and d2 is not None and d3 is not None
        assert len({d1.document_id, d2.document_id, d3.document_id}) == 3
        assert d1.revisions == ["d1"] and d1.overrides == []
        assert d1.projects == ["a", "b"] and d1.tags == ["mcu"]
        assert d1.fields == {"mpn": ["X1", "X2"], "acme-id": "D-1"}
        assert d1.project_fields == {"a": {"subsystem": "power"}}
        assert d2.projects == ["a"] and d2.fields == {"manufacturer": "TI"}
        assert d3.projects == [] and d3.fields == {}

        assert count_chunks(c, project_id="b") == 1
        assert [m.doc_id for m in list_docs(c, fields={"mpn": "x2"})] == ["d1"]
        assert [m.doc_id for m in list_docs(c, project_id="a", fields={"subsystem": "power"})] == [
            "d1"
        ]

        # A second process that read v5 before this one migrated does nothing.
        _migrate_to_v6(c)
        again = get_metadata(c, "d1")
        assert again is not None and again.document_id == d1.document_id
    finally:
        c.close()


@pytest.mark.parametrize("name", ["revision", "document_id", "newest"])
def test_v6_refuses_a_user_field_with_a_reserved_name(tmp_path: Path, name: str) -> None:
    path = tmp_path / "rag.sqlite"
    _make_v5_store(path)
    raw = sqlite3.connect(path)
    raw.execute("INSERT INTO metadata_fields (name, shape) VALUES (?, 'single')", (name,))
    raw.commit()
    raw.close()

    with pytest.raises(MigrationError, match=name):
        connect(path, embedding_dim=EMBED_DIM)

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("SELECT version FROM schema_version").fetchone()[0] == 5
        tables = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "doc_projects" in tables
    finally:
        raw.close()
