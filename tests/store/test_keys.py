"""Registered metadata keys (GH #120): registry, values, and the v2→v3 migration."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from datasheet_rag.store.keys import (
    BUILTIN_KEYS,
    MetadataKeyError,
    list_keys,
    register_key,
    remove_key,
)
from datasheet_rag.store.metadata import get_metadata, list_docs, set_metadata
from datasheet_rag.store.schema import SCHEMA_VERSION, connect

EMBED_DIM = 4


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    try:
        yield c
    finally:
        c.close()


def test_builtin_keys_ship_registered(conn: sqlite3.Connection) -> None:
    keys = {k.name: k for k in list_keys(conn)}
    assert {k.name for k in BUILTIN_KEYS} <= set(keys)
    assert keys["mpn"].shape == "list" and keys["mpn"].roles == ["part"]
    assert keys["manufacturer"].shape == "single" and keys["manufacturer"].builtin


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("Acme", "invalid key name"),
        ("-acme", "invalid key name"),
        ("acme:id", "invalid key name"),
        ("tags", "fixed metadata field"),
        ("version", "fixed metadata field"),
        ("project", "fixed metadata field"),
        ("keys", "fixed metadata field"),
        ("title_source", "reserved"),
        ("mpn", "already registered"),
    ],
)
def test_register_rejects_bad_names(conn: sqlite3.Connection, name: str, message: str) -> None:
    with pytest.raises(MetadataKeyError, match=message):
        register_key(conn, name)


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
    with pytest.raises(MetadataKeyError, match="unknown shape"):
        register_key(conn, "a", shape="many")  # type: ignore[arg-type]
    with pytest.raises(MetadataKeyError, match="unknown role"):
        register_key(conn, "a", roles=["boss"])


def test_set_metadata_validates_keys_before_writing(conn: sqlite3.Connection) -> None:
    register_key(conn, "acme-id")
    with pytest.raises(MetadataKeyError, match="unregistered key"):
        set_metadata(conn, "doc", project_id="p", keys={"nope": "x"})
    with pytest.raises(MetadataKeyError, match="expected a string"):
        set_metadata(conn, "doc", project_id="p", keys={"acme-id": ["a", "b"]})
    with pytest.raises(MetadataKeyError, match="registered key"):
        set_metadata(conn, "doc", project_id="p", attributes={"acme-id": "x"})
    # Nothing was written by any of the refused calls.
    assert get_metadata(conn, "doc") is None


def test_register_promotes_attribute_values(conn: sqlite3.Connection) -> None:
    set_metadata(conn, "d1", attributes={"acme-pn": "P1", "rev": "A"})
    set_metadata(conn, "d2", attributes={"rev": "B"})

    spec, promoted = register_key(conn, "acme-pn", shape="list", roles=["part"])

    assert spec.shape == "list" and promoted == 1
    d1 = get_metadata(conn, "d1")
    assert d1 is not None
    assert d1.keys == {"acme-pn": ["P1"]}
    assert d1.attributes == {"rev": "A"}
    assert [m.doc_id for m in list_docs(conn, keys={"acme-pn": "p1"})] == ["d1"]


def test_register_refuses_when_an_attribute_does_not_fit_the_shape(
    conn: sqlite3.Connection,
) -> None:
    set_metadata(conn, "d1", attributes={"acme-id": ["a", "b"]})
    with pytest.raises(MetadataKeyError, match="d1"):
        register_key(conn, "acme-id", shape="single")
    # Nothing registered, nothing moved.
    assert "acme-id" not in {k.name for k in list_keys(conn)}
    d1 = get_metadata(conn, "d1")
    assert d1 is not None and d1.attributes == {"acme-id": ["a", "b"]}


def test_remove_key(conn: sqlite3.Connection) -> None:
    register_key(conn, "acme-pn", shape="list")
    set_metadata(conn, "d1", keys={"acme-pn": ["P1", "P2"]})

    with pytest.raises(MetadataKeyError, match="built in"):
        remove_key(conn, "mpn")
    with pytest.raises(MetadataKeyError, match="not registered"):
        remove_key(conn, "nope")
    with pytest.raises(MetadataKeyError, match="demote"):
        remove_key(conn, "acme-pn")

    assert remove_key(conn, "acme-pn", demote=True) == 1
    d1 = get_metadata(conn, "d1")
    assert d1 is not None
    assert d1.keys == {}
    assert d1.attributes == {"acme-pn": ["P1", "P2"]}
    # And it round-trips: registering again promotes the list back.
    _, promoted = register_key(conn, "acme-pn", shape="list")
    assert promoted == 1
    d1 = get_metadata(conn, "d1")
    assert d1 is not None and d1.keys == {"acme-pn": ["P1", "P2"]}


def test_remove_key_without_values_needs_no_demote(conn: sqlite3.Connection) -> None:
    register_key(conn, "acme-id")
    assert remove_key(conn, "acme-id") == 0


# ---------------------------------------------------------------------------
# v2 → v3 migration
# ---------------------------------------------------------------------------


def _make_v2_store(path: Path) -> None:
    """A store shaped as v2 left it: mpn/manufacturer/doc_type as columns."""
    c = connect(path, embedding_dim=EMBED_DIM)
    c.execute("DROP TABLE doc_key_values")
    c.execute("DROP TABLE metadata_keys")
    for col in ("mpn", "manufacturer", "doc_type"):
        c.execute(f"ALTER TABLE doc_metadata ADD COLUMN {col} TEXT")
    c.execute("CREATE INDEX ix_doc_metadata_mpn ON doc_metadata(mpn)")
    rows = [
        ("d1", "A, B,,C", "TI", "datasheet", json.dumps({"rev": "A"})),
        ("d2", "LM317", None, "", None),
        ("d3", None, None, None, None),
    ]
    c.executemany(
        "INSERT INTO doc_metadata(doc_id, mpn, manufacturer, doc_type, tags, attributes) "
        "VALUES (?, ?, ?, ?, '[]', ?)",
        [(d, m, mf, dt, a) for d, m, mf, dt, a in rows],
    )
    c.execute("UPDATE schema_version SET version = 2")
    c.commit()
    c.close()


def test_v2_store_migrates_columns_onto_keys(tmp_path: Path) -> None:
    path = tmp_path / "rag.sqlite"
    _make_v2_store(path)

    c = connect(path, embedding_dim=EMBED_DIM)
    try:
        assert c.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        cols = {r["name"] for r in c.execute("PRAGMA table_info(doc_metadata)")}
        assert not cols & {"mpn", "manufacturer", "doc_type"}
        indexes = {r["name"] for r in c.execute("PRAGMA index_list(doc_metadata)")}
        assert "ix_doc_metadata_mpn" not in indexes

        d1 = get_metadata(c, "d1")
        assert d1 is not None
        assert d1.keys == {"mpn": ["A", "B", "C"], "manufacturer": "TI", "doc_type": "datasheet"}
        assert d1.attributes == {"rev": "A"}
        d2 = get_metadata(c, "d2")
        assert d2 is not None and d2.keys == {"mpn": ["LM317"]}  # "" doc_type is no value
        d3 = get_metadata(c, "d3")
        assert d3 is not None and d3.keys == {}
        assert [m.doc_id for m in list_docs(c, keys={"mpn": "B"})] == ["d1"]
    finally:
        c.close()

    # Opening again is a no-op.
    c = connect(path, embedding_dim=EMBED_DIM)
    try:
        d1 = get_metadata(c, "d1")
        assert d1 is not None and d1.keys["mpn"] == ["A", "B", "C"]
    finally:
        c.close()
