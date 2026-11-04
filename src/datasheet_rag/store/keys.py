"""Registered metadata keys (GH #120).

A *key* is a named, typed, filterable piece of document metadata. Users
register their own — an internal document number (``acme-id``), a vendor's
document number (``intel-id``), an internal part number (``acme-pn``) — and
the pipeline ships three of its own pre-registered: ``mpn``, ``manufacturer``
and ``doc_type``. Built-in and user keys share one code path; "built-in" only
means the key ships registered and cannot be removed.

Two tables back this:

* ``metadata_keys``  – the registry: name, description, shape (``single`` or
                       ``list``), roles, and whether the key is built in.
* ``doc_key_values`` – one row per (document, key, value). A single-valued key
                       has at most one row per document; a list-valued key
                       keeps its order in ``position``.

Values compare case-insensitively (``COLLATE NOCASE``): part and document
numbers are identifiers, and ``stm32h743`` should find ``STM32H743``.

Free-form ``attributes`` on ``doc_metadata`` stay as they were. Registering a
key whose name is already used as an attribute *promotes* those values into
the key; removing a key with ``demote=True`` moves them back.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

#: A key holds one value per document, or an ordered list of them.
KeyShape = Literal["single", "list"]
KEY_SHAPES: tuple[KeyShape, ...] = get_args(KeyShape)

#: What a key is used for, beyond being filterable. ``part`` marks a key whose
#: values are part numbers (BOM import matches on these — GH #125).
KeyRole = Literal["part"]
KEY_ROLES: tuple[KeyRole, ...] = get_args(KeyRole)

#: A key's value as stored and returned: a string, or a list of strings.
KeyValue = str | list[str]

#: Key names: lower case, digits, ``-`` and ``_``, starting with a letter or
#: digit. Kept narrow so a name is safe on a command line, in a TOML table and
#: in a query string without quoting.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

#: Names a key may not take: the fixed ``doc_metadata`` fields, the fields
#: planned to become fixed (``version``, GH #124), and the built-in column
#: names of ``rag list``, which would shadow the key. A key with one of these
#: names would be ambiguous in every listing and filter.
FIXED_FIELDS = frozenset(
    {
        *("doc_id", "title", "project_id", "group_name", "subsystem", "tags", "attributes"),
        *("keys", "version"),
        *("project", "group", "updated", "stale", "chunks", "pages", "ingested"),
    }
)


class KeySpec(BaseModel):
    """One row of the key registry."""

    name: str
    description: str = ""
    shape: KeyShape = "single"
    roles: list[KeyRole] = Field(default_factory=list)
    builtin: bool = False


#: The keys every store ships with. Their values used to be fixed columns on
#: ``doc_metadata`` (moved here by the v3 migration in ``store.schema``).
BUILTIN_KEYS: tuple[KeySpec, ...] = (
    KeySpec(
        name="mpn",
        description="Manufacturer part number(s) the document covers, e.g. STM32H743VIT6.",
        shape="list",
        roles=["part"],
        builtin=True,
    ),
    KeySpec(
        name="manufacturer",
        description="Manufacturer name, e.g. STMicroelectronics.",
        shape="single",
        builtin=True,
    ),
    KeySpec(
        name="doc_type",
        description="Kind of document: datasheet, reference-manual, errata, app-note, …",
        shape="single",
        builtin=True,
    ),
)


class MetadataKeyError(ValueError):
    """A key operation that cannot be applied as asked (unknown key, bad
    shape, name clash). A ``ValueError`` so the HTTP layer maps it to 400."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def ensure_key_tables(conn: sqlite3.Connection) -> None:
    """Create the registry and value tables and seed the built-in keys.

    Idempotent, and safe on every open: ``IF NOT EXISTS`` for the DDL and
    ``INSERT OR IGNORE`` for the seed rows, so an older store gains the tables
    and a newer one is left alone. Does not commit — the caller does, so this
    can run inside the v3 migration's transaction.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata_keys (
            name        TEXT PRIMARY KEY,
            description TEXT NOT NULL DEFAULT '',
            shape       TEXT NOT NULL CHECK (shape IN ('single', 'list')),
            roles       TEXT NOT NULL DEFAULT '[]',   -- JSON array of strings
            builtin     INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_key_values (
            doc_id   TEXT NOT NULL,
            key      TEXT NOT NULL REFERENCES metadata_keys(name),
            value    TEXT NOT NULL COLLATE NOCASE,
            position INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (doc_id, key, value)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_doc_key_values_kv ON doc_key_values(key, value)")
    # Insert only the missing seeds: even an INSERT OR IGNORE that inserts
    # nothing takes the write lock, and this runs on every open.
    present = {r[0] for r in conn.execute("SELECT name FROM metadata_keys")}
    for spec in BUILTIN_KEYS:
        if spec.name in present:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO metadata_keys(name, description, shape, roles, builtin) "
            "VALUES (?, ?, ?, ?, 1)",
            (spec.name, spec.description, spec.shape, json.dumps(spec.roles)),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _row_to_spec(row: sqlite3.Row) -> KeySpec:
    return KeySpec(
        name=row["name"],
        description=row["description"],
        shape=row["shape"],
        roles=json.loads(row["roles"] or "[]"),
        builtin=bool(row["builtin"]),
    )


def list_keys(conn: sqlite3.Connection) -> list[KeySpec]:
    """Every registered key, built-ins first, then by name."""
    rows = conn.execute("SELECT * FROM metadata_keys ORDER BY builtin DESC, name").fetchall()
    return [_row_to_spec(r) for r in rows]


def get_key(conn: sqlite3.Connection, name: str) -> KeySpec | None:
    row = conn.execute("SELECT * FROM metadata_keys WHERE name = ?", (name,)).fetchone()
    return _row_to_spec(row) if row is not None else None


def key_shapes(conn: sqlite3.Connection) -> dict[str, KeyShape]:
    """Map every registered key name to its shape."""
    return {r["name"]: r["shape"] for r in conn.execute("SELECT name, shape FROM metadata_keys")}


def validate_key_name(name: str) -> None:
    """Raise :class:`MetadataKeyError` unless ``name`` may be registered."""
    from datasheet_rag.store.metadata import RESERVED_ATTRIBUTES

    if not _NAME_RE.match(name):
        raise MetadataKeyError(
            f"invalid key name {name!r}: use lower-case letters, digits, '-' and '_', "
            "starting with a letter or digit"
        )
    if name in FIXED_FIELDS:
        raise MetadataKeyError(f"{name!r} is a fixed metadata field and cannot be a key")
    if name in RESERVED_ATTRIBUTES:
        raise MetadataKeyError(f"{name!r} is reserved for the pipeline and cannot be a key")


def register_key(
    conn: sqlite3.Connection,
    name: str,
    *,
    shape: KeyShape = "single",
    description: str = "",
    roles: Iterable[str] = (),
) -> tuple[KeySpec, int]:
    """Register a new key and promote any attribute of the same name into it.

    Returns the new spec and the number of documents whose ``attributes[name]``
    moved into the key. The whole operation is one transaction: if any
    document's attribute cannot be read as the key's shape, nothing is
    registered and the error names those documents.
    """
    validate_key_name(name)
    if shape not in KEY_SHAPES:
        raise MetadataKeyError(f"unknown shape {shape!r} (expected one of {KEY_SHAPES})")
    role_list = list(dict.fromkeys(roles))
    bad_roles = [r for r in role_list if r not in KEY_ROLES]
    if bad_roles:
        raise MetadataKeyError(f"unknown role(s) {bad_roles} (expected some of {KEY_ROLES})")
    # Read and write under one write lock, so a concurrent write to a
    # document's attributes cannot land between the read and the write-back.
    _begin(conn)
    try:
        if get_key(conn, name) is not None:
            raise MetadataKeyError(f"key {name!r} is already registered")

        # Find attribute values to promote before writing anything.
        promote: list[tuple[str, dict[str, Any], list[str]]] = []
        bad_docs: list[str] = []
        rows = conn.execute(
            "SELECT doc_id, attributes FROM doc_metadata WHERE attributes IS NOT NULL"
        ).fetchall()
        for row in rows:
            attrs = json.loads(row["attributes"] or "{}")
            if name not in attrs:
                continue
            try:
                values = _coerce(attrs[name], shape)
            except MetadataKeyError:
                bad_docs.append(row["doc_id"])
                continue
            promote.append((row["doc_id"], attrs, values))
        if bad_docs:
            raise MetadataKeyError(
                f"cannot register {name!r} as a {shape} key: attribute {name!r} on "
                f"{len(bad_docs)} document(s) is not a {_shape_noun(shape)}: {', '.join(bad_docs)}"
            )

        conn.execute(
            "INSERT INTO metadata_keys(name, description, shape, roles, builtin) "
            "VALUES (?, ?, ?, ?, 0)",
            (name, description, shape, json.dumps(role_list)),
        )
        for doc_id, attrs, values in promote:
            attrs.pop(name)
            conn.execute(
                "UPDATE doc_metadata SET attributes = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE doc_id = ?",
                (json.dumps(attrs), doc_id),
            )
            _write_values(conn, doc_id, name, values)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    spec = get_key(conn, name)
    assert spec is not None
    return spec, len(promote)


def remove_key(conn: sqlite3.Connection, name: str, *, demote: bool = False) -> int:
    """Unregister ``name``. Returns the number of documents that held a value.

    Refuses while any document has a value for the key, unless ``demote`` is
    set: then each value moves back into that document's ``attributes`` (a
    list key's values as a JSON list), so unregistering loses nothing.
    Built-in keys cannot be removed.
    """
    _begin(conn)
    try:
        spec = get_key(conn, name)
        if spec is None:
            raise MetadataKeyError(f"key {name!r} is not registered")
        if spec.builtin:
            raise MetadataKeyError(f"key {name!r} is built in and cannot be removed")

        held = get_values_for_docs(conn, None, keys=[name])
        if held and not demote:
            raise MetadataKeyError(
                f"key {name!r} has values on {len(held)} document(s); pass demote to "
                "move them back into attributes, or clear them first"
            )

        for doc_id, values in held.items():
            row = conn.execute(
                "SELECT attributes FROM doc_metadata WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            attrs = json.loads(row["attributes"] or "{}") if row is not None else {}
            attrs[name] = values[name]
            conn.execute(
                "UPDATE doc_metadata SET attributes = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE doc_id = ?",
                (json.dumps(attrs), doc_id),
            )
        conn.execute("DELETE FROM doc_key_values WHERE key = ?", (name,))
        conn.execute("DELETE FROM metadata_keys WHERE name = ?", (name,))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return len(held)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def display_value(value: KeyValue | None) -> str | None:
    """A key's value as one line of text: a list joined with ``", "``."""
    if isinstance(value, list):
        return ", ".join(value) or None
    return value or None


def _shape_noun(shape: KeyShape) -> str:
    return "string" if shape == "single" else "string or list of strings"


def _coerce(value: Any, shape: KeyShape) -> list[str]:
    """Normalise one key value to the stored list form.

    A single key yields at most one value; a list key yields its entries,
    stripped, empties dropped, duplicates (case-insensitive, matching the
    column collation) dropped, first occurrence kept. An empty result means
    "no value". A list key accepts a bare string as a one-entry list.
    """
    if shape == "single":
        if not isinstance(value, str):
            raise MetadataKeyError(f"expected a string, got {type(value).__name__}")
        value = value.strip()
        return [value] if value else []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise MetadataKeyError(
            f"expected a string or a list of strings, got {type(value).__name__}"
        )
    out: list[str] = []
    seen: set[str] = set()
    for v in value:
        v = v.strip()
        if v and v.casefold() not in seen:
            seen.add(v.casefold())
            out.append(v)
    return out


def normalize_key_patch(conn: sqlite3.Connection, keys: Mapping[str, Any]) -> dict[str, list[str]]:
    """Validate a ``{key: value}`` patch against the registry.

    Returns ``{key: [values]}`` where an empty list means "remove this key's
    value". ``None``, ``""`` and ``[]`` all mean remove. Raises
    :class:`MetadataKeyError` naming every unregistered key, or the first value whose
    type does not fit its key's shape.
    """
    shapes = key_shapes(conn)
    unknown = sorted(k for k in keys if k not in shapes)
    if unknown:
        raise MetadataKeyError(
            f"unregistered key(s) {', '.join(unknown)} — register with "
            "`rag keys add <name>` (registered: " + ", ".join(sorted(shapes)) + ")"
        )
    out: dict[str, list[str]] = {}
    for name, value in keys.items():
        if value is None:
            out[name] = []
            continue
        try:
            out[name] = _coerce(value, shapes[name])
        except MetadataKeyError as e:
            raise MetadataKeyError(f"key {name!r}: {e}") from None
    return out


def _write_values(conn: sqlite3.Connection, doc_id: str, key: str, values: list[str]) -> None:
    """Replace ``doc_id``'s values for ``key``. Does not commit."""
    conn.execute("DELETE FROM doc_key_values WHERE doc_id = ? AND key = ?", (doc_id, key))
    conn.executemany(
        "INSERT INTO doc_key_values(doc_id, key, value, position) VALUES (?, ?, ?, ?)",
        [(doc_id, key, v, i) for i, v in enumerate(values)],
    )


def write_key_values(conn: sqlite3.Connection, doc_id: str, patch: Mapping[str, list[str]]) -> None:
    """Apply a normalised patch (see :func:`normalize_key_patch`). Does not commit."""
    for key, values in patch.items():
        _write_values(conn, doc_id, key, values)


def get_values_for_docs(
    conn: sqlite3.Connection,
    doc_ids: Iterable[str] | None,
    *,
    keys: Iterable[str] | None = None,
) -> dict[str, dict[str, KeyValue]]:
    """Return ``{doc_id: {key: value}}`` for the given documents (all if None).

    Each value takes its key's shape: a string for a single key, a list for a
    list key. Documents with no values are absent from the result.
    """
    sql = (
        "SELECT v.doc_id, v.key, v.value, k.shape FROM doc_key_values v "
        "JOIN metadata_keys k ON k.name = v.key"
    )
    clauses: list[str] = []
    params: list[object] = []
    if doc_ids is not None:
        ids = list(doc_ids)
        if not ids:
            return {}
        clauses.append(f"v.doc_id IN ({','.join('?' * len(ids))})")
        params.extend(ids)
    if keys is not None:
        names = list(keys)
        clauses.append(f"v.key IN ({','.join('?' * len(names))})")
        params.extend(names)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY v.doc_id, v.key, v.position"

    out: dict[str, dict[str, KeyValue]] = {}
    for row in conn.execute(sql, params):
        doc = out.setdefault(row["doc_id"], {})
        if row["shape"] == "list":
            existing = doc.setdefault(row["key"], [])
            assert isinstance(existing, list)
            existing.append(row["value"])
        else:
            doc[row["key"]] = row["value"]
    return out


def delete_doc_values(conn: sqlite3.Connection, doc_id: str) -> None:
    """Drop every key value of ``doc_id``. Does not commit."""
    conn.execute("DELETE FROM doc_key_values WHERE doc_id = ?", (doc_id,))


def key_filter_sql(
    keys: Mapping[str, str], *, doc_id_column: str
) -> tuple[list[str], list[object]]:
    """SQL clauses restricting ``doc_id_column`` to documents matching every key.

    A list key matches when any one of its values equals the filter value.
    Comparison is case-insensitive (the column's ``NOCASE`` collation). The
    clauses are meant to be AND-ed into the caller's WHERE.
    """
    clauses: list[str] = []
    params: list[object] = []
    for key, value in keys.items():
        clauses.append(
            f"{doc_id_column} IN (SELECT doc_id FROM doc_key_values WHERE key = ? AND value = ?)"
        )
        params.extend([key, value.strip()])
    return clauses, params


def _begin(conn: sqlite3.Connection) -> None:
    """Start an explicit transaction, committing any implicit one first.

    The module's writes mix DML with reads; an explicit BEGIN makes the
    multi-statement changes atomic whatever the connection's isolation mode.
    """
    if conn.in_transaction:
        conn.commit()
    # IMMEDIATE takes the write lock now, so the reads that follow see the
    # rows the writes are based on.
    conn.execute("BEGIN IMMEDIATE")
