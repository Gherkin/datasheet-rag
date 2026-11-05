"""Registered metadata fields (GH #120).

A *field* is a named, typed, filterable piece of document metadata. Users
register their own — an internal document number (``acme-id``), a vendor's
document number (``intel-id``), an internal part number (``acme-pn``) — and
the pipeline ships three of its own pre-registered: ``mpn``, ``manufacturer``
and ``doc_type``. Built-in and user fields share one code path; "built-in"
only means the field ships registered and cannot be removed.

A name must be registered before any document can hold a value for it. There
is no free-form tier.

Two tables back this:

* ``metadata_fields``  – the registry: name, description, shape (``single``
                         or ``list``), roles, and whether the field is built in.
* ``doc_field_values`` – one row per (document, field, value). A single-valued
                         field has at most one row per document; a list-valued
                         field keeps its order in ``position``.

Values compare case-insensitively (``COLLATE NOCASE``): part and document
numbers are identifiers, and ``stm32h743`` should find ``STM32H743``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

#: A field holds one value per document, or an ordered list of them.
FieldShape = Literal["single", "list"]
FIELD_SHAPES: tuple[FieldShape, ...] = get_args(FieldShape)

#: What a field is used for, beyond being filterable. ``part`` marks a field
#: whose values are part numbers (BOM import matches on these — GH #125).
FieldRole = Literal["part"]
FIELD_ROLES: tuple[FieldRole, ...] = get_args(FieldRole)

#: A field's value as stored and returned: a string, or a list of strings.
FieldValue = str | list[str]

#: Field names: lower case, digits, ``-`` and ``_``, starting with a letter or
#: digit. Kept narrow so a name is safe on a command line, in a TOML table and
#: in a query string without quoting.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

#: Names a field may not take: the ``doc_metadata`` columns (including the
#: pipeline's bookkeeping columns), the fields planned to become fixed
#: (``version``, GH #124), and the built-in column names of ``rag list``,
#: which would shadow the field. A field with one of these names would be
#: ambiguous in every listing and filter.
RESERVED_NAMES = frozenset(
    {
        *("doc_id", "title", "project_id", "projects", "group_name", "subsystem", "tags"),
        "updated_at",
        *("title_source", "needs_reembed", "pdf_meta_title", "running_header"),
        *("fields", "version"),
        *("project", "group", "updated", "stale", "chunks", "pages", "ingested"),
        # `rag list --columns` aliases (model attribute spellings).
        *("id", "doc_title", "chunk_count", "page_count", "ingested_at"),
    }
)


class FieldSpec(BaseModel):
    """One row of the field registry."""

    name: str
    description: str = ""
    shape: FieldShape = "single"
    roles: list[FieldRole] = Field(default_factory=list)
    builtin: bool = False


#: The fields every store ships with. Their values used to be fixed columns on
#: ``doc_metadata`` (moved here by the v3 migration in ``store.schema``).
BUILTIN_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="mpn",
        description="Manufacturer part number(s) the document covers, e.g. STM32H743VIT6.",
        shape="list",
        roles=["part"],
        builtin=True,
    ),
    FieldSpec(
        name="manufacturer",
        description="Manufacturer name, e.g. STMicroelectronics.",
        shape="single",
        builtin=True,
    ),
    FieldSpec(
        name="doc_type",
        description="Kind of document: datasheet, reference-manual, errata, app-note, …",
        shape="single",
        builtin=True,
    ),
)


class MetadataFieldError(ValueError):
    """A field operation that cannot be applied as asked (unknown field, bad
    shape, name clash). A ``ValueError`` so the HTTP layer maps it to 400."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def ensure_field_tables(conn: sqlite3.Connection) -> None:
    """Create the registry and value tables and seed the built-in fields.

    Idempotent, and safe on every open: ``IF NOT EXISTS`` for the DDL and
    ``INSERT OR IGNORE`` for the seed rows. Does not commit — the caller does,
    so this can run inside the v3 migration's transaction.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata_fields (
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
        CREATE TABLE IF NOT EXISTS doc_field_values (
            doc_id   TEXT NOT NULL,
            field    TEXT NOT NULL REFERENCES metadata_fields(name),
            value    TEXT NOT NULL COLLATE NOCASE,
            position INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (doc_id, field, value)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_doc_field_values_fv ON doc_field_values(field, value)"
    )
    # Insert only the missing seeds: even an INSERT OR IGNORE that inserts
    # nothing takes the write lock, and this runs on every open.
    present = {r[0] for r in conn.execute("SELECT name FROM metadata_fields")}
    for spec in BUILTIN_FIELDS:
        if spec.name in present:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO metadata_fields(name, description, shape, roles, builtin) "
            "VALUES (?, ?, ?, ?, 1)",
            (spec.name, spec.description, spec.shape, json.dumps(spec.roles)),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _row_to_spec(row: sqlite3.Row) -> FieldSpec:
    return FieldSpec(
        name=row["name"],
        description=row["description"],
        shape=row["shape"],
        roles=json.loads(row["roles"] or "[]"),
        builtin=bool(row["builtin"]),
    )


def list_fields(conn: sqlite3.Connection) -> list[FieldSpec]:
    """Every registered field, built-ins first, then by name."""
    rows = conn.execute("SELECT * FROM metadata_fields ORDER BY builtin DESC, name").fetchall()
    return [_row_to_spec(r) for r in rows]


def get_field(conn: sqlite3.Connection, name: str) -> FieldSpec | None:
    row = conn.execute("SELECT * FROM metadata_fields WHERE name = ?", (name,)).fetchone()
    return _row_to_spec(row) if row is not None else None


def field_shapes(conn: sqlite3.Connection) -> dict[str, FieldShape]:
    """Map every registered field name to its shape."""
    return {r["name"]: r["shape"] for r in conn.execute("SELECT name, shape FROM metadata_fields")}


def shapes_of(specs: Iterable[FieldSpec]) -> dict[str, FieldShape]:
    """:func:`field_shapes` for a registry already read (``list_fields`` or an API reply)."""
    return {f.name: f.shape for f in specs}


def validate_field_name(name: str) -> None:
    """Raise :class:`MetadataFieldError` unless ``name`` may be registered."""
    if not _NAME_RE.match(name):
        raise MetadataFieldError(
            f"invalid field name {name!r}: use lower-case letters, digits, '-' and '_', "
            "starting with a letter or digit"
        )
    if name in RESERVED_NAMES:
        raise MetadataFieldError(f"{name!r} is a reserved metadata name and cannot be a field")


def register_field(
    conn: sqlite3.Connection,
    name: str,
    *,
    shape: FieldShape = "single",
    description: str = "",
    roles: Iterable[str] = (),
) -> FieldSpec:
    """Register a new field and return its spec."""
    validate_field_name(name)
    if shape not in FIELD_SHAPES:
        raise MetadataFieldError(f"unknown shape {shape!r} (expected one of {FIELD_SHAPES})")
    role_list = list(dict.fromkeys(roles))
    bad_roles = [r for r in role_list if r not in FIELD_ROLES]
    if bad_roles:
        raise MetadataFieldError(f"unknown role(s) {bad_roles} (expected some of {FIELD_ROLES})")
    _begin(conn)
    try:
        if get_field(conn, name) is not None:
            raise MetadataFieldError(f"field {name!r} is already registered")
        conn.execute(
            "INSERT INTO metadata_fields(name, description, shape, roles, builtin) "
            "VALUES (?, ?, ?, ?, 0)",
            (name, description, shape, json.dumps(role_list)),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    spec = get_field(conn, name)
    assert spec is not None
    return spec


def remove_field(conn: sqlite3.Connection, name: str, *, clear: bool = False) -> int:
    """Unregister ``name``. Returns the number of documents that held a value.

    Refuses while any document has a value for the field, unless ``clear`` is
    set: then those values are deleted with it. Built-in fields cannot be
    removed.
    """
    _begin(conn)
    try:
        spec = get_field(conn, name)
        if spec is None:
            raise MetadataFieldError(f"field {name!r} is not registered")
        if spec.builtin:
            raise MetadataFieldError(f"field {name!r} is built in and cannot be removed")
        held = int(
            conn.execute(
                "SELECT COUNT(DISTINCT doc_id) FROM doc_field_values WHERE field = ?", (name,)
            ).fetchone()[0]
        )
        if held and not clear:
            raise MetadataFieldError(
                f"field {name!r} has values on {held} document(s); pass clear to "
                "delete them with the field"
            )
        conn.execute("DELETE FROM doc_field_values WHERE field = ?", (name,))
        conn.execute("DELETE FROM metadata_fields WHERE name = ?", (name,))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return held


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def display_value(value: FieldValue | None) -> str | None:
    """A field's value as one line of text: a list joined with ``", "``."""
    if isinstance(value, list):
        return ", ".join(value) or None
    return value or None


def _coerce(value: Any, shape: FieldShape) -> list[str]:
    """Normalise one field value to the stored list form.

    A single field yields at most one value; a list field yields its entries,
    stripped, empties dropped, duplicates (case-insensitive, matching the
    column collation) dropped, first occurrence kept. An empty result means
    "no value". A list field accepts a bare string as a one-entry list.
    """
    if shape == "single":
        if not isinstance(value, str):
            raise MetadataFieldError(f"expected a string, got {type(value).__name__}")
        value = value.strip()
        return [value] if value else []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise MetadataFieldError(
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


def normalize_field_patch(
    conn: sqlite3.Connection, fields: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Validate a ``{field: value}`` patch against the registry.

    Returns ``{field: [values]}`` where an empty list means "remove this
    field's value". ``None``, ``""`` and ``[]`` all mean remove. Raises
    :class:`MetadataFieldError` naming every unregistered field, or the first
    value whose type does not fit its field's shape.
    """
    return normalize_field_patch_for(field_shapes(conn), fields)


def normalize_field_patch_for(
    shapes: Mapping[str, FieldShape], fields: Mapping[str, Any]
) -> dict[str, list[str]]:
    """:func:`normalize_field_patch` against a registry already read (``{name: shape}``),
    so a client can check a patch without a store connection."""
    unknown = sorted(f for f in fields if f not in shapes)
    if unknown:
        raise MetadataFieldError(
            f"unregistered field(s) {', '.join(unknown)} — register with "
            "`rag fields add <name>` (registered: " + ", ".join(sorted(shapes)) + ")"
        )
    out: dict[str, list[str]] = {}
    for name, value in fields.items():
        if value is None:
            out[name] = []
            continue
        try:
            out[name] = _coerce(value, shapes[name])
        except MetadataFieldError as e:
            raise MetadataFieldError(f"field {name!r}: {e}") from None
    return out


def _write_values(conn: sqlite3.Connection, doc_id: str, field: str, values: list[str]) -> None:
    """Replace ``doc_id``'s values for ``field``. Does not commit."""
    conn.execute("DELETE FROM doc_field_values WHERE doc_id = ? AND field = ?", (doc_id, field))
    conn.executemany(
        "INSERT INTO doc_field_values(doc_id, field, value, position) VALUES (?, ?, ?, ?)",
        [(doc_id, field, v, i) for i, v in enumerate(values)],
    )


def write_field_values(
    conn: sqlite3.Connection, doc_id: str, patch: Mapping[str, list[str]]
) -> None:
    """Apply a normalised patch (see :func:`normalize_field_patch`). Does not commit."""
    for field, values in patch.items():
        _write_values(conn, doc_id, field, values)


def get_values_for_docs(
    conn: sqlite3.Connection,
    doc_ids: Iterable[str] | None,
    *,
    fields: Iterable[str] | None = None,
) -> dict[str, dict[str, FieldValue]]:
    """Return ``{doc_id: {field: value}}`` for the given documents (all if None).

    Each value takes its field's shape: a string for a single field, a list
    for a list field. Documents with no values are absent from the result.
    """
    sql = (
        "SELECT v.doc_id, v.field, v.value, f.shape FROM doc_field_values v "
        "JOIN metadata_fields f ON f.name = v.field"
    )
    clauses: list[str] = []
    params: list[object] = []
    if doc_ids is not None:
        ids = list(doc_ids)
        if not ids:
            return {}
        clauses.append(f"v.doc_id IN ({','.join('?' * len(ids))})")
        params.extend(ids)
    if fields is not None:
        names = list(fields)
        clauses.append(f"v.field IN ({','.join('?' * len(names))})")
        params.extend(names)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY v.doc_id, v.field, v.position"

    out: dict[str, dict[str, FieldValue]] = {}
    for row in conn.execute(sql, params):
        doc = out.setdefault(row["doc_id"], {})
        if row["shape"] == "list":
            existing = doc.setdefault(row["field"], [])
            assert isinstance(existing, list)
            existing.append(row["value"])
        else:
            doc[row["field"]] = row["value"]
    return out


def delete_doc_values(conn: sqlite3.Connection, doc_id: str) -> None:
    """Drop every field value of ``doc_id``. Does not commit."""
    conn.execute("DELETE FROM doc_field_values WHERE doc_id = ?", (doc_id,))


def check_filter_fields(registered: Iterable[str], fields: Iterable[str]) -> None:
    """Raise :class:`MetadataFieldError` if a filter names an unregistered field.

    A filter on a misspelt field would otherwise match nothing and look like
    "the store has no such documents".
    """
    names = set(registered)
    unknown = sorted(f for f in fields if f not in names)
    if unknown:
        raise MetadataFieldError(
            f"cannot filter on unregistered field(s) {', '.join(unknown)} "
            f"(registered: {', '.join(sorted(names))})"
        )


def field_filter_sql(
    fields: Mapping[str, str], *, doc_id_column: str
) -> tuple[list[str], list[object]]:
    """SQL clauses restricting ``doc_id_column`` to documents matching every field.

    A list field matches when any one of its values equals the filter value.
    Comparison is case-insensitive (the column's ``NOCASE`` collation). The
    clauses are meant to be AND-ed into the caller's WHERE.
    """
    clauses: list[str] = []
    params: list[object] = []
    for field, value in fields.items():
        clauses.append(
            f"{doc_id_column} IN "
            "(SELECT doc_id FROM doc_field_values WHERE field = ? AND value = ?)"
        )
        params.extend([field, value.strip()])
    return clauses, params


def _begin(conn: sqlite3.Connection) -> None:
    """Start an explicit transaction, committing any implicit one first.

    IMMEDIATE takes the write lock now, so the reads that follow see the rows
    the writes are based on.
    """
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
