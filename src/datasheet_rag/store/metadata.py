"""Document-level metadata sidecar.

Sits next to ``chunks`` and stores per-document fields (project, group,
tags, and the values of registered fields such as MPN and manufacturer —
see ``store.fields``) that are independent of any single chunk. It also
holds the pipeline's own bookkeeping for each document: title provenance,
the ingest-time title hints and the re-embed marker. Those are typed
columns the pipeline writes; a metadata patch cannot set them.

Two properties of this sidecar matter:

* It can be written **before** ingestion completes — useful for the CLI
  to register a PDF up front so chunks land with the right
  ``project_id`` once they're produced.
* :func:`apply_metadata_to_chunks` back-fills ``chunks.project_id`` and
  ``chunks.group_name`` from the sidecar after chunks have been
  inserted, so existing rows pick up changes without a full re-ingest.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field

from datasheet_rag.store.fields import (
    FieldShape,
    check_filter_fields,
    delete_doc_values,
    field_filter_sql,
    field_shapes,
    get_values_for_docs,
    normalize_field_patch_for,
    write_field_values,
)


class DocMetadata(BaseModel):
    """Per-document metadata row (mirrors the ``doc_metadata`` table)."""

    doc_id: str
    project_id: str | None = None
    group_name: str | None = None
    subsystem: str | None = None
    tags: list[str] = Field(default_factory=list)
    #: Values of registered fields (``store.fields``), including the built-in
    #: ``mpn`` (a list), ``manufacturer`` and ``doc_type``. A field with no
    #: value on this document is absent.
    fields: dict[str, str | list[str]] = Field(default_factory=dict)
    #: Pipeline bookkeeping, read-only through the metadata API: where the
    #: title came from (see below), whether `rag repair` marked the document
    #: for re-embedding, and the two title hints captured at ingest.
    title_source: str = "auto"
    needs_reembed: bool = False
    pdf_meta_title: str | None = None
    running_header: str | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# Title provenance
#
# Where a document's title came from decides whether the next write may
# replace it. Ranked lowest to highest:
#
#   auto      derived by the layout parser at ingest, from the first
#             title-ish text on page 1 — frequently "Contents", a running
#             header, or a marketing strapline
#   inferred  backfilled by `rag repair titles` from a first-page LLM read
#   manual    set by a human via `rag metadata <doc> --title`
#
# A write only lands if its source ranks at least as high as the stored
# one, so re-ingesting a document cannot demote a title a human or the LLM
# chose. Enforced in `store.sqlite.set_doc_title` / `insert_chunks`, not in
# the callers, so the CLI, MCP and server ingest routes all inherit it.
#
# Provenance lives here rather than on `chunks` because it is one fact per
# document: a column alongside `chunks.doc_title` would be duplicated
# across every row and could go inconsistent between them.
# ---------------------------------------------------------------------------

#: Where a document's title came from, weakest first. The order *is* the
#: precedence: a later source overwrites an earlier one, never the reverse.
TitleSource = Literal["auto", "inferred", "manual"]

TITLE_SOURCES: tuple[TitleSource, ...] = get_args(TitleSource)

_TITLE_RANK: dict[str, int] = {source: rank for rank, source in enumerate(TITLE_SOURCES)}


def title_rank(source: str) -> int:
    """Return the precedence rank of *source* (unknown values rank lowest)."""
    return _TITLE_RANK.get(source, 0)


def validate_title_source(source: str) -> None:
    """Raise ``ValueError`` unless *source* is a known provenance.

    Callers that are about to *write* a title must check first: an unknown
    source ranks 0, the same as ``"auto"``, so it slips past the precedence
    guard and would otherwise be caught only after the title itself had
    already been overwritten.
    """
    if source not in _TITLE_RANK:
        raise ValueError(f"unknown title source {source!r} (expected one of {TITLE_SOURCES})")


def get_title_source(conn: sqlite3.Connection, doc_id: str) -> str:
    """Return the recorded provenance of ``doc_id``'s title (``auto`` if none)."""
    row = conn.execute(
        "SELECT title_source FROM doc_metadata WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return (row["title_source"] if row is not None else None) or "auto"


def _set_bookkeeping(conn: sqlite3.Connection, doc_id: str, **columns: object) -> None:
    """Upsert pipeline bookkeeping columns for ``doc_id``. Does not commit.

    ``columns`` names come from this module only, never from user input.
    """
    names = list(columns)
    conn.execute(
        f"INSERT INTO doc_metadata (doc_id, {', '.join(names)}, updated_at) "
        f"VALUES (?, {', '.join('?' * len(names))}, CURRENT_TIMESTAMP) "
        "ON CONFLICT(doc_id) DO UPDATE SET "
        + ", ".join(f"{n} = excluded.{n}" for n in names)
        + ", updated_at = CURRENT_TIMESTAMP",
        (doc_id, *columns.values()),
    )


def set_title_source(conn: sqlite3.Connection, doc_id: str, source: str) -> None:
    """Record ``doc_id``'s title provenance."""
    validate_title_source(source)
    _set_bookkeeping(conn, doc_id, title_source=source)
    conn.commit()


def set_title_hints(
    conn: sqlite3.Connection,
    doc_id: str,
    *,
    pdf_meta_title: str | None,
    running_header: str | None,
) -> None:
    """Record the ingest-time title hints, replacing both (``None`` clears one).

    With neither hint this only clears an existing row's hints; it does not
    create a sidecar row for a document that has none.
    """
    if not pdf_meta_title and not running_header:
        conn.execute(
            "UPDATE doc_metadata SET pdf_meta_title = NULL, running_header = NULL, "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE doc_id = ? AND (pdf_meta_title IS NOT NULL OR running_header IS NOT NULL)",
            (doc_id,),
        )
    else:
        _set_bookkeeping(
            conn,
            doc_id,
            pdf_meta_title=pdf_meta_title or None,
            running_header=running_header or None,
        )
    conn.commit()


def set_needs_reembed(conn: sqlite3.Connection, doc_id: str, value: bool) -> None:
    """Mark (or unmark) ``doc_id`` as needing its chunks re-embedded."""
    _set_bookkeeping(conn, doc_id, needs_reembed=1 if value else None)
    conn.commit()


def _row_to_metadata(
    row: sqlite3.Row, fields: dict[str, str | list[str]] | None = None
) -> DocMetadata:
    tags_raw = row["tags"]
    tags = json.loads(tags_raw) if tags_raw else []
    return DocMetadata(
        doc_id=row["doc_id"],
        project_id=row["project_id"],
        group_name=row["group_name"],
        subsystem=row["subsystem"],
        tags=tags,
        fields=dict(fields or {}),
        title_source=row["title_source"] or "auto",
        needs_reembed=bool(row["needs_reembed"]),
        pdf_meta_title=row["pdf_meta_title"],
        running_header=row["running_header"],
        updated_at=row["updated_at"],
    )


def get_metadata(conn: sqlite3.Connection, doc_id: str) -> DocMetadata | None:
    """Return the sidecar row for ``doc_id`` or ``None``."""
    row = conn.execute(
        "SELECT * FROM doc_metadata WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    if row is None:
        return None
    values = get_values_for_docs(conn, [doc_id])
    return _row_to_metadata(row, values.get(doc_id))


def check_patch(
    conn: sqlite3.Connection, *, fields: Mapping[str, Any] | None = None
) -> dict[str, list[str]]:
    """Validate the ``fields`` of a metadata write.

    Returns the normalised field patch (see
    ``store.fields.normalize_field_patch``). Raises :class:`MetadataFieldError`
    for an unregistered field or a value that does not fit its field's shape.
    Writes nothing, so a caller can check a patch before work it would be a
    waste to throw away (ingest does).
    """
    return check_patch_for(field_shapes(conn), fields=fields)


def check_patch_for(
    shapes: Mapping[str, FieldShape], *, fields: Mapping[str, Any] | None = None
) -> dict[str, list[str]]:
    """:func:`check_patch` against a registry already read (``{name: shape}``),
    so a client can reject a bad patch before parsing a PDF."""
    return normalize_field_patch_for(shapes, fields or {})


def set_metadata(
    conn: sqlite3.Connection,
    doc_id: str,
    *,
    project_id: str | None = None,
    group_name: str | None = None,
    subsystem: str | None = None,
    tags: list[str] | None = None,
    fields: Mapping[str, Any] | None = None,
) -> DocMetadata:
    """Upsert metadata for ``doc_id`` with **partial** merge semantics.

    A parameter left as ``None`` (its default) means "leave the existing
    value alone". To clear a string field, pass an empty string; to
    clear ``tags`` pass ``[]``.

    Merge rules
    -----------
    * Scalar fields: overwrite with the new value when not ``None``.
    * ``tags``: replace wholesale (not merged).
    * ``fields``: per field — a field named in the patch has its value
      replaced; ``None``, ``""`` or ``[]`` removes it; fields not named are
      left alone. Every field must be registered (``store.fields``) and its
      value must fit the field's shape, or :class:`MetadataFieldError` is
      raised before anything is written.

    Returns the resulting :class:`DocMetadata` after the upsert.
    """
    field_patch = check_patch(conn, fields=fields)

    existing = get_metadata(conn, doc_id)
    if existing is None:
        existing = DocMetadata(doc_id=doc_id)
    merged = existing.model_copy(
        update={
            "project_id": project_id if project_id is not None else existing.project_id,
            "group_name": group_name if group_name is not None else existing.group_name,
            "subsystem": subsystem if subsystem is not None else existing.subsystem,
            "tags": list(tags) if tags is not None else existing.tags,
        }
    )

    conn.execute(
        """
        INSERT INTO doc_metadata (
            doc_id, project_id, group_name, subsystem, tags, updated_at
        ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(doc_id) DO UPDATE SET
            project_id   = excluded.project_id,
            group_name   = excluded.group_name,
            subsystem    = excluded.subsystem,
            tags         = excluded.tags,
            updated_at   = CURRENT_TIMESTAMP
        """,
        (
            merged.doc_id,
            merged.project_id,
            merged.group_name,
            merged.subsystem,
            json.dumps(merged.tags),
        ),
    )
    write_field_values(conn, doc_id, field_patch)
    conn.commit()

    # Re-read so updated_at reflects the DB clock.
    refreshed = get_metadata(conn, doc_id)
    assert refreshed is not None  # we just wrote it
    return refreshed


def delete_metadata(conn: sqlite3.Connection, doc_id: str) -> int:
    """Delete the sidecar row for ``doc_id``, if any, and its field values.

    Not covered by the ``chunks`` cascade (`doc_metadata` is intentionally
    not FK'd to `chunks` — see module docstring), so callers that fully
    delete a document must call this alongside :func:`datasheet_rag.store.sqlite.delete_doc`.

    Returns the number of sidecar rows deleted (0 or 1).
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM doc_metadata WHERE doc_id = ?", (doc_id,))
    deleted = cur.rowcount
    delete_doc_values(conn, doc_id)
    conn.commit()
    return int(deleted)


def list_docs(
    conn: sqlite3.Connection,
    *,
    project_id: str | None = None,
    group_name: str | None = None,
    fields: dict[str, str] | None = None,
) -> list[DocMetadata]:
    """Return all sidecar rows matching the optional filters.

    ``fields`` filters on registered fields, AND-ed: ``{"mpn": "INA226"}``
    keeps documents whose mpn list contains INA226 (case-insensitive). Naming
    an unregistered field raises :class:`MetadataFieldError` rather than
    silently matching nothing.
    """
    clauses: list[str] = []
    params: list[object] = []
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)
    if group_name is not None:
        clauses.append("group_name = ?")
        params.append(group_name)
    if fields:
        check_filter_fields(field_shapes(conn), fields)
        field_clauses, field_params = field_filter_sql(fields, doc_id_column="doc_id")
        clauses.extend(field_clauses)
        params.extend(field_params)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM doc_metadata{where} ORDER BY doc_id",
        params,
    ).fetchall()
    # Unfiltered: read every value rather than binding one parameter per row.
    values = get_values_for_docs(conn, [r["doc_id"] for r in rows] if clauses else None)
    return [_row_to_metadata(r, values.get(r["doc_id"])) for r in rows]


def apply_metadata_to_chunks(conn: sqlite3.Connection, doc_id: str) -> int:
    """Back-fill ``chunks.project_id`` / ``chunks.group_name`` from sidecar.

    No-ops (returns 0) if there is no sidecar row, or if both
    ``project_id`` and ``group_name`` are unset on the sidecar. The
    chunks' other columns are not touched.

    Returns the number of chunk rows updated.
    """
    md = get_metadata(conn, doc_id)
    if md is None:
        return 0
    if md.project_id is None and md.group_name is None:
        return 0

    cur = conn.cursor()
    cur.execute(
        """
        UPDATE chunks
           SET project_id = ?,
               group_name = ?
         WHERE doc_id = ?
        """,
        (md.project_id, md.group_name, doc_id),
    )
    updated = cur.rowcount
    conn.commit()
    return int(updated)
