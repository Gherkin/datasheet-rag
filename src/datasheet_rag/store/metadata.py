"""Document-level metadata sidecar.

Sits next to ``chunks`` and stores per-document fields (projects, tags, and
the values of registered fields such as MPN, manufacturer and the
project-scoped subsystem — see ``store.fields``) that are independent of any
single chunk. Project membership lives in its own table (``store.projects``);
this module reads and writes it with the rest of the document's metadata. It
also holds the pipeline's own bookkeeping for each document: title
provenance, the ingest-time title hints and the re-embed marker. Those are
typed columns the pipeline writes; a metadata patch cannot set them.

The sidecar can be written **before** ingestion completes — useful for the
CLI to register a PDF up front so its chunks are in the right projects once
they're produced. Chunks carry none of it: every read scopes and filters
through these tables.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple, get_args

from pydantic import BaseModel, Field

from datasheet_rag.store.fields import (
    FieldScope,
    FieldShape,
    MetadataFieldError,
    check_filter_fields,
    check_patch_project,
    delete_doc_project_values,
    delete_doc_values,
    field_filter_sql,
    field_scopes,
    field_shapes,
    get_project_values_for_docs,
    get_values_for_docs,
    normalize_field_patch_for,
    write_field_values,
)
from datasheet_rag.store.projects import (
    change_projects,
    check_project_change,
    delete_doc_projects,
    get_projects_for_docs,
    project_scope_sql,
)


class DocMetadata(BaseModel):
    """Per-document metadata row (mirrors the ``doc_metadata`` table)."""

    doc_id: str
    #: Every project the document belongs to, sorted (``doc_projects``).
    projects: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    #: Values of document-scoped fields (``store.fields``), including the
    #: built-in ``mpn`` (a list), ``manufacturer`` and ``doc_type``. A field
    #: with no value on this document is absent.
    fields: dict[str, str | list[str]] = Field(default_factory=dict)
    #: Values of project-scoped fields such as ``subsystem``, per project:
    #: ``{"netdaq": {"subsystem": "power"}}``. A project with no values is
    #: absent.
    project_fields: dict[str, dict[str, str | list[str]]] = Field(default_factory=dict)
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
    row: sqlite3.Row,
    fields: dict[str, str | list[str]] | None = None,
    projects: list[str] | None = None,
    project_fields: dict[str, dict[str, str | list[str]]] | None = None,
) -> DocMetadata:
    tags_raw = row["tags"]
    tags = json.loads(tags_raw) if tags_raw else []
    return DocMetadata(
        doc_id=row["doc_id"],
        projects=list(projects or []),
        tags=tags,
        fields=dict(fields or {}),
        project_fields=dict(project_fields or {}),
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
    projects = get_projects_for_docs(conn, [doc_id])
    project_values = get_project_values_for_docs(conn, [doc_id])
    return _row_to_metadata(
        row, values.get(doc_id), projects.get(doc_id), project_values.get(doc_id)
    )


class CheckedWrite(NamedTuple):
    """A metadata write that :func:`check_write` accepted, normalised."""

    #: ``{field: [values]}``; an empty list removes the field's value.
    fields: dict[str, list[str]]
    scopes: dict[str, FieldScope]
    #: Where the project-scoped fields go, stripped; ``None`` when not named.
    project: str | None
    add: list[str]
    remove: list[str]


def check_write(
    conn: sqlite3.Connection,
    doc_id: str,
    *,
    add_projects: list[str] | None = None,
    remove_projects: list[str] | None = None,
    fields: Mapping[str, Any] | None = None,
    project: str | None = None,
) -> CheckedWrite:
    """Validate a metadata write (the arguments of :func:`set_metadata`).

    Raises :class:`MetadataFieldError` for an unregistered field, a value
    that does not fit its field's shape, a project-scoped field with no
    ``project``, or a project-scoped value set in a project the document is
    not in after this write. Clearing a project-scoped value needs no
    membership: the document holds no value outside its projects. Raises
    ``ValueError`` for a project both added and removed. Writes nothing, so
    ingest checks a patch before work it would be a waste to throw away.
    """
    project = project.strip() if project is not None and project.strip() else None
    scopes = field_scopes(conn)
    field_patch = check_patch_for(field_shapes(conn), scopes, fields=fields, project=project)
    add, remove = check_project_change(add_projects, remove_projects)
    sets_project_value = any(
        values and scopes.get(name) == "project" for name, values in field_patch.items()
    )
    if project is not None and sets_project_value:
        current = get_projects_for_docs(conn, [doc_id]).get(doc_id, [])
        if project in remove or (project not in current and project not in add):
            raise MetadataFieldError(
                f"{doc_id} is not in project {project!r}; add it to the project "
                "first (--add-project) to set its project fields there"
            )
    return CheckedWrite(field_patch, scopes, project, add, remove)


def check_patch_for(
    shapes: Mapping[str, FieldShape],
    scopes: Mapping[str, FieldScope],
    *,
    fields: Mapping[str, Any] | None = None,
    project: str | None = None,
) -> dict[str, list[str]]:
    """Check ``fields`` against a registry already read, so a client can
    reject a bad patch before parsing a PDF. Returns the normalised patch
    (see ``store.fields.normalize_field_patch``)."""
    patch = normalize_field_patch_for(shapes, fields or {})
    check_patch_project(scopes, patch, project)
    return patch


def set_metadata(
    conn: sqlite3.Connection,
    doc_id: str,
    *,
    add_projects: list[str] | None = None,
    remove_projects: list[str] | None = None,
    tags: list[str] | None = None,
    fields: Mapping[str, Any] | None = None,
    project: str | None = None,
) -> DocMetadata:
    """Upsert metadata for ``doc_id`` with **partial** merge semantics.

    A parameter left as ``None`` (its default) means "leave the existing
    value alone". To clear ``tags`` pass ``[]``.

    Merge rules
    -----------
    * ``add_projects`` / ``remove_projects``: the document joins / leaves
      each named project; its other projects are left alone. A project in
      both lists, or an empty name, raises ``ValueError`` before anything
      is written. Leaving a project deletes the document's project-scoped
      values in it.
    * ``tags``: replace wholesale (not merged).
    * ``fields``: per field — a field named in the patch has its value
      replaced; ``None``, ``""`` or ``[]`` removes it; fields not named are
      left alone. Every field must be registered (``store.fields``) and its
      value must fit the field's shape, or :class:`MetadataFieldError` is
      raised before anything is written.
    * ``project``: where the project-scoped fields in ``fields`` are
      written. Required when ``fields`` names one. To set a value there the
      document must be in that project after this write (an
      ``add_projects`` in the same call counts); clearing one needs no
      membership. Otherwise :class:`MetadataFieldError`, before anything is
      written (see :func:`check_write`).

    Returns the resulting :class:`DocMetadata` after the upsert.
    """
    checked = check_write(
        conn,
        doc_id,
        add_projects=add_projects,
        remove_projects=remove_projects,
        fields=fields,
        project=project,
    )

    existing = get_metadata(conn, doc_id)
    merged_tags = list(tags) if tags is not None else (existing.tags if existing else [])

    conn.execute(
        """
        INSERT INTO doc_metadata (doc_id, tags, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(doc_id) DO UPDATE SET
            tags         = excluded.tags,
            updated_at   = CURRENT_TIMESTAMP
        """,
        (doc_id, json.dumps(merged_tags)),
    )
    change_projects(conn, doc_id, add=checked.add, remove=checked.remove)
    delete_doc_project_values(conn, doc_id, checked.remove)
    write_field_values(conn, doc_id, checked.fields, scopes=checked.scopes, project=checked.project)
    conn.commit()

    # Re-read so updated_at reflects the DB clock.
    refreshed = get_metadata(conn, doc_id)
    assert refreshed is not None  # we just wrote it
    return refreshed


def join_projects(conn: sqlite3.Connection, doc_id: str, projects: list[str]) -> None:
    """Add ``doc_id`` to ``projects`` and make sure it has a sidecar row. Does not commit.

    Ingest calls this inside the chunk-insert transaction, so the projects
    commit with the chunks. The sidecar row is what document listings read;
    without it a document would be searchable in a project but not listed
    in it. ``projects`` are normalised already (``MetadataPatch``).
    """
    if not projects:
        return
    conn.execute("INSERT OR IGNORE INTO doc_metadata (doc_id) VALUES (?)", (doc_id,))
    change_projects(conn, doc_id, add=projects)


def delete_metadata(conn: sqlite3.Connection, doc_id: str) -> int:
    """Delete the sidecar row for ``doc_id``, if any, its field values and
    its project memberships.

    Not covered by the ``chunks`` cascade (`doc_metadata` is intentionally
    not FK'd to `chunks` — see module docstring), so callers that fully
    delete a document must call this alongside :func:`datasheet_rag.store.sqlite.delete_doc`.

    Returns the number of sidecar rows deleted (0 or 1).
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM doc_metadata WHERE doc_id = ?", (doc_id,))
    deleted = cur.rowcount
    delete_doc_values(conn, doc_id)
    delete_doc_projects(conn, doc_id)
    conn.commit()
    return int(deleted)


def list_docs(
    conn: sqlite3.Connection,
    *,
    project_id: str | None = None,
    fields: dict[str, str] | None = None,
) -> list[DocMetadata]:
    """Return all sidecar rows matching the optional filters.

    ``project_id`` keeps documents that belong to that project (among any
    others). ``fields`` filters on registered fields, AND-ed: ``{"mpn": "INA226"}``
    keeps documents whose mpn list contains INA226 (case-insensitive). A
    project-scoped field matches in ``project_id`` when one is given, else in
    any project. Naming an unregistered field raises :class:`MetadataFieldError`
    rather than silently matching nothing.
    """
    clauses: list[str] = []
    params: list[object] = []
    if project_id is not None:
        clause, clause_params = project_scope_sql(project_id, doc_id_column="doc_id")
        clauses.append(clause)
        params.extend(clause_params)
    if fields:
        check_filter_fields(field_shapes(conn), fields)
        field_clauses, field_params = field_filter_sql(
            fields, doc_id_column="doc_id", project_id=project_id
        )
        clauses.extend(field_clauses)
        params.extend(field_params)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM doc_metadata{where} ORDER BY doc_id",
        params,
    ).fetchall()
    # Unfiltered: read every value rather than binding one parameter per row.
    wanted = [r["doc_id"] for r in rows] if clauses else None
    values = get_values_for_docs(conn, wanted)
    projects = get_projects_for_docs(conn, wanted)
    project_values = get_project_values_for_docs(conn, wanted)
    return [
        _row_to_metadata(
            r,
            values.get(r["doc_id"]),
            projects.get(r["doc_id"]),
            project_values.get(r["doc_id"]),
        )
        for r in rows
    ]
