"""Document-level metadata sidecar.

Sits next to ``chunks`` and stores per-document fields (project, MPN,
manufacturer, …) that are independent of any single chunk. Two
properties of this sidecar matter:

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
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field


class DocMetadata(BaseModel):
    """Per-document metadata row (mirrors the ``doc_metadata`` table)."""

    doc_id: str
    project_id: str | None = None
    group_name: str | None = None
    mpn: str | None = None
    manufacturer: str | None = None
    subsystem: str | None = None
    doc_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
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


def title_source_of(attributes: dict[str, Any]) -> str:
    """Read title provenance out of a sidecar ``attributes`` dict.

    Defaults to ``"auto"`` when nothing is recorded — an unmarked title is
    one the parser produced, which is exactly the weakest case.

    Older stores recorded provenance as a boolean ``title_inferred: true``
    instead. That is read as ``"inferred"`` here, and rewritten into
    ``title_source`` the next time the document's title is set, so no
    separate migration step is needed.

    Split out from :func:`get_title_source` so callers holding metadata
    they already fetched — notably the CLI talking to a remote backend, which
    has no connection to query — can classify it without a second round trip.
    """
    source = attributes.get("title_source")
    if isinstance(source, str) and source in _TITLE_RANK:
        return source

    if attributes.get("title_inferred"):
        return "inferred"
    return "auto"


def get_title_source(conn: sqlite3.Connection, doc_id: str) -> str:
    """Return the recorded provenance of ``doc_id``'s title."""
    md = get_metadata(conn, doc_id)
    return title_source_of(md.attributes if md is not None else {})


def set_title_source(conn: sqlite3.Connection, doc_id: str, source: str) -> None:
    """Record ``doc_id``'s title provenance, retiring the legacy boolean."""
    validate_title_source(source)
    set_metadata(
        conn,
        doc_id,
        attributes={"title_source": source, "title_inferred": None},
    )


def _row_to_metadata(row: sqlite3.Row) -> DocMetadata:
    tags_raw = row["tags"]
    attrs_raw = row["attributes"]
    tags = json.loads(tags_raw) if tags_raw else []
    attributes = json.loads(attrs_raw) if attrs_raw else {}
    return DocMetadata(
        doc_id=row["doc_id"],
        project_id=row["project_id"],
        group_name=row["group_name"],
        mpn=row["mpn"],
        manufacturer=row["manufacturer"],
        subsystem=row["subsystem"],
        doc_type=row["doc_type"],
        tags=tags,
        attributes=attributes,
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
    return _row_to_metadata(row)


def set_metadata(
    conn: sqlite3.Connection,
    doc_id: str,
    *,
    project_id: str | None = None,
    group_name: str | None = None,
    mpn: str | None = None,
    manufacturer: str | None = None,
    subsystem: str | None = None,
    doc_type: str | None = None,
    tags: list[str] | None = None,
    attributes: dict[str, Any] | None = None,
) -> DocMetadata:
    """Upsert metadata for ``doc_id`` with **partial** merge semantics.

    A parameter left as ``None`` (its default) means "leave the existing
    value alone". To clear a string field, pass an empty string; to
    clear ``tags`` pass ``[]``; to drop an attribute pass an empty dict
    (note: ``attributes`` is **deep-merged** key-by-key, so passing
    ``{}`` keeps everything — see below).

    Merge rules
    -----------
    * Scalar fields: overwrite with the new value when not ``None``.
    * ``tags``: replace wholesale (not merged).
    * ``attributes``: shallow key-merge — new keys are added, existing
      keys are overwritten, omitted keys are preserved. To remove a
      single key set it to ``None`` and the merge will drop it.

    Returns the resulting :class:`DocMetadata` after the upsert.
    """
    existing = get_metadata(conn, doc_id)

    if existing is None:
        merged = DocMetadata(
            doc_id=doc_id,
            project_id=project_id,
            group_name=group_name,
            mpn=mpn,
            manufacturer=manufacturer,
            subsystem=subsystem,
            doc_type=doc_type,
            tags=list(tags) if tags is not None else [],
            # A None value means "drop this key" (see the merge branch
            # below). There is nothing to drop on a brand-new row, but the
            # key must not be stored as a null either.
            attributes={k: v for k, v in (attributes or {}).items() if v is not None},
        )
    else:
        merged_attrs = dict(existing.attributes)
        if attributes is not None:
            for k, v in attributes.items():
                if v is None:
                    merged_attrs.pop(k, None)
                else:
                    merged_attrs[k] = v

        merged = existing.model_copy(
            update={
                "project_id": project_id if project_id is not None else existing.project_id,
                "group_name": group_name if group_name is not None else existing.group_name,
                "mpn": mpn if mpn is not None else existing.mpn,
                "manufacturer": (
                    manufacturer if manufacturer is not None else existing.manufacturer
                ),
                "subsystem": subsystem if subsystem is not None else existing.subsystem,
                "doc_type": doc_type if doc_type is not None else existing.doc_type,
                "tags": list(tags) if tags is not None else existing.tags,
                "attributes": merged_attrs,
            }
        )

    conn.execute(
        """
        INSERT INTO doc_metadata (
            doc_id, project_id, group_name, mpn, manufacturer,
            subsystem, doc_type, tags, attributes, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(doc_id) DO UPDATE SET
            project_id   = excluded.project_id,
            group_name   = excluded.group_name,
            mpn          = excluded.mpn,
            manufacturer = excluded.manufacturer,
            subsystem    = excluded.subsystem,
            doc_type     = excluded.doc_type,
            tags         = excluded.tags,
            attributes   = excluded.attributes,
            updated_at   = CURRENT_TIMESTAMP
        """,
        (
            merged.doc_id,
            merged.project_id,
            merged.group_name,
            merged.mpn,
            merged.manufacturer,
            merged.subsystem,
            merged.doc_type,
            json.dumps(merged.tags),
            json.dumps(merged.attributes),
        ),
    )
    conn.commit()

    # Re-read so updated_at reflects the DB clock.
    refreshed = get_metadata(conn, doc_id)
    assert refreshed is not None  # we just wrote it
    return refreshed


def delete_metadata(conn: sqlite3.Connection, doc_id: str) -> int:
    """Delete the sidecar row for ``doc_id``, if any.

    Not covered by the ``chunks`` cascade (`doc_metadata` is intentionally
    not FK'd to `chunks` — see module docstring), so callers that fully
    delete a document must call this alongside :func:`datasheet_rag.store.sqlite.delete_doc`.

    Returns the number of rows deleted (0 or 1).
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM doc_metadata WHERE doc_id = ?", (doc_id,))
    deleted = cur.rowcount
    conn.commit()
    return int(deleted)


def list_docs(
    conn: sqlite3.Connection,
    *,
    project_id: str | None = None,
    group_name: str | None = None,
    mpn: str | None = None,
) -> list[DocMetadata]:
    """Return all sidecar rows matching the optional filters."""
    clauses: list[str] = []
    params: list[object] = []
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)
    if group_name is not None:
        clauses.append("group_name = ?")
        params.append(group_name)
    if mpn is not None:
        # mpn column may hold comma-separated aliases; match any token
        clauses.append("(mpn = ? OR mpn LIKE ? OR mpn LIKE ? OR mpn LIKE ?)")
        params.extend([mpn, f"{mpn},%", f"%,{mpn},%", f"%,{mpn}"])

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM doc_metadata{where} ORDER BY doc_id",
        params,
    ).fetchall()
    return [_row_to_metadata(r) for r in rows]


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
