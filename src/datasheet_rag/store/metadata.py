"""Document metadata, and the per-revision sidecar (GH #124).

Metadata belongs to a *document* (``D12``, see ``store.documents``): its
projects, its tags and the values of registered fields such as MPN,
manufacturer and the project-scoped subsystem (``store.fields``). Every
revision of the document has them. A revision (one PDF, ``rev_id``) may
override a field or the tag list for itself only (``this_revision``); its
*effective* metadata is then the override.

Each revision has a ``revisions`` row. It holds the revision label
(``"Rev C"``), the revision's place in its document, its tag override, and
the pipeline's own bookkeeping: title provenance, the ingest-time title
hints and the re-embed marker. Bookkeeping is typed columns the pipeline
writes; a metadata patch cannot set them.

Metadata can be written **before** ingestion completes — useful for the CLI
to register a PDF up front so its chunks are in the right projects once
they're produced. Chunks carry none of it: every read scopes and filters
through these tables.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NamedTuple, get_args

from pydantic import BaseModel, Field

from datasheet_rag.store.documents import (
    RevisionError,
    delete_revision,
    document_exists,
    ensure_revision,
    format_document_id,
    lookup,
    newest_revision,
    older_revisions_sql,
    parse_document_ref,
    require_document,
    resolve_target,
)
from datasheet_rag.store.fields import (
    FieldScope,
    FieldShape,
    FieldValue,
    MetadataFieldError,
    check_filter_fields,
    check_patch_project,
    delete_document_project_values,
    field_filter_sql,
    field_scopes,
    field_shapes,
    get_document_values,
    get_override_values,
    get_project_values,
    normalize_field_patch_for,
    write_field_values,
)
from datasheet_rag.store.projects import (
    change_projects,
    check_project_change,
    get_projects_for_documents,
    project_scope_sql,
)


class DocMetadata(BaseModel):
    """The metadata of one revision, or of a document (``D12``) as a whole.

    For a revision, ``projects`` / ``tags`` / ``fields`` / ``project_fields``
    are its effective values: its document's, with its own overrides in
    place (named in ``overrides``). For a document, they are the document's
    own values and ``rev_id`` is its newest revision.
    """

    #: The revision (one PDF). For a document view, its newest revision.
    rev_id: str
    #: The document, ``"D12"``.
    document_id: str = ""
    #: The revision label, e.g. ``"Rev C"``; free text, never parsed.
    revision: str | None = None
    #: Every revision of the document, newest first.
    revisions: list[str] = Field(default_factory=list)
    #: Every project the document belongs to, sorted.
    projects: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    #: Values of document-scoped fields (``store.fields``), including the
    #: built-in ``mpn`` (a list), ``manufacturer`` and ``doc_type``. A field
    #: with no value is absent.
    fields: dict[str, str | list[str]] = Field(default_factory=dict)
    #: Values of project-scoped fields such as ``subsystem``, per project:
    #: ``{"netdaq": {"subsystem": "power"}}``. A project with no values is
    #: absent.
    project_fields: dict[str, dict[str, str | list[str]]] = Field(default_factory=dict)
    #: Names of the fields (and ``"tags"``) set on this revision only.
    overrides: list[str] = Field(default_factory=list)
    #: Pipeline bookkeeping of the revision, read-only through the metadata
    #: API: where the title came from (see below), whether `rag repair`
    #: marked it for re-embedding, and the two title hints captured at ingest.
    title_source: str = "auto"
    needs_reembed: bool = False
    pdf_meta_title: str | None = None
    running_header: str | None = None
    updated_at: str | None = None

    @property
    def is_newest(self) -> bool:
        return not self.revisions or self.revisions[0] == self.rev_id


# ---------------------------------------------------------------------------
# Title provenance
#
# Where a revision's title came from decides whether the next write may
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
# revision: a column alongside `chunks.doc_title` would be duplicated
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


def get_title_source(conn: sqlite3.Connection, rev_id: str) -> str:
    """Return the recorded provenance of ``rev_id``'s title (``auto`` if none)."""
    row = conn.execute("SELECT title_source FROM revisions WHERE rev_id = ?", (rev_id,)).fetchone()
    return (row["title_source"] if row is not None else None) or "auto"


def _set_bookkeeping(conn: sqlite3.Connection, rev_id: str, **columns: object) -> None:
    """Set pipeline bookkeeping columns of revision ``rev_id``. Does not commit.

    A revision seen for the first time gets its row (and a document).
    ``columns`` names come from this module only, never from user input.
    """
    ensure_revision(conn, rev_id)
    conn.execute(
        "UPDATE revisions SET "
        + ", ".join(f"{n} = ?" for n in columns)
        + ", updated_at = CURRENT_TIMESTAMP WHERE rev_id = ?",
        (*columns.values(), rev_id),
    )


def set_title_source(conn: sqlite3.Connection, rev_id: str, source: str) -> None:
    """Record ``rev_id``'s title provenance."""
    validate_title_source(source)
    _set_bookkeeping(conn, rev_id, title_source=source)
    conn.commit()


def set_title_hints(
    conn: sqlite3.Connection,
    rev_id: str,
    *,
    pdf_meta_title: str | None,
    running_header: str | None,
) -> None:
    """Record the ingest-time title hints, replacing both (``None`` clears one).

    With neither hint this only clears an existing row's hints; it does not
    create a revision row for a PDF that has none.
    """
    if not pdf_meta_title and not running_header:
        conn.execute(
            "UPDATE revisions SET pdf_meta_title = NULL, running_header = NULL, "
            "updated_at = CURRENT_TIMESTAMP "
            "WHERE rev_id = ? AND (pdf_meta_title IS NOT NULL OR running_header IS NOT NULL)",
            (rev_id,),
        )
    else:
        _set_bookkeeping(
            conn,
            rev_id,
            pdf_meta_title=pdf_meta_title or None,
            running_header=running_header or None,
        )
    conn.commit()


def set_needs_reembed(conn: sqlite3.Connection, rev_id: str, value: bool) -> None:
    """Mark (or unmark) ``rev_id`` as needing its chunks re-embedded."""
    _set_bookkeeping(conn, rev_id, needs_reembed=1 if value else None)
    conn.commit()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

#: Above this many keys a batched read reads the whole table instead of
#: binding one parameter per key.
_READ_ALL_ABOVE = 500


def _keys(ids: Sequence[Any]) -> Sequence[Any] | None:
    return None if len(ids) > _READ_ALL_ABOVE else ids


def _views(
    conn: sqlite3.Connection, rows: Sequence[sqlite3.Row], *, document_view: bool = False
) -> list[DocMetadata]:
    """Build :class:`DocMetadata` for revision rows, in a fixed number of queries."""
    if not rows:
        return []
    rev_ids = [r["rev_id"] for r in rows]
    dids = sorted({int(r["document_id"]) for r in rows if r["document_id"] is not None})
    dvals = get_document_values(conn, _keys(dids))
    ovals = {} if document_view else get_override_values(conn, _keys(rev_ids))
    projects = get_projects_for_documents(conn, _keys(dids))
    pvals = get_project_values(conn, _keys(dids))
    doc_tags: dict[int, list[str]] = {}
    revs: dict[int, list[str]] = {}
    for start in range(0, len(dids), 500):
        batch = dids[start : start + 500]
        marks = ",".join("?" * len(batch))
        for r in conn.execute(f"SELECT id, tags FROM documents WHERE id IN ({marks})", batch):
            doc_tags[int(r[0])] = json.loads(r[1]) if r[1] else []
        for r in conn.execute(
            f"SELECT document_id, rev_id FROM revisions WHERE document_id IN ({marks}) "
            "ORDER BY position DESC, rev_id",
            batch,
        ):
            revs.setdefault(int(r[0]), []).append(r[1])

    out: list[DocMetadata] = []
    for row in rows:
        did = int(row["document_id"]) if row["document_id"] is not None else None
        values: dict[str, FieldValue] = dict(dvals.get(did, {})) if did is not None else {}
        overridden: list[str] = []
        tags = list(doc_tags.get(did, [])) if did is not None else []
        if not document_view:
            own = ovals.get(row["rev_id"], {})
            values.update(own)
            overridden = sorted(own)
            if row["tags"] is not None:
                tags = json.loads(row["tags"])
                overridden.append("tags")
        out.append(
            DocMetadata(
                rev_id=row["rev_id"],
                document_id=format_document_id(did) if did is not None else "",
                revision=row["revision"],
                revisions=list(revs.get(did, [row["rev_id"]])) if did is not None else [],
                projects=list(projects.get(did, [])) if did is not None else [],
                tags=tags,
                fields=values,
                project_fields=dict(pvals.get(did, {})) if did is not None else {},
                overrides=overridden,
                title_source=row["title_source"] or "auto",
                needs_reembed=bool(row["needs_reembed"]),
                pdf_meta_title=row["pdf_meta_title"],
                running_header=row["running_header"],
                updated_at=row["updated_at"],
            )
        )
    return out


def get_metadata(conn: sqlite3.Connection, ref: str) -> DocMetadata | None:
    """The metadata of a revision (``rev_id``) or a document (``D12``), or ``None``.

    Read-only: an unknown rev_id returns ``None`` rather than creating a row.
    """
    n = parse_document_ref(ref)
    if n is not None:
        newest = newest_revision(conn, n) if document_exists(conn, n) else None
        if newest is None:
            return None
        row = conn.execute("SELECT * FROM revisions WHERE rev_id = ?", (newest,)).fetchone()
        return _views(conn, [row], document_view=True)[0]
    row = conn.execute("SELECT * FROM revisions WHERE rev_id = ?", (ref,)).fetchone()
    if row is None or row["document_id"] is None:
        return None
    return _views(conn, [row])[0]


def list_docs(
    conn: sqlite3.Connection,
    *,
    project_id: str | None = None,
    fields: dict[str, str] | None = None,
    all_revisions: bool = False,
) -> list[DocMetadata]:
    """Every revision matching the optional filters, as revision views.

    By default only the newest revision of each document is returned;
    ``all_revisions`` returns every one. ``project_id`` keeps revisions whose
    document belongs to that project (among any others). ``fields`` filters
    on registered fields, AND-ed, by each revision's effective value:
    ``{"mpn": "INA226"}`` keeps revisions whose mpn list contains INA226
    (case-insensitive). A project-scoped field matches in ``project_id`` when
    one is given, else in any project. Naming an unregistered field raises
    :class:`MetadataFieldError` rather than silently matching nothing.
    """
    clauses: list[str] = ["document_id IS NOT NULL"]
    params: list[object] = []
    if not all_revisions:
        clauses.append(older_revisions_sql("rev_id"))
    if project_id is not None:
        clause, clause_params = project_scope_sql(project_id, rev_id_column="rev_id")
        clauses.append(clause)
        params.extend(clause_params)
    if fields:
        check_filter_fields(field_shapes(conn), fields)
        field_clauses, field_params = field_filter_sql(
            fields, rev_id_column="rev_id", project_id=project_id
        )
        clauses.extend(field_clauses)
        params.extend(field_params)
    rows = conn.execute(
        f"SELECT * FROM revisions WHERE {' AND '.join(clauses)} ORDER BY rev_id",
        params,
    ).fetchall()
    return _views(conn, rows)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class CheckedWrite(NamedTuple):
    """A metadata write that :func:`check_write` accepted, normalised."""

    #: ``{field: [values]}``; an empty list removes the field's value.
    fields: dict[str, list[str]]
    scopes: dict[str, FieldScope]
    #: Where the project-scoped fields go, stripped; ``None`` when not named.
    project: str | None
    add: list[str]
    remove: list[str]
    #: Default projects (``.rag.toml``) not also in ``add``: joined without
    #: claiming, so an existing membership keeps its source.
    join: list[str]


def check_write(
    conn: sqlite3.Connection,
    ref: str,
    *,
    add_projects: list[str] | None = None,
    remove_projects: list[str] | None = None,
    tags: list[str] | None = None,
    fields: Mapping[str, Any] | None = None,
    project: str | None = None,
    this_revision: bool = False,
    revision: str | None = None,
    into: int | None = None,
    default_projects: list[str] | None = None,
) -> CheckedWrite:
    """Validate a metadata write (the arguments of :func:`set_metadata`).

    Raises :class:`MetadataFieldError` for an unregistered field, a value
    that does not fit its field's shape, a project-scoped field with no
    ``project``, or a project-scoped value set in a project the document is
    not in after this write. Clearing a project-scoped value needs no
    membership: the document holds no value outside its projects. Raises
    :class:`RevisionError` for ``this_revision`` or ``revision`` on a ``D12``
    ref, or ``this_revision`` with project changes. Raises ``ValueError``
    for a project both added and removed. Writes nothing, so ingest checks a
    patch before work it would be a waste to throw away. ``into`` is the
    document a rev_id not stored yet will join (an ingest with a link): the
    check runs against it, so nothing is written before the chunks are.
    """
    document_id, rev_id = lookup(conn, ref)
    if document_id is None and into is not None:
        document_id = require_document(conn, into)
    if rev_id is None and (this_revision or revision is not None):
        what = "--this-revision" if this_revision else "a revision label"
        raise RevisionError(
            f"{ref} names a whole document; {what} needs one revision's rev_id "
            f"(see rag revisions {ref})"
        )
    project = project.strip() if project is not None and project.strip() else None
    scopes = field_scopes(conn)
    field_patch = check_patch_for(field_shapes(conn), scopes, fields=fields, project=project)
    add, remove = check_project_change(add_projects, remove_projects)
    join = [p for p in check_project_change(default_projects, remove)[0] if p not in add]
    if this_revision:
        if add or remove or join:
            raise RevisionError(
                "projects belong to the whole document; drop --this-revision to change them"
            )
        scoped = [n for n in field_patch if scopes.get(n) == "project"]
        if scoped:
            raise MetadataFieldError(
                f"field(s) {', '.join(sorted(scoped))} hold one value per project for "
                "the whole document; they cannot be set on one revision"
            )
    sets_project_value = any(
        values and scopes.get(name) == "project" for name, values in field_patch.items()
    )
    if project is not None and sets_project_value:
        current = (
            get_projects_for_documents(conn, [document_id]).get(document_id, [])
            if document_id is not None
            else []
        )
        if project in remove or (project not in current and project not in add + join):
            raise MetadataFieldError(
                f"{ref} is not in project {project!r}; add it to the project "
                "first (--add-project) to set its project fields there"
            )
    return CheckedWrite(field_patch, scopes, project, add, remove, join)


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
    ref: str,
    *,
    add_projects: list[str] | None = None,
    remove_projects: list[str] | None = None,
    tags: list[str] | None = None,
    fields: Mapping[str, Any] | None = None,
    project: str | None = None,
    this_revision: bool = False,
    revision: str | None = None,
    default_projects: list[str] | None = None,
) -> DocMetadata:
    """Upsert metadata with **partial** merge semantics.

    ``ref`` is a document (``D12``) or a revision's rev_id. Either way the
    write lands on the document, unless ``this_revision`` is set: then tags
    and fields are written as overrides on that revision only. A parameter
    left as ``None`` (its default) means "leave the existing value alone".

    Merge rules
    -----------
    * ``add_projects`` / ``remove_projects``: the document joins / leaves
      each named project; its other projects are left alone. A project it
      is already in becomes a ``manual`` membership (a person asked for
      it, so a BOM sync keeps it).
    * ``default_projects``: projects the document joins when it is not in
      them yet, such as the ``.rag.toml`` project. An existing membership
      keeps its source. A project in
      both lists, or an empty name, raises ``ValueError`` before anything
      is written. Leaving a project deletes the document's project-scoped
      values in it. Not allowed with ``this_revision``.
    * ``tags``: replace wholesale (not merged); ``[]`` clears. With
      ``this_revision``, ``[]`` removes the override (the revision inherits
      the document's tags again).
    * ``fields``: per field — a field named in the patch has its value
      replaced; ``None``, ``""`` or ``[]`` removes it (with
      ``this_revision``: removes the override); fields not named are left
      alone. Every field must be registered (``store.fields``) and its value
      must fit the field's shape, or :class:`MetadataFieldError` is raised
      before anything is written.
    * ``project``: where the project-scoped fields in ``fields`` are
      written. Required when ``fields`` names one (see :func:`check_write`).
    * ``revision``: the revision label (``"Rev C"``); ``""`` clears it.
      Needs a rev_id ref.

    Returns the resulting :class:`DocMetadata`: the revision view for a
    rev_id ref, the document view for a ``D12``.
    """
    checked = check_write(
        conn,
        ref,
        add_projects=add_projects,
        remove_projects=remove_projects,
        tags=tags,
        fields=fields,
        project=project,
        this_revision=this_revision,
        revision=revision,
        default_projects=default_projects,
    )
    try:
        document_id, rev = resolve_target(conn, ref)
        if tags is not None:
            if this_revision:
                conn.execute(
                    "UPDATE revisions SET tags = ? WHERE rev_id = ?",
                    (json.dumps(list(tags)) if tags else None, rev),
                )
            else:
                conn.execute(
                    "UPDATE documents SET tags = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (json.dumps(list(tags)), document_id),
                )
        if revision is not None:
            conn.execute(
                "UPDATE revisions SET revision = ? WHERE rev_id = ?",
                (revision.strip() or None, rev),
            )
        # An explicit add claims the membership, so a BOM sync does not
        # remove it later; a default project only joins (GH #125).
        change_projects(conn, document_id, add=checked.add, remove=checked.remove, claim=True)
        change_projects(conn, document_id, add=checked.join)
        delete_document_project_values(conn, document_id, checked.remove)
        write_field_values(
            conn,
            document_id,
            checked.fields,
            scopes=checked.scopes,
            project=checked.project,
            revision=rev if this_revision else None,
        )
        if rev is not None:
            conn.execute(
                "UPDATE revisions SET updated_at = CURRENT_TIMESTAMP WHERE rev_id = ?",
                (rev,),
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    refreshed = get_metadata(conn, ref)
    assert refreshed is not None  # we just wrote it
    return refreshed


def join_projects(
    conn: sqlite3.Connection, rev_id: str, projects: list[str], *, claim: Sequence[str] = ()
) -> None:
    """Add ``rev_id``'s document to ``projects``. Does not commit.

    Ingest calls this inside the chunk-insert transaction, so the projects
    commit with the chunks. ``projects`` are normalised already
    (``MetadataPatch``). The ones also in ``claim`` were asked for
    explicitly and become ``manual`` memberships; the rest (the ingest's
    default project) keep an existing membership's source (GH #125).
    """
    if not projects:
        return
    document_id = ensure_revision(conn, rev_id)
    change_projects(conn, document_id, add=[p for p in projects if p in claim], claim=True)
    change_projects(conn, document_id, add=[p for p in projects if p not in claim])


def delete_metadata(conn: sqlite3.Connection, rev_id: str) -> int:
    """Delete revision ``rev_id``'s row and overrides; the document goes with
    its last revision, with its projects and values.

    Not covered by the ``chunks`` cascade (``revisions`` is intentionally
    not FK'd to ``chunks`` — see module docstring), so callers that fully
    delete a revision must call this alongside
    :func:`datasheet_rag.store.sqlite.delete_doc`.

    Returns the number of revision rows deleted (0 or 1).
    """
    deleted = delete_revision(conn, rev_id)
    conn.commit()
    return deleted
