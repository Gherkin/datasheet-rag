"""Sync a project's documents with its BOM (GH #125).

Another system (a PLM, an ERP, a KiCad export) knows the parts a board uses.
:func:`sync_bom` takes that list, the whole BOM of one project, and makes
the project's BOM memberships equal to the documents it matches:

* A BOM line names a part by one or more identifiers, ``{field: value}``,
  e.g. an internal part number and a manufacturer part number. Each field
  must be registered with the ``part`` role (``store.fields``) and hold one
  value per document.
* A line matches a document when one of its values equals one of the
  document's values of that field, ignoring case. A document matches when
  any of its revisions does: a revision's override is its value, otherwise
  the document's is.
* A line that matches nothing gets *suggestions*: documents whose value of
  an ``mfr-part`` field is the start of the line's value (``TLA2518`` for
  ``TLA2518IRTER``). A suggestion never joins the project. Accept one by
  adding the BOM's part number to the document's field.
* Matched documents join the project, marked as added by the BOM. Documents
  the BOM added before that no longer match leave it, with their
  project-scoped values. Documents added by hand (app notes, standards) are
  never removed, and a matched one stays marked as added by hand.

Sending the same BOM twice changes nothing the second time.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, Field

from datasheet_rag.store.documents import format_document_id
from datasheet_rag.store.fields import (
    FieldSpec,
    MetadataFieldError,
    _begin,
    delete_document_project_values,
    list_fields,
)
from datasheet_rag.store.projects import normalize_projects

#: A stored value shorter than this never makes a prefix suggestion: a short
#: value is the start of too many unrelated part numbers.
MIN_PREFIX_LEN = 4

_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _fold(value: str) -> str:
    """Fold ASCII case only, as SQLite's NOCASE does, so a BOM line matches
    the same documents as ``rag list --field``."""
    return value.translate(_ASCII_LOWER)


class BomLine(BaseModel):
    """One line of a BOM."""

    #: The caller's label for the line (a refdes list, a line number), echoed
    #: in the result so it can be traced back. Not matched on.
    ref: str | None = None
    #: ``{field: part number}``; every field has the ``part`` role.
    parts: dict[str, str]


class BomDocument(BaseModel):
    """A document named in a sync result."""

    document_id: str
    title: str | None = None


class BomMatch(BomDocument):
    """A document a line matched, and the fields that matched."""

    fields: list[str] = Field(default_factory=list)


class BomSuggestion(BomDocument):
    """A document whose ``field`` value is the start of the line's value."""

    field: str
    value: str


class BomLineResult(BaseModel):
    ref: str | None = None
    parts: dict[str, str]
    matches: list[BomMatch] = Field(default_factory=list)
    suggestions: list[BomSuggestion] = Field(default_factory=list)


class BomSyncResult(BaseModel):
    """What a BOM sync did, or with ``dry_run`` would do."""

    project: str
    dry_run: bool = False
    lines: list[BomLineResult] = Field(default_factory=list)
    #: Documents that joined the project.
    added: list[BomDocument] = Field(default_factory=list)
    #: Documents the BOM had added that left the project.
    removed: list[BomDocument] = Field(default_factory=list)
    #: Matched documents already in the project by hand; left as they are.
    kept_manual: list[BomDocument] = Field(default_factory=list)


def check_lines(specs: Iterable[FieldSpec], lines: Sequence[BomLine]) -> list[BomLine]:
    """Validate ``lines`` against the registry; return them with blank values dropped.

    Raises :class:`MetadataFieldError` for a field without the ``part``
    role, a project-scoped part field, or a line with no part number left.
    """
    by_name = {s.name: s for s in specs}
    part_fields = sorted(n for n, s in by_name.items() if "part" in s.roles)
    out: list[BomLine] = []
    for i, line in enumerate(lines, start=1):
        label = f"BOM line {i}" + (f" ({line.ref})" if line.ref else "")
        parts = {k.strip(): v.strip() for k, v in line.parts.items() if v and v.strip()}
        if not parts:
            raise MetadataFieldError(f"{label} has no part number")
        for name in parts:
            spec = by_name.get(name)
            if spec is None or "part" not in spec.roles:
                raise MetadataFieldError(
                    f"{label}: {name!r} is not a part field (part fields: "
                    f"{', '.join(part_fields) or 'none'}; register one with "
                    "rag fields add <name> --role part)"
                )
            if spec.scope != "document":
                raise MetadataFieldError(
                    f"{label}: {name!r} holds one value per project; a BOM matches "
                    "only fields with one value per document"
                )
        out.append(BomLine(ref=line.ref, parts=parts))
    return out


def _effective_values(
    conn: sqlite3.Connection, fields: Sequence[str]
) -> dict[tuple[str, str], dict[int, str]]:
    """``{(field, folded value): {document_id: stored value}}`` (see :func:`_fold`).

    A document holds a value when one of its revisions does: the revision's
    override when it has one for the field, else the document's value.
    """
    if not fields:
        return {}
    marks = ",".join("?" * len(fields))
    doc_vals: dict[int, dict[str, list[str]]] = {}
    for r in conn.execute(
        f"SELECT document_id, field, value FROM document_field_values WHERE field IN ({marks})",
        list(fields),
    ):
        doc_vals.setdefault(int(r[0]), {}).setdefault(r[1], []).append(r[2])
    overrides: dict[str, dict[str, list[str]]] = {}
    for r in conn.execute(
        f"SELECT rev_id, field, value FROM revision_field_values WHERE field IN ({marks})",
        list(fields),
    ):
        overrides.setdefault(r[0], {}).setdefault(r[1], []).append(r[2])
    revisions: dict[int, list[str]] = {}
    for r in conn.execute(
        "SELECT rev_id, document_id FROM revisions WHERE document_id IS NOT NULL"
    ):
        revisions.setdefault(int(r[1]), []).append(r[0])

    index: dict[tuple[str, str], dict[int, str]] = {}

    def add(document_id: int, field: str, values: Iterable[str]) -> None:
        for v in values:
            index.setdefault((field, _fold(v)), {}).setdefault(document_id, v)

    for document_id in set(doc_vals) | set(revisions):
        own = doc_vals.get(document_id, {})
        revs = revisions.get(document_id, [])
        for field in fields:
            if not revs:
                add(document_id, field, own.get(field, []))
            for rev in revs:
                values = overrides.get(rev, {}).get(field)
                add(document_id, field, own.get(field, []) if values is None else values)
    return index


def _titles(conn: sqlite3.Connection, document_ids: Iterable[int]) -> dict[int, str | None]:
    """The title of each document's newest revision, or None when it has none."""
    ids = sorted(set(document_ids))
    out: dict[int, str | None] = dict.fromkeys(ids)
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        marks = ",".join("?" * len(batch))
        rows = conn.execute(
            "SELECT m.document_id, (SELECT c.doc_title FROM chunks c WHERE c.rev_id = m.rev_id "
            "AND NULLIF(c.doc_title, '') IS NOT NULL LIMIT 1) FROM revisions m "
            f"WHERE m.document_id IN ({marks}) AND m.position = (SELECT MAX(o.position) "
            "FROM revisions o WHERE o.document_id = m.document_id)",
            batch,
        ).fetchall()
        for r in rows:
            out[int(r[0])] = r[1]
    return out


def sync_bom(
    conn: sqlite3.Connection,
    project: str,
    lines: Sequence[BomLine],
    *,
    dry_run: bool = False,
) -> BomSyncResult:
    """Make ``project``'s BOM memberships match ``lines``, the project's full BOM.

    See the module docstring for the rules. With ``dry_run`` nothing is
    written. Raises ``ValueError`` (:class:`MetadataFieldError`) for a blank
    project or a bad line, before anything is written. Commits.
    """
    (project,) = normalize_projects([project])  # raises for a blank name
    specs = list_fields(conn)
    checked = check_lines(specs, lines)
    mfr_fields = {s.name for s in specs if "mfr-part" in s.roles}
    used = sorted({f for line in checked for f in line.parts})

    # Take the write lock before reading, so the plan is made on the rows
    # it is applied to. A dry run only reads.
    if not dry_run:
        _begin(conn)
    try:
        index = _effective_values(conn, used)
        results: list[tuple[BomLine, dict[int, list[str]], list[tuple[int, str, str]]]] = []
        matched: set[int] = set()
        for line in checked:
            hits: dict[int, list[str]] = {}
            for field, value in line.parts.items():
                for document_id in index.get((field, _fold(value)), {}):
                    hits.setdefault(document_id, []).append(field)
            suggestions: list[tuple[int, str, str]] = []
            if not hits:
                for field, value in line.parts.items():
                    if field not in mfr_fields:
                        continue
                    wanted = _fold(value)
                    # Look up each proper prefix of the line's value.
                    for k in range(MIN_PREFIX_LEN, len(wanted)):
                        docs = index.get((field, wanted[:k]), {})
                        suggestions.extend((d, field, v) for d, v in docs.items())
            matched.update(hits)
            results.append((line, hits, sorted(set(suggestions))))

        members = {
            int(r[0]): r[1]
            for r in conn.execute(
                "SELECT document_id, source FROM document_projects WHERE project_id = ?",
                (project,),
            )
        }
        to_add = sorted(matched - set(members))
        to_remove = sorted(d for d, src in members.items() if src == "bom" and d not in matched)
        kept_manual = sorted(d for d in matched if members.get(d) == "manual")

        if not dry_run:
            conn.executemany(
                "INSERT INTO document_projects (document_id, project_id, source) "
                "VALUES (?, ?, 'bom')",
                [(d, project) for d in to_add],
            )
            conn.executemany(
                "DELETE FROM document_projects WHERE document_id = ? AND project_id = ?",
                [(d, project) for d in to_remove],
            )
            for d in to_remove:
                delete_document_project_values(conn, d, [project])

        shown = {d for _, hits, sugg in results for d in [*hits, *(s[0] for s in sugg)]}
        titles = _titles(conn, shown | set(to_remove) | set(to_add))
        if not dry_run:
            conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise

    def doc(d: int) -> BomDocument:
        return BomDocument(document_id=format_document_id(d), title=titles.get(d))

    return BomSyncResult(
        project=project,
        dry_run=dry_run,
        lines=[
            BomLineResult(
                ref=line.ref,
                parts=line.parts,
                matches=[
                    BomMatch(document_id=format_document_id(d), title=titles.get(d), fields=fields)
                    for d, fields in sorted(hits.items())
                ],
                suggestions=[
                    BomSuggestion(
                        document_id=format_document_id(d), title=titles.get(d), field=f, value=v
                    )
                    for d, f, v in sugg
                ],
            )
            for line, hits, sugg in results
        ],
        added=[doc(d) for d in to_add],
        removed=[doc(d) for d in to_remove],
        kept_manual=[doc(d) for d in kept_manual],
    )
