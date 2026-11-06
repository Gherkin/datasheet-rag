"""Documents and their revisions (GH #124).

A *revision* is one PDF, named by its ``rev_id`` (the SHA-256 of the file).
A *document* groups the revisions of one datasheet — Rev A, Rev B, Rev C —
and is named ``D12``. Metadata (projects, tags, registered fields) belongs to
the document, so it applies to every revision. A revision holds its own
label (``"Rev C"``), title and pipeline bookkeeping, and may override a
document's field values or tags for itself only (``store.metadata``).

The revisions of a document are kept in an explicit order: ``position`` on
``revisions``, higher is newer. Nothing parses revision labels. Search and
listings show the newest revision of each document unless asked for all.

Tables:

* ``documents``      – one row per document; ``tags`` is the document's tag list.
* ``revisions``   – one row per revision (see ``store.schema``), carrying
                       ``document_id``, ``position`` and the ``revision`` label.
* ``store_settings`` – store-wide settings; holds the identity rule
                       (``store.identity``).

Every ingested PDF has a ``revisions`` row and so a document:
:func:`ensure_revision` creates one on first sight (``insert_chunks`` calls it).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from typing import NamedTuple

_REF_RE = re.compile(r"^D(\d+)$")


class RevisionError(ValueError):
    """A revision operation that cannot be applied as asked. A ``ValueError``
    so the HTTP layer maps it to 400."""


def ensure_documents_tables(conn: sqlite3.Connection) -> None:
    """Create ``documents`` and ``store_settings`` if absent. Does not commit."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            tags       TEXT,                 -- JSON array of strings
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS store_settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------


def format_document_id(document_id: int) -> str:
    """The user-facing form of a document id: ``D12``."""
    return f"D{int(document_id)}"


def parse_document_ref(ref: str) -> int | None:
    """``12`` for ``"D12"``; ``None`` when ``ref`` is not a document id.

    Only an upper-case ``D`` counts. A rev_id is lower-case hex, so
    ``d12`` may well be the start of one (``d12f…``) and is read as such.
    """
    m = _REF_RE.match(ref.strip())
    return int(m.group(1)) if m else None


def document_exists(conn: sqlite3.Connection, document_id: int) -> bool:
    return (
        conn.execute("SELECT 1 FROM documents WHERE id = ?", (document_id,)).fetchone() is not None
    )


class Target(NamedTuple):
    """What a reference names: a document, and the revision when it named one."""

    #: ``None`` for a rev_id that has no revision row (yet).
    document_id: int | None
    #: The revision's rev_id, or ``None`` when the reference was a ``D12``.
    rev_id: str | None


def require_document(conn: sqlite3.Connection, document_id: int) -> int:
    """``document_id`` itself; raises :class:`RevisionError` when it does not exist."""
    if not document_exists(conn, document_id):
        raise RevisionError(f"no document {format_document_id(document_id)}")
    return document_id


def lookup(conn: sqlite3.Connection, ref: str) -> Target:
    """Resolve ``D12`` or an exact rev_id to its document, without creating
    anything. Every store and backend path that turns a ref into a document
    goes through here.

    A ``D12`` that does not exist raises :class:`RevisionError`. A rev_id with
    no revision row gives ``document_id=None``. Prefix resolution of rev_ids
    happens in the backend (``resolve_rev_id``), before this is called.
    """
    n = parse_document_ref(ref)
    if n is not None:
        return Target(require_document(conn, n), None)
    return Target(document_of(conn, ref), ref)


def resolve_target(conn: sqlite3.Connection, ref: str) -> tuple[int, str | None]:
    """:func:`lookup` for a write: a rev_id with no revision row yet gets one,
    in a new document of its own (metadata may be written before a PDF is
    ingested). Does not commit. Returns ``(document_id, rev_id)``."""
    target = lookup(conn, ref)
    if target.document_id is not None:
        return target.document_id, target.rev_id
    assert target.rev_id is not None
    return ensure_revision(conn, target.rev_id), target.rev_id


def document_of(conn: sqlite3.Connection, rev_id: str) -> int | None:
    row = conn.execute("SELECT document_id FROM revisions WHERE rev_id = ?", (rev_id,)).fetchone()
    return int(row[0]) if row is not None and row[0] is not None else None


def documents_of(conn: sqlite3.Connection, rev_ids: Iterable[str] | None) -> dict[str, int]:
    """``{rev_id: document_id}``; ``None`` reads every revision."""
    if rev_ids is None:
        rows = conn.execute(
            "SELECT rev_id, document_id FROM revisions WHERE document_id IS NOT NULL"
        ).fetchall()
    else:
        ids = list(rev_ids)
        rows = []
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            rows.extend(
                conn.execute(
                    "SELECT rev_id, document_id FROM revisions "
                    f"WHERE document_id IS NOT NULL AND rev_id IN ({','.join('?' * len(batch))})",
                    batch,
                ).fetchall()
            )
    return {r[0]: int(r[1]) for r in rows}


def revisions_of(conn: sqlite3.Connection, document_id: int) -> list[str]:
    """The rev_ids of a document's revisions, newest first."""
    rows = conn.execute(
        "SELECT rev_id FROM revisions WHERE document_id = ? ORDER BY position DESC, rev_id",
        (document_id,),
    ).fetchall()
    return [r[0] for r in rows]


def newest_revision(conn: sqlite3.Connection, document_id: int) -> str | None:
    revs = revisions_of(conn, document_id)
    return revs[0] if revs else None


def older_revisions_sql(rev_id_column: str) -> str:
    """A WHERE clause dropping rows of revisions that have a newer sibling.

    Written as "not older" rather than "is newest" so a row with no revision
    record at all (chunks written straight to the table) is kept, not hidden.
    ``rev_id_column`` comes from code, never from user input.
    """
    return (
        f"{rev_id_column} NOT IN (SELECT m.rev_id FROM revisions m "
        "JOIN revisions n ON n.document_id = m.document_id AND n.position > m.position)"
    )


# ---------------------------------------------------------------------------
# Creating and placing revisions
# ---------------------------------------------------------------------------


def new_document(conn: sqlite3.Connection, *, tags: Sequence[str] | None = None) -> int:
    """Create an empty document and return its id. Does not commit."""
    cur = conn.execute(
        "INSERT INTO documents (tags) VALUES (?)", (json.dumps(list(tags)) if tags else None,)
    )
    assert cur.lastrowid is not None
    return int(cur.lastrowid)


def _renumber(conn: sqlite3.Connection, newest_first: Sequence[str]) -> None:
    """Write positions so ``newest_first[0]`` is newest. Does not commit."""
    n = len(newest_first)
    conn.executemany(
        "UPDATE revisions SET position = ? WHERE rev_id = ?",
        [(n - 1 - i, d) for i, d in enumerate(newest_first)],
    )


def ensure_revision(
    conn: sqlite3.Connection, rev_id: str, *, document_id: int | None = None
) -> int:
    """Make sure ``rev_id`` has a revision row and return its document id.

    A new revision goes on top of ``document_id`` when given, else into a new
    document of its own. An existing revision is left where it is (and
    ``document_id``, if given, must be the document it is in). Does not commit.
    """
    row = conn.execute("SELECT document_id FROM revisions WHERE rev_id = ?", (rev_id,)).fetchone()
    if row is not None and row[0] is not None:
        current = int(row[0])
        if document_id is not None and document_id != current:
            raise RevisionError(
                f"{rev_id[:12]} is already a revision of {format_document_id(current)}"
            )
        return current
    if document_id is None:
        document_id = new_document(conn)
    elif not document_exists(conn, document_id):
        raise RevisionError(f"no document {format_document_id(document_id)}")
    top = conn.execute(
        "SELECT COALESCE(MAX(position) + 1, 0) FROM revisions WHERE document_id = ?",
        (document_id,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO revisions (rev_id, document_id, position) VALUES (?, ?, ?) "
        "ON CONFLICT(rev_id) DO UPDATE SET document_id = excluded.document_id, "
        "position = excluded.position",
        (rev_id, document_id, int(top)),
    )
    return document_id


def ensure_revisions(conn: sqlite3.Connection, rev_ids: Iterable[str]) -> None:
    """:func:`ensure_revision` for each rev_id not yet known. Does not commit."""
    known = set(documents_of(conn, list(set(rev_ids))))
    for rev_id in dict.fromkeys(rev_ids):
        if rev_id not in known:
            ensure_revision(conn, rev_id)


def _tags_of(conn: sqlite3.Connection, document_id: int) -> list[str]:
    row = conn.execute("SELECT tags FROM documents WHERE id = ?", (document_id,)).fetchone()
    return json.loads(row[0]) if row is not None and row[0] else []


def _same_values(a: str | list[str], b: str | list[str]) -> bool:
    def norm(v: str | list[str]) -> set[str]:
        return {x.casefold() for x in ([v] if isinstance(v, str) else v)}

    return norm(a) == norm(b)


def _merge_documents(conn: sqlite3.Connection, source: int, target: int) -> None:
    """Move ``source``'s metadata onto ``target`` and delete ``source``.

    Projects and tags are joined. A field both hold with different values
    stops the merge (nothing is written): which one is right is the user's
    call. Does not commit.
    """
    from datasheet_rag.store.fields import get_document_values, get_project_values
    from datasheet_rag.store.projects import get_projects_for_documents

    src_vals = get_document_values(conn, [source]).get(source, {})
    dst_vals = get_document_values(conn, [target]).get(target, {})
    src_pvals = get_project_values(conn, [source]).get(source, {})
    dst_pvals = get_project_values(conn, [target]).get(target, {})
    conflicts = [
        f"{name} ({format_document_id(target)}: {dst_vals[name]!r}, "
        f"{format_document_id(source)}: {value!r})"
        for name, value in src_vals.items()
        if name in dst_vals and not _same_values(dst_vals[name], value)
    ]
    for project, values in src_pvals.items():
        for name, value in values.items():
            other = dst_pvals.get(project, {}).get(name)
            if other is not None and not _same_values(other, value):
                conflicts.append(
                    f"{name} in {project} ({format_document_id(target)}: {other!r}, "
                    f"{format_document_id(source)}: {value!r})"
                )
    if conflicts:
        raise RevisionError(
            "the two documents hold different values for: "
            + "; ".join(conflicts)
            + ". Make them agree (rag metadata <doc> --field name=value) and try again."
        )

    for name in src_vals:
        if name not in dst_vals:
            conn.execute(
                "UPDATE document_field_values SET document_id = ? WHERE document_id = ? "
                "AND field = ?",
                (target, source, name),
            )
    for project, values in src_pvals.items():
        for name in values:
            if name not in dst_pvals.get(project, {}):
                conn.execute(
                    "UPDATE document_project_field_values SET document_id = ? "
                    "WHERE document_id = ? AND project_id = ? AND field = ?",
                    (target, source, project, name),
                )
    projects = get_projects_for_documents(conn, [source]).get(source, [])
    conn.executemany(
        "INSERT OR IGNORE INTO document_projects (document_id, project_id) VALUES (?, ?)",
        [(target, p) for p in projects],
    )
    tags = list(dict.fromkeys([*_tags_of(conn, target), *_tags_of(conn, source)]))
    conn.execute(
        "UPDATE documents SET tags = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (json.dumps(tags) if tags else None, target),
    )
    _delete_document_rows(conn, source)


def _delete_document_rows(conn: sqlite3.Connection, document_id: int) -> None:
    for table in ("document_field_values", "document_project_field_values", "document_projects"):
        conn.execute(f"DELETE FROM {table} WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))


def ingest_times(conn: sqlite3.Connection, rev_ids: Iterable[str]) -> dict[str, str]:
    """``{rev_id: time}``: when each revision's chunks were first written.

    There is no ingest-time column; the oldest chunk stands in for it (the
    same value ``rag revisions`` shows as "ingested"). Chunk upserts keep
    ``created_at``, so a re-ingest keeps the time of the chunks it keeps.
    A revision with no chunks is absent.
    """
    ids = list(dict.fromkeys(rev_ids))
    out: dict[str, str] = {}
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        for rev_id, at in conn.execute(
            "SELECT rev_id, MIN(created_at) FROM chunks "
            f"WHERE rev_id IN ({','.join('?' * len(batch))}) GROUP BY rev_id",
            batch,
        ):
            if at is not None:
                out[rev_id] = at
    return out


def merge_group(
    conn: sqlite3.Connection, target: int, sources: Sequence[int], newest_first: Sequence[str]
) -> None:
    """Merge every document in ``sources`` into ``target`` and set the order.

    Each source's revisions move to ``target`` and its metadata is merged as
    in :func:`add_revision` (a field with different values stops it with a
    :class:`RevisionError`). ``newest_first`` must name every revision of the
    merged document. Does not commit, and does not roll back on error: the
    caller holds the transaction.
    """
    for source in sources:
        conn.execute("UPDATE revisions SET document_id = ? WHERE document_id = ?", (target, source))
        _merge_documents(conn, source, target)
    if sorted(newest_first) != sorted(revisions_of(conn, target)):
        # A caller bug, not a user error: never reported as a conflict.
        raise RuntimeError(f"the new order of {format_document_id(target)} is incomplete")
    _renumber(conn, newest_first)


class AddResult(NamedTuple):
    document_id: int
    #: True when the revision's own document was merged into this one.
    merged: bool


def add_revision(conn: sqlite3.Connection, document_id: int, rev_id: str) -> AddResult:
    """Put ``rev_id`` on top of ``document_id``'s revisions, as its newest.

    * ``rev_id`` already in that document: it moves to the top.
    * ``rev_id`` alone in its own document: that document is merged in
      (see :func:`_merge_documents`) and deleted.
    * ``rev_id`` in another document with other revisions: refused, so a
      whole family is never moved by accident.

    Commits.
    """
    if not document_exists(conn, document_id):
        raise RevisionError(f"no document {format_document_id(document_id)}")
    current = ensure_revision(conn, rev_id)
    merged = False
    try:
        if current != document_id:
            siblings = revisions_of(conn, current)
            if len(siblings) > 1:
                raise RevisionError(
                    f"{rev_id[:12]} is one of {len(siblings)} revisions of "
                    f"{format_document_id(current)}; unlink it first "
                    f"(rag revisions unlink {rev_id[:12]})"
                )
            # Move the revision first: its old document is deleted by the merge.
            conn.execute(
                "UPDATE revisions SET document_id = ? WHERE rev_id = ?", (document_id, rev_id)
            )
            _merge_documents(conn, current, document_id)
            merged = True
        order = [d for d in revisions_of(conn, document_id) if d != rev_id]
        _renumber(conn, [rev_id, *order])
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return AddResult(document_id, merged)


def move_revision(
    conn: sqlite3.Connection,
    rev_id: str,
    *,
    above: str | None = None,
    below: str | None = None,
) -> int:
    """Move ``rev_id`` directly above (newer than) or below (older than) a sibling.

    Both must be revisions of the same document. Returns that document's id.
    Commits.
    """
    if (above is None) == (below is None):
        raise RevisionError("give exactly one of above / below")
    other = above if above is not None else below
    assert other is not None
    if other == rev_id:
        raise RevisionError("cannot move a revision relative to itself")
    document_id = document_of(conn, rev_id)
    if document_id is None or document_of(conn, other) != document_id:
        raise RevisionError(
            f"{rev_id[:12]} and {other[:12]} are not revisions of the same document"
        )
    order = [d for d in revisions_of(conn, document_id) if d != rev_id]
    i = order.index(other)
    order.insert(i if above is not None else i + 1, rev_id)
    _renumber(conn, order)
    conn.commit()
    return document_id


def unlink_revision(conn: sqlite3.Connection, rev_id: str) -> int:
    """Take ``rev_id`` out of its document into a new one, and return the new id.

    The new document starts with a copy of the old one's projects, tags and
    field values, so the revision stays in the same projects and keeps the
    same values. Its own overrides stay with it. Refused when the revision is
    the document's only one. Commits.
    """
    old = document_of(conn, rev_id)
    if old is None:
        raise RevisionError(f"{rev_id[:12]} is not a known revision")
    siblings = revisions_of(conn, old)
    if len(siblings) < 2:
        raise RevisionError(
            f"{rev_id[:12]} is the only revision of {format_document_id(old)}; nothing to unlink"
        )
    try:
        new = new_document(conn, tags=_tags_of(conn, old))
        conn.execute(
            "INSERT INTO document_projects (document_id, project_id) "
            "SELECT ?, project_id FROM document_projects WHERE document_id = ?",
            (new, old),
        )
        conn.execute(
            "INSERT INTO document_field_values (document_id, field, value, position) "
            "SELECT ?, field, value, position FROM document_field_values WHERE document_id = ?",
            (new, old),
        )
        conn.execute(
            "INSERT INTO document_project_field_values "
            "(document_id, project_id, field, value, position) "
            "SELECT ?, project_id, field, value, position FROM document_project_field_values "
            "WHERE document_id = ?",
            (new, old),
        )
        conn.execute(
            "UPDATE revisions SET document_id = ?, position = 0 WHERE rev_id = ?",
            (new, rev_id),
        )
        _renumber(conn, [d for d in siblings if d != rev_id])
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return new


def delete_revision(conn: sqlite3.Connection, rev_id: str) -> int:
    """Drop ``rev_id``'s revision row and overrides. Does not commit.

    The document goes with its last revision, with all its metadata. Returns
    the number of revision rows deleted (0 or 1).
    """
    document_id = document_of(conn, rev_id)
    conn.execute("DELETE FROM revision_field_values WHERE rev_id = ?", (rev_id,))
    cur = conn.execute("DELETE FROM revisions WHERE rev_id = ?", (rev_id,))
    if document_id is not None:
        remaining = revisions_of(conn, document_id)
        if remaining:
            _renumber(conn, remaining)
        else:
            _delete_document_rows(conn, document_id)
    return int(cur.rowcount)


# ---------------------------------------------------------------------------
# Store settings
# ---------------------------------------------------------------------------


def get_setting(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM store_settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row is not None else None


def set_setting(conn: sqlite3.Connection, key: str, value: str | None) -> None:
    """Set (or with ``None`` remove) a store setting. Does not commit."""
    if value is None:
        conn.execute("DELETE FROM store_settings WHERE key = ?", (key,))
    else:
        conn.execute(
            "INSERT INTO store_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
