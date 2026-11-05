"""Project membership: which projects a document belongs to (GH #121).

A document can belong to any number of projects — one datasheet is often
used by several boards. ``doc_projects`` holds one row per (document,
project). Every project-scoped read (search, document listings, stats,
chunk counts, figure listings) scopes through it with
:func:`project_scope_sql`; chunks carry no project of their own.

Membership is a fixed table, not a registered field (``store.fields``): it
is the scoping mechanism behind ``/mcp/<project>``, ``.rag.toml`` and the
default project.

Writes go through ``store.metadata.set_metadata`` (``add_projects`` /
``remove_projects``), which also keeps the document's ``doc_metadata`` row.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence


def ensure_projects_table(conn: sqlite3.Connection) -> None:
    """Create ``doc_projects`` and its index if absent (idempotent)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_projects (
            doc_id     TEXT NOT NULL,
            project_id TEXT NOT NULL,
            PRIMARY KEY (doc_id, project_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_doc_projects_project ON doc_projects(project_id)")


def normalize_projects(names: Iterable[str] | None) -> list[str]:
    """Strip each project name and drop duplicates, keeping order.

    Raises ``ValueError`` for an empty name: membership in "" is never what
    the caller meant, and it could not be removed by name later.
    """
    out: list[str] = []
    for raw in names or ():
        name = raw.strip()
        if not name:
            raise ValueError("a project name cannot be empty")
        if name not in out:
            out.append(name)
    return out


def check_project_change(
    add: Iterable[str] | None, remove: Iterable[str] | None
) -> tuple[list[str], list[str]]:
    """Normalise an add/remove pair; raise ``ValueError`` if a project is in both."""
    add_n = normalize_projects(add)
    remove_n = normalize_projects(remove)
    both = sorted(set(add_n) & set(remove_n))
    if both:
        raise ValueError(f"project(s) {', '.join(both)} given both to add and to remove")
    return add_n, remove_n


def project_scope_sql(project_id: str, *, doc_id_column: str) -> tuple[str, list[object]]:
    """A WHERE clause keeping rows whose document belongs to ``project_id``.

    ``doc_id_column`` names the column to test (``"doc_id"``, ``"c.doc_id"``)
    and comes from code, never from user input.
    """
    return (
        f"{doc_id_column} IN (SELECT doc_id FROM doc_projects WHERE project_id = ?)",
        [project_id],
    )


def get_projects_for_docs(
    conn: sqlite3.Connection, doc_ids: Sequence[str] | None = None
) -> dict[str, list[str]]:
    """``{doc_id: sorted projects}``; ``doc_ids=None`` reads every document.

    A document with no project is absent from the result.
    """
    if doc_ids is None:
        rows = conn.execute("SELECT doc_id, project_id FROM doc_projects").fetchall()
    else:
        rows = []
        ids = list(doc_ids)
        # SQLite's parameter limit is 999 on older builds; batch well under it.
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            marks = ",".join("?" * len(batch))
            rows.extend(
                conn.execute(
                    f"SELECT doc_id, project_id FROM doc_projects WHERE doc_id IN ({marks})",
                    batch,
                ).fetchall()
            )
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(row[0], []).append(row[1])
    for projects in out.values():
        projects.sort()
    return out


def change_projects(
    conn: sqlite3.Connection,
    doc_id: str,
    *,
    add: Sequence[str] = (),
    remove: Sequence[str] = (),
) -> None:
    """Add and remove project memberships of ``doc_id``. Does not commit.

    Adding a project the document already has, or removing one it does not
    have, is a no-op. Callers normalise the names first
    (:func:`check_project_change`).
    """
    conn.executemany(
        "INSERT OR IGNORE INTO doc_projects (doc_id, project_id) VALUES (?, ?)",
        [(doc_id, p) for p in add],
    )
    conn.executemany(
        "DELETE FROM doc_projects WHERE doc_id = ? AND project_id = ?",
        [(doc_id, p) for p in remove],
    )


def delete_doc_projects(conn: sqlite3.Connection, doc_id: str) -> None:
    """Remove every project membership of ``doc_id``. Does not commit."""
    conn.execute("DELETE FROM doc_projects WHERE doc_id = ?", (doc_id,))
