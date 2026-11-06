"""Project membership: which projects a document belongs to (GH #121, #124).

A document can belong to any number of projects — one datasheet is often
used by several boards. Membership belongs to the *document* (``D12``, see
``store.documents``), so every revision of a datasheet is in the same
projects. ``document_projects`` holds one row per (document, project).
Every project-scoped read (search, document listings, stats, chunk counts,
figure listings) scopes through it with :func:`project_scope_sql`; chunks
carry no project of their own.

Membership is a fixed table, not a registered field (``store.fields``): it
is the scoping mechanism behind ``/mcp/<project>``, ``.rag.toml`` and the
default project.

Writes go through ``store.metadata.set_metadata`` (``add_projects`` /
``remove_projects``) or, for ingest, ``store.metadata.join_projects``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence


def ensure_projects_table(conn: sqlite3.Connection) -> None:
    """Create ``document_projects`` and its index if absent (idempotent)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_projects (
            document_id INTEGER NOT NULL,
            project_id  TEXT NOT NULL,
            PRIMARY KEY (document_id, project_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_projects_project ON document_projects(project_id)"
    )


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


def project_scope_sql(project_id: str, *, rev_id_column: str) -> tuple[str, list[object]]:
    """A WHERE clause keeping rows whose revision's document belongs to ``project_id``.

    ``rev_id_column`` names the column to test (``"rev_id"``, ``"c.rev_id"``)
    and comes from code, never from user input.
    """
    return (
        f"{rev_id_column} IN (SELECT m.rev_id FROM revisions m "
        "JOIN document_projects p ON p.document_id = m.document_id WHERE p.project_id = ?)",
        [project_id],
    )


def get_projects_for_documents(
    conn: sqlite3.Connection, document_ids: Sequence[int] | None = None
) -> dict[int, list[str]]:
    """``{document_id: sorted projects}``; ``None`` reads every document.

    A document with no project is absent from the result.
    """
    if document_ids is None:
        rows = conn.execute("SELECT document_id, project_id FROM document_projects").fetchall()
    else:
        rows = []
        ids = list(document_ids)
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            marks = ",".join("?" * len(batch))
            rows.extend(
                conn.execute(
                    "SELECT document_id, project_id FROM document_projects "
                    f"WHERE document_id IN ({marks})",
                    batch,
                ).fetchall()
            )
    out: dict[int, list[str]] = {}
    for row in rows:
        out.setdefault(int(row[0]), []).append(row[1])
    for projects in out.values():
        projects.sort()
    return out


def get_projects_for_docs(
    conn: sqlite3.Connection, rev_ids: Sequence[str] | None = None
) -> dict[str, list[str]]:
    """``{rev_id: sorted projects of its document}``; ``None`` reads every revision.

    A revision whose document is in no project is absent from the result.
    """
    from datasheet_rag.store.documents import documents_of

    docs = documents_of(conn, rev_ids)
    projects = get_projects_for_documents(conn, sorted(set(docs.values())))
    return {d: list(projects[n]) for d, n in docs.items() if n in projects}


def change_projects(
    conn: sqlite3.Connection,
    document_id: int,
    *,
    add: Sequence[str] = (),
    remove: Sequence[str] = (),
) -> None:
    """Add and remove project memberships of a document. Does not commit.

    Adding a project the document already has, or removing one it does not
    have, is a no-op. Callers normalise the names first
    (:func:`check_project_change`).
    """
    conn.executemany(
        "INSERT OR IGNORE INTO document_projects (document_id, project_id) VALUES (?, ?)",
        [(document_id, p) for p in add],
    )
    conn.executemany(
        "DELETE FROM document_projects WHERE document_id = ? AND project_id = ?",
        [(document_id, p) for p in remove],
    )
