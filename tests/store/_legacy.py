"""Turn a fresh store into the v5 layout, so tests can build older stores.

The migration tests start from a store opened at the current version, write
data through today's API, then reshape it into the layout of an older
version. This step undoes v6 (GH #124): metadata moves from documents back
onto each PDF's doc_id, and the v6 tables, columns and names go.
"""

from __future__ import annotations

import sqlite3


def to_v5_layout(c: sqlite3.Connection) -> None:
    """Reshape a v6 store into v5. Each document must have one revision."""
    # v6 named a PDF a revision: undo rev_id / revisions first.
    c.execute("ALTER TABLE revisions RENAME TO doc_metadata")
    c.execute("ALTER TABLE doc_metadata RENAME COLUMN rev_id TO doc_id")
    c.execute("ALTER TABLE chunks RENAME COLUMN rev_id TO doc_id")
    c.execute(
        "UPDATE chunks SET metadata_json = json_set(json_remove(metadata_json, '$.rev_id'), "
        "'$.doc_id', json_extract(metadata_json, '$.rev_id')) "
        "WHERE json_type(metadata_json, '$.rev_id') IS NOT NULL"
    )
    c.execute("ALTER TABLE revision_field_values RENAME TO doc_field_values")
    c.execute("ALTER TABLE doc_field_values RENAME COLUMN rev_id TO doc_id")
    audit = {r[1] for r in c.execute("PRAGMA table_info(audit_log)")}
    if "rev_id" in audit:
        c.execute("ALTER TABLE audit_log RENAME COLUMN rev_id TO doc_id")
    c.execute(
        "CREATE TABLE doc_projects (doc_id TEXT NOT NULL, project_id TEXT NOT NULL, "
        "PRIMARY KEY (doc_id, project_id))"
    )
    c.execute(
        "INSERT INTO doc_projects SELECT m.doc_id, p.project_id FROM document_projects p "
        "JOIN doc_metadata m ON m.document_id = p.document_id"
    )
    c.execute(
        """
        CREATE TABLE doc_project_field_values (
            doc_id     TEXT NOT NULL,
            project_id TEXT NOT NULL,
            field      TEXT NOT NULL REFERENCES metadata_fields(name),
            value      TEXT NOT NULL COLLATE NOCASE,
            position   INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (doc_id, project_id, field, value)
        )
        """
    )
    c.execute(
        "INSERT INTO doc_project_field_values SELECT m.doc_id, v.project_id, v.field, "
        "v.value, v.position FROM document_project_field_values v "
        "JOIN doc_metadata m ON m.document_id = v.document_id"
    )
    c.execute(
        "INSERT OR IGNORE INTO doc_field_values SELECT m.doc_id, v.field, v.value, v.position "
        "FROM document_field_values v JOIN doc_metadata m ON m.document_id = v.document_id"
    )
    c.execute(
        "UPDATE doc_metadata SET tags = COALESCE(tags, "
        "(SELECT d.tags FROM documents d WHERE d.id = doc_metadata.document_id), '[]')"
    )
    c.execute("DROP INDEX ix_revisions_document")
    for col in ("document_id", "position", "revision"):
        c.execute(f"ALTER TABLE doc_metadata DROP COLUMN {col}")
    for table in (
        "document_projects",
        "document_field_values",
        "document_project_field_values",
        "documents",
        "store_settings",
    ):
        c.execute(f"DROP TABLE {table}")
    c.execute("UPDATE schema_version SET version = 5")
    c.commit()
