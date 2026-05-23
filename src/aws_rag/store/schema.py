"""SQLite schema and connection factory for the chunk + vector store.

Tables created by :func:`init_schema`:

* ``chunks``         – primary chunk store (one row per chunk).
* ``chunk_vecs``     – ``sqlite-vec`` ``vec0`` virtual table holding the
                       embedding for each chunk, keyed by ``chunk_id``.
* ``chunk_fts``      – FTS5 virtual table providing BM25 keyword search
                       over ``context_text`` and ``text``. Content-linked
                       to ``chunks`` via triggers.
* ``doc_metadata``   – sidecar table for document-level metadata
                       (project, mpn, manufacturer, …). Independent of
                       ``chunks`` so metadata can be registered before
                       ingestion completes.
* ``schema_version`` – single-row table recording the schema version and
                       the embedding dimension the DB was created with.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec
from rich.console import Console

from aws_rag.config import get_settings

console = Console()

SCHEMA_VERSION = 1


def connect(
    db_path: Path | str | None = None,
    *,
    embedding_dim: int | None = None,
) -> sqlite3.Connection:
    """Open (or create) the SQLite store and return a ready-to-use connection.

    Parameters
    ----------
    db_path:
        Path to the SQLite file. Defaults to ``settings.sqlite_db_path``.
        Use ``":memory:"`` for tests.
    embedding_dim:
        Dimension of the embedding vectors stored in ``chunk_vecs``.
        Defaults to ``settings.embedding_dimensions``. If the DB already
        exists with a different dimension a :class:`RuntimeError` is
        raised — there is no automatic migration.
    """
    settings = get_settings()
    if db_path is None:
        db_path = settings.sqlite_db_path
    if embedding_dim is None:
        embedding_dim = settings.embedding_dimensions

    # Coerce to str — sqlite3 accepts both but path objects need a parent.
    is_memory = str(db_path) == ":memory:"
    if not is_memory:
        path_obj = Path(db_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        db_path_str = str(path_obj)
    else:
        db_path_str = ":memory:"

    conn = sqlite3.connect(db_path_str, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # sqlite-vec is a loadable extension — enable then load.
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Pragmas. WAL is a no-op on in-memory DBs and SQLite warns, so skip.
    conn.execute("PRAGMA foreign_keys = ON")
    if not is_memory:
        conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    # Bootstrap tables if they are not present yet.
    if not _schema_initialized(conn):
        init_schema(conn, embedding_dim=embedding_dim)
    else:
        _check_embedding_dim(conn, embedding_dim)

    return conn


def _schema_initialized(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    return row is not None


def _check_embedding_dim(conn: sqlite3.Connection, embedding_dim: int) -> None:
    row = conn.execute(
        "SELECT embedding_dim FROM schema_version WHERE id = 1"
    ).fetchone()
    if row is None:
        # Table exists but no row — shouldn't happen, but treat as fresh.
        return
    stored_dim = int(row["embedding_dim"])
    if stored_dim != embedding_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: database was created with "
            f"dim={stored_dim} but caller requested dim={embedding_dim}. "
            f"Re-create the DB or change settings.embedding_dimensions."
        )


def init_schema(conn: sqlite3.Connection, *, embedding_dim: int) -> None:
    """Create all tables, indexes and triggers if they do not already exist.

    Safe to call multiple times — all DDL uses ``IF NOT EXISTS``.
    """
    cur = conn.cursor()

    # ---- chunks ----------------------------------------------------------
    # Holds one row per chunk. metadata_json round-trips the full
    # `ChunkMetadata` blob so we never lose information that doesn't fit
    # into the flat columns.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id              TEXT PRIMARY KEY,
            doc_id          TEXT NOT NULL,
            project_id      TEXT,
            group_name      TEXT,
            level           INTEGER NOT NULL,
            text            TEXT NOT NULL,
            context_text    TEXT NOT NULL,
            token_count     INTEGER,
            layout_type     TEXT,
            page_numbers    TEXT,            -- JSON array of ints
            chapter_title   TEXT,
            section_title   TEXT,
            doc_title       TEXT,
            parent_id       TEXT,
            prev_id         TEXT,
            next_id         TEXT,
            chapter_root_id TEXT,
            figure_s3_key   TEXT,
            figure_caption  TEXT,
            metadata_json   TEXT,            -- full ChunkMetadata JSON
            created_at      TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS ix_chunks_doc_id ON chunks(doc_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_chunks_project_id ON chunks(project_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_chunks_group_name ON chunks(group_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_chunks_level ON chunks(level)")

    # ---- chunk_vecs ------------------------------------------------------
    # sqlite-vec ``vec0`` virtual table. The dimension is baked into the
    # DDL and cannot be changed after creation — we therefore record it
    # in ``schema_version`` and refuse mismatched re-opens.
    cur.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vecs USING vec0(
            chunk_id TEXT PRIMARY KEY,
            embedding FLOAT[{int(embedding_dim)}]
        )
        """
    )

    # ---- chunk_fts -------------------------------------------------------
    # External-content FTS5 over chunks. The ``porter`` stemmer plus
    # ``unicode61 remove_diacritics 2`` tokenizer keeps technical tokens
    # (e.g. "3.3V", "I2C") reasonably intact while still folding case
    # and accents.
    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
            context_text,
            text,
            content='chunks',
            content_rowid='rowid',
            tokenize = 'porter unicode61 remove_diacritics 2'
        )
        """
    )

    # Triggers keep the FTS index in sync with ``chunks``. For external
    # content tables we issue ``INSERT INTO chunk_fts(chunk_fts, rowid, …)
    # VALUES('delete', …)`` to remove the old row before re-indexing.
    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunk_fts(rowid, context_text, text)
            VALUES (new.rowid, new.context_text, new.text);
        END
        """
    )
    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunk_fts(chunk_fts, rowid, context_text, text)
            VALUES ('delete', old.rowid, old.context_text, old.text);
            DELETE FROM chunk_vecs WHERE chunk_id = old.id;
        END
        """
    )
    cur.execute(
        """
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunk_fts(chunk_fts, rowid, context_text, text)
            VALUES ('delete', old.rowid, old.context_text, old.text);
            INSERT INTO chunk_fts(rowid, context_text, text)
            VALUES (new.rowid, new.context_text, new.text);
        END
        """
    )

    # ---- doc_metadata ----------------------------------------------------
    # Sidecar: not a FK to chunks so metadata can be registered before
    # ingestion completes (and survive a re-ingest).
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS doc_metadata (
            doc_id       TEXT PRIMARY KEY,
            project_id   TEXT,
            group_name   TEXT,
            mpn          TEXT,
            manufacturer TEXT,
            subsystem    TEXT,
            doc_type     TEXT,
            tags         TEXT,             -- JSON array of strings
            attributes   TEXT,             -- JSON object (free-form)
            updated_at   TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_doc_metadata_project ON doc_metadata(project_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS ix_doc_metadata_group ON doc_metadata(group_name)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS ix_doc_metadata_mpn ON doc_metadata(mpn)")

    # ---- schema_version --------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            id            INTEGER PRIMARY KEY CHECK (id = 1),
            version       INTEGER NOT NULL,
            embedding_dim INTEGER NOT NULL,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        "INSERT OR IGNORE INTO schema_version(id, version, embedding_dim) VALUES (1, ?, ?)",
        (SCHEMA_VERSION, int(embedding_dim)),
    )

    conn.commit()
