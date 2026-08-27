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
from typing import NamedTuple

import sqlite_vec
from rich.console import Console

from datasheet_rag.config import get_settings

console = Console()

SCHEMA_VERSION = 2

# Columns added in each schema version, used for in-place migration of
# databases created at an older version. Each entry is (column_name, DDL
# fragment to feed ALTER TABLE chunks ADD COLUMN).
_CHUNKS_COLUMNS_BY_VERSION: dict[int, list[tuple[str, str]]] = {
    2: [
        ("figure_image_path", "figure_image_path TEXT"),
        ("figure_description", "figure_description TEXT"),
    ],
}


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
        _migrate_if_needed(conn)

    # Control-plane tables (api keys + audit log) are independent of the
    # chunk schema version and embedding dim, so ensure them on every open
    # — IF NOT EXISTS makes this idempotent and brings older DBs forward.
    _ensure_control_tables(conn)
    # Same reasoning: the FTS index is ensured on every open, not only on a
    # fresh DB, so a store that predates it gets one instead of an error.
    _ensure_fts(conn)
    _warn_if_fts_stale(conn)

    return conn


def _schema_initialized(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    return row is not None


def _migrate_if_needed(conn: sqlite3.Connection) -> None:
    """Bring an older DB up to the current ``SCHEMA_VERSION`` in place.

    Idempotent: re-checks columns via ``PRAGMA table_info`` so re-running
    on an already-migrated DB is a no-op. SQLite has no ``ADD COLUMN IF
    NOT EXISTS``, hence the explicit check.
    """
    row = conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
    if row is None:
        return
    current = int(row["version"])
    if current >= SCHEMA_VERSION:
        return

    cur = conn.cursor()
    existing_cols = {r["name"] for r in cur.execute("PRAGMA table_info(chunks)")}

    for target_version in range(current + 1, SCHEMA_VERSION + 1):
        for col_name, ddl_frag in _CHUNKS_COLUMNS_BY_VERSION.get(target_version, []):
            if col_name not in existing_cols:
                cur.execute(f"ALTER TABLE chunks ADD COLUMN {ddl_frag}")
                existing_cols.add(col_name)
        console.print(f"[yellow]Migrated chunks store {current} → {target_version}[/]")

    cur.execute(
        "UPDATE schema_version SET version = ? WHERE id = 1",
        (SCHEMA_VERSION,),
    )
    conn.commit()


def stored_embedding_dim(conn: sqlite3.Connection) -> int | None:
    """The vector width this database was created with, or None if unrecorded.

    The authority on what a vector has to look like to be usable here — more
    so than ``settings.embedding_dimensions``, which is only what *this*
    process is configured for. Callers validating a vector handed in from
    elsewhere (a client that embeds its own — GH #43) should ask the store.
    """
    try:
        row = conn.execute("SELECT embedding_dim FROM schema_version WHERE id = 1").fetchone()
    except sqlite3.Error:
        return None
    return int(row["embedding_dim"]) if row is not None else None


def _check_embedding_dim(conn: sqlite3.Connection, embedding_dim: int) -> None:
    stored = stored_embedding_dim(conn)
    if stored is None:
        # Table exists but no row — shouldn't happen, but treat as fresh.
        return
    stored_dim = stored
    if stored_dim != embedding_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: database was created with "
            f"dim={stored_dim} but caller requested dim={embedding_dim}. "
            f"Re-create the DB or change settings.embedding_dimensions."
        )


def _ensure_control_tables(conn: sqlite3.Connection) -> None:
    """Create the auth + audit tables if absent (idempotent).

    These hold the server's control plane — per-client API keys and the
    ingest-path audit trail — and are unrelated to the chunk/vector schema,
    so they live outside ``init_schema``/``SCHEMA_VERSION`` and are ensured
    on every connection open.
    """
    cur = conn.cursor()
    # ---- api_keys --------------------------------------------------------
    # Only the SHA-256 hash of the token is stored; the plaintext is shown
    # once at creation and never persisted. ``scopes`` is a JSON array.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id           TEXT PRIMARY KEY,
            label        TEXT NOT NULL,
            token_sha256 TEXT NOT NULL UNIQUE,
            scopes       TEXT NOT NULL,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            revoked_at   TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS ix_api_keys_sha ON api_keys(token_sha256)")
    # ---- audit_log -------------------------------------------------------
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT NOT NULL,
            key_label   TEXT,
            client_ip   TEXT,
            action      TEXT NOT NULL,
            doc_id      TEXT,
            project_id  TEXT,
            detail_json TEXT,
            status      TEXT NOT NULL,
            error       TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS ix_audit_doc ON audit_log(doc_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_audit_ts ON audit_log(ts)")
    conn.commit()


# ---------------------------------------------------------------------------
# chunk_fts: the keyword half of hybrid search
# ---------------------------------------------------------------------------


def _ensure_fts(conn: sqlite3.Connection) -> None:
    """Create ``chunk_fts`` and its sync triggers if they are absent.

    Ensured on every open (like :func:`_ensure_control_tables`) rather than
    only on a fresh DB: a store written before the FTS feature landed has no
    ``chunk_fts`` at all, and ``init_schema`` never runs again on an existing
    DB, so keyword search would raise "no such table" forever.

    Creating the table on a store that already holds chunks leaves it *empty*
    — ``CREATE … IF NOT EXISTS`` indexes nothing retroactively, and the
    triggers only fire on rows written from here on. That is precisely the
    silent-degradation state of GH #23, so we backfill immediately.
    """
    cur = conn.cursor()
    fresh = (
        cur.execute("SELECT name FROM sqlite_master WHERE name = 'chunk_fts'").fetchone() is None
    )

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

    if fresh and _count(conn, "chunks"):
        cur.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')")
        console.print(
            "[yellow]Built the keyword (FTS5) index for a store that predates "
            "it[/] — hybrid search now has its BM25 half back."
        )
    conn.commit()


def _count(conn: sqlite3.Connection, table: str) -> int:
    """Row count for a known-safe table name (never user input)."""
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


class FtsStatus(NamedTuple):
    """How much of ``chunks`` the ``chunk_fts`` index actually covers.

    ``indexed`` is ``None`` when coverage cannot be determined (see
    :func:`fts_status`) — that is "unknown", not "broken".
    """

    chunks: int
    indexed: int | None

    @property
    def healthy(self) -> bool:
        return self.indexed is None or self.indexed == self.chunks

    @property
    def missing(self) -> int:
        """Chunks absent from the index (0 when healthy or unknown)."""
        if self.indexed is None:
            return 0
        return max(self.chunks - self.indexed, 0)


def fts_status(conn: sqlite3.Connection) -> FtsStatus:
    """Compare the number of indexed rows against the number of chunks.

    The obvious check — ``SELECT count(*) FROM chunk_fts`` — cannot see this
    failure: on an external-content FTS5 table an unconstrained scan is
    answered from the *content* table, so it returns the number of chunks
    even when the index holds nothing. ``INSERT INTO chunk_fts(chunk_fts)
    VALUES('integrity-check')`` is no help either; it verifies that the index
    is self-consistent, not that it covers every content row, and passes
    happily on a half-empty index.

    What does tell the truth is the ``chunk_fts_docsize`` shadow table: one
    row per indexed document. It is absent only for ``columnsize=0`` tables,
    which we never create — if it is missing anyway we report ``None`` rather
    than guess, so an unrecognised store shape stays quiet instead of
    crying wolf.
    """
    chunks = _count(conn, "chunks")
    has_docsize = (
        conn.execute("SELECT name FROM sqlite_master WHERE name = 'chunk_fts_docsize'").fetchone()
        is not None
    )
    indexed = _count(conn, "chunk_fts_docsize") if has_docsize else None
    return FtsStatus(chunks=chunks, indexed=indexed)


def rebuild_fts(conn: sqlite3.Connection) -> FtsStatus:
    """Repopulate ``chunk_fts`` from ``chunks`` and return the status after.

    FTS5's own ``'rebuild'`` command discards the index and re-derives it
    from the content table, so this repairs any degree of desync — empty,
    partial, or stale — without touching ``chunks`` or re-running ingest.
    """
    _ensure_fts(conn)
    conn.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('rebuild')")
    conn.commit()
    return fts_status(conn)


def _warn_if_fts_stale(conn: sqlite3.Connection) -> None:
    """Say so, loudly, when the keyword index does not cover the chunks.

    A desynced index makes ``keyword_search`` return ``[]`` for every query
    and ``hybrid_search`` silently collapse to vector-only — same results,
    same shape, no error, just without the BM25 half that catches exact part
    numbers and register names (GH #23). Nothing downstream can tell; the
    only place that can is here, where the store is opened.
    """
    try:
        status = fts_status(conn)
    except sqlite3.Error:
        return  # Never let a health check stop the store from opening.
    if status.healthy or not status.chunks:
        return
    lost = (
        "Keyword search finds nothing" if not status.indexed else "Keyword search is missing hits"
    )
    console.print(
        f"[yellow]Keyword index out of sync[/]: chunk_fts covers "
        f"{status.indexed} of {status.chunks} chunks. {lost} and hybrid "
        f"search has silently lost its BM25 half. Run "
        f"[cyan]rag repair fts[/] to rebuild it."
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
            figure_image_path  TEXT,         -- local cropped figure file (added v2)
            figure_description TEXT,         -- vision-LLM description (added v2)
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
    _ensure_fts(conn)

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
    cur.execute("CREATE INDEX IF NOT EXISTS ix_doc_metadata_project ON doc_metadata(project_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS ix_doc_metadata_group ON doc_metadata(group_name)")
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
