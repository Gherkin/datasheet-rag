"""CRUD helpers for the chunk + vector store.

All functions take an already-open :class:`sqlite3.Connection` (see
:func:`aws_rag.store.schema.connect`) and operate in a single transaction
where it makes sense. They mirror the columns declared in
:mod:`aws_rag.store.schema`.

The ``chunk_vecs`` virtual table is keyed by ``chunk_id`` (TEXT). Vectors
are serialized to ``float32`` little-endian bytes via ``numpy.tobytes()``
which is the binary format ``sqlite-vec``'s ``vec0`` table expects.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from aws_rag.models.chunk import (
    Chunk,
    ChunkGraph,
    ChunkLevel,
    ChunkMetadata,
    LayoutType,
)

# Columns inserted in the order below — keeping the SQL and the parameter
# tuple aligned is the single source of truth for the row shape.
_INSERT_SQL = """
INSERT INTO chunks (
    id, doc_id, project_id, group_name, level,
    text, context_text, token_count, layout_type, page_numbers,
    chapter_title, section_title, doc_title,
    parent_id, prev_id, next_id, chapter_root_id,
    figure_s3_key, figure_caption, figure_image_path, figure_description,
    metadata_json
) VALUES (
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?
)
ON CONFLICT(id) DO UPDATE SET
    doc_id            = excluded.doc_id,
    project_id        = excluded.project_id,
    group_name        = excluded.group_name,
    level             = excluded.level,
    text              = excluded.text,
    context_text      = excluded.context_text,
    token_count       = excluded.token_count,
    layout_type       = excluded.layout_type,
    page_numbers      = excluded.page_numbers,
    chapter_title     = excluded.chapter_title,
    section_title     = excluded.section_title,
    doc_title         = excluded.doc_title,
    parent_id         = excluded.parent_id,
    prev_id           = excluded.prev_id,
    next_id           = excluded.next_id,
    chapter_root_id   = excluded.chapter_root_id,
    figure_s3_key     = excluded.figure_s3_key,
    figure_caption    = excluded.figure_caption,
    figure_image_path = excluded.figure_image_path,
    figure_description = COALESCE(excluded.figure_description, chunks.figure_description),
    metadata_json     = excluded.metadata_json
"""

_VEC_DELETE_SQL = "DELETE FROM chunk_vecs WHERE chunk_id = ?"
_VEC_INSERT_SQL = "INSERT INTO chunk_vecs(chunk_id, embedding) VALUES (?, ?)"


def _vector_to_bytes(vector: Sequence[float]) -> bytes:
    """Serialize a vector to the ``float32`` byte layout ``vec0`` expects."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def _chunk_row(
    chunk: Chunk,
    *,
    project_id: str | None,
    group_name: str | None,
) -> tuple[object, ...]:
    """Flatten a :class:`Chunk` into the parameter tuple for :data:`_INSERT_SQL`.

    ``project_id`` and ``group_name`` override whatever was on the chunk's
    metadata when supplied (non-None); otherwise we fall back to the
    chunk's own values (currently always None on the model — callers wire
    this in at insert time).
    """
    md = chunk.metadata
    page_numbers_json = json.dumps(md.page_numbers)
    metadata_json = chunk.metadata.model_dump_json()

    return (
        chunk.id,
        chunk.doc_id,
        project_id,
        group_name,
        int(chunk.level),
        chunk.text,
        chunk.context_text,
        int(chunk.token_count),
        md.layout_type.value if isinstance(md.layout_type, LayoutType) else md.layout_type,
        page_numbers_json,
        md.chapter_title,
        md.section_title,
        md.doc_title,
        chunk.parent_id,
        chunk.prev_id,
        chunk.next_id,
        chunk.chapter_root_id,
        chunk.figure_s3_key,
        chunk.figure_caption,
        chunk.figure_image_path,
        chunk.figure_description,
        metadata_json,
    )


def insert_chunks(
    conn: sqlite3.Connection,
    chunks: Iterable[Chunk],
    vectors: Mapping[str, Sequence[float]] | None = None,
    *,
    project_id: str | None = None,
    group_name: str | None = None,
) -> int:
    """Bulk-insert chunks (and optionally their vectors) in one transaction.

    Parameters
    ----------
    conn:
        Open SQLite connection from :func:`connect`.
    chunks:
        Any iterable of :class:`Chunk`. Consumed once.
    vectors:
        Optional ``chunk_id -> embedding`` mapping. Only the chunks that
        have a vector here get a row in ``chunk_vecs``; the rest are
        text-only (useful for partial re-ingest).
    project_id, group_name:
        If provided, override the column for every inserted row.

    Returns
    -------
    int
        Number of chunks inserted (== number of rows the INSERT touched).
    """
    cur = conn.cursor()
    count = 0
    try:
        for chunk in chunks:
            row = _chunk_row(
                chunk,
                project_id=project_id,
                group_name=group_name,
            )
            cur.execute(_INSERT_SQL, row)

            if vectors is not None and chunk.id in vectors:
                cur.execute(_VEC_DELETE_SQL, (chunk.id,))
                cur.execute(
                    _VEC_INSERT_SQL,
                    (chunk.id, _vector_to_bytes(vectors[chunk.id])),
                )
            count += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return count


def insert_chunk_graph(
    conn: sqlite3.Connection,
    graph: ChunkGraph,
    vectors: Mapping[str, Sequence[float]] | None = None,
    *,
    project_id: str | None = None,
    group_name: str | None = None,
) -> int:
    """Convenience wrapper: insert every chunk in a :class:`ChunkGraph`."""
    return insert_chunks(
        conn,
        graph.chunks.values(),
        vectors=vectors,
        project_id=project_id,
        group_name=group_name,
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    """Reconstruct a :class:`Chunk` from a ``chunks`` row.

    ``children_ids`` is not denormalized onto the row — callers that need
    it should query ``SELECT id FROM chunks WHERE parent_id = ?`` (or use
    the in-memory :class:`ChunkGraph` returned by chunking).
    """
    metadata_json = row["metadata_json"]
    if metadata_json:
        metadata = ChunkMetadata.model_validate_json(metadata_json)
    else:
        # Defensive fallback: synthesize from columns. Should not happen
        # for rows written by insert_chunks().
        layout_value = row["layout_type"] or LayoutType.TEXT.value
        page_numbers_raw = row["page_numbers"]
        page_numbers = json.loads(page_numbers_raw) if page_numbers_raw else []
        metadata = ChunkMetadata(
            doc_id=row["doc_id"],
            doc_title=row["doc_title"] or "",
            chapter_title=row["chapter_title"] or "",
            section_title=row["section_title"] or "",
            page_numbers=page_numbers,
            layout_type=LayoutType(layout_value),
        )

    # figure_image_path / figure_description were added in schema v2; older
    # rows return None for them after migration, which is the correct default.
    figure_image_path = row["figure_image_path"] if "figure_image_path" in row.keys() else None
    figure_description = row["figure_description"] if "figure_description" in row.keys() else None

    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        level=ChunkLevel(int(row["level"])),
        text=row["text"],
        context_text=row["context_text"] or "",
        token_count=int(row["token_count"] or 0),
        metadata=metadata,
        parent_id=row["parent_id"],
        children_ids=[],
        prev_id=row["prev_id"],
        next_id=row["next_id"],
        chapter_root_id=row["chapter_root_id"],
        figure_s3_key=row["figure_s3_key"],
        figure_caption=row["figure_caption"],
        figure_image_path=figure_image_path,
        figure_description=figure_description,
    )


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> Chunk | None:
    """Fetch one chunk by ID, or ``None`` if not present.

    Notes
    -----
    ``children_ids`` on the returned :class:`Chunk` is always ``[]`` —
    children are not denormalized into the row. Use
    ``SELECT id FROM chunks WHERE parent_id = ?`` to materialize them on
    demand.
    """
    row = conn.execute(
        "SELECT * FROM chunks WHERE id = ?",
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_chunk(row)


def delete_doc(conn: sqlite3.Connection, doc_id: str) -> int:
    """Delete every chunk belonging to ``doc_id``.

    The ``chunks_ad`` AFTER DELETE trigger (see
    :mod:`aws_rag.store.schema`) cascades the delete to ``chunk_vecs``
    and removes the FTS5 entries.

    Returns the number of chunk rows deleted.
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    deleted = cur.rowcount
    conn.commit()
    return int(deleted)


def list_figure_chunks(
    conn: sqlite3.Connection,
    *,
    doc_id: str | None = None,
    project_id: str | None = None,
    only_with_image: bool = True,
) -> list[Chunk]:
    """Return all chunks whose layout_type is 'figure'.

    Useful for both the MCP ``get_figure`` tool and offline workflows like
    "generate descriptions for every figure that doesn't have one yet."

    Parameters
    ----------
    only_with_image:
        When True (default), filter out figure chunks that don't have a
        usable image source (no ``figure_image_path`` and no ``figure_s3_key``).
    """
    sql = "SELECT * FROM chunks WHERE layout_type = ?"
    params: list[object] = [LayoutType.FIGURE.value]
    if doc_id:
        sql += " AND doc_id = ?"
        params.append(doc_id)
    if project_id:
        sql += " AND project_id = ?"
        params.append(project_id)
    if only_with_image:
        sql += " AND (figure_image_path IS NOT NULL OR figure_s3_key IS NOT NULL)"
    sql += " ORDER BY doc_id, rowid"

    rows = conn.execute(sql, params).fetchall()
    return [_row_to_chunk(row) for row in rows]


def update_figure_description(
    conn: sqlite3.Connection,
    chunk_id: str,
    description: str,
    *,
    update_context_text: bool = True,
) -> bool:
    """Set ``figure_description`` on a chunk and (optionally) re-fold it
    into ``context_text`` so future re-embeds pick it up.

    Returns True if a row was updated.
    """
    if update_context_text:
        # Append the description to context_text — the embedding pipeline
        # uses context_text verbatim. We don't try to dedup if called twice
        # because the splitter's enrich_context already structures the
        # blob; callers running this multiple times should re-run the full
        # chunking pipeline instead.
        cur = conn.execute(
            """UPDATE chunks
               SET figure_description = ?,
                   context_text = CASE
                       WHEN context_text LIKE '%' || ? || '%' THEN context_text
                       ELSE context_text || char(10) || 'Description: ' || ?
                   END
               WHERE id = ?""",
            (description, description, description, chunk_id),
        )
    else:
        cur = conn.execute(
            "UPDATE chunks SET figure_description = ? WHERE id = ?",
            (description, chunk_id),
        )
    conn.commit()
    return cur.rowcount > 0


def get_doc_titles(conn: sqlite3.Connection) -> dict[str, str]:
    """Return a mapping of doc_id → doc_title from the chunks table."""
    rows = conn.execute(
        "SELECT doc_id, doc_title FROM chunks WHERE doc_title IS NOT NULL GROUP BY doc_id"
    ).fetchall()
    return {row["doc_id"]: row["doc_title"] for row in rows}


def resolve_doc_id(conn: sqlite3.Connection, doc_id: str) -> str:
    """Resolve a possibly-abbreviated doc_id (à la `git` short SHAs) to its full form.

    doc_ids are full SHA-256 content hashes (64 hex chars) — too long to
    read or copy comfortably. Any unambiguous prefix that matches exactly
    one ingested document resolves to that document's full doc_id.

    Raises ``ValueError`` if the prefix matches zero or more than one document.
    """
    rows = conn.execute(
        "SELECT DISTINCT doc_id FROM chunks WHERE doc_id LIKE ? || '%'",
        (doc_id,),
    ).fetchall()
    matches = [row["doc_id"] for row in rows]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No ingested document matches doc_id '{doc_id}'.")
    raise ValueError(
        f"doc_id '{doc_id}' is ambiguous — matches {len(matches)} documents: "
        + ", ".join(m[:12] for m in matches)
    )


def get_ingested_docs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return one summary row per ingested document.

    A document only appears here once it has actually been chunked and
    embedded into the store — unlike S3, which lists every upload regardless
    of ingestion status.
    """
    rows = conn.execute(
        """
        SELECT doc_id,
               COUNT(*) AS chunk_count,
               MIN(created_at) AS ingested_at
          FROM chunks
         GROUP BY doc_id
         ORDER BY doc_id
        """
    ).fetchall()

    titles = get_doc_titles(conn)
    docs: list[dict[str, Any]] = []
    for row in rows:
        doc_id = row["doc_id"]
        page_row = conn.execute(
            """
            SELECT page_numbers FROM chunks
             WHERE doc_id = ? AND page_numbers IS NOT NULL AND page_numbers != '[]'
             ORDER BY rowid DESC
             LIMIT 1
            """,
            (doc_id,),
        ).fetchone()
        page_count: int | None = None
        if page_row:
            try:
                pages = json.loads(page_row["page_numbers"])
                if pages:
                    page_count = max(pages)
            except (ValueError, TypeError):
                pass

        docs.append({
            "doc_id": doc_id,
            "doc_title": titles.get(doc_id, "—"),
            "chunk_count": row["chunk_count"],
            "page_count": page_count,
            "ingested_at": row["ingested_at"],
        })
    return docs


def count_chunks(
    conn: sqlite3.Connection,
    *,
    doc_id: str | None = None,
    project_id: str | None = None,
) -> int:
    """Count rows in ``chunks`` matching the optional filters."""
    clauses: list[str] = []
    params: list[object] = []
    if doc_id is not None:
        clauses.append("doc_id = ?")
        params.append(doc_id)
    if project_id is not None:
        clauses.append("project_id = ?")
        params.append(project_id)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM chunks{where}",
        params,
    ).fetchone()
    return int(row["n"])
