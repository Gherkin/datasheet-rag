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
INSERT OR REPLACE INTO chunks (
    id, doc_id, project_id, group_name, level,
    text, context_text, token_count, layout_type, page_numbers,
    chapter_title, section_title, doc_title,
    parent_id, prev_id, next_id, chapter_root_id,
    figure_s3_key, figure_caption, metadata_json
) VALUES (
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?
)
"""

_VEC_INSERT_SQL = (
    "INSERT OR REPLACE INTO chunk_vecs(chunk_id, embedding) VALUES (?, ?)"
)


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
