"""Vector, keyword and hybrid retrieval over the chunk store.

Three search modes are exposed:

* :func:`vector_search`  – KNN over ``chunk_vecs`` (``sqlite-vec``).
* :func:`keyword_search` – BM25 over the FTS5 ``chunk_fts`` index.
* :func:`hybrid_search`  – Reciprocal Rank Fusion of the two above.

All three return a homogeneous list of :class:`SearchResult` ranked
descending by ``score`` (larger = more relevant).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel

from aws_rag.models.chunk import Chunk, ChunkLevel, LayoutType
from aws_rag.store.sqlite import get_chunk


class SearchFilters(BaseModel):
    """Optional metadata filters applied on top of a search query."""

    doc_ids: list[str] | None = None
    project_id: str | None = None
    group_name: str | None = None
    level: ChunkLevel | list[ChunkLevel] | None = None
    layout_types: list[LayoutType] | None = None
    min_page: int | None = None
    max_page: int | None = None

    def _where_clause(self) -> tuple[str, list[object]]:
        """Build the SQL fragment to AND-append to a query on ``chunks c``.

        Returns
        -------
        (where_sql, params)
            ``where_sql`` is either an empty string or ``" AND <expr>..."``
            ready to splice into an existing WHERE. ``params`` lines up
            with the ``?`` placeholders inside ``where_sql``.
        """
        clauses: list[str] = []
        params: list[object] = []

        if self.doc_ids:
            placeholders = ",".join(["?"] * len(self.doc_ids))
            clauses.append(f"c.doc_id IN ({placeholders})")
            params.extend(self.doc_ids)

        if self.project_id is not None:
            clauses.append("c.project_id = ?")
            params.append(self.project_id)

        if self.group_name is not None:
            clauses.append("c.group_name = ?")
            params.append(self.group_name)

        if self.level is not None:
            if isinstance(self.level, list):
                placeholders = ",".join(["?"] * len(self.level))
                clauses.append(f"c.level IN ({placeholders})")
                params.extend(int(lvl) for lvl in self.level)
            else:
                clauses.append("c.level = ?")
                params.append(int(self.level))

        if self.layout_types:
            placeholders = ",".join(["?"] * len(self.layout_types))
            clauses.append(f"c.layout_type IN ({placeholders})")
            params.extend(lt.value for lt in self.layout_types)

        # page_numbers is a JSON array on the row; use the first page as
        # the canonical position. This is consistent with how chunks are
        # built (lowest page first).
        if self.min_page is not None:
            clauses.append("json_extract(c.page_numbers, '$[0]') >= ?")
            params.append(self.min_page)
        if self.max_page is not None:
            clauses.append("json_extract(c.page_numbers, '$[0]') <= ?")
            params.append(self.max_page)

        if not clauses:
            return "", []
        return " AND " + " AND ".join(clauses), params


class SearchResult(BaseModel):
    """A single retrieved chunk with its score and provenance."""

    chunk_id: str
    score: float
    chunk: Chunk
    match_source: Literal["vector", "keyword", "hybrid"]


def _vector_to_bytes(vector: Sequence[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def vector_search(
    conn: sqlite3.Connection,
    query_embedding: Sequence[float],
    k: int = 10,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """KNN search against ``chunk_vecs`` (``sqlite-vec``).

    Scoring
    -------
    ``sqlite-vec`` returns the raw distance (cosine by default for vec0
    when vectors are L2-normalized). We map it to a bounded "larger is
    better" score with ``1.0 / (1.0 + distance)`` so the result list is
    comparable to the keyword branch.

    Filtering
    ---------
    ``vec0`` doesn't natively support metadata pre-filtering in the
    version pinned by ``sqlite-vec``. When ``filters`` are supplied we
    over-fetch (``max(k * 5, 50)``) from ``chunk_vecs`` and re-rank /
    truncate after JOINing on ``chunks`` with the WHERE filter applied.
    """
    embedding_bytes = _vector_to_bytes(query_embedding)

    if filters is None:
        rows = conn.execute(
            """
            SELECT chunk_id, distance
            FROM chunk_vecs
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (embedding_bytes, int(k)),
        ).fetchall()
    else:
        over_fetch = max(k * 5, 50)
        where_sql, where_params = filters._where_clause()
        # NOTE: vec0 doesn't support metadata pre-filtering, so we
        # over-fetch from the vector index then post-filter on chunks.
        sql = f"""
            SELECT v.chunk_id AS chunk_id, v.distance AS distance
            FROM chunk_vecs v
            JOIN chunks c ON c.id = v.chunk_id
            WHERE v.embedding MATCH ? AND v.k = ?
            {where_sql}
            ORDER BY v.distance
            LIMIT ?
        """
        params: list[object] = [embedding_bytes, int(over_fetch)]
        params.extend(where_params)
        params.append(int(k))
        rows = conn.execute(sql, params).fetchall()

    results: list[SearchResult] = []
    for row in rows:
        chunk = get_chunk(conn, row["chunk_id"])
        if chunk is None:
            continue
        distance = float(row["distance"])
        score = 1.0 / (1.0 + distance)
        results.append(
            SearchResult(
                chunk_id=row["chunk_id"],
                score=score,
                chunk=chunk,
                match_source="vector",
            )
        )
    return results


# FTS5 reserves these characters; the simplest robust escape is to wrap
# each whitespace-separated term in double quotes.
_FTS_SPLIT = re.compile(r"\s+")


def _escape_fts_query(query: str) -> str:
    """Quote each term so FTS5 treats them as literals.

    Handles embedded double quotes by doubling them (FTS5's own escape).
    Returns an empty string if the query is whitespace-only.
    """
    terms = [t for t in _FTS_SPLIT.split(query.strip()) if t]
    if not terms:
        return ""
    quoted = [f'"{t.replace(chr(34), chr(34) * 2)}"' for t in terms]
    return " ".join(quoted)


def keyword_search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 10,
    filters: SearchFilters | None = None,
) -> list[SearchResult]:
    """BM25 search via the FTS5 ``chunk_fts`` virtual table.

    Scoring
    -------
    SQLite FTS5's ``bm25()`` returns *negative* numbers where lower
    (more negative) = better. We flip the sign so the returned ``score``
    is "larger = better" and consistent with :func:`vector_search`.

    Query escaping
    --------------
    Tokens are wrapped in double quotes to dodge FTS5 operators such as
    ``*``, ``:``, ``"``. This is important for technical queries like
    ``I2C: 3.3V*`` which would otherwise raise a parse error.
    """
    escaped = _escape_fts_query(query)
    if not escaped:
        return []

    where_sql = ""
    extra_params: list[object] = []
    if filters is not None:
        where_sql, extra_params = filters._where_clause()

    sql = f"""
        SELECT c.*, bm25(chunk_fts) AS bm25_score
        FROM chunk_fts
        JOIN chunks c ON c.rowid = chunk_fts.rowid
        WHERE chunk_fts MATCH ?
        {where_sql}
        ORDER BY bm25_score
        LIMIT ?
    """
    params: list[object] = [escaped]
    params.extend(extra_params)
    params.append(int(k))

    rows = conn.execute(sql, params).fetchall()

    # Late import to avoid a cycle (sqlite.py imports nothing from here).
    from aws_rag.store.sqlite import _row_to_chunk

    results: list[SearchResult] = []
    for row in rows:
        chunk = _row_to_chunk(row)
        # Flip sign: FTS5 bm25 is negative-better, we want positive-better.
        score = -float(row["bm25_score"])
        results.append(
            SearchResult(
                chunk_id=chunk.id,
                score=score,
                chunk=chunk,
                match_source="keyword",
            )
        )
    return results


def hybrid_search(
    conn: sqlite3.Connection,
    query_embedding: Sequence[float],
    query_text: str,
    k: int = 10,
    filters: SearchFilters | None = None,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion of vector + keyword results.

    For each candidate ``c`` we compute::

        score(c) = vector_weight  / (rrf_k + vec_rank(c))
                 + keyword_weight / (rrf_k + kw_rank(c))

    where ranks are **1-indexed** (best result has rank 1). If a
    candidate appears in only one branch the missing term contributes
    zero. The constant ``rrf_k`` (default 60, the value from the
    original RRF paper) flattens differences at the head of the list.

    We over-fetch ``max(k * 4, 40)`` from each branch to give the fusion
    room to promote items that ranked weakly on one side but strongly on
    the other.
    """
    over_fetch = max(k * 4, 40)

    vec_results = vector_search(conn, query_embedding, k=over_fetch, filters=filters)
    kw_results = keyword_search(conn, query_text, k=over_fetch, filters=filters)

    ranks: dict[str, dict[str, int | None]] = {}
    for i, r in enumerate(vec_results, start=1):
        ranks.setdefault(r.chunk_id, {"vec_rank": None, "kw_rank": None})
        ranks[r.chunk_id]["vec_rank"] = i
    for i, r in enumerate(kw_results, start=1):
        ranks.setdefault(r.chunk_id, {"vec_rank": None, "kw_rank": None})
        ranks[r.chunk_id]["kw_rank"] = i

    fused: list[tuple[str, float]] = []
    for chunk_id, r in ranks.items():
        score = 0.0
        if r["vec_rank"] is not None:
            score += vector_weight / (rrf_k + r["vec_rank"])
        if r["kw_rank"] is not None:
            score += keyword_weight / (rrf_k + r["kw_rank"])
        fused.append((chunk_id, score))

    fused.sort(key=lambda x: x[1], reverse=True)

    results: list[SearchResult] = []
    for chunk_id, score in fused[:k]:
        chunk = get_chunk(conn, chunk_id)
        if chunk is None:
            continue
        results.append(
            SearchResult(
                chunk_id=chunk_id,
                score=score,
                chunk=chunk,
                match_source="hybrid",
            )
        )
    return results
