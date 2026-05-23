"""End-to-end tests for the SQLite chunk + vector store.

These tests use an in-memory database (``:memory:``) and the real
``sqlite-vec`` extension — no mocks. They verify the contracts that
the rest of the pipeline depends on:

* schema bootstrap is idempotent and dimension-aware
* chunks round-trip (including metadata + nav links)
* delete cascades through FTS5 and vec0
* vector / keyword / hybrid search produce the expected ordering
* metadata filters apply end-to-end
* the doc_metadata sidecar back-fills chunks correctly
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from aws_rag.models.chunk import (
    Chunk,
    ChunkLevel,
    ChunkMetadata,
    LayoutType,
)
from aws_rag.store.metadata import (
    apply_metadata_to_chunks,
    get_metadata,
    list_docs,
    set_metadata,
)
from aws_rag.store.schema import _check_embedding_dim, connect, init_schema
from aws_rag.store.search import (
    SearchFilters,
    hybrid_search,
    keyword_search,
    vector_search,
)
from aws_rag.store.sqlite import (
    count_chunks,
    delete_doc,
    get_chunk,
    insert_chunks,
)

# Small embedding dim keeps the tests fast and makes hand-built unit
# vectors easy to reason about.
EMBED_DIM = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    """Fresh in-memory store with schema bootstrapped at EMBED_DIM."""
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    try:
        yield c
    finally:
        c.close()


def _unit_vec(position: int, dim: int = EMBED_DIM) -> list[float]:
    """One-hot vector with the 1 at ``position`` — handy for KNN tests."""
    v = [0.0] * dim
    v[position % dim] = 1.0
    return v


def _make_chunk(
    chunk_id: str,
    *,
    doc_id: str,
    level: ChunkLevel,
    text: str,
    context_text: str | None = None,
    page: int = 1,
    layout: LayoutType = LayoutType.TEXT,
    chapter: str = "",
    section: str = "",
    parent_id: str | None = None,
    prev_id: str | None = None,
    next_id: str | None = None,
    chapter_root_id: str | None = None,
) -> Chunk:
    md = ChunkMetadata(
        doc_id=doc_id,
        doc_title=f"Doc {doc_id}",
        chapter_title=chapter,
        section_title=section,
        page_numbers=[page],
        layout_type=layout,
        context_string=f"{chapter} > {section}".strip(" >"),
    )
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        level=level,
        text=text,
        context_text=context_text or text,
        token_count=len(text.split()),
        metadata=md,
        parent_id=parent_id,
        prev_id=prev_id,
        next_id=next_id,
        chapter_root_id=chapter_root_id,
    )


def _seed_chunks() -> list[Chunk]:
    """Eight deterministic, semantically-different chunks for retrieval tests."""
    return [
        _make_chunk(
            "doc1:0:0",
            doc_id="doc1",
            level=ChunkLevel.MACRO,
            text="I2C clock stretching specification overview chapter.",
            page=1,
            chapter="I2C Interface",
        ),
        _make_chunk(
            "doc1:2:0",
            doc_id="doc1",
            level=ChunkLevel.MICRO,
            text="The SCL line may be held low to indicate clock stretching by the slave device.",
            page=2,
            chapter="I2C Interface",
            section="Timing",
            parent_id="doc1:0:0",
            chapter_root_id="doc1:0:0",
        ),
        _make_chunk(
            "doc1:2:1",
            doc_id="doc1",
            level=ChunkLevel.MICRO,
            text="ESD HBM rating is 2 kV minimum per JEDEC JS-001.",
            page=3,
            chapter="ESD Protection",
            section="Ratings",
            layout=LayoutType.TABLE,
            parent_id="doc1:0:0",
            chapter_root_id="doc1:0:0",
            prev_id="doc1:2:0",
        ),
        _make_chunk(
            "doc1:2:2",
            doc_id="doc1",
            level=ChunkLevel.MICRO,
            text="Dropout voltage curve at full load current versus temperature.",
            page=4,
            chapter="LDO Performance",
            section="Dropout",
            layout=LayoutType.FIGURE,
            parent_id="doc1:0:0",
            chapter_root_id="doc1:0:0",
            prev_id="doc1:2:1",
        ),
        _make_chunk(
            "doc1:2:3",
            doc_id="doc1",
            level=ChunkLevel.MICRO,
            text="Quiescent current is 50 microamps typical at 25 degrees Celsius.",
            page=5,
            chapter="LDO Performance",
            section="Quiescent",
            parent_id="doc1:0:0",
            chapter_root_id="doc1:0:0",
        ),
        _make_chunk(
            "doc2:0:0",
            doc_id="doc2",
            level=ChunkLevel.MACRO,
            text="Buck converter reference design and bill of materials.",
            page=1,
            chapter="Buck Converter",
        ),
        _make_chunk(
            "doc2:2:0",
            doc_id="doc2",
            level=ChunkLevel.MICRO,
            text="Switching frequency is selectable between 500 kHz and 2 MHz via the FSEL pin.",
            page=2,
            chapter="Buck Converter",
            section="Operation",
            parent_id="doc2:0:0",
            chapter_root_id="doc2:0:0",
        ),
        _make_chunk(
            "doc2:2:1",
            doc_id="doc2",
            level=ChunkLevel.MICRO,
            text="Thermal shutdown engages at 150 degrees Celsius junction temperature.",
            page=3,
            chapter="Buck Converter",
            section="Protection",
            parent_id="doc2:0:0",
            chapter_root_id="doc2:0:0",
            prev_id="doc2:2:0",
        ),
    ]


def _seed_vectors(chunks: list[Chunk]) -> dict[str, list[float]]:
    """Map each chunk to a distinct one-hot vector so KNN order is predictable."""
    return {chunk.id: _unit_vec(i) for i, chunk in enumerate(chunks)}


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------


def test_schema_tables_created(conn: sqlite3.Connection) -> None:
    names = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        ).fetchall()
    }
    for required in {"chunks", "chunk_vecs", "chunk_fts", "doc_metadata", "schema_version"}:
        assert required in names, f"missing table: {required}"


def test_init_schema_idempotent(conn: sqlite3.Connection) -> None:
    # Calling init_schema again on the same connection must not raise.
    init_schema(conn, embedding_dim=EMBED_DIM)
    init_schema(conn, embedding_dim=EMBED_DIM)
    # Still exactly one row in schema_version.
    rows = conn.execute("SELECT COUNT(*) AS n FROM schema_version").fetchone()
    assert rows["n"] == 1


def test_check_embedding_dim_mismatch_raises(conn: sqlite3.Connection) -> None:
    # Schema was created with EMBED_DIM. Requesting a different dim must fail.
    with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
        _check_embedding_dim(conn, EMBED_DIM + 4)


# ---------------------------------------------------------------------------
# 2. Insert + get_chunk round-trip
# ---------------------------------------------------------------------------


def test_insert_chunks_and_round_trip(conn: sqlite3.Connection) -> None:
    chunks = _seed_chunks()
    vectors = _seed_vectors(chunks)

    n = insert_chunks(conn, chunks, vectors=vectors, project_id="proj-A")
    assert n == len(chunks)
    assert count_chunks(conn) == len(chunks)
    assert count_chunks(conn, doc_id="doc1") == 5
    assert count_chunks(conn, project_id="proj-A") == len(chunks)
    assert count_chunks(conn, doc_id="doc2", project_id="proj-A") == 3

    # Round-trip a chunk with non-default fields.
    fetched = get_chunk(conn, "doc1:2:1")
    assert fetched is not None
    assert fetched.text.startswith("ESD HBM rating")
    assert fetched.metadata.layout_type == LayoutType.TABLE
    assert fetched.metadata.page_numbers == [3]
    assert fetched.metadata.chapter_title == "ESD Protection"
    assert fetched.parent_id == "doc1:0:0"
    assert fetched.prev_id == "doc1:2:0"
    assert fetched.chapter_root_id == "doc1:0:0"
    assert fetched.level == ChunkLevel.MICRO
    assert fetched.children_ids == []  # not denormalized — documented behavior

    # Missing chunk returns None.
    assert get_chunk(conn, "does-not-exist") is None


# ---------------------------------------------------------------------------
# 3. Delete cascades through vec0 and FTS5
# ---------------------------------------------------------------------------


def test_delete_doc_cascades(conn: sqlite3.Connection) -> None:
    chunks = _seed_chunks()
    vectors = _seed_vectors(chunks)
    insert_chunks(conn, chunks, vectors=vectors)

    before = conn.execute("SELECT COUNT(*) AS n FROM chunk_vecs").fetchone()["n"]
    assert before == len(chunks)

    deleted = delete_doc(conn, "doc1")
    assert deleted == 5
    assert count_chunks(conn, doc_id="doc1") == 0
    assert count_chunks(conn) == 3  # only doc2 left

    # vec0 should have lost the doc1 rows via the AFTER DELETE trigger.
    after = conn.execute("SELECT COUNT(*) AS n FROM chunk_vecs").fetchone()["n"]
    assert after == 3

    # FTS5 must no longer return doc1 results.
    kw_hits = keyword_search(conn, "clock stretching", k=10)
    assert all(r.chunk.doc_id != "doc1" for r in kw_hits)


# ---------------------------------------------------------------------------
# 4. Vector search ordering
# ---------------------------------------------------------------------------


def test_vector_search_ordering(conn: sqlite3.Connection) -> None:
    chunks = _seed_chunks()
    vectors = _seed_vectors(chunks)
    insert_chunks(conn, chunks, vectors=vectors)

    # Query == vector of chunk index 2 ("ESD HBM rating") — exact match.
    target = chunks[2]
    results = vector_search(conn, vectors[target.id], k=3)
    assert len(results) >= 1
    assert results[0].chunk_id == target.id
    assert results[0].match_source == "vector"
    # Score transform 1/(1+distance) — exact match → score == 1.0.
    assert results[0].score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. Keyword search
# ---------------------------------------------------------------------------


def test_keyword_search_finds_phrase(conn: sqlite3.Connection) -> None:
    chunks = _seed_chunks()
    insert_chunks(conn, chunks, vectors=_seed_vectors(chunks))

    results = keyword_search(conn, "clock stretching", k=5)
    assert results, "expected at least one BM25 hit"
    assert results[0].chunk_id == "doc1:2:0"
    assert results[0].match_source == "keyword"
    # Flipped sign → score must be positive for a match.
    assert results[0].score > 0


def test_keyword_search_escapes_special_chars(conn: sqlite3.Connection) -> None:
    chunks = _seed_chunks()
    insert_chunks(conn, chunks, vectors=_seed_vectors(chunks))
    # These characters would crash a raw FTS5 query but our escape wraps
    # each term in quotes.
    results = keyword_search(conn, 'I2C: "stretching"*', k=5)
    # We don't assert a specific result here, only that it doesn't raise
    # and returns a list.
    assert isinstance(results, list)


def test_keyword_search_empty_query(conn: sqlite3.Connection) -> None:
    chunks = _seed_chunks()
    insert_chunks(conn, chunks, vectors=_seed_vectors(chunks))
    assert keyword_search(conn, "   ", k=5) == []


# ---------------------------------------------------------------------------
# 6. Hybrid search
# ---------------------------------------------------------------------------


def test_hybrid_search_promotes_dual_match(conn: sqlite3.Connection) -> None:
    chunks = _seed_chunks()
    vectors = _seed_vectors(chunks)
    insert_chunks(conn, chunks, vectors=vectors)

    # The query text matches the "clock stretching" chunk via BM25, and
    # we hand it the vector for that same chunk. RRF should rank it #1.
    target = next(c for c in chunks if c.id == "doc1:2:0")
    results = hybrid_search(
        conn,
        vectors[target.id],
        "clock stretching",
        k=5,
    )
    assert results, "hybrid_search returned nothing"
    assert results[0].chunk_id == target.id
    assert results[0].match_source == "hybrid"
    # Scores must be sorted descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 7. Filters apply end-to-end
# ---------------------------------------------------------------------------


def test_filters_restrict_searches(conn: sqlite3.Connection) -> None:
    chunks = _seed_chunks()
    vectors = _seed_vectors(chunks)
    # Insert with project_id baked onto rows so we can filter by it.
    doc1_chunks = [c for c in chunks if c.doc_id == "doc1"]
    doc2_chunks = [c for c in chunks if c.doc_id == "doc2"]
    insert_chunks(conn, doc1_chunks, vectors=vectors, project_id="alpha")
    insert_chunks(conn, doc2_chunks, vectors=vectors, project_id="beta")

    # Vector filter — only doc1 / alpha may appear.
    target = doc1_chunks[1]
    filt = SearchFilters(project_id="alpha")
    vec_hits = vector_search(conn, vectors[target.id], k=10, filters=filt)
    assert vec_hits
    assert all(r.chunk.doc_id == "doc1" for r in vec_hits)

    # Keyword filter — restrict by doc_id.
    kw_hits = keyword_search(
        conn,
        "frequency",
        k=10,
        filters=SearchFilters(doc_ids=["doc2"]),
    )
    assert kw_hits
    assert all(r.chunk.doc_id == "doc2" for r in kw_hits)

    # Hybrid filter — restrict by level.
    hyb_hits = hybrid_search(
        conn,
        vectors[target.id],
        "clock stretching",
        k=10,
        filters=SearchFilters(level=ChunkLevel.MICRO),
    )
    assert hyb_hits
    assert all(r.chunk.level == ChunkLevel.MICRO for r in hyb_hits)

    # Page filter — only pages 1-2.
    page_hits = vector_search(
        conn,
        vectors[target.id],
        k=10,
        filters=SearchFilters(min_page=1, max_page=2),
    )
    for r in page_hits:
        first_page = r.chunk.metadata.page_numbers[0]
        assert 1 <= first_page <= 2


# ---------------------------------------------------------------------------
# 8. Metadata sidecar
# ---------------------------------------------------------------------------


def test_metadata_set_get_update_and_apply(conn: sqlite3.Connection) -> None:
    # Insert chunks for doc1 first WITHOUT a project_id.
    chunks = _seed_chunks()
    doc1_chunks = [c for c in chunks if c.doc_id == "doc1"]
    insert_chunks(conn, doc1_chunks, vectors=_seed_vectors(chunks))
    assert count_chunks(conn, project_id="proj-1") == 0

    # set_metadata creates a row.
    md = set_metadata(
        conn,
        "doc1",
        project_id="proj-1",
        group_name="grp-1",
        mpn="MPN-123",
        manufacturer="Acme",
        tags=["analog", "ldo"],
        attributes={"revision": "B", "notes": "draft"},
    )
    assert md.project_id == "proj-1"
    assert md.tags == ["analog", "ldo"]
    assert md.attributes == {"revision": "B", "notes": "draft"}
    assert md.updated_at is not None

    # get_metadata round-trips.
    fetched = get_metadata(conn, "doc1")
    assert fetched is not None
    assert fetched.mpn == "MPN-123"
    assert fetched.manufacturer == "Acme"

    # Partial update — only mpn changes, attributes deep-merge,
    # `notes` is removed via None sentinel.
    updated = set_metadata(
        conn,
        "doc1",
        mpn="MPN-456",
        attributes={"revision": "C", "notes": None, "extra": "yes"},
    )
    assert updated.mpn == "MPN-456"
    # Manufacturer was not passed -> preserved.
    assert updated.manufacturer == "Acme"
    # Project / group preserved.
    assert updated.project_id == "proj-1"
    # Attributes merged: revision overwritten, notes dropped, extra added.
    assert updated.attributes == {"revision": "C", "extra": "yes"}
    assert updated.tags == ["analog", "ldo"]

    # list_docs by project includes doc1.
    listed = list_docs(conn, project_id="proj-1")
    assert [m.doc_id for m in listed] == ["doc1"]
    assert list_docs(conn, project_id="non-existent") == []

    # apply_metadata_to_chunks back-fills the rows.
    n_updated = apply_metadata_to_chunks(conn, "doc1")
    assert n_updated == len(doc1_chunks)
    assert count_chunks(conn, project_id="proj-1") == len(doc1_chunks)

    # And searches now respect the back-filled project.
    target = doc1_chunks[0]
    vectors = _seed_vectors(chunks)
    hits = vector_search(
        conn,
        vectors[target.id],
        k=10,
        filters=SearchFilters(project_id="proj-1"),
    )
    assert hits
    assert all(r.chunk.doc_id == "doc1" for r in hits)


def test_apply_metadata_noops_when_missing(conn: sqlite3.Connection) -> None:
    # No sidecar row at all.
    assert apply_metadata_to_chunks(conn, "doc-unknown") == 0
    # Sidecar exists but no project / group set.
    set_metadata(conn, "doc-bare", mpn="X")
    assert apply_metadata_to_chunks(conn, "doc-bare") == 0
