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

from datasheet_rag.models.chunk import (
    Chunk,
    ChunkGraph,
    ChunkLevel,
    ChunkMetadata,
    LayoutType,
)
from datasheet_rag.store.metadata import (
    apply_metadata_to_chunks,
    delete_metadata,
    get_metadata,
    list_docs,
    set_metadata,
)
from datasheet_rag.store.schema import _check_embedding_dim, connect, init_schema
from datasheet_rag.store.search import (
    SearchFilters,
    hybrid_search,
    keyword_search,
    vector_search,
)
from datasheet_rag.store.sqlite import (
    count_chunks,
    delete_doc,
    get_chunk,
    insert_chunk_graph,
    insert_chunks,
    insert_chunks_stats,
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
            text="I2C interface specification overview chapter.",
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


def test_delete_metadata(conn: sqlite3.Connection) -> None:
    # doc_metadata is a sidecar (not FK'd to chunks), so it needs its own
    # delete — this covers the gap `delete_doc`'s cascade doesn't reach.
    set_metadata(conn, "doc1", mpn="M1", manufacturer="Acme")
    assert get_metadata(conn, "doc1") is not None

    deleted = delete_metadata(conn, "doc1")
    assert deleted == 1
    assert get_metadata(conn, "doc1") is None

    # No-op (returns 0) when there's no sidecar row for that doc_id.
    assert delete_metadata(conn, "doc1") == 0


# ---------------------------------------------------------------------------
# 3b. Pruning a re-ingest that re-chunked (GH #44)
# ---------------------------------------------------------------------------


def _shorter_doc1_graph() -> ChunkGraph:
    """doc1 re-chunked into 3 chunks — the seed's last two ids disappear.

    This is what a real re-ingest does: ids are positional, so dropping a
    figure or changing --micro-tokens renumbers everything after it and the
    tail of the previous graph has no counterpart in the new one.
    """
    graph = ChunkGraph(doc_id="doc1")
    for chunk in _seed_chunks():
        if chunk.doc_id == "doc1" and chunk.id not in ("doc1:2:2", "doc1:2:3"):
            graph.add(chunk)
    return graph


def test_prune_drops_chunks_the_new_graph_does_not_carry(
    conn: sqlite3.Connection,
) -> None:
    chunks = _seed_chunks()
    insert_chunks(conn, chunks, vectors=_seed_vectors(chunks))
    assert count_chunks(conn, doc_id="doc1") == 5

    graph = _shorter_doc1_graph()
    stats = insert_chunk_graph(conn, graph, prune=True)
    assert stats.inserted == 3
    assert stats.pruned == 2

    assert count_chunks(conn, doc_id="doc1") == 3
    assert get_chunk(conn, "doc1:2:2") is None
    assert get_chunk(conn, "doc1:2:3") is None

    # The AFTER DELETE trigger has to have carried the prune into both
    # indexes, or the rows stay searchable — the actual bug in GH #44.
    vec_rows = conn.execute("SELECT COUNT(*) AS n FROM chunk_vecs").fetchone()["n"]
    assert vec_rows == 6  # 3 surviving doc1 + 3 doc2
    assert keyword_search(conn, "quiescent current", k=10) == []


def test_prune_leaves_other_documents_alone(conn: sqlite3.Connection) -> None:
    chunks = _seed_chunks()
    insert_chunks(conn, chunks, vectors=_seed_vectors(chunks))

    insert_chunk_graph(conn, _shorter_doc1_graph(), prune=True)

    # Only the doc_ids the incoming chunks cover are pruned.
    assert count_chunks(conn, doc_id="doc2") == 3
    assert get_chunk(conn, "doc2:2:1") is not None


def test_insert_chunks_does_not_prune_by_default(conn: sqlite3.Connection) -> None:
    # insert_chunks takes any iterable, so it must assume a partial one: a
    # caller shipping a handful of chunks must not wipe the rest of the doc.
    chunks = _seed_chunks()
    insert_chunks(conn, chunks, vectors=_seed_vectors(chunks))

    survivors = [c for c in chunks if c.id not in ("doc1:2:2", "doc1:2:3")]
    stats = insert_chunks_stats(conn, survivors)
    assert stats.pruned == 0
    assert count_chunks(conn, doc_id="doc1") == 5


def test_insert_chunk_graph_prunes_without_being_asked(
    conn: sqlite3.Connection,
) -> None:
    # A ChunkGraph is a whole document, so the cleanup is the default —
    # a caller that forgets to ask for it would re-open GH #44.
    chunks = _seed_chunks()
    insert_chunks(conn, chunks, vectors=_seed_vectors(chunks))

    stats = insert_chunk_graph(conn, _shorter_doc1_graph())
    assert stats.pruned == 2
    assert count_chunks(conn, doc_id="doc1") == 3

    # ...and a deliberately partial graph can still opt out.
    insert_chunks(conn, chunks, vectors=_seed_vectors(chunks))
    stats = insert_chunk_graph(conn, _shorter_doc1_graph(), prune=False)
    assert stats.pruned == 0
    assert count_chunks(conn, doc_id="doc1") == 5


def test_prune_keeps_curated_fields_on_surviving_chunks(
    conn: sqlite3.Connection,
) -> None:
    # Pruning must not become "delete the doc, then re-insert": everything the
    # upsert deliberately preserves for ids that survive has to still be there.
    chunks = _seed_chunks()
    insert_chunks(conn, chunks)
    conn.execute(
        "UPDATE chunks SET figure_description = ?, figure_image_path = ? "
        "WHERE id = 'doc1:2:0'",
        ("A described figure", "doc1/fig.png"),
    )
    conn.commit()

    # The fresh graph carries neither field, exactly like a re-chunk.
    insert_chunk_graph(conn, _shorter_doc1_graph(), prune=True)

    survivor = get_chunk(conn, "doc1:2:0")
    assert survivor is not None
    assert survivor.figure_description == "A described figure"
    assert survivor.figure_image_path == "doc1/fig.png"


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


def _append_mpn_aliases(
    conn: sqlite3.Connection, doc_id: str, *aliases: str
) -> str:
    """Replicate the CLI --mpn-alias merge logic in isolation."""
    existing = get_metadata(conn, doc_id)
    base = existing.mpn if existing else None
    all_mpns: list[str] = [t.strip() for t in base.split(",") if t.strip()] if base else []
    for alias in aliases:
        if alias not in all_mpns:
            all_mpns.append(alias)
    merged = ",".join(all_mpns)
    set_metadata(conn, doc_id, mpn=merged)
    return merged


def test_mpn_alias_append_and_dedup(conn: sqlite3.Connection) -> None:
    set_metadata(conn, "doc-x", mpn="INA226")

    # Append two aliases
    _append_mpn_aliases(conn, "doc-x", "INA226A", "INA226B")
    md = get_metadata(conn, "doc-x")
    assert md is not None
    assert md.mpn == "INA226,INA226A,INA226B"

    # Appending a duplicate is a no-op
    _append_mpn_aliases(conn, "doc-x", "INA226A")
    md = get_metadata(conn, "doc-x")
    assert md is not None
    assert md.mpn == "INA226,INA226A,INA226B"

    # All three aliases are now findable via list_docs
    for token in ("INA226", "INA226A", "INA226B"):
        matches = list_docs(conn, mpn=token)
        assert any(m.doc_id == "doc-x" for m in matches), f"token {token!r} not found"


def test_list_docs_mpn_filter_comma_separated(conn: sqlite3.Connection) -> None:
    set_metadata(conn, "doc-a", mpn="INA226")
    set_metadata(conn, "doc-b", mpn="INA226,INA226A")
    set_metadata(conn, "doc-c", mpn="INA226A,INA226B")
    set_metadata(conn, "doc-d", mpn="INA226B")
    set_metadata(conn, "doc-e", mpn="OTHER")

    def ids(mpn: str) -> list[str]:
        return [m.doc_id for m in list_docs(conn, mpn=mpn)]

    # exact single match
    assert ids("INA226") == ["doc-a", "doc-b"]
    # alias in the middle / at the end
    assert ids("INA226A") == ["doc-b", "doc-c"]
    # alias only at start of multi-value
    assert ids("INA226B") == ["doc-c", "doc-d"]
    # unrelated MPN returns nothing
    assert ids("NOTHERE") == []
    # token that is a prefix of another MPN must NOT match (INA226 ≠ INA226A)
    assert "doc-c" not in ids("INA226")


def test_apply_metadata_noops_when_missing(conn: sqlite3.Connection) -> None:
    # No sidecar row at all.
    assert apply_metadata_to_chunks(conn, "doc-unknown") == 0
    # Sidecar exists but no project / group set.
    set_metadata(conn, "doc-bare", mpn="X")
    assert apply_metadata_to_chunks(conn, "doc-bare") == 0


# ---------------------------------------------------------------------------
# Figure persistence (schema v2 — figure_image_path + figure_description)
# ---------------------------------------------------------------------------


def _make_figure_chunk(
    chunk_id: str = "doc-fig:2:0",
    *,
    image_path: str | None = "/tmp/figs/fig-1.png",
    description: str | None = None,
    caption: str = "Figure 3-2: SPI4 timing diagram",
) -> Chunk:
    md = ChunkMetadata(
        doc_id="doc-fig",
        doc_title="STM32H7 RM",
        chapter_title="SPI",
        section_title="Timing",
        page_numbers=[42],
        layout_type=LayoutType.FIGURE,
    )
    return Chunk(
        id=chunk_id,
        doc_id="doc-fig",
        level=ChunkLevel.MICRO,
        text="[Figure]",
        context_text="SPI > Timing > [Figure] " + caption,
        token_count=5,
        metadata=md,
        figure_image_path=image_path,
        figure_s3_key="figures/doc-fig/page-42-fig-1.png",
        figure_caption=caption,
        figure_description=description,
    )


def test_figure_columns_round_trip(conn: sqlite3.Connection) -> None:
    """figure_image_path and figure_description must survive insert/fetch."""
    chunk = _make_figure_chunk(
        image_path="/abs/path/cropped.png",
        description="Block diagram showing APB2 → SPI4 TX FIFO → MOSI pad.",
    )
    insert_chunks(conn, [chunk])

    round_tripped = get_chunk(conn, chunk.id)
    assert round_tripped is not None
    assert round_tripped.figure_image_path == "/abs/path/cropped.png"
    assert round_tripped.figure_s3_key == "figures/doc-fig/page-42-fig-1.png"
    assert round_tripped.figure_caption.startswith("Figure 3-2")
    assert round_tripped.figure_description is not None
    assert "APB2" in round_tripped.figure_description


def test_list_figure_chunks_filters_and_image_required(
    conn: sqlite3.Connection,
) -> None:
    from datasheet_rag.store import list_figure_chunks

    fig_with_image = _make_figure_chunk("fig:has-image")
    fig_no_image = _make_figure_chunk("fig:orphan", image_path=None)
    # Strip the S3 key too so this one has no usable source.
    fig_no_image.figure_s3_key = None
    text_chunk = _make_chunk(
        "doc-fig:2:1",
        doc_id="doc-fig",
        level=ChunkLevel.MICRO,
        text="Body paragraph about SPI clock polarity.",
        layout=LayoutType.TEXT,
    )
    insert_chunks(conn, [fig_with_image, fig_no_image, text_chunk])

    # Default: only chunks with a usable image source come back.
    only_with_image = list_figure_chunks(conn)
    ids = {c.id for c in only_with_image}
    assert ids == {"fig:has-image"}

    # When the flag is off, both figure chunks come back (but never the
    # plain text one).
    all_figures = list_figure_chunks(conn, only_with_image=False)
    assert {c.id for c in all_figures} == {"fig:has-image", "fig:orphan"}

    # doc_id filter still applies.
    by_doc = list_figure_chunks(conn, doc_id="does-not-exist")
    assert by_doc == []


def test_update_figure_description_persists_and_folds_into_context(
    conn: sqlite3.Connection,
) -> None:
    from datasheet_rag.store import update_figure_description

    chunk = _make_figure_chunk("fig:to-describe", description=None)
    insert_chunks(conn, [chunk])

    description = "State diagram of the I2C bus: IDLE → START → ADDR → ACK."
    updated = update_figure_description(conn, chunk.id, description)
    assert updated is True

    after = get_chunk(conn, chunk.id)
    assert after is not None
    assert after.figure_description == description
    assert description in after.context_text  # folded in for re-embed

    # Idempotent: re-applying the same description should not duplicate it.
    update_figure_description(conn, chunk.id, description)
    after2 = get_chunk(conn, chunk.id)
    assert after2 is not None
    assert after2.context_text.count(description) == 1

    # Unknown chunk → False, no row touched.
    assert update_figure_description(conn, "nope", "x") is False


def test_schema_v1_database_is_migrated_to_v2() -> None:
    """A database created at schema v1 must gain the new figure columns
    without losing existing rows."""
    from datasheet_rag.store import schema as schema_mod

    # Open a real file-backed DB so we can re-open it. (:memory: dies on close.)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        db_path = f"{td}/v1.sqlite"

        # 1) Create the v1-flavoured DB by temporarily clamping SCHEMA_VERSION
        # and stripping the new columns out of the DDL via monkeypatching.
        real_version = schema_mod.SCHEMA_VERSION
        real_v2_cols = schema_mod._CHUNKS_COLUMNS_BY_VERSION.get(2, [])
        try:
            schema_mod.SCHEMA_VERSION = 1
            schema_mod._CHUNKS_COLUMNS_BY_VERSION = {
                k: v for k, v in schema_mod._CHUNKS_COLUMNS_BY_VERSION.items() if k > 1
            }
            c1 = connect(db_path, embedding_dim=EMBED_DIM)
            # Drop the v2 columns the new DDL bakes in, to truly emulate v1.
            c1.execute("ALTER TABLE chunks DROP COLUMN figure_image_path")
            c1.execute("ALTER TABLE chunks DROP COLUMN figure_description")
            c1.commit()
            c1.close()
        finally:
            schema_mod.SCHEMA_VERSION = real_version
            schema_mod._CHUNKS_COLUMNS_BY_VERSION[2] = real_v2_cols

        # 2) Re-open via the modern connect(): the migration must run and
        # the new columns must appear.
        c2 = connect(db_path, embedding_dim=EMBED_DIM)
        try:
            cols = {row["name"] for row in c2.execute("PRAGMA table_info(chunks)")}
            assert "figure_image_path" in cols
            assert "figure_description" in cols
            version = c2.execute(
                "SELECT version FROM schema_version WHERE id = 1"
            ).fetchone()["version"]
            assert version == schema_mod.SCHEMA_VERSION
        finally:
            c2.close()


# ---------------------------------------------------------------------------
# Figure sources: what the store promises must be what it can deliver (GH #41)
# ---------------------------------------------------------------------------


def test_figure_available_reflects_the_file_on_disk(
    conn: sqlite3.Connection, tmp_path: object
) -> None:
    from datasheet_rag.store.sqlite import figure_source_available

    real = tmp_path / "fig.png"  # type: ignore[operator]
    real.write_bytes(b"\x89PNGFAKE")

    here = _make_figure_chunk("fig:here", image_path=str(real))
    gone = _make_figure_chunk("fig:gone", image_path=str(tmp_path / "nope.png"))  # type: ignore[operator]
    gone.figure_s3_key = None
    bare = _make_figure_chunk("fig:bare", image_path=None)
    bare.figure_s3_key = None
    s3_only = _make_figure_chunk("fig:s3", image_path=None)  # keeps its s3 key
    text = _make_chunk(
        "doc-fig:2:9", doc_id="doc-fig", level=ChunkLevel.MICRO,
        text="Plain body text.", layout=LayoutType.TEXT,
    )
    insert_chunks(conn, [here, gone, bare, s3_only, text])

    assert get_chunk(conn, "fig:here").figure_available is True
    assert get_chunk(conn, "fig:gone").figure_available is False
    assert get_chunk(conn, "fig:bare").figure_available is False
    # An S3 key counts without a round-trip to the bucket.
    assert get_chunk(conn, "fig:s3").figure_available is True
    # Non-figure chunks are not stat'd at all.
    assert get_chunk(conn, "doc-fig:2:9").figure_available is None

    assert figure_source_available(str(real), None) is True
    assert figure_source_available(None, None) is False


def test_reinsert_without_a_figure_source_keeps_the_stored_one(
    conn: sqlite3.Connection,
) -> None:
    """A re-embed from a figure-less chunk graph must not strip images."""
    original = _make_figure_chunk("fig:keep", image_path="/tmp/figs/fig-1.png")
    insert_chunks(conn, [original])

    # Same chunk id, re-chunked with figures skipped: no path, no key, no caption.
    stripped = _make_figure_chunk("fig:keep", image_path=None, caption="")
    stripped.figure_s3_key = None
    insert_chunks(conn, [stripped])

    after = get_chunk(conn, "fig:keep")
    assert after is not None
    assert after.figure_image_path == "/tmp/figs/fig-1.png"
    assert after.figure_s3_key == "figures/doc-fig/page-42-fig-1.png"
    assert after.figure_caption.startswith("Figure 3-2")

    # A genuine re-crop still lands — it arrives with a path.
    recropped = _make_figure_chunk("fig:keep", image_path="/tmp/figs/fig-1-v2.png")
    insert_chunks(conn, [recropped])
    assert get_chunk(conn, "fig:keep").figure_image_path == "/tmp/figs/fig-1-v2.png"


def test_set_figure_source_relinks_and_preserves_a_curated_caption(
    conn: sqlite3.Connection, tmp_path: object
) -> None:
    from datasheet_rag.store import set_figure_source

    img = tmp_path / "relinked.png"  # type: ignore[operator]
    img.write_bytes(b"\x89PNGFAKE")

    orphan = _make_figure_chunk("fig:orphan", image_path=None, caption="")
    orphan.figure_s3_key = None
    curated = _make_figure_chunk("fig:curated", image_path=None, caption="Hand-written")
    curated.figure_s3_key = None
    insert_chunks(conn, [orphan, curated])

    assert set_figure_source(
        conn, "fig:orphan", image_path=img, caption="Figure 9: Power tree"
    ) is True
    assert set_figure_source(
        conn, "fig:curated", image_path=img, caption="Figure 9: Power tree"
    ) is True
    assert set_figure_source(conn, "fig:nope", image_path=img) is False

    relinked = get_chunk(conn, "fig:orphan")
    assert relinked.figure_available is True
    assert relinked.figure_caption == "Figure 9: Power tree"
    # An existing caption wins — a repair reattaches images, not prose.
    assert get_chunk(conn, "fig:curated").figure_caption == "Hand-written"
