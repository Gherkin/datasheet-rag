"""Tests for the MCP server tool implementations.

We test the ``_impl`` functions directly — they take ``conn`` and
``embedder`` as kwargs so we can bypass the module-level singletons and
the FastMCP transport. This keeps tests fast and free of the ``mcp``
SDK dependency.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aws_rag.mcp.server import (
    _get_chunk_impl,
    _get_document_metadata_impl,
    _list_documents_impl,
    _navigate_impl,
    _search_impl,
    _stats_impl,
)
from aws_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from aws_rag.store import (
    connect,
    insert_chunks,
    set_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMB_DIM = 8  # small but valid for sqlite-vec virtual table


def _vec(slot: int) -> list[float]:
    """One-hot vector at `slot` — makes cosine distance deterministic."""
    v = [0.0] * EMB_DIM
    v[slot % EMB_DIM] = 1.0
    return v


def _make_chunk(
    chunk_id: str,
    *,
    doc_id: str = "docA",
    level: ChunkLevel = ChunkLevel.MICRO,
    text: str = "",
    section: str = "",
    chapter: str = "",
    page: int = 1,
    parent_id: str | None = None,
    prev_id: str | None = None,
    next_id: str | None = None,
    chapter_root_id: str | None = None,
    layout_type: LayoutType = LayoutType.TEXT,
) -> Chunk:
    md = ChunkMetadata(
        doc_id=doc_id,
        chapter_title=chapter,
        section_title=section,
        page_numbers=[page],
        layout_type=layout_type,
    )
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        level=level,
        text=text or f"text for {chunk_id}",
        context_text=text or f"text for {chunk_id}",
        token_count=10,
        metadata=md,
        parent_id=parent_id,
        prev_id=prev_id,
        next_id=next_id,
        chapter_root_id=chapter_root_id,
    )


@pytest.fixture
def conn() -> Any:
    """In-memory store seeded with a small, deterministic chunk graph.

    Structure (doc=docA, project=p1):
      MACRO m1 ─┐
                ├── MESO   s1 ─── MICRO  c1 ("I2C clock stretching specification")
                │                 MICRO  c2 ("STM32H743 SPI4 CR1 register layout")
                └── MESO   s2 ─── MICRO  c3 ("ESD HBM rating for the input pin")

      docB has a single MICRO c4 in a different project (p2) for filter tests.
    """
    c = connect(":memory:", embedding_dim=EMB_DIM)

    chunks = [
        _make_chunk("m1", level=ChunkLevel.MACRO, chapter="Comm", section="Comm"),
        _make_chunk("s1", level=ChunkLevel.MESO, chapter="Comm", section="I2C",
                    parent_id="m1", chapter_root_id="m1", next_id="s2"),
        _make_chunk("s2", level=ChunkLevel.MESO, chapter="Comm", section="ESD",
                    parent_id="m1", chapter_root_id="m1", prev_id="s1"),
        _make_chunk("c1", text="I2C clock stretching specification",
                    section="I2C", chapter="Comm",
                    parent_id="s1", chapter_root_id="m1", next_id="c2"),
        _make_chunk("c2", text="STM32H743 SPI4 CR1 register layout",
                    section="I2C", chapter="Comm",
                    parent_id="s1", chapter_root_id="m1", prev_id="c1"),
        _make_chunk("c3", text="ESD HBM rating for the input pin",
                    section="ESD", chapter="Comm",
                    parent_id="s2", chapter_root_id="m1"),
        _make_chunk("c4", doc_id="docB",
                    text="dropout voltage versus load current curve",
                    section="LDO", chapter="Power"),
    ]
    vectors = {
        "m1": _vec(0), "s1": _vec(1), "s2": _vec(2),
        "c1": _vec(3),   # I2C
        "c2": _vec(4),   # SPI4
        "c3": _vec(5),   # ESD
        "c4": _vec(6),   # dropout
    }

    # docA gets project p1; docB gets p2 (insert in two calls so project
    # is correctly attached per doc).
    insert_chunks(
        c, [ch for ch in chunks if ch.doc_id == "docA"],
        vectors={k: v for k, v in vectors.items() if k != "c4"},
        project_id="p1",
    )
    insert_chunks(
        c, [ch for ch in chunks if ch.doc_id == "docB"],
        vectors={"c4": vectors["c4"]},
        project_id="p2",
    )
    return c


@pytest.fixture
def fake_embedder() -> Any:
    """Embedder that maps known queries to deterministic vectors."""
    emb = MagicMock()
    routing = {
        "i2c": _vec(3),
        "spi4": _vec(4),
        "esd": _vec(5),
        "dropout": _vec(6),
    }

    def embed_one(text: str) -> list[float]:
        key = text.lower()
        for k, v in routing.items():
            if k in key:
                return v
        return _vec(7)

    emb.embed_one.side_effect = embed_one
    return emb


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_keyword_finds_exact_phrase(conn: Any) -> None:
    out = _search_impl(
        "clock stretching", mode="keyword", k=3,
        project_id="p1", conn=conn,
    )
    assert out, "keyword search returned nothing"
    assert out[0]["chunk_id"] == "c1"


def test_search_vector_uses_embedder_and_orders_by_similarity(
    conn: Any, fake_embedder: Any
) -> None:
    out = _search_impl(
        "i2c question", mode="vector", k=3,
        project_id="p1", conn=conn, embedder=fake_embedder,
    )
    assert out[0]["chunk_id"] == "c1"
    fake_embedder.embed_one.assert_called_once()


def test_search_hybrid_combines_signals(conn: Any, fake_embedder: Any) -> None:
    # A query that hits c2 on both keyword (SPI4 CR1) and vector (slot 4).
    out = _search_impl(
        "SPI4 CR1", mode="hybrid", k=3,
        project_id="p1", conn=conn, embedder=fake_embedder,
    )
    assert out[0]["chunk_id"] == "c2"


def test_search_respects_project_filter(conn: Any, fake_embedder: Any) -> None:
    # The "dropout" chunk lives in project p2 — filtering to p1 must hide it.
    out = _search_impl(
        "dropout voltage", mode="keyword", k=5,
        project_id="p1", conn=conn,
    )
    ids = [r["chunk_id"] for r in out]
    assert "c4" not in ids


def test_search_doc_id_filter_narrows_to_doc(conn: Any) -> None:
    out = _search_impl(
        "dropout voltage", mode="keyword", k=5,
        doc_id="docB", conn=conn,
    )
    assert [r["chunk_id"] for r in out] == ["c4"]


def test_search_level_filter(conn: Any) -> None:
    out = _search_impl(
        "Comm", mode="keyword", k=10, level="macro",
        project_id="p1", conn=conn,
    )
    assert all(r["level"] == "MACRO" for r in out)


def test_search_empty_query_raises(conn: Any) -> None:
    with pytest.raises(ValueError, match="empty"):
        _search_impl("   ", mode="keyword", conn=conn)


def test_search_invalid_level_raises(conn: Any) -> None:
    with pytest.raises(ValueError, match="level"):
        _search_impl("anything", mode="keyword", level="nano", conn=conn)


def test_search_invalid_layout_type_raises(conn: Any) -> None:
    with pytest.raises(ValueError, match="layout_type"):
        _search_impl("anything", mode="keyword",
                     layout_types=["pixelart"], conn=conn)


# ---------------------------------------------------------------------------
# get_chunk
# ---------------------------------------------------------------------------


def test_get_chunk_returns_compact_dict(conn: Any) -> None:
    out = _get_chunk_impl("c1", conn=conn)
    assert out is not None
    assert out["chunk_id"] == "c1"
    assert out["level"] == "MICRO"
    assert out["section"] == "I2C"
    assert out["text"].startswith("I2C clock stretching")
    assert "neighbors" not in out


def test_get_chunk_missing_returns_none(conn: Any) -> None:
    assert _get_chunk_impl("does-not-exist", conn=conn) is None


def test_get_chunk_with_neighbors(conn: Any) -> None:
    out = _get_chunk_impl("c1", include_neighbors=True, conn=conn)
    assert out is not None
    nb = out["neighbors"]
    assert nb["parent"]["chunk_id"] == "s1"
    assert nb["next"]["chunk_id"] == "c2"
    assert nb["prev"] is None  # c1 is first in its section


# ---------------------------------------------------------------------------
# navigate / zoom
# ---------------------------------------------------------------------------


def test_navigate_parent(conn: Any) -> None:
    out = _navigate_impl("c1", "parent", conn=conn)
    assert [r["chunk_id"] for r in out] == ["s1"]


def test_navigate_children_queries_by_parent_id(conn: Any) -> None:
    out = _navigate_impl("s1", "children", conn=conn)
    ids = sorted(r["chunk_id"] for r in out)
    assert ids == ["c1", "c2"]


def test_navigate_chapter_root(conn: Any) -> None:
    out = _navigate_impl("c3", "chapter_root", conn=conn)
    assert [r["chunk_id"] for r in out] == ["m1"]


def test_navigate_next_and_prev(conn: Any) -> None:
    assert [r["chunk_id"] for r in _navigate_impl("c1", "next", conn=conn)] == ["c2"]
    assert [r["chunk_id"] for r in _navigate_impl("c2", "prev", conn=conn)] == ["c1"]


def test_navigate_missing_link_returns_empty(conn: Any) -> None:
    # c3 has no prev_id set
    assert _navigate_impl("c3", "prev", conn=conn) == []


def test_navigate_invalid_direction_raises(conn: Any) -> None:
    with pytest.raises(ValueError, match="direction"):
        _navigate_impl("c1", "sideways", conn=conn)  # type: ignore[arg-type]


def test_navigate_missing_chunk_returns_empty(conn: Any) -> None:
    assert _navigate_impl("nope", "parent", conn=conn) == []


# ---------------------------------------------------------------------------
# documents / metadata / stats
# ---------------------------------------------------------------------------


def test_list_documents_filters_by_project(conn: Any) -> None:
    set_metadata(conn, "docA", project_id="p1", mpn="STM32H743",
                 manufacturer="ST", subsystem="mcu")
    set_metadata(conn, "docB", project_id="p2", mpn="LM317",
                 manufacturer="TI", subsystem="power")

    out = _list_documents_impl(project_id="p1", conn=conn)
    assert [d["doc_id"] for d in out] == ["docA"]
    assert out[0]["mpn"] == "STM32H743"


def test_list_documents_manufacturer_filter(conn: Any) -> None:
    set_metadata(conn, "docA", project_id="p1", manufacturer="ST")
    set_metadata(conn, "docB", project_id="p2", manufacturer="TI")
    out = _list_documents_impl(manufacturer="TI", conn=conn)
    assert [d["doc_id"] for d in out] == ["docB"]


def test_get_document_metadata_returns_sidecar_row(conn: Any) -> None:
    set_metadata(conn, "docA", mpn="STM32H743", tags=["mcu", "arm"])
    out = _get_document_metadata_impl("docA", conn=conn)
    assert out is not None
    assert out["mpn"] == "STM32H743"
    assert "mcu" in out["tags"]


def test_get_document_metadata_missing_returns_none(conn: Any) -> None:
    assert _get_document_metadata_impl("nope", conn=conn) is None


def test_stats_total_and_by_level(conn: Any) -> None:
    out = _stats_impl(project_id="p1", conn=conn)
    assert out["total_chunks"] == 6  # 1 MACRO + 2 MESO + 3 MICRO
    assert out["by_level"]["MICRO"] == 3
    assert out["by_level"]["MESO"] == 2
    assert out["by_level"]["MACRO"] == 1
    assert out["project_id"] == "p1"


def test_stats_doc_id_scope(conn: Any) -> None:
    out = _stats_impl(doc_id="docB", conn=conn)
    assert out["total_chunks"] == 1
    assert out["by_level"] == {"MICRO": 1}
