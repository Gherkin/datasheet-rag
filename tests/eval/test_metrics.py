"""Unit tests for retrieval metrics + multi-scale hit matching.

Deterministic, no AWS. Builds SearchResult lists by hand.
"""

from __future__ import annotations

from aws_rag.eval.dataset import GoldenItem
from aws_rag.eval.metrics import (
    CategoryMetrics,
    QueryOutcome,
    aggregate_by_category,
    build_doc_graph,
    hit_at_k,
    is_hit,
    is_hit_lineage,
    lineage_relevant_ids,
    ndcg_at_k,
    reciprocal_rank,
    relevance_vector,
)
from aws_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from aws_rag.store.search import SearchResult


def _chunk(
    chunk_id: str,
    *,
    doc_id: str = "doc1",
    pages: list[int] | None = None,
    level: ChunkLevel = ChunkLevel.MICRO,
    parent_id: str | None = None,
    prev_id: str | None = None,
    next_id: str | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        level=level,
        text="x",
        context_text="x",
        parent_id=parent_id,
        prev_id=prev_id,
        next_id=next_id,
        metadata=ChunkMetadata(
            doc_id=doc_id, page_numbers=pages or [1], layout_type=LayoutType.TEXT
        ),
    )


def _result(chunk: Chunk, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk_id=chunk.id, score=score, chunk=chunk, match_source="vector"
    )


def _item(**kw) -> GoldenItem:
    base = dict(question="q", category="identifier", doc_id="doc1")
    base.update(kw)
    return GoldenItem(**base)  # type: ignore[arg-type]


# ---- is_hit ---------------------------------------------------------------


def test_is_hit_exact_chunk_id() -> None:
    item = _item(gold_chunk_ids=["doc1:2:5"])
    assert is_hit(_chunk("doc1:2:5"), item)
    assert not is_hit(_chunk("doc1:2:6"), item)


def test_is_hit_page_overlap_same_doc() -> None:
    item = _item(gold_chunk_ids=[], gold_pages=[12])
    # Different chunk id, but shares page 12 in the same doc — credit.
    assert is_hit(_chunk("doc1:1:0", pages=[11, 12]), item)
    # Same page but different doc — no credit.
    assert not is_hit(_chunk("docX:1:0", doc_id="docX", pages=[12]), item)
    # Same doc, non-overlapping page — no credit.
    assert not is_hit(_chunk("doc1:1:9", pages=[99]), item)


def test_is_hit_no_gold_pages_falls_back_to_ids_only() -> None:
    item = _item(gold_chunk_ids=["doc1:2:5"], gold_pages=[])
    assert not is_hit(_chunk("doc1:1:0", pages=[1]), item)


# ---- lineage relevance ----------------------------------------------------
#
#   macro M ── meso S ──┬── micro a (gold)
#                       ├── micro b (sibling of a)
#                       └── micro c
#   plus an unrelated micro z on the same page as a.


def _family_graph():
    chunks = [
        _chunk("M", level=ChunkLevel.MACRO),
        _chunk("S", level=ChunkLevel.MESO, parent_id="M"),
        _chunk("a", parent_id="S", prev_id=None, next_id="b", pages=[4]),
        _chunk("b", parent_id="S", prev_id="a", next_id="c", pages=[4]),
        _chunk("c", parent_id="S", prev_id="b", next_id=None, pages=[4]),
        _chunk("z", parent_id="S2", pages=[4]),  # unrelated, same page
    ]
    return build_doc_graph(chunks)


def test_lineage_includes_ancestors_descendants_siblings() -> None:
    graph = _family_graph()
    rel = lineage_relevant_ids(["a"], graph)
    assert "a" in rel              # exact
    assert "S" in rel and "M" in rel  # ancestors
    assert "b" in rel              # immediate reading-order sibling
    # 'c' is a non-adjacent sibling: included via descendants-of-S only if S
    # is gold; here it should NOT be credited from gold 'a'.
    assert "c" not in rel
    assert "z" not in rel          # unrelated, despite sharing page 4


def test_lineage_descendants_when_gold_is_macro() -> None:
    graph = _family_graph()
    rel = lineage_relevant_ids(["S"], graph)
    # Gold is the meso; all its micro children are lineage-relevant.
    assert {"S", "M", "a", "b", "c"} <= rel
    assert "z" not in rel


def test_is_hit_lineage_uses_relevant_set() -> None:
    rel = {"a", "S", "M", "b"}
    assert is_hit_lineage(_chunk("S"), rel)
    assert not is_hit_lineage(_chunk("z"), rel)


def test_score_lineage_vs_loose_page_gap() -> None:
    # Gold micro 'a' on page 4. Retrieval returns unrelated 'z' (same page).
    item = _item(gold_chunk_ids=["a"], gold_pages=[4])
    graph = _family_graph()
    rel_ids = lineage_relevant_ids(["a"], graph)
    results = [_result(_chunk("z", pages=[4]))]

    o = QueryOutcome.score(item, results, ks=(1,), ndcg_k=1, relevant_ids=rel_ids)
    # Strict: 'z' is not in 'a' lineage -> miss. Loose: shares page 4 -> hit.
    assert o.hit_at_ks == {1: 0.0}
    assert o.hit_at_ks_loose == {1: 1.0}


# ---- rank metrics ---------------------------------------------------------


def test_hit_at_k_and_reciprocal_rank() -> None:
    rel = [False, False, True, False]
    assert hit_at_k(rel, 1) == 0.0
    assert hit_at_k(rel, 3) == 1.0
    assert reciprocal_rank(rel) == 1.0 / 3.0
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_rewards_higher_placement() -> None:
    top = ndcg_at_k([True, False, False], 3)
    low = ndcg_at_k([False, False, True], 3)
    assert top == 1.0  # single relevant at rank 1 == ideal
    assert 0.0 < low < top


def test_ndcg_zero_when_no_relevant() -> None:
    assert ndcg_at_k([False, False], 5) == 0.0


def test_relevance_vector_order_preserved() -> None:
    item = _item(gold_chunk_ids=["b"])
    results = [_result(_chunk("a")), _result(_chunk("b")), _result(_chunk("c"))]
    assert relevance_vector(results, item) == [False, True, False]


# ---- aggregation ----------------------------------------------------------


def test_query_outcome_score_and_aggregate() -> None:
    item_a = _item(category="identifier", gold_chunk_ids=["hit"])
    item_b = _item(category="conceptual", gold_chunk_ids=["nope"])

    res_hit = [_result(_chunk("x")), _result(_chunk("hit"))]  # first-relevant rank 2
    res_miss = [_result(_chunk("x")), _result(_chunk("y"))]

    o_a = QueryOutcome.score(item_a, res_hit, ks=(1, 3), ndcg_k=3)
    o_b = QueryOutcome.score(item_b, res_miss, ks=(1, 3), ndcg_k=3)

    assert o_a.first_relevant_rank == 2
    assert o_a.hit_at_ks == {1: 0.0, 3: 1.0}
    assert o_b.first_relevant_rank is None

    report = aggregate_by_category([o_a, o_b], ks=(1, 3))
    assert set(report) == {"identifier", "conceptual", "overall"}
    assert report["identifier"].hit_rate_at_k[3] == 1.0
    assert report["conceptual"].hit_rate_at_k[3] == 0.0
    assert report["overall"].n == 2
    assert report["overall"].hit_rate_at_k[3] == 0.5


def test_aggregate_empty() -> None:
    m = CategoryMetrics.aggregate([], ks=(1, 5))
    assert m.n == 0
    assert m.hit_rate_at_k == {1: 0.0, 5: 0.0}
