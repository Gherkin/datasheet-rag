"""Retrieval metrics with multi-scale-aware hit matching.

The central subtlety is **what counts as a hit**. Because the chunker
splits each page into micro/meso/macro chunks, the exact chunk a question
was labeled against is often *not* the one retrieval returns — it may
return the parent meso, a sibling micro, or the macro summary covering
the same material. Scoring only exact ``chunk_id`` matches would unfairly
punish the multi-scale design we are trying to evaluate.

We support two relevance policies, and report both:

**lineage (strict, primary)** — a retrieved chunk is relevant iff it is a
gold chunk, an ancestor/descendant of a gold chunk (the same content at a
different zoom level), or an immediate reading-order sibling (the same span
split at a chunk boundary). This credits the genuine multi-scale case
*without* crediting unrelated content that merely shares a page. The
lineage set is precomputed per item from the document's chunk graph (see
:func:`lineage_relevant_ids`).

**page (loose, diagnostic)** — a retrieved chunk is relevant iff it is a
gold chunk, or it shares a ``doc_id`` **and** a page number with the gold
location. This is the old behavior; in a dense technical document a page
holds many unrelated facts (e.g. every row of an Electrical Characteristics
table), so it over-credits "landed on the right page" as if it were "found
the right fact". We keep it only as an upper bound — the gap between strict
and loose hit-rate is itself a useful signal of how page-inflated a number
is, and the inflation is uneven across categories (worst for dense tables).

From the (strict) boolean relevance judgment we compute the usual rank
metrics:

* **hit_rate@k** — fraction of queries with ≥1 relevant chunk in the top k
  (a.k.a. success@k). The primary, most-interpretable signal.
* **MRR** — mean of ``1 / rank`` of the first relevant chunk.
* **nDCG@k** — ranking quality. With unknown global relevance counts we
  compute the ideal DCG from the relevant items present in the *retrieved*
  list (packed at the top), so nDCG measures how well retrieval ordered
  what it found. Documented here so the number is not over-read.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from pydantic import BaseModel, Field

from datasheet_rag.eval.dataset import Category, GoldenItem
from datasheet_rag.models.chunk import Chunk
from datasheet_rag.store.search import SearchResult

# Default k values for the recall/hit curve (the "knee" sweep).
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10, 20)


class GraphNode(NamedTuple):
    """The lineage links of one chunk, enough to walk the chunk graph."""

    parent_id: str | None
    prev_id: str | None
    next_id: str | None


def build_doc_graph(chunks: Iterable[Chunk]) -> dict[str, GraphNode]:
    """Map ``chunk_id -> GraphNode`` for one document's chunks."""
    return {c.id: GraphNode(c.parent_id, c.prev_id, c.next_id) for c in chunks}


def _ancestors(chunk_id: str, graph: dict[str, GraphNode]) -> list[str]:
    """Walk parent links upward from ``chunk_id`` (cycle-safe)."""
    out: list[str] = []
    seen: set[str] = set()
    node = graph.get(chunk_id)
    while node and node.parent_id and node.parent_id not in seen:
        seen.add(node.parent_id)
        out.append(node.parent_id)
        node = graph.get(node.parent_id)
    return out


def lineage_relevant_ids(
    gold_chunk_ids: Sequence[str], graph: dict[str, GraphNode]
) -> set[str]:
    """Expand gold chunk ids into the set of lineage-equivalent chunk ids.

    A chunk is lineage-relevant to a gold chunk if it is the gold chunk
    itself, an ancestor or descendant of it (same content, different zoom
    level), or an immediate reading-order sibling (``prev``/``next`` — the
    same span split at a chunk boundary). Chunks not in ``graph`` (e.g. a
    gold id from another document) are still seeded so labels never get
    dropped.
    """
    relevant: set[str] = set(gold_chunk_ids)
    gold_set = set(gold_chunk_ids)
    # Ancestors of each gold chunk.
    for gid in gold_chunk_ids:
        relevant.update(_ancestors(gid, graph))
    # Descendants of any gold chunk = any chunk with a gold ancestor.
    for cid in graph:
        if gold_set.intersection(_ancestors(cid, graph)):
            relevant.add(cid)
    # Immediate reading-order siblings of each gold chunk.
    for gid in gold_chunk_ids:
        node = graph.get(gid)
        if node:
            if node.prev_id:
                relevant.add(node.prev_id)
            if node.next_id:
                relevant.add(node.next_id)
    return relevant


def is_hit(chunk: Chunk, item: GoldenItem) -> bool:
    """Loose (page) relevance test for one retrieved chunk.

    Relevant iff the chunk is an exact labeled gold chunk, or it shares a
    page with the gold location within the same document. This is the
    diagnostic upper bound — see the module docstring; prefer
    :func:`is_hit_lineage` for the strict, primary judgment.
    """
    if chunk.id in item.gold_chunk_ids:
        return True
    if item.gold_pages and chunk.doc_id == item.doc_id:
        if set(chunk.metadata.page_numbers) & set(item.gold_pages):
            return True
    return False


def is_hit_lineage(chunk: Chunk, relevant_ids: set[str]) -> bool:
    """Strict relevance test: chunk is in the precomputed lineage set."""
    return chunk.id in relevant_ids


def relevance_vector(results: Sequence[SearchResult], item: GoldenItem) -> list[bool]:
    """Loose (page) boolean relevance per result, in rank order (best first)."""
    return [is_hit(r.chunk, item) for r in results]


def lineage_relevance_vector(
    results: Sequence[SearchResult], relevant_ids: set[str]
) -> list[bool]:
    """Strict (lineage) boolean relevance per result, in rank order."""
    return [is_hit_lineage(r.chunk, relevant_ids) for r in results]


def hit_at_k(rel: Sequence[bool], k: int) -> float:
    """1.0 if any of the first ``k`` results is relevant, else 0.0."""
    return 1.0 if any(rel[:k]) else 0.0


def first_relevant_rank(rel: Sequence[bool]) -> int | None:
    """1-indexed rank of the first relevant result, or None."""
    for i, hit in enumerate(rel, start=1):
        if hit:
            return i
    return None


def reciprocal_rank(rel: Sequence[bool]) -> float:
    """``1 / rank`` of the first relevant result, else 0.0."""
    rank = first_relevant_rank(rel)
    return 1.0 / rank if rank is not None else 0.0


def _dcg(rel: Sequence[bool], k: int) -> float:
    return sum(
        (1.0 / math.log2(i + 1)) for i, hit in enumerate(rel[:k], start=1) if hit
    )


def ndcg_at_k(rel: Sequence[bool], k: int) -> float:
    """Binary nDCG@k.

    IDCG is computed from the number of relevant items present in the
    retrieved list (packed at the top), so this measures how well the
    retriever ordered what it found — see module docstring.
    """
    dcg = _dcg(rel, k)
    num_relevant = sum(1 for hit in rel if hit)
    if num_relevant == 0:
        return 0.0
    ideal = [True] * min(num_relevant, k)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


class QueryOutcome(BaseModel):
    """Per-query scored result, ready for aggregation."""

    question: str
    category: Category
    doc_id: str
    num_retrieved: int
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    first_relevant_rank: int | None = None
    reciprocal_rank: float = 0.0
    ndcg: float = 0.0
    # Primary (strict/lineage) hit-rate, plus the loose (page) upper bound.
    hit_at_ks: dict[int, float] = Field(default_factory=dict)
    hit_at_ks_loose: dict[int, float] = Field(default_factory=dict)
    latency_ms: float = 0.0

    @classmethod
    def score(
        cls,
        item: GoldenItem,
        results: Sequence[SearchResult],
        *,
        ks: Sequence[int] = DEFAULT_KS,
        ndcg_k: int,
        latency_ms: float = 0.0,
        relevant_ids: set[str] | None = None,
    ) -> QueryOutcome:
        """Score one query.

        When ``relevant_ids`` (the precomputed lineage set) is given, the
        primary metrics use strict lineage relevance and ``hit_at_ks_loose``
        carries the page-based upper bound. When omitted, both fall back to
        page relevance (keeps direct callers / unit tests simple).
        """
        rel_loose = relevance_vector(results, item)
        rel = (
            lineage_relevance_vector(results, relevant_ids)
            if relevant_ids is not None
            else rel_loose
        )
        return cls(
            question=item.question,
            category=item.category,
            doc_id=item.doc_id,
            num_retrieved=len(results),
            retrieved_chunk_ids=[r.chunk_id for r in results],
            first_relevant_rank=first_relevant_rank(rel),
            reciprocal_rank=reciprocal_rank(rel),
            ndcg=ndcg_at_k(rel, ndcg_k),
            hit_at_ks={k: hit_at_k(rel, k) for k in ks},
            hit_at_ks_loose={k: hit_at_k(rel_loose, k) for k in ks},
            latency_ms=latency_ms,
        )


class CategoryMetrics(BaseModel):
    """Mean metrics over a group of queries (one category, or overall)."""

    n: int
    mrr: float
    ndcg: float
    # Strict (lineage) hit-rate, and the loose (page) upper bound beside it.
    hit_rate_at_k: dict[int, float] = Field(default_factory=dict)
    hit_rate_at_k_loose: dict[int, float] = Field(default_factory=dict)
    mean_latency_ms: float = 0.0

    @classmethod
    def aggregate(
        cls,
        outcomes: Sequence[QueryOutcome],
        *,
        ks: Sequence[int] = DEFAULT_KS,
    ) -> CategoryMetrics:
        n = len(outcomes)
        if n == 0:
            return cls(
                n=0,
                mrr=0.0,
                ndcg=0.0,
                hit_rate_at_k={k: 0.0 for k in ks},
                hit_rate_at_k_loose={k: 0.0 for k in ks},
            )
        mrr = sum(o.reciprocal_rank for o in outcomes) / n
        ndcg = sum(o.ndcg for o in outcomes) / n
        hit_rate = {
            k: sum(o.hit_at_ks.get(k, 0.0) for o in outcomes) / n for k in ks
        }
        hit_rate_loose = {
            k: sum(o.hit_at_ks_loose.get(k, 0.0) for o in outcomes) / n for k in ks
        }
        latency = sum(o.latency_ms for o in outcomes) / n
        return cls(
            n=n,
            mrr=mrr,
            ndcg=ndcg,
            hit_rate_at_k=hit_rate,
            hit_rate_at_k_loose=hit_rate_loose,
            mean_latency_ms=latency,
        )


def aggregate_by_category(
    outcomes: Sequence[QueryOutcome],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, CategoryMetrics]:
    """Group outcomes by category and add an ``"overall"`` bucket."""
    buckets: dict[str, list[QueryOutcome]] = {}
    for o in outcomes:
        buckets.setdefault(o.category, []).append(o)
    report = {
        cat: CategoryMetrics.aggregate(group, ks=ks) for cat, group in buckets.items()
    }
    report["overall"] = CategoryMetrics.aggregate(outcomes, ks=ks)
    return report
