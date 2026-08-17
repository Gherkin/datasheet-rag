"""Run a golden set through one search configuration and score it.

The harness is the deterministic core of the retrieval eval: given an
:class:`EvalSet` and a :class:`RunConfig`, it dispatches each question to
the matching ``datasheet_rag.store.search`` function, scores the results with
:mod:`datasheet_rag.eval.metrics`, and returns a per-category + overall report.

Every query also produces one JSONL trace record (question, config,
retrieved chunk ids + scores + match source, latency). That trace file is
both production-style observability and the substrate the deferred
agent-layer eval will build on.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from datasheet_rag.eval.dataset import EvalSet
from datasheet_rag.eval.metrics import (
    DEFAULT_KS,
    CategoryMetrics,
    GraphNode,
    QueryOutcome,
    aggregate_by_category,
    lineage_relevant_ids,
)
from datasheet_rag.models.chunk import ChunkLevel
from datasheet_rag.store.search import (
    SearchFilters,
    SearchResult,
    hybrid_search,
    keyword_search,
    vector_search,
)

SearchMode = Literal["hybrid", "vector", "keyword"]

_LEVEL_BY_NAME: dict[str, ChunkLevel] = {
    "macro": ChunkLevel.MACRO,
    "meso": ChunkLevel.MESO,
    "micro": ChunkLevel.MICRO,
}


class Embedder(Protocol):
    """Minimal embedding interface the harness needs (lets tests mock)."""

    def embed_one(self, text: str) -> list[float]: ...


class RunConfig(BaseModel):
    """One retrieval operating point to evaluate."""

    mode: SearchMode = "hybrid"
    # Headline k: nDCG cutoff and the trace's reported result count.
    k: int = 5
    level: Literal["macro", "meso", "micro"] | None = None
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    # k values for the hit-rate / recall curve.
    ks: tuple[int, ...] = DEFAULT_KS
    label: str = ""

    def filters(self) -> SearchFilters | None:
        if self.level is None:
            return None
        return SearchFilters(level=_LEVEL_BY_NAME[self.level])

    def describe(self) -> str:
        if self.label:
            return self.label
        parts = [f"mode={self.mode}", f"k={self.k}"]
        if self.level:
            parts.append(f"level={self.level}")
        if self.mode == "hybrid" and (
            self.vector_weight != 1.0
            or self.keyword_weight != 1.0
            or self.rrf_k != 60
        ):
            parts.append(
                f"rrf_k={self.rrf_k},vw={self.vector_weight},kw={self.keyword_weight}"
            )
        return " ".join(parts)


class RunReport(BaseModel):
    """Result of running one config over a golden set."""

    config: RunConfig
    outcomes: list[QueryOutcome] = Field(default_factory=list)
    by_category: dict[str, CategoryMetrics] = Field(default_factory=dict)


def _search(
    conn: sqlite3.Connection,
    config: RunConfig,
    question: str,
    *,
    embedder: Embedder | None,
    fetch_n: int,
) -> list[SearchResult]:
    filters = config.filters()
    if config.mode == "keyword":
        return keyword_search(conn, question, k=fetch_n, filters=filters)

    if embedder is None:
        raise ValueError(f"mode={config.mode!r} requires an embedder")
    query_vec = embedder.embed_one(question)

    if config.mode == "vector":
        return vector_search(conn, query_vec, k=fetch_n, filters=filters)

    return hybrid_search(
        conn,
        query_vec,
        question,
        k=fetch_n,
        filters=filters,
        rrf_k=config.rrf_k,
        vector_weight=config.vector_weight,
        keyword_weight=config.keyword_weight,
    )


def _load_doc_graph(
    conn: sqlite3.Connection, doc_id: str
) -> dict[str, GraphNode]:
    """Load the lineage links for one document's chunks from the store."""
    rows = conn.execute(
        "SELECT id, parent_id, prev_id, next_id FROM chunks WHERE doc_id = ?",
        (doc_id,),
    ).fetchall()
    return {r[0]: GraphNode(r[1], r[2], r[3]) for r in rows}


def run_eval(
    conn: sqlite3.Connection,
    eval_set: EvalSet,
    config: RunConfig,
    *,
    embedder: Embedder | None = None,
    trace_path: Path | str | None = None,
) -> RunReport:
    """Score ``eval_set`` under ``config`` and (optionally) write a trace.

    Retrieves ``max(config.k, max(config.ks))`` results per query so the
    full hit-rate curve is populated from a single run; nDCG uses
    ``config.k`` as its cutoff. Relevance is judged with strict lineage
    matching (see :mod:`datasheet_rag.eval.metrics`); the page-based loose number
    is reported alongside as a diagnostic.
    """
    ks = config.ks
    fetch_n = max(config.k, max(ks))

    # Per-doc lineage graphs, loaded once and reused across questions.
    graph_cache: dict[str, dict[str, GraphNode]] = {}
    relevant_by_item: list[set[str]] = []
    for item in eval_set.items:
        graph = graph_cache.get(item.doc_id)
        if graph is None:
            graph = _load_doc_graph(conn, item.doc_id)
            graph_cache[item.doc_id] = graph
        relevant_by_item.append(lineage_relevant_ids(item.gold_chunk_ids, graph))
    trace_fh = None
    if trace_path is not None:
        tp = Path(trace_path)
        tp.parent.mkdir(parents=True, exist_ok=True)
        trace_fh = tp.open("a", encoding="utf-8")

    outcomes: list[QueryOutcome] = []
    try:
        for item, relevant_ids in zip(eval_set.items, relevant_by_item):
            t0 = time.perf_counter()
            results = _search(
                conn, config, item.question, embedder=embedder, fetch_n=fetch_n
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            outcome = QueryOutcome.score(
                item,
                results,
                ks=ks,
                ndcg_k=config.k,
                latency_ms=latency_ms,
                relevant_ids=relevant_ids,
            )
            outcomes.append(outcome)

            if trace_fh is not None:
                trace_fh.write(
                    json.dumps(
                        {
                            "question": item.question,
                            "category": item.category,
                            "doc_id": item.doc_id,
                            "config": config.model_dump(),
                            "results": [
                                {
                                    "chunk_id": r.chunk_id,
                                    "score": r.score,
                                    "match_source": r.match_source,
                                    "doc_id": r.chunk.doc_id,
                                    "pages": r.chunk.metadata.page_numbers,
                                    "level": r.chunk.level.name,
                                }
                                for r in results
                            ],
                            "first_relevant_rank": outcome.first_relevant_rank,
                            "latency_ms": latency_ms,
                        }
                    )
                    + "\n"
                )
    finally:
        if trace_fh is not None:
            trace_fh.close()

    return RunReport(
        config=config,
        outcomes=outcomes,
        by_category=aggregate_by_category(outcomes, ks=ks),
    )
