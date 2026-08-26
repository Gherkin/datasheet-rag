"""Harness wiring tests: in-memory SQLite + mock embedder, no AWS.

Mirrors tests/store/test_sqlite_store.py's one-hot-vector fixture style so
vector ranking is deterministic and hand-checkable.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from datasheet_rag.eval.dataset import EvalSet, GoldenItem
from datasheet_rag.eval.harness import RunConfig, run_eval
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

EMBED_DIM = 8


def _unit(pos: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[pos % EMBED_DIM] = 1.0
    return v


def _chunk(chunk_id: str, text: str, *, page: int, pos: int) -> tuple[Chunk, list[float]]:
    chunk = Chunk(
        id=chunk_id,
        doc_id="doc1",
        level=ChunkLevel.MICRO,
        text=text,
        context_text=text,
        token_count=len(text.split()),
        metadata=ChunkMetadata(doc_id="doc1", page_numbers=[page], layout_type=LayoutType.TEXT),
    )
    return chunk, _unit(pos)


class _MockEmbedder:
    """Maps questions to one-hot vectors so KNN is deterministic."""

    def embed_one(self, text: str) -> list[float]:
        t = text.lower()
        if "i2c" in t or "clock stretching" in t:
            return _unit(0)
        if "thermal" in t or "junction" in t:
            return _unit(1)
        return _unit(2)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    specs = [
        _chunk("doc1:2:0", "I2C clock stretching on the SCL line register", page=5, pos=0),
        _chunk("doc1:2:1", "Thermal characteristics and junction temperature", page=7, pos=1),
        _chunk("doc1:2:2", "SPI mode timing diagram and chip select", page=9, pos=2),
    ]
    chunks = [s[0] for s in specs]
    vectors = {s[0].id: s[1] for s in specs}
    insert_chunks(c, chunks, vectors=vectors, project_id="p1")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture()
def eval_set() -> EvalSet:
    return EvalSet(
        items=[
            GoldenItem(
                question="How does I2C clock stretching work?",
                category="conceptual",
                doc_id="doc1",
                gold_chunk_ids=["doc1:2:0"],
                gold_pages=[5],
            ),
            GoldenItem(
                question="thermal junction temperature limit",
                category="identifier",
                doc_id="doc1",
                gold_chunk_ids=["doc1:2:1"],
                gold_pages=[7],
            ),
        ]
    )


def test_vector_mode_ranks_target_first(conn, eval_set) -> None:
    report = run_eval(
        conn,
        eval_set,
        RunConfig(mode="vector", k=3, ks=(1, 3)),
        embedder=_MockEmbedder(),
    )
    for o in report.outcomes:
        assert o.first_relevant_rank == 1
    assert report.by_category["overall"].hit_rate_at_k[1] == 1.0
    assert report.by_category["overall"].mrr == 1.0


def test_keyword_mode_no_embedder_needed(conn) -> None:
    # FTS5 ANDs the quoted terms, so keyword queries must use terms that
    # actually appear in the target chunk (a real property of this branch).
    es = EvalSet(
        items=[
            GoldenItem(
                question="I2C clock stretching SCL",
                category="identifier",
                doc_id="doc1",
                gold_chunk_ids=["doc1:2:0"],
                gold_pages=[5],
            ),
            GoldenItem(
                question="thermal junction temperature",
                category="identifier",
                doc_id="doc1",
                gold_chunk_ids=["doc1:2:1"],
                gold_pages=[7],
            ),
        ]
    )
    report = run_eval(conn, es, RunConfig(mode="keyword", k=3, ks=(1, 3)))
    assert report.by_category["overall"].hit_rate_at_k[3] == 1.0


def test_hybrid_mode_runs(conn, eval_set) -> None:
    report = run_eval(
        conn,
        eval_set,
        RunConfig(mode="hybrid", k=3, ks=(1, 3)),
        embedder=_MockEmbedder(),
    )
    assert report.by_category["overall"].hit_rate_at_k[3] == 1.0


def test_page_only_label_scores_loose_not_strict(conn) -> None:
    # Gold labels only a page, no exact chunk id. Strict lineage has no chunk
    # to anchor on, so it cannot credit; the loose page metric still does.
    es = EvalSet(
        items=[
            GoldenItem(
                question="How does I2C clock stretching work?",
                category="conceptual",
                doc_id="doc1",
                gold_chunk_ids=[],
                gold_pages=[5],
            )
        ]
    )
    report = run_eval(conn, es, RunConfig(mode="vector", k=3, ks=(1,)), embedder=_MockEmbedder())
    outcome = report.outcomes[0]
    assert outcome.first_relevant_rank is None  # strict: no anchor
    assert outcome.hit_at_ks == {1: 0.0}
    assert outcome.hit_at_ks_loose == {1: 1.0}  # loose: page-overlap credit


def test_vector_mode_requires_embedder(conn, eval_set) -> None:
    with pytest.raises(ValueError):
        run_eval(conn, eval_set, RunConfig(mode="vector", k=3))


def test_trace_file_written(conn, eval_set, tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    run_eval(
        conn,
        eval_set,
        RunConfig(mode="vector", k=3, ks=(1, 3)),
        embedder=_MockEmbedder(),
        trace_path=trace,
    )
    lines = [json.loads(ln) for ln in trace.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert lines[0]["results"]
    assert "match_source" in lines[0]["results"][0]
    assert lines[0]["config"]["mode"] == "vector"
