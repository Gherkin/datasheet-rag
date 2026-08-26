"""Tests for the review backend (no HTTP, no AWS)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from datasheet_rag.eval.dataset import EvalSet, GoldenItem
from datasheet_rag.eval.review import (
    ReviewState,
    apply_update,
    chunks_on_pages,
)
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

EMBED_DIM = 8


def _chunk(cid: str, *, level: ChunkLevel, page: int, text: str) -> Chunk:
    return Chunk(
        id=cid,
        doc_id="doc1",
        level=level,
        text=text,
        context_text=text,
        metadata=ChunkMetadata(doc_id="doc1", page_numbers=[page], layout_type=LayoutType.TEXT),
    )


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    insert_chunks(
        c,
        [
            _chunk("doc1:L2:0", level=ChunkLevel.MICRO, page=14, text="Input slew rate notes"),
            _chunk("doc1:L1:0", level=ChunkLevel.MESO, page=14, text="Oscillation near threshold"),
            _chunk("doc1:L2:9", level=ChunkLevel.MICRO, page=99, text="Unrelated content"),
        ],
        project_id="p1",
    )
    try:
        yield c
    finally:
        c.close()


def test_chunks_on_pages_overlap(conn) -> None:
    rows = chunks_on_pages(conn, "doc1", [14])
    ids = {r["chunk_id"] for r in rows}
    assert ids == {"doc1:L2:0", "doc1:L1:0"}
    assert all("preview" in r and "level" in r for r in rows)


def test_chunks_on_pages_empty_pages(conn) -> None:
    assert chunks_on_pages(conn, "doc1", []) == []


def test_apply_update_edits_fields() -> None:
    item = GoldenItem(
        question="old", category="identifier", doc_id="doc1", gold_pages=[1], source="auto"
    )
    updated = apply_update(
        item,
        {
            "question": "new",
            "category": "figure",
            "gold_pages": [14, 15],
            "gold_chunk_ids": ["doc1:L2:0"],
            "source": "human",
        },
    )
    assert updated.question == "new"
    assert updated.category == "figure"
    assert updated.gold_pages == [14, 15]
    assert updated.gold_chunk_ids == ["doc1:L2:0"]
    assert updated.source == "human"


def test_apply_update_partial_keeps_other_fields() -> None:
    item = GoldenItem(
        question="q", category="identifier", doc_id="doc1", gold_pages=[3], answer_notes="keep me"
    )
    updated = apply_update(item, {"gold_pages": [7]})
    assert updated.gold_pages == [7]
    assert updated.answer_notes == "keep me"
    assert updated.question == "q"


def test_review_state_save_roundtrip(conn, tmp_path: Path) -> None:
    set_path = tmp_path / "golden.jsonl"
    EvalSet(
        items=[
            GoldenItem(question="q1", category="identifier", doc_id="doc1", gold_pages=[14]),
            GoldenItem(question="q2", category="figure", doc_id="doc1", gold_pages=[99]),
        ]
    ).save(set_path)

    state = ReviewState(set_path, conn)
    assert state.summary() == {"total": 2, "reviewed": 0}

    state.eval_set.items[0] = apply_update(
        state.eval_set.items[0],
        {
            "gold_pages": [14, 15],
            "source": "human",
        },
    )
    state.save()

    reloaded = EvalSet.load(set_path)
    assert reloaded.items[0].gold_pages == [14, 15]
    assert reloaded.items[0].source == "human"
    assert ReviewState(set_path, conn).summary() == {"total": 2, "reviewed": 1}
