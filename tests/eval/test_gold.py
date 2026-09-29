"""Gold chunks resolved from evidence quotes, not stored chunk ids (GH #92)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from datasheet_rag.eval.dataset import EvalSet, Evidence, GoldenItem
from datasheet_rag.eval.gold import normalize, resolve_gold_chunk_ids
from datasheet_rag.eval.harness import RunConfig, run_eval
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

EMBED_DIM = 4


def _chunk(cid: str, text: str, *, pages: list[int], level: ChunkLevel) -> Chunk:
    return Chunk(
        id=cid,
        doc_id="doc1",
        level=level,
        text=text,
        context_text=text,
        token_count=len(text.split()),
        metadata=ChunkMetadata(doc_id="doc1", page_numbers=pages, layout_type=LayoutType.TABLE),
    )


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    chunks = [
        _chunk(
            "doc1:L0:0",
            "VBUS | -0.3 | 5.25 | V and much more",
            pages=[51, 52],
            level=ChunkLevel.MACRO,
        ),
        _chunk(
            "doc1:L1:0",
            "Absolute maximum\nVBUS | Voltage on VBUS | -0.3 | 5.25 | V",
            pages=[52],
            level=ChunkLevel.MESO,
        ),
        _chunk(
            "doc1:L2:7",
            "VBUS | Voltage on VBUS | -0.3 | 5.25 | V",
            pages=[52],
            level=ChunkLevel.MICRO,
        ),
        _chunk(
            "doc1:L2:8",
            "VBUS | Voltage on VBUS | -0.3 | 5.25 | V",
            pages=[60],
            level=ChunkLevel.MICRO,
        ),
        _chunk(
            "doc1:L2:9",
            "Thermal resistance junction to ambient",
            pages=[52],
            level=ChunkLevel.MICRO,
        ),
    ]
    insert_chunks(c, chunks, vectors={ch.id: [1.0, 0.0, 0.0, 0.0] for ch in chunks})
    try:
        yield c
    finally:
        c.close()


def _item(**kw: object) -> GoldenItem:
    base: dict[str, object] = {
        "question": "VBUS absolute maximum",
        "category": "table_spec",
        "doc_id": "doc1",
        "source": "mined",
        "evidence": [Evidence(page=52, quote="VBUS   Voltage on VBUS   -0.3   5.25   V")],
    }
    base.update(kw)
    return GoldenItem(**base)  # type: ignore[arg-type]


def test_resolves_meso_and_micro_on_the_evidence_page(conn) -> None:
    # Not the MACRO summary (it spans chapters), not the same row on another
    # page, not an unrelated chunk on the right page.
    assert resolve_gold_chunk_ids(conn, _item()) == ["doc1:L1:0", "doc1:L2:7"]


def test_quote_the_store_lost_resolves_to_nothing(conn) -> None:
    item = _item(evidence=[Evidence(page=52, quote="Storage temperature -65 to 150 °C")])
    assert resolve_gold_chunk_ids(conn, item) == []


def test_normalize_folds_extraction_variants() -> None:
    assert normalize("10 μA – 5 Ω") == normalize("10  µA - 5 Ω")
    # pdftotext renders an unmapped glyph as a control character.
    assert normalize("200 k\x02   C3*") == normalize("200 k C3*")


def test_harness_uses_evidence_over_stale_chunk_ids(conn) -> None:
    # A re-chunk moved the row; the stored id now names the thermal chunk.
    item = _item(gold_chunk_ids=["doc1:L2:9"])
    report = run_eval(conn, EvalSet(items=[item]), RunConfig(mode="keyword", k=5, ks=(1, 5)))
    top = report.outcomes[0]
    assert top.first_relevant_rank is not None
    assert top.retrieved_chunk_ids[top.first_relevant_rank - 1] != "doc1:L2:9"


def test_harness_reports_evidence_the_store_lost(conn) -> None:
    lost = _item(
        question="storage temp",
        evidence=[Evidence(page=52, quote="Storage temperature -65 to 150")],
    )
    report = run_eval(conn, EvalSet(items=[_item(), lost]), RunConfig(mode="keyword", k=5, ks=(1,)))
    assert report.unresolved_gold == ["storage temp"]


def test_harness_skips_unanswerable_items(conn) -> None:
    items = [_item(), _item(question="DF2S30FS clamp", doc_id=None, answerable=False, evidence=[])]
    report = run_eval(conn, EvalSet(items=items), RunConfig(mode="keyword", k=5, ks=(1,)))
    assert [o.question for o in report.outcomes] == ["VBUS absolute maximum"]
