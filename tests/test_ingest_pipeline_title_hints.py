"""`parse_pdf_to_graph` reports title hints only when a Docling outline was read.

``None`` tells the store "no parse computed hints, keep what you have". A dict,
even an empty one, is what the parse found and replaces the stored hints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasheet_rag.chunking.layout_parser import DocumentOutline
from datasheet_rag.chunking.pipeline import save_chunk_graph
from datasheet_rag.config import get_settings
from datasheet_rag.ingest_pipeline import parse_pdf_to_graph
from datasheet_rag.models.chunk import Chunk, ChunkGraph, ChunkLevel, ChunkMetadata

DID = "d" * 64


@pytest.fixture
def pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Each storage path is set on its own: they are filled from rag_home only
    # when the settings are built, and the suite shares one RAG_HOME.
    settings = get_settings()
    for name in ("pdf_dir", "figures_dir", "output_dir"):
        d = tmp_path / name
        d.mkdir()
        monkeypatch.setattr(settings, name, d)
    monkeypatch.setattr("datasheet_rag.docling_parser.is_native_pdf", lambda _p: True)
    path = tmp_path / "part.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def _write_outline(outline: DocumentOutline) -> None:
    path = get_settings().output_dir / f"{DID}_outline.json"
    path.write_text(json.dumps({"outline": outline.to_dict(), "figure_regions": []}))


def _write_chunk_cache() -> None:
    g = ChunkGraph(doc_id=DID)
    g.add(
        Chunk(
            id=f"{DID}:L2:0",
            doc_id=DID,
            level=ChunkLevel.MICRO,
            text="text",
            metadata=ChunkMetadata(doc_id=DID, page_numbers=[1]),
        )
    )
    save_chunk_graph(g, get_settings().output_dir / f"{DID}_chunks.json")


def test_cached_chunk_graph_without_outline_reports_no_hints(pdf: Path) -> None:
    _write_chunk_cache()
    parsed = parse_pdf_to_graph(pdf, doc_id=DID, skip_figures=True)
    assert parsed.title_hints is None


def test_cached_chunk_graph_reads_hints_from_its_outline(pdf: Path) -> None:
    # A --dry-run or a failed embed leaves both caches; the next ingest must
    # still store the hints that parse found.
    _write_chunk_cache()
    _write_outline(DocumentOutline(doc_id=DID, running_header="LM317"))
    parsed = parse_pdf_to_graph(pdf, doc_id=DID, skip_figures=True)
    assert parsed.title_hints == {"running_header": "LM317"}


def test_cached_chunk_graph_with_hintless_outline_reports_an_empty_dict(pdf: Path) -> None:
    _write_chunk_cache()
    _write_outline(DocumentOutline(doc_id=DID))
    parsed = parse_pdf_to_graph(pdf, doc_id=DID, skip_figures=True)
    assert parsed.title_hints == {}


def test_outline_without_hints_reports_an_empty_dict(pdf: Path) -> None:
    _write_outline(DocumentOutline(doc_id=DID))
    parsed = parse_pdf_to_graph(pdf, doc_id=DID, skip_figures=True)
    assert parsed.title_hints == {}


def test_outline_hints_are_reported(pdf: Path) -> None:
    _write_outline(DocumentOutline(doc_id=DID, running_header="LM317", pdf_meta_title="Guide"))
    parsed = parse_pdf_to_graph(pdf, doc_id=DID, skip_figures=True)
    assert parsed.title_hints == {"running_header": "LM317", "pdf_meta_title": "Guide"}
