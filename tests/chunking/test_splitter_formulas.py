"""Formula handling in the multi-scale splitter (GH #19).

The splitter had no FORMULA branch, so every formula Docling found — its
extracted text and its crop — was dropped before it became a chunk. These
tests pin that a formula now becomes a chunk of its own type, linked to its
crop, and that nothing about it depends on its size.
"""

from __future__ import annotations

from typing import Any

import pytest

from datasheet_rag.chunking.layout_parser import (
    BoundingBox,
    ContentElement,
    DocumentOutline,
    DocumentSection,
    ElementType,
)
from datasheet_rag.chunking.splitter import split_document
from datasheet_rag.models.chunk import ChunkLevel, LayoutType


def _formula_element(block_id: str, text: str, page: int = 3) -> ContentElement:
    return ContentElement(
        element_type=ElementType.FORMULA,
        text=text,
        block_id=f"blk-{block_id}",
        page=page,
        bbox=BoundingBox(),
        figure_block_id=block_id,
    )


def _text_element(text: str, page: int = 3) -> ContentElement:
    return ContentElement(
        element_type=ElementType.TEXT,
        text=text,
        block_id=f"blk-{text[:8]}",
        page=page,
        bbox=BoundingBox(),
    )


def _outline(*elements: ContentElement) -> DocumentOutline:
    section = DocumentSection(
        title="Output voltage",
        level=0,
        page_start=3,
        page_end=3,
        elements=list(elements),
    )
    return DocumentOutline(title="Regulator", doc_id="doc1", total_pages=4, sections=[section])


def _manifest(block_id: str, image_path: str, *, width: int = 180, height: int = 40) -> dict:
    # Deliberately far below SplitterConfig.min_figure_px: formula crops are
    # small, and the logo filter must not apply to them.
    return {
        "figures": [
            {
                "block_id": block_id,
                "page": 3,
                "caption": "",
                "image_path": image_path,
                "width_px": width,
                "height_px": height,
            }
        ]
    }


def _formulas(graph: Any, level: ChunkLevel) -> list[Any]:
    return [
        c
        for c in graph.chunks.values()
        if c.level == level and c.metadata.layout_type == LayoutType.FORMULA
    ]


def test_a_formula_becomes_a_chunk_with_its_extracted_text() -> None:
    graph = split_document(
        _outline(_formula_element("docling_formula_1", "V_OUT = V_REF (1 + R1/R2)")),
        figure_manifest=None,
    )

    micro = _formulas(graph, ChunkLevel.MICRO)
    assert [c.text for c in micro] == ["V_OUT = V_REF (1 + R1/R2)"]


@pytest.mark.parametrize("text", ["[Formula]", ""])
def test_a_formula_with_no_extracted_text_still_becomes_a_chunk(text: str) -> None:
    graph = split_document(_outline(_formula_element("docling_formula_1", text)))

    micro = _formulas(graph, ChunkLevel.MICRO)
    assert [c.text for c in micro] == ["[Formula]"]


def test_a_small_formula_crop_is_linked_not_filtered() -> None:
    graph = split_document(
        _outline(_formula_element("docling_formula_1", "[Formula]")),
        figure_manifest=_manifest("docling_formula_1", "/figs/p003_formula000.png"),
    )

    micro = _formulas(graph, ChunkLevel.MICRO)
    assert [c.figure_image_path for c in micro] == ["/figs/p003_formula000.png"]


def test_meso_wrapping_one_formula_carries_its_image() -> None:
    graph = split_document(
        _outline(_formula_element("docling_formula_1", "[Formula]")),
        figure_manifest=_manifest("docling_formula_1", "/figs/p003_formula000.png"),
    )

    meso = _formulas(graph, ChunkLevel.MESO)
    assert [c.figure_image_path for c in meso] == ["/figs/p003_formula000.png"]


def test_a_formula_between_paragraphs_keeps_its_place_in_reading_order() -> None:
    graph = split_document(
        _outline(
            _text_element("The output is set by a divider."),
            _formula_element("docling_formula_1", "V_OUT = V_REF (1 + R1/R2)"),
            _text_element("Pick R2 below 100 kOhm."),
        )
    )

    micro = sorted(
        (c for c in graph.chunks.values() if c.level == ChunkLevel.MICRO),
        key=lambda c: int(c.id.rsplit(":", 1)[1]),
    )
    assert [c.metadata.layout_type for c in micro] == [
        LayoutType.TEXT,
        LayoutType.FORMULA,
        LayoutType.TEXT,
    ]
