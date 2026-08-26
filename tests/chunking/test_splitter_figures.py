"""Figure handling in the multi-scale splitter (GH #41).

The store can only serve a figure the chunker linked, and search can only be
honest about a figure chunk that carries what it claims to. These tests pin
the two places that used to drop information on the floor: captions when no
crops were made, and the image on the MESO chunk that wraps a lone figure.
"""

from __future__ import annotations

from typing import Any

from datasheet_rag.chunking.layout_parser import (
    BoundingBox,
    ContentElement,
    DocumentOutline,
    DocumentSection,
    ElementType,
)
from datasheet_rag.chunking.splitter import SplitterConfig, split_document
from datasheet_rag.models.chunk import ChunkLevel, LayoutType


def _figure_element(block_id: str, caption: str, page: int = 3) -> ContentElement:
    return ContentElement(
        element_type=ElementType.FIGURE,
        text=caption or "[Figure]",
        block_id=f"blk-{block_id}",
        page=page,
        bbox=BoundingBox(),
        figure_block_id=block_id,
        figure_caption=caption,
    )


def _outline(*elements: ContentElement) -> DocumentOutline:
    section = DocumentSection(
        title="Mechanical", level=0, page_start=3, page_end=3,
        elements=list(elements),
    )
    return DocumentOutline(title="Widget Manual", doc_id="doc1", total_pages=4,
                           sections=[section])


def _manifest(*entries: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "figures": [
            {
                "block_id": block_id,
                "page": 3,
                "caption": caption,
                "image_path": image_path,
                "width_px": 900,
                "height_px": 700,
            }
            for block_id, caption, image_path in entries
        ]
    }


def _figures(graph: Any, level: ChunkLevel) -> list[Any]:
    return [
        c for c in graph.chunks.values()
        if c.level == level and c.metadata.layout_type == LayoutType.FIGURE
    ]


def test_caption_survives_when_no_figures_were_cropped() -> None:
    """`--skip-figures` means no manifest — the caption is still known."""
    graph = split_document(
        _outline(_figure_element("f1", "Figure 4: DT connector pinout")),
        figure_manifest=None,
    )

    micro = _figures(graph, ChunkLevel.MICRO)
    assert len(micro) == 1
    assert micro[0].figure_caption == "Figure 4: DT connector pinout"
    assert micro[0].figure_image_path is None


def test_micro_figure_links_to_its_crop() -> None:
    graph = split_document(
        _outline(_figure_element("f1", "Figure 4: DT connector pinout")),
        figure_manifest=_manifest(("f1", "Figure 4: DT connector pinout", "/figs/p003.png")),
    )

    micro = _figures(graph, ChunkLevel.MICRO)
    assert [c.figure_image_path for c in micro] == ["/figs/p003.png"]


def test_meso_wrapping_one_figure_carries_its_image() -> None:
    """The coarser zoom level of a lone figure IS that figure."""
    graph = split_document(
        _outline(_figure_element("f1", "Figure 4: DT connector pinout")),
        figure_manifest=_manifest(("f1", "Figure 4: DT connector pinout", "/figs/p003.png")),
    )

    meso = _figures(graph, ChunkLevel.MESO)
    assert len(meso) == 1
    assert meso[0].figure_image_path == "/figs/p003.png"
    assert meso[0].figure_caption == "Figure 4: DT connector pinout"


def test_meso_wrapping_several_figures_claims_none_of_them() -> None:
    """No single right image to advertise — so it advertises nothing."""
    graph = split_document(
        _outline(
            _figure_element("f1", "Figure 4: Front view"),
            _figure_element("f2", "Figure 5: Rear view"),
        ),
        figure_manifest=_manifest(
            ("f1", "Figure 4: Front view", "/figs/p003a.png"),
            ("f2", "Figure 5: Rear view", "/figs/p003b.png"),
        ),
        config=SplitterConfig(micro_max_tokens=512, meso_max_tokens=512),
    )

    meso = _figures(graph, ChunkLevel.MESO)
    assert len(meso) == 1
    assert meso[0].children_ids and len(meso[0].children_ids) == 2
    assert meso[0].figure_image_path is None
    assert meso[0].figure_s3_key is None
