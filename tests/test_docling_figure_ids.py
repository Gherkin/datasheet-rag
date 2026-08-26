"""The two Docling traversals must agree on figure ids (GH #41).

``_build_outline`` stamps each PICTURE with ``docling_figure_N`` and
``_build_figure_regions`` names the crops the same way. Nothing links them but
the counters, so both walks have to count the *same* items — including ones
neither can use. When they drift, every later figure chunk points at its
neighbour's picture and the last one points at nothing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from datasheet_rag.chunking.layout_parser import ElementType

DocItemLabel = pytest.importorskip("docling_core.types.doc").DocItemLabel


class _BBox:
    l, t, r, b = 10.0, 700.0, 300.0, 500.0  # noqa: E741

    def to_top_left_origin(self, page_h: float) -> Any:
        return SimpleNamespace(l=self.l, t=page_h - self.t, r=self.r, b=page_h - self.b)


def _item(label: Any, *, page: int | None = 1, text: str = "") -> Any:
    prov = [] if page is None else [SimpleNamespace(page_no=page, bbox=_BBox())]
    return SimpleNamespace(label=label, prov=prov, text=text)


def _doc(items: list[Any]) -> Any:
    page = SimpleNamespace(size=SimpleNamespace(width=612.0, height=792.0))
    return SimpleNamespace(
        pages={1: page, 2: page},
        iterate_items=lambda: ((it, 1) for it in items),
    )


def _outline_figure_ids(items: list[Any]) -> list[str]:
    from datasheet_rag.docling_parser import _build_outline

    outline = _build_outline(_doc(items), "doc1")
    return [
        e.figure_block_id
        for section in outline.all_sections_flat
        for e in section.elements
        if e.element_type == ElementType.FIGURE
    ]


def _region_ids(items: list[Any]) -> list[str]:
    from datasheet_rag.docling_parser import _build_figure_regions

    return [r.block_id for r in _build_figure_regions(_doc(items))]


def test_figure_ids_match_across_both_traversals() -> None:
    items = [
        _item(DocItemLabel.TITLE, text="Widget Manual"),
        _item(DocItemLabel.PICTURE),
        _item(DocItemLabel.CAPTION, text="Figure 1: front view"),
        _item(DocItemLabel.TEXT, text="Some prose about the widget."),
        _item(DocItemLabel.PICTURE),
        _item(DocItemLabel.CAPTION, text="Figure 2: rear view"),
    ]
    assert _outline_figure_ids(items) == ["docling_figure_1", "docling_figure_2"]
    assert _region_ids(items) == ["docling_figure_1", "docling_figure_2"]


def test_a_picture_without_provenance_does_not_shift_later_ids() -> None:
    """The outline keeps such a picture (page 1); the cropper cannot."""
    items = [
        _item(DocItemLabel.PICTURE, page=None),  # no bbox → nothing to crop
        _item(DocItemLabel.PICTURE),
        _item(DocItemLabel.CAPTION, text="Figure 2: rear view"),
    ]
    assert _outline_figure_ids(items) == ["docling_figure_1", "docling_figure_2"]
    # The unusable one is dropped, but the id it consumed is not reused — so
    # the second picture is still figure_2 on both sides.
    assert _region_ids(items) == ["docling_figure_2"]


def test_a_caption_after_a_dropped_picture_does_not_move_to_its_neighbour() -> None:
    from datasheet_rag.docling_parser import _build_figure_regions

    items = [
        _item(DocItemLabel.PICTURE),
        _item(DocItemLabel.PICTURE, page=None),
        _item(DocItemLabel.CAPTION, text="Caption for the dropped picture"),
    ]
    regions = _build_figure_regions(_doc(items))
    assert [r.block_id for r in regions] == ["docling_figure_1"]
    assert regions[0].caption == ""


def test_formula_ids_use_their_own_counter() -> None:
    items = [
        _item(DocItemLabel.PICTURE),
        _item(DocItemLabel.FORMULA, text="V = I * R"),
        _item(DocItemLabel.PICTURE),
    ]
    assert _region_ids(items) == [
        "docling_figure_1", "docling_formula_1", "docling_figure_2",
    ]
