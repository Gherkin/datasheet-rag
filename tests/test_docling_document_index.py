"""DOCUMENT_INDEX items are kept unless they read as a table of contents (GH #98).

Docling's layout model labels dotted-leader spec lists — the MP6004's
Absolute Maximum Ratings block, for one — as DOCUMENT_INDEX, the class it
uses for tables of contents. ``_build_outline`` used to handle only TABLE, so
these items fell through and their values never reached the store.
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


def _index_item(rows: list[list[str]]) -> Any:
    grid = [[SimpleNamespace(text=text) for text in row] for row in rows]
    return SimpleNamespace(
        label=DocItemLabel.DOCUMENT_INDEX,
        prov=[SimpleNamespace(page_no=1, bbox=_BBox())],
        text="",
        data=SimpleNamespace(grid=grid),
    )


def _doc(items: list[Any]) -> Any:
    page = SimpleNamespace(size=SimpleNamespace(width=612.0, height=792.0))
    return SimpleNamespace(pages={1: page}, iterate_items=lambda: ((it, 1) for it in items))


def _table_texts(items: list[Any]) -> list[str]:
    from datasheet_rag.docling_parser import _build_outline

    outline = _build_outline(_doc(items), "doc1")
    return [
        e.text
        for section in outline.all_sections_flat
        for e in section.elements
        if e.element_type == ElementType.TABLE
    ]


def test_dotted_leader_ratings_list_is_kept_without_leaders() -> None:
    item = _index_item(
        [
            ["VIN............................-0.3 V to +100 V", ""],
            ["SW", "............................-0.7 V to +180 V"],
            ["All other pins", "...........-0.3 V to +6.5 V (3)"],
            ["Junction temperature", "150°C"],
        ]
    )
    [text] = _table_texts([item])
    assert "VIN -0.3 V to +100 V" in text
    assert "SW | -0.7 V to +180 V" in text
    assert "All other pins | -0.3 V to +6.5 V (3)" in text
    assert "...." not in text


def test_table_of_contents_index_is_dropped() -> None:
    item = _index_item(
        [
            ["Features", "........ 1"],
            ["Pin Description", "........ 4"],
            ["Electrical Characteristics", "5"],
            ["Package Information", "........ 22"],
        ]
    )
    assert _table_texts([item]) == []
