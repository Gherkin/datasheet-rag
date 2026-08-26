"""Parse Textract blocks into a structured document outline.

Takes raw Textract blocks and produces a tree of DocumentSection objects
that represent the logical structure: chapters → sections → content elements
(paragraphs, tables, figures, key-value sets).

This is the bridge between Textract's flat block list and our hierarchical
chunk model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from datasheet_rag.textract import layout_reading_order


class ElementType(StrEnum):
    """Type of content element within a section."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    KEY_VALUE = "key_value"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


@dataclass
class BoundingBox:
    """Normalised bounding box (0..1) from Textract geometry."""

    left: float = 0.0
    top: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @classmethod
    def from_textract(cls, bbox: dict[str, float]) -> BoundingBox:
        return cls(
            left=bbox.get("Left", 0),
            top=bbox.get("Top", 0),
            width=bbox.get("Width", 0),
            height=bbox.get("Height", 0),
        )

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class ContentElement:
    """A single content element extracted from Textract layout blocks.

    This is an atomic unit — paragraphs, tables, figures, etc. — that
    should never be split across chunks.
    """

    element_type: ElementType
    text: str
    block_id: str
    page: int
    bbox: BoundingBox
    # For tables: structured cell data
    table_cells: list[dict[str, Any]] = field(default_factory=list)
    table_title: str = ""
    # Set when docling_parser.table_structure_untrustworthy() flagged this
    # table's row/column/header structure as unsafe to assert (see
    # docs/table-structure-repair/{problem,plan}.md) — None means trusted.
    table_structure_warning: str | None = None
    # Cached LLM-repaired structure (Stage 3 — table_repair.TableRepairer),
    # same grid-filled shape as table_cells. None means "not (yet) repaired";
    # an empty list would mean "repaired and found to have no cells", which
    # cannot happen — a flagged table always has origin-cell text to repair,
    # so empty vs. None is never ambiguous. Populated opt-in by `rag
    # repair-tables`, not on the ingest hot path — see table_repair_concurrency.
    table_repaired_cells: list[dict[str, Any]] | None = None
    # For figures: reference to the extracted image
    figure_block_id: str = ""
    figure_caption: str = ""
    # For key-value: the key and value separately
    kv_key: str = ""
    kv_value: str = ""
    # Child block IDs from Textract (for traceability)
    child_block_ids: list[str] = field(default_factory=list)


@dataclass
class DocumentSection:
    """A section of the document, forming a tree structure.

    A section has a title (from LAYOUT_SECTION_HEADER or LAYOUT_TITLE),
    content elements, and potentially child sub-sections.
    """

    title: str
    level: int  # 0 = document/chapter, 1 = section, 2+ = subsection
    page_start: int = 1
    page_end: int = 1
    elements: list[ContentElement] = field(default_factory=list)
    children: list[DocumentSection] = field(default_factory=list)
    # Block ID of the header that started this section
    header_block_id: str = ""

    @property
    def all_text(self) -> str:
        """Concatenate all element text in reading order."""
        parts = [e.text for e in self.elements if e.text.strip()]
        return "\n\n".join(parts)

    @property
    def all_pages(self) -> list[int]:
        """All page numbers covered by this section and its children."""
        pages: set[int] = set()
        for e in self.elements:
            pages.add(e.page)
        for child in self.children:
            pages.update(child.all_pages)
        if self.page_start:
            pages.add(self.page_start)
        return sorted(pages)

    @property
    def element_count(self) -> int:
        """Total elements including children recursively."""
        count = len(self.elements)
        for child in self.children:
            count += child.element_count
        return count

    def walk_elements(self) -> list[ContentElement]:
        """Yield all elements in reading order, depth-first."""
        result: list[ContentElement] = []
        result.extend(self.elements)
        for child in self.children:
            result.extend(child.walk_elements())
        return result


@dataclass
class DocumentOutline:
    """The full parsed document structure."""

    title: str = ""
    doc_id: str = ""
    total_pages: int = 0
    running_header: str = ""
    pdf_meta_title: str = ""
    sections: list[DocumentSection] = field(default_factory=list)

    @property
    def all_sections_flat(self) -> list[DocumentSection]:
        """All sections at all levels, flattened depth-first."""
        result: list[DocumentSection] = []

        def _walk(section: DocumentSection) -> None:
            result.append(section)
            for child in section.children:
                _walk(child)

        for s in self.sections:
            _walk(s)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-able dict (used to cache Docling's output —
        layout analysis is the slowest step on large PDFs, so persisting its
        result lets a crash during figure extraction or chunking resume
        without re-running it)."""
        return {
            "title": self.title,
            "doc_id": self.doc_id,
            "total_pages": self.total_pages,
            "running_header": self.running_header,
            "pdf_meta_title": self.pdf_meta_title,
            "sections": [_section_to_dict(s) for s in self.sections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentOutline:
        return cls(
            title=data.get("title", ""),
            doc_id=data.get("doc_id", ""),
            total_pages=data.get("total_pages", 0),
            running_header=data.get("running_header", ""),
            pdf_meta_title=data.get("pdf_meta_title", ""),
            sections=[_section_from_dict(s) for s in data.get("sections", [])],
        )

    def summary(self) -> dict[str, Any]:
        all_sections = self.all_sections_flat
        all_elements: list[ContentElement] = []
        for s in self.sections:
            all_elements.extend(s.walk_elements())

        return {
            "title": self.title,
            "total_pages": self.total_pages,
            "top_level_sections": len(self.sections),
            "total_sections": len(all_sections),
            "total_elements": len(all_elements),
            "elements_by_type": {
                et.value: sum(1 for e in all_elements if e.element_type == et)
                for et in ElementType
                if any(e.element_type == et for e in all_elements)
            },
        }


# ---------------------------------------------------------------------------
# (De)serialization helpers for DocumentOutline.to_dict / from_dict
# ---------------------------------------------------------------------------


def _bbox_to_dict(bbox: BoundingBox) -> dict[str, float]:
    return {"left": bbox.left, "top": bbox.top, "width": bbox.width, "height": bbox.height}


def _bbox_from_dict(data: dict[str, Any]) -> BoundingBox:
    return BoundingBox(
        left=data.get("left", 0.0),
        top=data.get("top", 0.0),
        width=data.get("width", 0.0),
        height=data.get("height", 0.0),
    )


def _element_to_dict(element: ContentElement) -> dict[str, Any]:
    return {
        "element_type": element.element_type.value,
        "text": element.text,
        "block_id": element.block_id,
        "page": element.page,
        "bbox": _bbox_to_dict(element.bbox),
        "table_cells": element.table_cells,
        "table_title": element.table_title,
        "table_structure_warning": element.table_structure_warning,
        "table_repaired_cells": element.table_repaired_cells,
        "figure_block_id": element.figure_block_id,
        "figure_caption": element.figure_caption,
        "kv_key": element.kv_key,
        "kv_value": element.kv_value,
        "child_block_ids": element.child_block_ids,
    }


def _element_from_dict(data: dict[str, Any]) -> ContentElement:
    return ContentElement(
        element_type=ElementType(data["element_type"]),
        text=data.get("text", ""),
        block_id=data.get("block_id", ""),
        page=data.get("page", 1),
        bbox=_bbox_from_dict(data.get("bbox", {})),
        table_cells=data.get("table_cells", []),
        table_title=data.get("table_title", ""),
        table_structure_warning=data.get("table_structure_warning"),
        table_repaired_cells=data.get("table_repaired_cells"),
        figure_block_id=data.get("figure_block_id", ""),
        figure_caption=data.get("figure_caption", ""),
        kv_key=data.get("kv_key", ""),
        kv_value=data.get("kv_value", ""),
        child_block_ids=data.get("child_block_ids", []),
    )


def _section_to_dict(section: DocumentSection) -> dict[str, Any]:
    return {
        "title": section.title,
        "level": section.level,
        "page_start": section.page_start,
        "page_end": section.page_end,
        "header_block_id": section.header_block_id,
        "elements": [_element_to_dict(e) for e in section.elements],
        "children": [_section_to_dict(c) for c in section.children],
    }


def _section_from_dict(data: dict[str, Any]) -> DocumentSection:
    return DocumentSection(
        title=data.get("title", ""),
        level=data.get("level", 0),
        page_start=data.get("page_start", 1),
        page_end=data.get("page_end", 1),
        header_block_id=data.get("header_block_id", ""),
        elements=[_element_from_dict(e) for e in data.get("elements", [])],
        children=[_section_from_dict(c) for c in data.get("children", [])],
    )


# ---------------------------------------------------------------------------
# Parsing logic
# ---------------------------------------------------------------------------


def parse_textract_blocks(
    blocks: list[dict[str, Any]],
    *,
    doc_id: str = "",
) -> DocumentOutline:
    """Parse Textract blocks into a structured DocumentOutline.

    Strategy:
    1. Build an ID map for block lookups
    2. Identify all layout blocks and sort by page/position
    3. Use LAYOUT_TITLE and LAYOUT_SECTION_HEADER as section boundaries
    4. Assign content blocks (text, tables, figures) to sections
    5. Build the section tree based on header nesting
    """
    id_map = {b["Id"]: b for b in blocks if "Id" in b}

    # Get total page count
    page_blocks = [b for b in blocks if b.get("BlockType") == "PAGE"]
    total_pages = len(page_blocks) if page_blocks else 1

    # Collect layout blocks in Textract's column-aware reading order.
    layout_blocks = layout_reading_order(blocks)

    # Build table lookup: TABLE blocks indexed by ID
    table_blocks = {b["Id"]: b for b in blocks if b.get("BlockType") == "TABLE"}

    # Track which layout blocks are children of other layout blocks to avoid duplication.
    # Textract sometimes gives LAYOUT_LIST containing LAYOUT_TEXT children.
    child_layout_ids = _find_child_layout_ids(layout_blocks)

    # Parse into sections
    doc_title = ""
    sections: list[DocumentSection] = []
    current_section: DocumentSection | None = None
    section_stack: list[DocumentSection] = []  # For nesting

    for block in layout_blocks:
        bt = block.get("BlockType", "")
        block_id = block.get("Id", "")
        page = block.get("Page", 1)
        bbox = BoundingBox.from_textract(block.get("Geometry", {}).get("BoundingBox", {}))

        # Skip blocks that are children of other layout blocks (avoid duplication)
        if block_id in child_layout_ids:
            continue

        # Skip non-content elements
        if bt in ("LAYOUT_FOOTER", "LAYOUT_HEADER", "LAYOUT_PAGE_NUMBER"):
            continue

        if bt == "LAYOUT_TITLE":
            text = _collect_text(block, id_map).strip()
            if not doc_title:
                doc_title = text
            # Start a new top-level section
            section = DocumentSection(
                title=text,
                level=0,
                page_start=page,
                page_end=page,
                header_block_id=block_id,
            )
            # Close current section stack
            _flush_section_stack(section_stack, sections)
            section_stack = [section]
            current_section = section

        elif bt == "LAYOUT_SECTION_HEADER":
            text = _collect_text(block, id_map).strip()
            section = DocumentSection(
                title=text,
                level=1,
                page_start=page,
                page_end=page,
                header_block_id=block_id,
            )

            if not section_stack:
                # No parent section yet — create an implicit top-level
                section_stack = [section]
                current_section = section
            else:
                # Add as child of current top-level section
                parent = section_stack[0]
                parent.children.append(section)
                parent.page_end = page
                if len(section_stack) > 1:
                    section_stack[1] = section
                else:
                    section_stack.append(section)
                current_section = section

        elif bt == "LAYOUT_TABLE":
            element = _parse_table_element(block, id_map, table_blocks, page, bbox)
            if current_section is None:
                current_section = DocumentSection(
                    title="(Untitled)",
                    level=0,
                    page_start=page,
                    page_end=page,
                )
                section_stack = [current_section]
            current_section.elements.append(element)
            current_section.page_end = page

        elif bt == "LAYOUT_FIGURE":
            element = ContentElement(
                element_type=ElementType.FIGURE,
                text="[Figure]",
                block_id=block_id,
                page=page,
                bbox=bbox,
                figure_block_id=block_id,
                figure_caption=_find_figure_caption(block, layout_blocks, id_map),
            )
            if current_section is None:
                current_section = DocumentSection(
                    title="(Untitled)",
                    level=0,
                    page_start=page,
                    page_end=page,
                )
                section_stack = [current_section]
            current_section.elements.append(element)
            current_section.page_end = page

        elif bt == "LAYOUT_LIST":
            text = _collect_text(block, id_map).strip()
            if text:
                element = ContentElement(
                    element_type=ElementType.LIST,
                    text=text,
                    block_id=block_id,
                    page=page,
                    bbox=bbox,
                )
                if current_section is None:
                    current_section = DocumentSection(
                        title="(Untitled)",
                        level=0,
                        page_start=page,
                        page_end=page,
                    )
                    section_stack = [current_section]
                current_section.elements.append(element)
                current_section.page_end = page

        elif bt == "LAYOUT_TEXT":
            text = _collect_text(block, id_map).strip()
            if text:
                element = ContentElement(
                    element_type=ElementType.TEXT,
                    text=text,
                    block_id=block_id,
                    page=page,
                    bbox=bbox,
                )
                if current_section is None:
                    current_section = DocumentSection(
                        title="(Untitled)",
                        level=0,
                        page_start=page,
                        page_end=page,
                    )
                    section_stack = [current_section]
                current_section.elements.append(element)
                current_section.page_end = page

    # Flush remaining
    _flush_section_stack(section_stack, sections)

    return DocumentOutline(
        title=doc_title,
        doc_id=doc_id,
        total_pages=total_pages,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_child_layout_ids(layout_blocks: list[dict[str, Any]]) -> set[str]:
    """Find layout block IDs that are children of other layout blocks.

    Textract sometimes nests e.g. LAYOUT_TEXT inside LAYOUT_LIST. We want
    to process only the parent to avoid duplicate text.
    """
    child_ids: set[str] = set()
    layout_ids = {b["Id"] for b in layout_blocks if "Id" in b}

    for block in layout_blocks:
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for cid in rel["Ids"]:
                    if cid in layout_ids:
                        child_ids.add(cid)
    return child_ids


def _collect_text(block: dict[str, Any], id_map: dict[str, dict[str, Any]]) -> str:
    """Recursively collect text from a block and its children."""
    if "Text" in block:
        text: str = block["Text"]
        return text

    child_ids = [rel["Ids"] for rel in block.get("Relationships", []) if rel["Type"] == "CHILD"]
    flat_ids = [cid for ids in child_ids for cid in ids]

    texts: list[str] = []
    for cid in flat_ids:
        child = id_map.get(cid)
        if child:
            t = _collect_text(child, id_map)
            if t.strip():
                texts.append(t)
    return " ".join(texts)


def _parse_table_element(
    layout_block: dict[str, Any],
    id_map: dict[str, dict[str, Any]],
    table_blocks: dict[str, dict[str, Any]],
    page: int,
    bbox: BoundingBox,
) -> ContentElement:
    """Parse a LAYOUT_TABLE block and its associated TABLE block."""
    block_id = layout_block.get("Id", "")

    # Find the child TABLE block
    table_data: list[dict[str, Any]] = []
    table_title = ""
    table_text_parts: list[str] = []

    for rel in layout_block.get("Relationships", []):
        if rel["Type"] == "CHILD":
            for cid in rel["Ids"]:
                child = id_map.get(cid)
                if not child:
                    continue
                child_bt = child.get("BlockType", "")

                if child_bt == "TABLE":
                    cells = _extract_table_cells(child, id_map)
                    table_data.extend(cells)
                    table_text_parts.append(_table_cells_to_text(cells))

                elif child_bt == "TABLE_TITLE":
                    table_title = _collect_text(child, id_map).strip()

    # Fallback: try to find text directly
    if not table_text_parts:
        table_text_parts.append(_collect_text(layout_block, id_map).strip())

    text = (
        table_title + "\n" + "\n".join(table_text_parts)
        if table_title
        else "\n".join(table_text_parts)
    )

    return ContentElement(
        element_type=ElementType.TABLE,
        text=text.strip(),
        block_id=block_id,
        page=page,
        bbox=bbox,
        table_cells=table_data,
        table_title=table_title,
    )


def _extract_table_cells(
    table_block: dict[str, Any],
    id_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract structured cell data from a TABLE block."""
    cells: list[dict[str, Any]] = []

    for rel in table_block.get("Relationships", []):
        if rel["Type"] != "CHILD":
            continue
        for cid in rel["Ids"]:
            child = id_map.get(cid)
            if not child:
                continue
            if child.get("BlockType") not in ("CELL", "MERGED_CELL"):
                continue

            cell_text = _collect_text(child, id_map).strip()
            cells.append(
                {
                    "row": child.get("RowIndex", 0),
                    "col": child.get("ColumnIndex", 0),
                    "row_span": child.get("RowSpan", 1),
                    "col_span": child.get("ColumnSpan", 1),
                    "text": cell_text,
                    "is_header": child.get("EntityTypes", []) == ["COLUMN_HEADER"],
                }
            )

    return cells


def _table_cells_to_text(cells: list[dict[str, Any]]) -> str:
    """Convert structured table cells to a readable text representation."""
    if not cells:
        return ""

    # Group by row
    rows: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        row_idx = cell.get("row", 0)
        rows.setdefault(row_idx, []).append(cell)

    # Sort each row by column
    lines: list[str] = []
    for row_idx in sorted(rows.keys()):
        row_cells = sorted(rows[row_idx], key=lambda c: c.get("col", 0))
        line = " | ".join(c["text"] for c in row_cells)
        lines.append(line)

    return "\n".join(lines)


def _find_figure_caption(
    fig_block: dict[str, Any],
    layout_blocks: list[dict[str, Any]],
    id_map: dict[str, dict[str, Any]],
) -> str:
    """Find caption text near a figure block."""
    fig_page = fig_block.get("Page", 1)
    fig_bottom = fig_block.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0) + fig_block.get(
        "Geometry", {}
    ).get("BoundingBox", {}).get("Height", 0)

    fig_idx = None
    for i, lb in enumerate(layout_blocks):
        if lb.get("Id") == fig_block.get("Id"):
            fig_idx = i
            break

    if fig_idx is None:
        return ""

    for j in range(fig_idx + 1, min(fig_idx + 4, len(layout_blocks))):
        candidate = layout_blocks[j]
        if candidate.get("Page", 1) != fig_page:
            break
        bt = candidate.get("BlockType", "")
        if bt in ("LAYOUT_SECTION_HEADER", "LAYOUT_TITLE", "LAYOUT_FIGURE"):
            break

        cand_top = candidate.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0)
        if bt == "LAYOUT_TEXT" and (cand_top - fig_bottom) < 0.05:
            text = _collect_text(candidate, id_map).strip()
            if text and (text.lower().startswith(("figure", "fig.", "fig ")) or len(text) < 200):
                return text
    return ""


def _flush_section_stack(
    stack: list[DocumentSection],
    sections: list[DocumentSection],
) -> None:
    """Push completed section from stack to the sections list."""
    if stack:
        sections.append(stack[0])
    stack.clear()
