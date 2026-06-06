"""Docling-based layout parsing for native (text-embedded) PDFs.

Produces a DocumentOutline and FigureRegion list using the same data
structures as the Textract path, so all downstream pipeline steps
(chunking, embedding, storage) are backend-agnostic.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from rich.console import Console

from aws_rag.chunking.layout_parser import (
    BoundingBox,
    ContentElement,
    DocumentOutline,
    DocumentSection,
    ElementType,
)
from aws_rag.figures import FigureRegion

console = Console()

_NATIVE_CHAR_THRESHOLD = 100  # total chars across sample pages to confirm native PDF
_MULTI_SPACE_RE = re.compile(r" {2,}")


def is_native_pdf(pdf_path: Path, sample_pages: int = 5) -> bool:
    """Return True if the PDF has selectable embedded text (not a scan)."""
    try:
        import fitz  # pymupdf
    except ImportError as exc:
        raise ImportError(
            "pymupdf is required for native PDF detection. "
            "Install: pip install 'aws-rag[docling]'"
        ) from exc

    doc = fitz.open(str(pdf_path))
    total_chars = 0
    for i in range(min(sample_pages, len(doc))):
        total_chars += len(doc[i].get_text().strip())
    doc.close()
    return total_chars > _NATIVE_CHAR_THRESHOLD


def content_hash(pdf_path: Path) -> str:
    """SHA-256 content hash — same algorithm used by storage.upload_pdf."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_pdf(
    pdf_path: Path,
    *,
    doc_id: str = "",
) -> tuple[DocumentOutline, list[FigureRegion]]:
    """Run Docling on a native PDF and return (DocumentOutline, figure_regions).

    figure_regions includes both PICTURE and FORMULA regions so they can be
    cropped by the same figure-extraction pipeline used for the Textract path.
    Requires: pip install 'aws-rag[docling]'
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise ImportError(
            "docling is required for native PDF parsing. "
            "Install: pip install 'aws-rag[docling]'"
        ) from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    console.print(f"[blue]Docling analysing[/] {pdf_path.name} …")
    result = converter.convert(str(pdf_path))
    doc = result.document
    console.print(f"[green]Docling done[/] — {len(doc.pages)} pages")

    outline = _build_outline(doc, doc_id=doc_id)
    regions = _build_figure_regions(doc)
    return outline, regions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Collapse runs of spaces from PDF justified-text positioning."""
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def _page_size(doc: Any, page_no: int) -> tuple[float, float]:
    """Return (width, height) in points for a 1-indexed page."""
    page = doc.pages.get(page_no)
    if page and page.size:
        return float(page.size.width), float(page.size.height)
    return 612.0, 792.0  # US Letter fallback


def _norm_bbox(bbox: Any, page_w: float, page_h: float) -> BoundingBox:
    """Convert a Docling BoundingBox to our top-left-origin, 0..1 normalised form."""
    try:
        # Docling default: bottom-left origin. Convert to top-left for consistency.
        tl = bbox.to_top_left_origin(page_h)
        left = tl.l / page_w
        top = tl.t / page_h
        width = (tl.r - tl.l) / page_w
        height = (tl.b - tl.t) / page_h
    except AttributeError:
        # Fallback for older docling versions without to_top_left_origin
        left = bbox.l / page_w
        top = (page_h - bbox.t) / page_h
        width = (bbox.r - bbox.l) / page_w
        height = (bbox.t - bbox.b) / page_h

    return BoundingBox(
        left=max(0.0, min(1.0, left)),
        top=max(0.0, min(1.0, top)),
        width=max(0.0, min(1.0, width)),
        height=max(0.0, min(1.0, height)),
    )


def _item_prov(item: Any) -> tuple[int, Any] | tuple[None, None]:
    """Return (page_no, bbox) from an item's provenance, or (None, None)."""
    if item.prov:
        p = item.prov[0]
        return p.page_no, p.bbox
    return None, None


def _table_to_cells(table_item: Any) -> list[dict[str, Any]]:
    """Convert Docling table grid to our table_cells list format."""
    cells: list[dict[str, Any]] = []
    try:
        for row_idx, row in enumerate(table_item.data.grid):
            for col_idx, cell in enumerate(row):
                cells.append({
                    "row": row_idx + 1,
                    "col": col_idx + 1,
                    "row_span": getattr(cell, "row_span", 1),
                    "col_span": getattr(cell, "col_span", 1),
                    "text": getattr(cell, "text", ""),
                    "is_header": (
                        getattr(cell, "column_header", False)
                        or getattr(cell, "row_header", False)
                    ),
                })
    except (AttributeError, TypeError):
        pass
    return cells


def _ensure_section(
    page_no: int,
    current_section: DocumentSection | None,
    section_stack: list[DocumentSection],
) -> DocumentSection:
    if current_section is not None:
        return current_section
    s = DocumentSection(title="(Untitled)", level=0, page_start=page_no, page_end=page_no)
    section_stack.append(s)
    return s


def _build_outline(doc: Any, doc_id: str) -> DocumentOutline:
    """Walk the DoclingDocument in reading order and build a DocumentOutline."""
    try:
        from docling_core.types.doc import DocItemLabel
    except ImportError as exc:
        raise ImportError("docling-core is required") from exc

    _SKIP = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
    try:
        _SKIP.add(DocItemLabel.FOOTNOTE)
    except AttributeError:
        pass

    doc_title = ""
    sections: list[DocumentSection] = []
    current_section: DocumentSection | None = None
    section_stack: list[DocumentSection] = []
    counter = 0
    figure_counter = 0   # tracks PICTURE items only — must match _build_figure_regions
    formula_counter = 0  # tracks FORMULA items only — must match _build_figure_regions

    # Last PICTURE/FORMULA element waiting for a CAPTION to be assigned.
    pending_caption_element: ContentElement | None = None

    for item, _level in doc.iterate_items():
        label = item.label

        # --- Captions: assign to the immediately preceding figure/formula ----
        # Docling emits CAPTION items right after their parent PICTURE in reading
        # order.  We grab the text here and attach it instead of creating a
        # standalone content element (which would create a spurious text chunk).
        if label == DocItemLabel.CAPTION:
            text = _clean_text(getattr(item, "text", ""))
            if text and pending_caption_element is not None:
                pending_caption_element.figure_caption = text
            pending_caption_element = None
            continue  # do not add caption as a content element

        if label in _SKIP:
            continue

        page_no, raw_bbox = _item_prov(item)
        if page_no is None:
            page_no = 1
            norm_bbox = BoundingBox()
        else:
            pw, ph = _page_size(doc, page_no)
            norm_bbox = _norm_bbox(raw_bbox, pw, ph)

        counter += 1
        block_id = f"docling_{counter}"

        # --- Section headers ---
        if label == DocItemLabel.TITLE:
            text = _clean_text(getattr(item, "text", ""))
            pending_caption_element = None
            if not doc_title:
                doc_title = text
            section = DocumentSection(
                title=text, level=0, page_start=page_no, page_end=page_no,
                header_block_id=block_id,
            )
            _flush(section_stack, sections)
            section_stack = [section]
            current_section = section
            continue

        if label == DocItemLabel.SECTION_HEADER:
            text = _clean_text(getattr(item, "text", ""))
            pending_caption_element = None
            section = DocumentSection(
                title=text, level=1, page_start=page_no, page_end=page_no,
                header_block_id=block_id,
            )
            if not section_stack:
                section_stack = [section]
                current_section = section
            else:
                parent = section_stack[0]
                parent.children.append(section)
                parent.page_end = page_no
                if len(section_stack) > 1:
                    section_stack[1] = section
                else:
                    section_stack.append(section)
                current_section = section
            continue

        # --- Content elements ---
        current_section = _ensure_section(page_no, current_section, section_stack)
        element: ContentElement | None = None

        if label == DocItemLabel.TABLE:
            pending_caption_element = None
            cells = _table_to_cells(item)
            try:
                table_md = item.export_to_markdown(doc=doc)
            except Exception:
                table_md = "\n".join(c["text"] for c in cells if c["text"])
            caption = _clean_text(getattr(item, "text", "") or "")
            text = (caption + "\n" + table_md).strip() if caption else table_md.strip()
            element = ContentElement(
                element_type=ElementType.TABLE,
                text=text,
                block_id=block_id,
                page=page_no,
                bbox=norm_bbox,
                table_cells=cells,
                table_title=caption,
            )

        elif label == DocItemLabel.PICTURE:
            figure_counter += 1
            fig_block_id = f"docling_figure_{figure_counter}"
            element = ContentElement(
                element_type=ElementType.FIGURE,
                text="[Figure]",
                block_id=block_id,
                page=page_no,
                bbox=norm_bbox,
                figure_block_id=fig_block_id,
                figure_caption="",  # filled in when we hit the following CAPTION
            )
            pending_caption_element = element

        elif label == DocItemLabel.FORMULA:
            formula_counter += 1
            formula_block_id = f"docling_formula_{formula_counter}"
            text = _clean_text(getattr(item, "text", "")) or "[Formula]"
            element = ContentElement(
                element_type=ElementType.FORMULA,
                text=text,
                block_id=block_id,
                page=page_no,
                bbox=norm_bbox,
                figure_block_id=formula_block_id,
            )
            pending_caption_element = element

        elif label in (DocItemLabel.TEXT, DocItemLabel.LIST_ITEM,
                       DocItemLabel.CODE):
            pending_caption_element = None
            text = _clean_text(getattr(item, "text", ""))
            if text:
                etype = (ElementType.LIST if label == DocItemLabel.LIST_ITEM
                         else ElementType.TEXT)
                element = ContentElement(
                    element_type=etype,
                    text=text,
                    block_id=block_id,
                    page=page_no,
                    bbox=norm_bbox,
                )
        else:
            pending_caption_element = None

        if element is not None:
            current_section.elements.append(element)
            current_section.page_end = page_no

    _flush(section_stack, sections)

    return DocumentOutline(
        title=doc_title,
        doc_id=doc_id,
        total_pages=len(doc.pages),
        sections=sections,
    )


def _build_figure_regions(doc: Any) -> list[FigureRegion]:
    """Collect PICTURE and FORMULA items as FigureRegion objects for cropping.

    Iterates all items so that CAPTION items immediately following a figure
    can be assigned to it.  IDs use separate per-kind counters that stay in
    sync with the figure_counter / formula_counter in _build_outline.
    """
    try:
        from docling_core.types.doc import DocItemLabel
    except ImportError as exc:
        raise ImportError("docling-core is required") from exc

    regions: list[FigureRegion] = []
    figure_counter = 0
    formula_counter = 0
    pending_region: FigureRegion | None = None

    for item, _level in doc.iterate_items():
        # Assign caption text to the preceding figure/formula region.
        if item.label == DocItemLabel.CAPTION:
            text = _clean_text(getattr(item, "text", ""))
            if text and pending_region is not None:
                pending_region.caption = text
            pending_region = None
            continue

        if item.label not in (DocItemLabel.PICTURE, DocItemLabel.FORMULA):
            # Any non-caption, non-figure item breaks the figure→caption adjacency.
            if item.label not in (
                DocItemLabel.PAGE_HEADER,
                DocItemLabel.PAGE_FOOTER,
            ):
                try:
                    if item.label != DocItemLabel.FOOTNOTE:
                        pending_region = None
                except AttributeError:
                    pending_region = None
            continue

        page_no, raw_bbox = _item_prov(item)
        if page_no is None:
            continue

        pw, ph = _page_size(doc, page_no)
        nb = _norm_bbox(raw_bbox, pw, ph)

        if item.label == DocItemLabel.FORMULA:
            formula_counter += 1
            block_id = f"docling_formula_{formula_counter}"
            kind = "formula"
        else:
            figure_counter += 1
            block_id = f"docling_figure_{figure_counter}"
            kind = "figure"

        region = FigureRegion(
            block_id=block_id,
            page=page_no,
            left=nb.left,
            top=nb.top,
            width=nb.width,
            height=nb.height,
            caption="",
            kind=kind,
        )
        regions.append(region)
        pending_region = region

    return regions


def _flush(stack: list[DocumentSection], sections: list[DocumentSection]) -> None:
    if stack:
        sections.append(stack[0])
    stack.clear()
