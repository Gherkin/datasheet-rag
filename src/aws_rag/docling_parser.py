"""Docling-based layout parsing for native (text-embedded) PDFs.

Produces a DocumentOutline and FigureRegion list using the same data
structures as the Textract path, so all downstream pipeline steps
(chunking, embedding, storage) are backend-agnostic.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
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
_TOC_TITLES = frozenset({"table of contents", "contents", "index", "toc"})
# "Section heading ........ 12" or "Section heading        12"
_TOC_ENTRY_RE = re.compile(r"^.+[.\s]{2,}\d+\s*$")

# Dotted numeric heading prefixes, e.g. "2." / "2.10." / "2.10.1.2." — used to
# recover chapter structure when Docling emits no TITLE items and flattens
# every heading to the same SECTION_HEADER level (see _restructure_flat_chapter).
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
# A flat chapter is only a candidate for restructuring once it's this large —
# small flat chapters are just... small chapters.
_FLAT_CHAPTER_MIN_CHILDREN = 50
_FLAT_CHAPTER_MIN_NUMBERED = 5


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


def _pdf_embedded_title(pdf_path: Path) -> str:
    """Combine the PDF's embedded ``/Title`` and ``/Subject`` metadata, if any.

    Some publishers split a document's real name across these two fields
    (e.g. Title="Programmer's Guide", Subject="CC Linux") — Docling never
    sees this, since it isn't part of the rendered page content. Returned
    as a single hint string, or "" if neither field is usable.
    """
    try:
        import fitz  # pymupdf

        with fitz.open(str(pdf_path)) as doc:
            meta = doc.metadata or {}
    except Exception:
        return ""

    title = (meta.get("title") or "").strip()
    subject = (meta.get("subject") or "").strip()
    parts = [p for p in (subject, title) if p and p.lower() not in ("untitled", "")]
    # Drop an exact duplicate (Subject == Title) rather than repeating it.
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():
        parts = parts[:1]
    return " ".join(parts)


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
    accurate_tables: bool = False,
    page_range: tuple[int, int] | None = None,
) -> tuple[DocumentOutline, list[FigureRegion]]:
    """Run Docling on a native PDF and return (DocumentOutline, figure_regions).

    figure_regions includes both PICTURE and FORMULA regions so they can be
    cropped by the same figure-extraction pipeline used for the Textract path.
    Requires: pip install 'aws-rag[docling]'

    accurate_tables: use TableFormerMode.ACCURATE instead of FAST.
      FAST (default) is 44% faster with negligible quality loss for RAG use;
      ACCURATE adds precise cell-boundary detection useful for post-processing
      table structure but costs ~2.4× the total pipeline time.

    page_range: optional 1-based inclusive ``(start, end)`` to convert only
      part of the PDF — used by ``reconvert_tables_in_range`` to selectively
      re-run a few pages with ACCURATE tables instead of the whole document.
      Page numbers in the resulting outline are the PDF's true page numbers
      (Docling's provenance tracks absolute page numbers regardless of range).
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            PdfPipelineOptions,
            TableFormerMode,
            TableStructureOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise ImportError(
            "docling is required for native PDF parsing. "
            "Install: pip install 'aws-rag[docling]'"
        ) from exc

    tmode = TableFormerMode.ACCURATE if accurate_tables else TableFormerMode.FAST
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(mode=tmode)
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CUDA,
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    mode_label = "accurate tables" if accurate_tables else "fast tables"
    range_label = f", pages {page_range[0]}-{page_range[1]}" if page_range else ""
    console.print(f"[blue]Docling analysing[/] {pdf_path.name} ({mode_label}{range_label}) …")
    convert_kwargs: dict[str, Any] = {}
    if page_range is not None:
        convert_kwargs["page_range"] = page_range
    result = converter.convert(str(pdf_path), **convert_kwargs)
    doc = result.document
    console.print(f"[green]Docling done[/] — {len(doc.pages)} pages")

    regions = _build_figure_regions(doc)
    skip_figure_ids, regions = _dedup_repeating_figures(regions, pdf_path, len(doc.pages))
    outline = _build_outline(doc, doc_id=doc_id, skip_figure_ids=skip_figure_ids)
    outline.pdf_meta_title = _pdf_embedded_title(pdf_path)
    return outline, regions


def _bbox_overlap(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-union of two normalised (0..1) bounding boxes."""
    ix = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    iy = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def reconvert_tables_in_range(
    pdf_path: Path,
    outline: DocumentOutline,
    *,
    doc_id: str,
    page_range: tuple[int, int],
    accurate_tables: bool = True,
) -> list[dict[str, Any]]:
    """Re-run Docling table-structure recognition for a page range and patch
    matching TABLE elements into ``outline`` in place.

    Re-running layout analysis on an entire multi-thousand-page document just
    to fix one or two misparsed tables is wasteful — this instead converts
    only ``page_range`` (via Docling's native ``page_range`` support, which
    keeps true PDF page numbers in the resulting provenance) and patches the
    TABLE elements it finds onto their counterparts in ``outline``, matched by
    page number plus bounding-box overlap (tables are leaf elements with a
    clear geometric identity, unlike sections/headings — matching at that
    granularity avoids the coordinate-frame and tree-merge guesswork that
    made whole-section reconciliation infeasible, see the multi-page-table
    warning in _build_outline).

    Returns a report: one dict per cached table in range, each noting whether
    a geometric match was found and how its size/garbled-header status changed
    — so the caller can decide whether to keep the patch (e.g. abort a dry run
    if nothing actually improved).
    """
    target_pages = set(range(page_range[0], page_range[1] + 1))

    cached_tables: list[ContentElement] = [
        el
        for section in outline.all_sections_flat
        for el in section.elements
        if el.element_type == ElementType.TABLE and el.page in target_pages
    ]

    partial_outline, _regions = convert_pdf(
        pdf_path, doc_id=doc_id, accurate_tables=accurate_tables, page_range=page_range
    )
    fresh_tables: list[ContentElement] = [
        el
        for section in partial_outline.all_sections_flat
        for el in section.elements
        if el.element_type == ElementType.TABLE
    ]

    report: list[dict[str, Any]] = []
    used_fresh: set[int] = set()
    for cached in cached_tables:
        candidates = [
            (i, fresh)
            for i, fresh in enumerate(fresh_tables)
            if i not in used_fresh and fresh.page == cached.page
        ]
        best_idx, best_overlap = None, 0.0
        for i, fresh in candidates:
            overlap = _bbox_overlap(cached.bbox, fresh.bbox)
            if overlap > best_overlap:
                best_idx, best_overlap = i, overlap

        entry: dict[str, Any] = {
            "page": cached.page,
            "caption": cached.table_title,
            "old_chars": len(cached.text),
            "old_garbled": _detect_garbled_header(cached.table_cells) is not None,
        }
        if best_idx is None or best_overlap < 0.1:
            entry["matched"] = False
            report.append(entry)
            continue

        fresh = fresh_tables[best_idx]
        used_fresh.add(best_idx)
        cached.table_cells = fresh.table_cells
        cached.table_title = fresh.table_title
        cached.text = fresh.text
        cached.bbox = fresh.bbox
        entry.update(
            matched=True,
            overlap=best_overlap,
            new_chars=len(fresh.text),
            new_garbled=_detect_garbled_header(fresh.table_cells) is not None,
        )
        report.append(entry)

    return report


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
    """Convert Docling table grid to our table_cells list format.

    Docling's ``grid`` repeats a spanning cell's text at every (row, col)
    position it covers — that's how the grid is filled in. We additionally
    record ``is_origin`` (true only at the cell's top-left position, per its
    ``start_row/col_offset_idx``) so renderers can emit each cell's text
    exactly once instead of duplicating it across every position it spans.
    """
    cells: list[dict[str, Any]] = []
    try:
        for row_idx, row in enumerate(table_item.data.grid):
            for col_idx, cell in enumerate(row):
                is_origin = (
                    row_idx == getattr(cell, "start_row_offset_idx", row_idx)
                    and col_idx == getattr(cell, "start_col_offset_idx", col_idx)
                )
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
                    "is_origin": is_origin,
                })
    except (AttributeError, TypeError):
        pass
    return cells


# A repeated header cell needs to be at least this long to count as a
# meaningful duplication signal — short tokens ("-", "Pin", "0x0") legitimately
# repeat across columns without indicating a parsing failure.
_GARBLED_HEADER_MIN_CHARS = 12
# How many times a single header cell's text must repeat before we treat it
# as TableFormer having misparsed a complex (multi-level / rotated-text)
# header rather than correctly recognising the table's real column structure.
_GARBLED_HEADER_MIN_REPEATS = 3


def _detect_garbled_header(cells: list[dict[str, Any]]) -> str | None:
    """Return the repeated text if this table's header looks misparsed.

    TableFormer can fail on complex multi-level / rotated-text headers (the
    kind found in dense pinout and register-summary tables) by stamping one
    cell's text across many header positions instead of recognising the
    actual nested structure — e.g. "Oscillator I/O PMUX Values 0x0 0x1 ..."
    repeated across 15 separate header cells in PIC32CK1025GC01100's Table
    5-3, each claiming colspan=15. That's not a real 15-column-group table;
    it's a parse failure that (a) misleads anything reading the header and
    (b) inflates every data row's rendered width via markdown-style padding.

    We can't reconstruct the correct header from its own corrupted output —
    detection only, so the table can be flagged for manual review (e.g.
    re-running with --accurate-tables) and the bad header text can be
    suppressed from what gets embedded.
    """
    header_texts = [
        c["text"].strip()
        for c in cells
        if c.get("is_header") and c.get("is_origin", True)
        and len(c["text"].strip()) >= _GARBLED_HEADER_MIN_CHARS
    ]
    for text, count in Counter(header_texts).most_common(1):
        if count >= _GARBLED_HEADER_MIN_REPEATS:
            return text
    return None


def _table_cells_to_compact_text(cells: list[dict[str, Any]], *, omit_header: bool = False) -> str:
    """Render structured table cells as compact, unpadded pipe-delimited rows.

    Unlike Docling's ``export_to_markdown`` (which pads every cell to a
    uniform column width for visual alignment), this emits each cell's text
    as-is. Alignment padding is pure display convenience — it costs real
    embedding-token budget without adding any retrievable signal, and a
    single oversized header cell can inflate the padding width applied to
    *every* row and column (a 50-row × 17-col table rendered to 67KB here —
    9× its actual ~7.5KB content — purely from padding "PB09" out to ~80
    chars to match a misparsed header cell; see _detect_garbled_header).

    Spanning cells are emitted once, at their origin position, to avoid
    repeating their text at every grid position the span covers.

    ``omit_header`` drops detected-garbled header rows entirely — feeding a
    known-wrong column structure to the embedding is worse than feeding none.
    """
    if not cells:
        return ""
    rows: dict[int, list[dict[str, Any]]] = {}
    header_rows: set[int] = set()
    for cell in cells:
        rows.setdefault(cell["row"], []).append(cell)
        if cell.get("is_header"):
            header_rows.add(cell["row"])

    lines: list[str] = []
    for row_idx in sorted(rows):
        if omit_header and row_idx in header_rows:
            continue
        row_cells = sorted(rows[row_idx], key=lambda c: c["col"])
        texts = [c["text"].strip() if c.get("is_origin", True) else "" for c in row_cells]
        if any(texts):
            lines.append(" | ".join(texts))
    return "\n".join(lines)


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


def _build_outline(
    doc: Any,
    doc_id: str,
    skip_figure_ids: set[str] | None = None,
) -> DocumentOutline:
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
    header_texts: Counter[str] = Counter()
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
            if label == DocItemLabel.PAGE_HEADER:
                text = _clean_text(getattr(item, "text", ""))
                if text:
                    header_texts[text] += 1
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
            caption = _clean_text(getattr(item, "text", "") or "")

            prov = list(getattr(item, "prov", []) or [])
            if len(prov) > 1:
                pages = sorted({p.page_no for p in prov})
                where = f"'{caption}'" if caption else f"starting on page {page_no}"
                console.print(
                    f"[yellow]Table parsing warning:[/] table {where} is a "
                    f"single Docling item physically spanning pages {pages} — "
                    f"continuation pages are normally separate items that "
                    f"chunk independently, but this one was merged. Splitting "
                    f"it at its true page boundaries would require reconciling "
                    f"mismatched coordinate frames (table provenance is "
                    f"BOTTOMLEFT-origin, cell bboxes are TOPLEFT-origin and "
                    f"often absent for empty cells) with no real instance to "
                    f"validate against — rather than guess, it's kept as one "
                    f"chunk and is subject to the hard-size check below."
                )

            garbled = _detect_garbled_header(cells)
            if garbled is not None:
                where = f"'{caption}'" if caption else f"on page {page_no}"
                console.print(
                    f"[yellow]Table parsing warning:[/] table {where} has a "
                    f"header cell repeated across multiple columns "
                    f"({garbled[:60]!r}…) — Docling's table-structure "
                    f"recognition likely misparsed a complex multi-level "
                    f"header. The garbled header is being dropped from the "
                    f"embedded text (this safety net applies regardless of "
                    f"table-structure mode). Inspect page {page_no} to "
                    f"confirm the real structure; re-running with "
                    f"--accurate-tables (or table_structure_mode=accurate in "
                    f"the global config) sometimes helps but is NOT a "
                    f"guaranteed fix — empirically it can produce a "
                    f"differently-garbled header for complex nested tables "
                    f"like this one."
                )
            table_text = _table_cells_to_compact_text(cells, omit_header=garbled is not None)
            if not table_text:
                try:
                    table_text = item.export_to_markdown(doc=doc)
                except Exception:
                    table_text = "\n".join(c["text"] for c in cells if c["text"])

            text = (caption + "\n" + table_text).strip() if caption else table_text.strip()
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
            if skip_figure_ids and fig_block_id in skip_figure_ids:
                # Repeating header/footer image — drop it; clear pending so the
                # following CAPTION (if any) is consumed without being attached.
                pending_caption_element = None
            else:
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
    sections = _restructure_degenerate_outline(sections)
    sections = _filter_toc_sections(sections)

    running_header = ""
    if header_texts:
        text, count = header_texts.most_common(1)[0]
        if count >= 2:
            running_header = text

    return DocumentOutline(
        title=doc_title,
        doc_id=doc_id,
        total_pages=len(doc.pages),
        running_header=running_header,
        sections=sections,
    )


def _dedup_repeating_figures(
    regions: list[FigureRegion],
    pdf_path: Path,
    total_pages: int,
    min_repeat_pages: int = 3,
    header_top_thresh: float = 0.10,
    footer_top_thresh: float = 0.90,
) -> tuple[set[str], list[FigureRegion]]:
    """Detect figures that repeat across pages and are in the header/footer zone.

    Strategy:
      1. Size bucket — only compare regions with similar normalised dimensions.
      2. Pixel hash — MD5 of a 72-dpi crop from PyMuPDF; identical images match.
      3. Frequency + position — if >= min_repeat_pages occurrences share the same
         hash AND each occurrence sits in the header (top < header_top_thresh) or
         footer (top+height > footer_top_thresh), all are dropped.

    Returns (skipped_block_ids, kept_regions).  skipped_block_ids contains the
    figure_block_ids of dropped regions so _build_outline can omit them too.
    """
    import fitz
    from collections import defaultdict

    if not regions:
        return set(), regions

    # Stage 1: group by normalised size (rounded to 2 dp ≈ 1 % of page)
    size_groups: dict[tuple[float, float], list[FigureRegion]] = defaultdict(list)
    for r in regions:
        size_groups[round(r.width, 2), round(r.height, 2)].append(r)

    skip_ids: set[str] = set()
    doc = fitz.open(str(pdf_path))

    for group in size_groups.values():
        if len(group) < min_repeat_pages:
            continue  # too few to be a repeating logo

        # Stage 2: pixel hash for each member of the size-matched group
        hash_to_regions: dict[bytes, list[FigureRegion]] = defaultdict(list)
        for r in group:
            page = doc[r.page - 1]
            pw, ph = page.rect.width, page.rect.height
            rect = fitz.Rect(
                r.left * pw,
                r.top * ph,
                (r.left + r.width) * pw,
                (r.top + r.height) * ph,
            )
            pix = page.get_pixmap(clip=rect, dpi=72)
            h = hashlib.md5(pix.samples).digest()
            hash_to_regions[h].append(r)

        # Stage 3: frequency + position gate
        for matched in hash_to_regions.values():
            pages_seen = {r.page for r in matched}
            if len(pages_seen) < min_repeat_pages:
                continue
            for r in matched:
                in_header = r.top < header_top_thresh
                in_footer = (r.top + r.height) > footer_top_thresh
                if in_header or in_footer:
                    skip_ids.add(r.block_id)

    doc.close()

    if skip_ids:
        console.print(
            f"[yellow]Dedup:[/] dropped {len(skip_ids)} repeating header/footer "
            f"figure(s) (out of {len(regions)} total, {total_pages} pages)"
        )

    return skip_ids, [r for r in regions if r.block_id not in skip_ids]


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


def _parse_numbered_heading(title: str) -> tuple[int, int] | None:
    """Parse a dotted numeric heading prefix into ``(depth, leading_number)``.

    "1. Configuration Summary"             -> (1, 1)
    "2.1. Basic Connection Requirements"   -> (2, 2)
    "2.10.1.2. PCB Layout Recommendations" -> (4, 2)

    Returns ``None`` when the title carries no such prefix (marketing
    blurbs, bullet fragments, continuation headers, ...).
    """
    m = _NUMBERED_HEADING_RE.match(title.strip())
    if not m:
        return None
    segments = m.group(1).split(".")
    return len(segments), int(segments[0])


def _restructure_flat_chapter(chapter: DocumentSection) -> list[DocumentSection]:
    """Re-nest a chapter's flat children using their numeric heading prefixes.

    Triggered when Docling emits no usable TITLE/level hierarchy for a
    document and every heading collapses to one flat list of SECTION_HEADER
    siblings (seen on a 2200-page MCU datasheet: one "chapter" swallowed
    3949 sections). Real chapter headings in such documents are reliably
    numbered ("1.", "2.1.", "2.10.1.2., ..."), so we use that numbering as
    the depth signal and rebuild a proper tree from it.

    Children before the first numbered heading (cover-page subtitles,
    feature blurbs) are bucketed into a synthetic front-matter chapter so
    they don't pollute the real chapter list. Stray headings that match the
    numeric-prefix pattern but don't continue the chapter sequence — numbered
    list items or notes inside body text, e.g. "2. Configure the EVSYS:"
    appearing between chapters 33 and 34 — and un-numbered fragments
    ("· Component Placement:", "Notes:") are nested under whatever section
    is currently open — harmless leaf nodes, not structural problems.
    """
    front_matter = DocumentSection(
        title=chapter.title,
        level=0,
        page_start=chapter.page_start,
        page_end=chapter.page_end,
        elements=list(chapter.elements),
        header_block_id=chapter.header_block_id,
    )

    top_level: list[DocumentSection] = []
    stack: list[tuple[int, DocumentSection]] = []  # (numbered depth, section)
    seen_numbered = False
    next_chapter_num: int | None = None

    for child in chapter.children:
        parsed = _parse_numbered_heading(child.title)

        if parsed is not None:
            depth, leading_num = parsed
            # A depth-1 heading only starts a new chapter if it continues the
            # chapter-numbering sequence — numbered list items / notes in body
            # text match the same prefix pattern but break the sequence.
            if depth == 1 and next_chapter_num is not None and leading_num != next_chapter_num:
                parsed = None

        if parsed is None:
            if not seen_numbered:
                front_matter.children.append(child)
            else:
                target = stack[-1][1] if stack else (top_level[-1] if top_level else front_matter)
                target.children.append(child)
            continue

        depth, leading_num = parsed
        seen_numbered = True
        if depth == 1:
            next_chapter_num = leading_num + 1

        while stack and stack[-1][0] >= depth:
            stack.pop()

        if not stack:
            child.level = 0
            top_level.append(child)
        else:
            child.level = len(stack)
            stack[-1][1].children.append(child)
        stack.append((depth, child))

    result: list[DocumentSection] = []
    if front_matter.children or front_matter.elements:
        result.append(front_matter)
    result.extend(top_level)
    return result


def _restructure_degenerate_outline(sections: list[DocumentSection]) -> list[DocumentSection]:
    """Detect and fix a flattened single-chapter outline (see _restructure_flat_chapter).

    Only triggers when the outline is exactly one chapter with an unusually
    large, mostly-numbered flat child list — i.e. genuinely small
    single-chapter documents are left untouched.
    """
    if len(sections) != 1:
        return sections

    chapter = sections[0]
    if len(chapter.children) < _FLAT_CHAPTER_MIN_CHILDREN:
        return sections

    numbered = sum(1 for c in chapter.children if _parse_numbered_heading(c.title) is not None)
    if numbered < _FLAT_CHAPTER_MIN_NUMBERED:
        return sections

    console.print(
        f"[yellow]Outline restructure:[/] '{chapter.title}' looked flattened "
        f"({len(chapter.children)} flat children, {numbered} numbered) — "
        f"rebuilding chapter structure from numeric heading prefixes."
    )
    return _restructure_flat_chapter(chapter)


def _is_toc_section(section: DocumentSection) -> bool:
    if section.title.lower().strip() in _TOC_TITLES:
        return True
    text_els = [e for e in section.elements if e.element_type == ElementType.TEXT]
    if len(text_els) < 10:
        return False
    matches = sum(1 for e in text_els if _TOC_ENTRY_RE.match(e.text))
    return matches / len(text_els) >= 0.60


def _filter_toc_sections(sections: list[DocumentSection]) -> list[DocumentSection]:
    """Remove ToC sections at any nesting level. Applied after the outline is built."""
    kept: list[DocumentSection] = []
    for s in sections:
        if _is_toc_section(s):
            console.print(
                f"[yellow]ToC filter:[/] dropped '{s.title}' "
                f"({len(s.elements)} elements)"
            )
            continue
        s.children = _filter_toc_sections(s.children)
        kept.append(s)
    return kept


def _flush(stack: list[DocumentSection], sections: list[DocumentSection]) -> None:
    if stack:
        sections.append(stack[0])
    stack.clear()
