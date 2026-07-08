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
# "Foo (continued)" or "Foo (Continued)" — trailing page-continuation suffix
_CONTINUED_RE = re.compile(r"\s*\(continued\)\s*$", re.IGNORECASE)
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

# A header/data text overlap needs to be at least this long to count as a
# fusion signal — single-character tokens ("-", "x", "0") are common
# placeholder/footnote markers that legitimately appear in both header and
# data cells without indicating that a data row was tagged as a header.
_FUSED_HEADER_MIN_CHARS = 2


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


def _detect_fused_header_row(cells: list[dict[str, Any]]) -> str | None:
    """Return overlapping text if a header row looks fused with a data row.

    TableFormer can fuse a data row into the header band on complex tables —
    e.g. tagging a pin number ("30"), its part-style identifiers ("PB08",
    "SERCOM3_PAD3") and the *real* header cells ("Port", "SERCOM") all as
    ``is_header=True`` in adjacent rows. Unlike the garbled-header failure
    above, this output looks structurally clean — separated cells, no
    repetition — so only its *content* gives it away: a pin's actual values
    end up split across rows and shifted relative to the header (problem.md
    failure mode #3).

    Headers describe columns; they don't repeat the values a table reports.
    So a cell tagged ``is_header=True`` whose exact text also occurs as a
    *non-header* origin cell elsewhere in the same table is a candidate for
    "data value that leaked into the header band" — a structural
    self-consistency check that needs no shape heuristics or magic
    thresholds (cf. _GARBLED_HEADER_MIN_CHARS).

    That candidate signal alone over-fires on multi-block register tables —
    e.g. a 32-bit register rendered as four repeated "Bit / Access / Reset"
    sub-blocks (bits 31-24, 23-16, …): Docling tags the *first* block's row
    labels ``is_header=True`` but the repeats in later blocks
    ``is_header=False``, so "bit"/"access"/"reset" each show up in both sets
    despite the table being perfectly fine — they're legitimately-recurring
    *structural labels*, not leaked data.

    The distinguishing trait of *real* fusion (problem.md's Table 5-6: pin
    30's row — "30", "PB08", "SERCOM3_PAD3" — tagged as header alongside the
    genuine "Port"/"SERCOM" labels) is that it pulls a whole DATA ROW of
    several distinct, mutually-unrelated values into the header band
    *together*. An individually-recurring label can never do that — it is,
    by construction, the only overlap candidate in its row (surrounded by
    genuinely distinct header content, e.g. the bit-position numbers). So:
    flag fusion only when **multiple** distinct overlap candidates cluster in
    the *same* header row — that is what a leaked row looks like, and a lone
    recurring label structurally cannot produce it.
    """
    origin = [c for c in cells if c.get("is_origin", True)]
    header_texts = {
        c["text"].strip().casefold()
        for c in origin
        if c.get("is_header") and c["text"].strip()
    }
    data_texts = {
        c["text"].strip().casefold()
        for c in origin
        if not c.get("is_header") and c["text"].strip()
    }
    candidates = {t for t in (header_texts & data_texts) if len(t) >= _FUSED_HEADER_MIN_CHARS}
    if not candidates:
        return None

    candidates_by_row: dict[int, set[str]] = {}
    for c in origin:
        if not c.get("is_header"):
            continue
        text = c["text"].strip().casefold()
        if text in candidates:
            candidates_by_row.setdefault(c["row"], set()).add(text)

    leaked_row = next((texts for texts in candidates_by_row.values() if len(texts) >= 2), None)
    if leaked_row is not None:
        return ", ".join(sorted(leaked_row)[:3])
    return None


def _table_header_row_count(cells: list[dict[str, Any]]) -> int:
    """Number of rows (from row 1) that make up this table's header band.

    Used by the header-band repair to size the crop and the proposal grid —
    derived from Docling's ``is_header`` tagging, which (per
    :func:`_detect_fused_header_row`'s docstring) can over- or under-tag in
    pathological cases, so callers should sanity-check the result before
    relying on it as a crop boundary.
    """
    header_origin = [c for c in cells if c.get("is_origin", True) and c.get("is_header")]
    if not header_origin:
        return 0
    return max(c["row"] for c in header_origin)


# Header-band repair trusts the data grid's column count as ground truth for
# what the re-transcribed header must tile. If the data rows themselves
# disagree on width below this fraction, that trust is misplaced — e.g.
# PIC32CK1025GC01100 p.16's table has data rows of width 37/44/55 (mode
# 37 covers only 5/9 = 56% of rows) while the table visually has ~40
# columns; a header forced to tile 37 columns would be structurally valid
# (passes every check in validate_header_grid) but silently misaligned
# against the real data columns from partway through the table onward —
# worse than the honest reading-order fallback it would replace.
_COLUMN_COUNT_CONSISTENCY_THRESHOLD = 0.7


def _table_column_count(cells: list[dict[str, Any]]) -> int:
    """Trusted column count, derived from the table's data rows.

    Header bands are exactly the part of the grid that
    :func:`table_structure_untrustworthy` doesn't trust — but the data rows
    below them retain a reliable column count for *most* rows. A handful of
    data rows can still have their own row-level corruption (extra/merged
    cells widening or narrowing just that row), so taking ``max()`` over all
    data rows is fragile — one outlier row inflates ``C`` for the whole
    table. The width shared by the *most* data rows is the table's real
    column count; outlier rows are themselves a sign of trouble but don't
    change what the header band needs to tile. Falls back to the widest row
    in the table if there are no data rows (a degenerate all-header table).

    Returns ``0`` — "don't trust this table's column count at all" — if
    fewer than :data:`_COLUMN_COUNT_CONSISTENCY_THRESHOLD` of the data rows
    agree on a single width (see module comment above).
    """
    header_rows = _table_header_row_count(cells)
    data = [c for c in cells if c.get("is_origin", True) and c["row"] > header_rows]
    source = data if data else [c for c in cells if c.get("is_origin", True)]
    if not source:
        return 0
    widths_by_row: dict[int, int] = {}
    for c in source:
        right_edge = c["col"] + c["col_span"] - 1
        widths_by_row[c["row"]] = max(widths_by_row.get(c["row"], 0), right_edge)
    width, count = Counter(widths_by_row.values()).most_common(1)[0]
    if count / len(widths_by_row) < _COLUMN_COUNT_CONSISTENCY_THRESHOLD:
        return 0
    return width


def table_structure_untrustworthy(cells: list[dict[str, Any]]) -> str | None:
    """Return a human-readable reason if this table's structure is untrustworthy.

    Two independent signals feed this gate:

    - :func:`_detect_garbled_header` — TableFormer stamped one cell's text
      across many header positions (failure mode #1: loud, self-announcing
      via repetition).
    - :func:`_detect_fused_header_row` — TableFormer tagged data-row content
      as header content (failure mode #3: looks clean, silently wrong).

    Either one means the row/column/header grid Docling proposed cannot be
    trusted as-is. Callers should not assert that grid (e.g. via
    :func:`_table_cells_to_compact_text`'s pipe-delimited columns) — instead
    fall back to :func:`_table_cells_to_reading_order_text`, which makes no
    structural claims. See docs/table-structure-repair/{problem,plan}.md for
    the full investigation this responds to.
    """
    garbled = _detect_garbled_header(cells)
    if garbled is not None:
        return f"garbled header — {garbled[:60]!r} repeated across columns"
    fused = _detect_fused_header_row(cells)
    if fused is not None:
        return f"fused header/data row — {fused!r} tagged as both header and data"
    return None


def _table_cells_to_reading_order_text(cells: list[dict[str, Any]]) -> str:
    """Render table cells as plain text in geometric reading order, no grid asserted.

    Used when :func:`table_structure_untrustworthy` flags a table — emitting
    Docling's proposed row/column/header structure (even via the compact
    pipe-delimited form) would assert a grid that may be silently wrong, and
    a reader (human or LLM) would conclude a pin/register/field does
    something it doesn't (problem.md failure mode #3). This instead emits
    each origin cell's text once, in row-major reading order (deduplicated
    via ``is_origin`` — the same span-collapsing _table_cells_to_compact_text
    relies on), with no column alignment implied. Strictly worse for *quick
    visual scanning*, strictly better than confidently-wrong structure for
    retrieval and LLM reasoning to build on.

    Adjacent identical lines are collapsed to one: a garbled-header table can
    have the *same* misparsed string independently claim ``is_origin`` at
    several consecutive grid positions (the span-detection bug described in
    problem.md's failure mode #1) — repeating it verbatim back-to-back would
    carry the garbling straight through into "structure-free" text instead of
    actually neutralising it. This is a property of well-formed reading-order
    text in general (the same line twice in a row is never useful signal,
    regardless of *why* it repeated), not a heuristic tuned to this table.
    """
    texts = [
        cell["text"].strip()
        for cell in sorted(cells, key=lambda c: (c["row"], c["col"]))
        if cell.get("is_origin", True) and cell["text"].strip()
    ]
    deduped = [t for i, t in enumerate(texts) if i == 0 or t != texts[i - 1]]
    return "\n".join(deduped)


def _table_cells_to_compact_text(cells: list[dict[str, Any]]) -> str:
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

    Only used when :func:`table_structure_untrustworthy` finds nothing wrong
    — once a table is flagged, asserting *any* row/column grid (even this
    compact one) is unsafe; see :func:`_table_cells_to_reading_order_text`.
    """
    if not cells:
        return ""
    rows: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        rows.setdefault(cell["row"], []).append(cell)

    lines: list[str] = []
    for row_idx in sorted(rows):
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

    for item, doc_level in doc.iterate_items():
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
                title=text, level=doc_level, page_start=page_no, page_end=page_no,
                header_block_id=block_id,
            )
            # Pop back to a parent that is strictly shallower than this section.
            # This respects Docling's nesting level for documents where headings
            # carry meaningful depth (e.g. subsections inside major sections).
            while len(section_stack) > 1 and section_stack[-1].level >= doc_level:
                section_stack.pop()
            if not section_stack:
                # Nothing in the stack yet — start a new root branch.
                section_stack.append(section)
            elif section_stack[-1].level < doc_level:
                # Found a proper shallower parent.
                parent = section_stack[-1]
                parent.children.append(section)
                parent.page_end = page_no
                section_stack.append(section)
            else:
                # One item left at the same or deeper level — sibling of the
                # current root; flush the old root and start fresh.
                _flush(section_stack, sections)
                section_stack.clear()
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

            untrustworthy = table_structure_untrustworthy(cells)
            if untrustworthy is not None:
                where = f"'{caption}'" if caption else f"on page {page_no}"
                console.print(
                    f"[yellow]Table parsing warning:[/] table {where} has an "
                    f"untrustworthy structure ({untrustworthy}) — Docling's "
                    f"table-structure recognition likely misparsed a complex "
                    f"multi-level header or fused a data row into the header "
                    f"band. Rather than assert a row/column grid that may be "
                    f"silently wrong, this table is being rendered as plain "
                    f"reading-order text with no structure implied (this "
                    f"safety net applies regardless of table-structure mode — "
                    f"switching to --accurate-tables is NOT a guaranteed fix, "
                    f"see docs/table-structure-repair/problem.md). Inspect "
                    f"page {page_no} by hand to confirm the real structure."
                )
                table_text = _table_cells_to_reading_order_text(cells)
            else:
                table_text = _table_cells_to_compact_text(cells)
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
                table_structure_warning=untrustworthy,
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
    sections = _filter_repeated_section_headings(sections, total_pages=len(doc.pages))
    sections = _merge_continued_sections(sections)
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


def _merge_continued_sections(sections: list[DocumentSection]) -> list[DocumentSection]:
    """Collapse 'Foo (continued)' sections back into their preceding 'Foo' base section.

    PDFs that span a logical section across multiple pages re-emit the section
    heading on each new page with a "(continued)" suffix. Each re-emission
    becomes a separate DocumentSection in the outline, fragmenting what is
    logically one section into many. This function merges them:

    - Any section whose title ends with " (continued)" (case-insensitive) is
      matched to the most recent preceding sibling whose base title (with the
      suffix stripped) matches case-insensitively.
    - The continuation's elements and children are appended to the matched
      base section, and its page_end is extended accordingly.
    - If no matching base exists in the current sibling list, the continuation
      is kept as-is (its base title may be in a parent scope).

    Applied recursively so continuations at any nesting depth are merged.
    """

    def _merge(sects: list[DocumentSection]) -> list[DocumentSection]:
        result: list[DocumentSection] = []
        # Maps casefold(base_title) -> index into result for the latest base section
        title_to_idx: dict[str, int] = {}

        for s in sects:
            # Recurse into children before processing this section
            s.children = _merge(s.children)

            if _CONTINUED_RE.search(s.title):
                base_key = _CONTINUED_RE.sub("", s.title).strip().casefold()
                if base_key in title_to_idx:
                    target = result[title_to_idx[base_key]]
                    target.elements.extend(s.elements)
                    target.children.extend(s.children)
                    target.page_end = max(target.page_end, s.page_end)
                    console.print(
                        f"[dim]Merge:[/] folded '{s.title}' (p{s.page_start}) "
                        f"into '{target.title}'"
                    )
                    continue  # drop the continuation — absorbed into base
                # No matching base in this sibling list — keep and register
                # (so later continuations at this level can still find it)
                title_to_idx[_CONTINUED_RE.sub("", s.title).strip().casefold()] = len(result)
            else:
                title_to_idx[s.title.strip().casefold()] = len(result)

            result.append(s)

        return result

    return _merge(sections)


# Minimum number of times a zero-element section title must appear for it to
# be treated as a repeating page watermark rather than a real heading.  Both
# the absolute count and the fraction of total pages must be satisfied.
_REPEATED_SECTION_MIN_COUNT = 3
_REPEATED_SECTION_MIN_PAGE_FRACTION = 0.04


def _filter_repeated_section_headings(
    sections: list[DocumentSection], total_pages: int
) -> list[DocumentSection]:
    """Remove section headings that are repeating page watermarks misclassified as headings.

    Some PDFs repeat a product name or chapter title as a visual heading on
    every page (rather than in the true page-header zone that Docling would
    classify as PAGE_HEADER). When Docling classifies these as SECTION_HEADER
    they create an empty placeholder section at each page boundary that
    pollutes the chapter's child list with noise.

    A section heading is treated as a repeating watermark when its
    case-folded, stripped title appears at least
    ``max(_REPEATED_SECTION_MIN_COUNT, total_pages *
    _REPEATED_SECTION_MIN_PAGE_FRACTION)`` times across the outline AND
    *every* occurrence has zero direct elements and zero children — i.e. it
    carries no content whatsoever.  Any children an occurrence has accumulated
    are promoted to the parent list so no content is lost.
    """
    title_counts: Counter[str] = Counter()

    def _count(sects: list[DocumentSection]) -> None:
        for s in sects:
            if not s.elements and not s.children:
                title_counts[s.title.casefold().strip()] += 1
            _count(s.children)

    _count(sections)

    threshold = max(
        _REPEATED_SECTION_MIN_COUNT,
        int(total_pages * _REPEATED_SECTION_MIN_PAGE_FRACTION),
    )
    noise_titles = {t for t, c in title_counts.items() if c >= threshold}
    if not noise_titles:
        return sections

    for title in noise_titles:
        console.print(
            f"[yellow]Section filter:[/] removing {title_counts[title]} repeated "
            f"empty section headings {title!r:.60} (likely a running page watermark)."
        )

    def _filter(sects: list[DocumentSection]) -> list[DocumentSection]:
        result: list[DocumentSection] = []
        for s in sects:
            if s.title.casefold().strip() in noise_titles:
                result.extend(_filter(s.children))
            else:
                s.children = _filter(s.children)
                result.append(s)
        return result

    return _filter(sections)


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
