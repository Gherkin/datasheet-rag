"""Figure extraction from PDFs using Textract layout bounding boxes.

Renders PDF pages to images, then crops each LAYOUT_FIGURE region
using Textract's bounding box geometry. Supports optional padding,
S3 upload, and generates a manifest for downstream multi-modal embedding.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from threading import get_ident
from typing import Any

from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    track,
)

console = Console()


class PageRenderError(RuntimeError):
    """A page render, or the re-read of a cached render, failed.

    The underlying Pillow/PyMuPDF messages ("image file is truncated") name
    neither the file nor the stage that hit them, which made an interrupted
    ingest surface as a bare ``500: image file is truncated`` (GH #40). This
    adds both, plus the --force hint.
    """

    def __init__(self, *, stage: str, path: Path, page: int | None, cause: BaseException):
        self.stage = stage
        self.path = Path(path)
        self.page = page
        where = f" (page {page})" if page is not None else ""
        super().__init__(
            f"{type(cause).__name__}: {cause} — while {stage}{where} in "
            f"{self.path}. Re-run the ingest with --force to discard cached "
            "artifacts and redo every step."
        )


def _save_image_atomic(img: Image.Image, path: Path, image_format: str = "png") -> None:
    """Save *img* to *path* via a uniquely named temp file plus a rename.

    Saving straight to the final name is what leaves half-written images behind
    when a run is interrupted: the next ingest sees the name, trusts it, and
    only fails once something tries to decode it. With the rename, a reader
    sees either the previous file or a complete new one.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{get_ident()}.tmp")
    try:
        img.save(str(tmp), format=image_format.upper())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _load_cached_page(cache_file: Path) -> Image.Image | None:
    """Return the cached page render, or None if it is missing or unreadable.

    Existence is not integrity: a truncated PNG left by an interrupted render
    passes ``is_file()`` and then blows up deep in the crop step. Decoding it
    here — and deleting it when it fails — turns that hard failure into a
    re-render of the affected pages.
    """
    if not cache_file.is_file():
        return None
    try:
        img = Image.open(cache_file)
        img.load()  # force decode now, so the file handle can be released
        return img
    except Exception:
        cache_file.unlink(missing_ok=True)
        return None


def _clear_render_cache(cache_dir: Path) -> None:
    """Drop every cached page render (and stray temp file) under *cache_dir*."""
    stale = [*cache_dir.glob("*.png"), *cache_dir.glob("*.tmp")]
    for path in stale:
        path.unlink(missing_ok=True)
    if stale:
        console.print(f"[dim]Page render cache:[/] --force discarded {len(stale)} cached files")


@dataclass
class FigureRegion:
    """A detected figure/formula region with normalised bounding box (0..1)."""

    block_id: str
    page: int  # 1-indexed
    left: float
    top: float
    width: float
    height: float
    # Nearby text context (populated during extraction)
    caption: str = ""
    preceding_text: str = ""
    section_header: str = ""
    # "figure" or "formula" — downstream can treat them differently
    kind: str = "figure"


@dataclass
class ExtractedFigure:
    """An extracted figure image with metadata."""

    region: FigureRegion
    image_path: Path | None = None
    s3_key: str | None = None
    width_px: int = 0
    height_px: int = 0


@dataclass
class FigureManifest:
    """Manifest of all figures extracted from a document."""

    doc_id: str
    source_pdf: str
    figures: list[ExtractedFigure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_pdf": self.source_pdf,
            "figure_count": len(self.figures),
            "figures": [
                {
                    "block_id": f.region.block_id,
                    "page": f.region.page,
                    "bbox": {
                        "left": f.region.left,
                        "top": f.region.top,
                        "width": f.region.width,
                        "height": f.region.height,
                    },
                    "caption": f.region.caption,
                    "preceding_text": f.region.preceding_text,
                    "section_header": f.region.section_header,
                    "image_path": str(f.image_path) if f.image_path else None,
                    "s3_key": f.s3_key,
                    "width_px": f.width_px,
                    "height_px": f.height_px,
                }
                for f in self.figures
            ],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fp:
            json.dump(self.to_dict(), fp, indent=2)
        console.print(f"[green]Manifest saved[/] → {path}")
        return path


# ---------------------------------------------------------------------------
# Extract figure regions from Textract blocks
# ---------------------------------------------------------------------------


def find_figure_regions(blocks: list[dict[str, Any]]) -> list[FigureRegion]:
    """Parse Textract blocks and return all LAYOUT_FIGURE regions with context."""
    from datasheet_rag.textract import layout_reading_order

    id_map = {b["Id"]: b for b in blocks if "Id" in b}

    # Layout blocks in Textract's column-aware reading order, so caption and
    # nearest-header lookups walk the correct neighbours.
    layout_blocks = layout_reading_order(blocks)

    figures: list[FigureRegion] = []

    for i, block in enumerate(layout_blocks):
        if block.get("BlockType") != "LAYOUT_FIGURE":
            continue

        bbox = block.get("Geometry", {}).get("BoundingBox", {})
        region = FigureRegion(
            block_id=block["Id"],
            page=block.get("Page", 1),
            left=bbox.get("Left", 0),
            top=bbox.get("Top", 0),
            width=bbox.get("Width", 0),
            height=bbox.get("Height", 0),
        )

        # Look backwards for context: caption, section header, preceding text
        region.caption = _find_caption(block, layout_blocks, i, id_map)
        region.section_header = _find_nearest_header(layout_blocks, i, id_map)
        region.preceding_text = _find_preceding_text(layout_blocks, i, id_map)

        figures.append(region)

    return figures


def _collect_text_from_block(block: dict[str, Any], id_map: dict[str, dict[str, Any]]) -> str:
    """Recursively collect text from a block and its children."""
    if "Text" in block:
        text: str = block["Text"]
        return text

    child_ids = [rel["Ids"] for rel in block.get("Relationships", []) if rel["Type"] == "CHILD"]
    flat_ids = [cid for ids in child_ids for cid in ids]
    texts = []
    for cid in flat_ids:
        child = id_map.get(cid)
        if child:
            texts.append(_collect_text_from_block(child, id_map))
    return " ".join(texts)


def _find_caption(
    fig_block: dict[str, Any],
    layout_blocks: list[dict[str, Any]],
    fig_index: int,
    id_map: dict[str, dict[str, Any]],
) -> str:
    """Find a caption near the figure (text block immediately after on same page)."""
    fig_page = fig_block.get("Page", 1)
    fig_bottom = fig_block.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0) + fig_block.get(
        "Geometry", {}
    ).get("BoundingBox", {}).get("Height", 0)

    # Look at the next few layout blocks for caption-like text
    for j in range(fig_index + 1, min(fig_index + 4, len(layout_blocks))):
        candidate = layout_blocks[j]
        if candidate.get("Page", 1) != fig_page:
            break
        bt = candidate.get("BlockType", "")
        if bt in ("LAYOUT_SECTION_HEADER", "LAYOUT_TITLE", "LAYOUT_FIGURE"):
            break

        cand_top = candidate.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0)
        # Caption should be close below the figure
        if bt == "LAYOUT_TEXT" and (cand_top - fig_bottom) < 0.05:
            text = _collect_text_from_block(candidate, id_map).strip()
            # Heuristic: captions often start with "Figure" or "Fig."
            if text and (
                text.lower().startswith(("figure", "fig.", "fig "))
                or len(text) < 200  # short text near figure is likely a caption
            ):
                return text
    return ""


def _find_nearest_header(
    layout_blocks: list[dict[str, Any]],
    fig_index: int,
    id_map: dict[str, dict[str, Any]],
) -> str:
    """Walk backwards to find the nearest section header."""
    fig_page = layout_blocks[fig_index].get("Page", 1)

    for j in range(fig_index - 1, -1, -1):
        block = layout_blocks[j]
        bt = block.get("BlockType", "")
        if bt in ("LAYOUT_SECTION_HEADER", "LAYOUT_TITLE"):
            return _collect_text_from_block(block, id_map).strip()
        # Don't cross too many pages back
        if fig_page - block.get("Page", 1) > 1:
            break
    return ""


def _find_preceding_text(
    layout_blocks: list[dict[str, Any]],
    fig_index: int,
    id_map: dict[str, dict[str, Any]],
) -> str:
    """Get the text block immediately before the figure for context."""
    fig_page = layout_blocks[fig_index].get("Page", 1)

    for j in range(fig_index - 1, -1, -1):
        block = layout_blocks[j]
        if block.get("Page", 1) != fig_page:
            break
        bt = block.get("BlockType", "")
        if bt == "LAYOUT_TEXT":
            text = _collect_text_from_block(block, id_map).strip()
            if text:
                return text[:500]  # Truncate long preceding text
    return ""


# ---------------------------------------------------------------------------
# Render PDF pages and crop figures
# ---------------------------------------------------------------------------


# Rendered pages are the memory hot-spot of the whole ingest: one A4 page at
# 300 DPI is ~26 MB of RGB before anything is cropped from it. Everything below
# is written to keep a bounded number of them resident at once (GH #59).
_DEFAULT_RENDER_BUDGET_MB = 1024
_RGB_BYTES_PER_PX = 3
# A page costs more than its finished RGB buffer while it is being produced:
# the PyMuPDF pixmap and the PIL copy coexist, and the PNG encoder allocates on
# top of both. Three times the pixel data is the honest per-slot figure.
_RENDER_OVERHEAD_FACTOR = 3


def _render_budget_mb() -> int:
    """The configured page-render memory budget, or the built-in default.

    Settings are optional here on purpose: unit tests call the renderer
    directly without a configured RAG_HOME, and a missing config should mean
    "use the default budget", not "fail to render".
    """
    try:
        from datasheet_rag.config import get_settings

        return get_settings().render_memory_budget_mb
    except Exception:
        return _DEFAULT_RENDER_BUDGET_MB


def _page_slot_bytes(pdf_path: Path, pages: list[int], dpi: int) -> int:
    """Approximate peak bytes one in-flight page occupies, at *dpi*.

    Sized off the largest of the *target* pages rather than an assumed A4: a
    fold-out block diagram or a B4 reference manual is several times the area
    of a letter page, and the window has to shrink accordingly.
    """
    import fitz  # pymupdf

    scale = dpi / 72
    widest = 0
    with fitz.open(str(pdf_path)) as doc:
        for page_no in pages:
            if not 1 <= page_no <= len(doc):
                continue  # out-of-range pages fail later, with a proper error
            rect = doc[page_no - 1].rect
            widest = max(widest, int(rect.width * scale) * int(rect.height * scale))
    return max(widest, 1) * _RGB_BYTES_PER_PX * _RENDER_OVERHEAD_FACTOR


def _render_window_size(slot_bytes: int, budget_bytes: int, n_pages: int) -> int:
    """How many pages may be in flight at once without blowing the budget.

    One slot is reserved for the page the consumer is currently holding, so the
    pool only gets what is left over. Never returns less than one: a page too
    big for the entire budget still has to be rendered, just on its own.
    """
    slots = max(2, budget_bytes // max(slot_bytes, 1))
    return max(1, min(os.cpu_count() or 4, n_pages, 8, slots - 1))


def iter_pdf_pages(
    pdf_path: Path,
    *,
    dpi: int = 300,
    pages: list[int] | None = None,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    memory_budget_mb: int | None = None,
) -> Iterator[tuple[int, Image.Image]]:
    """Yield ``(page_number, image)`` for each requested page, in page order.

    Page numbers are 1-indexed to match Textract conventions.

    Three properties, in the order they were added:

    1. **Page render cache** — each rendered page is saved as a PNG under
       ``cache_dir / f'p{page:04d}_{dpi}dpi.png'``. On subsequent ingests
       cached pages are loaded from disk instead of re-rendered, cutting
       Step 3 to near-zero for pages that haven't changed. Writes go through
       a temp file and a rename, and every cached page is decoded before it
       is trusted, so an interrupted render can't poison later ingests.
       ``refresh_cache`` (the ingest's --force) empties the cache first.

    2. **Parallel rendering** — pages are dispatched to a
       ``ThreadPoolExecutor`` where each worker opens its own
       ``fitz.Document`` (PyMuPDF is not thread-safe when sharing one
       instance) and renders its assigned page independently. Wall-clock
       time scales roughly as 1/n_workers.

    3. **Bounded memory** (GH #59) — this is a generator, and only a small
       window of pages is ever decoded at once. The old shape returned
       ``dict[int, Image.Image]`` with every page resident, which scales
       linearly with page count and DPI and with nothing capping it: a
       900-page datasheet at 300 DPI is ~23 GB of RGB, so the kernel
       OOM-killed the server mid-ingest — no traceback, just SIGKILL. The
       window is now sized from ``render_memory_budget_mb`` and the largest
       target page, so peak footprint is bounded by the budget rather than by
       the document.

    The caller must therefore consume each image before asking for the next,
    and must not accumulate them — that would reintroduce the OOM.
    """
    import fitz  # pymupdf

    # Resolve target page list — open briefly if caller passed None
    if pages is None:
        with fitz.open(str(pdf_path)) as probe:
            target_pages: list[int] = list(range(1, len(probe) + 1))
    else:
        target_pages = sorted(set(pages))
    if not target_pages:
        return

    def _cache_file(page_no: int) -> Path | None:
        return None if cache_dir is None else cache_dir / f"p{page_no:04d}_{dpi}dpi.png"

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if refresh_cache:
            _clear_render_cache(cache_dir)
        # Existence-only pre-scan: cheap, and just for the progress line. Whether
        # a cached page actually decodes is settled per page, in the worker.
        on_disk = sum(1 for p in target_pages if (f := _cache_file(p)) and f.is_file())
        if on_disk:
            console.print(
                f"[dim]Page render cache:[/] {on_disk} of {len(target_pages)} "
                f"pages already cached, {len(target_pages) - on_disk} to render."
            )

    # ---- Size the window from the memory budget ----------------------------
    budget_bytes = (memory_budget_mb or _render_budget_mb()) * 1024 * 1024
    slot_bytes = _page_slot_bytes(pdf_path, target_pages, dpi)
    n_workers = _render_window_size(slot_bytes, budget_bytes, len(target_pages))
    if slot_bytes * 2 > budget_bytes:
        console.print(
            f"[yellow]Page render:[/] one page at {dpi} DPI needs about "
            f"{slot_bytes / 1024 / 1024:.0f} MB, against a "
            f"{budget_bytes / 1024 / 1024:.0f} MB budget — rendering serially. "
            "Lower --dpi, or raise RAG_RENDER_MEMORY_BUDGET_MB."
        )

    mat_scale = dpi / 72
    pdf_path_str = str(pdf_path)

    def _page(page_no: int) -> tuple[int, Image.Image, str]:
        """Return one page: from the cache when it decodes, rendered otherwise."""
        cache_file = _cache_file(page_no)
        stale = False
        if cache_file is not None:
            stale = cache_file.is_file()
            cached = _load_cached_page(cache_file)
            if cached is not None:
                return page_no, cached, "cached"

        # Each worker owns its own fitz.Document — required for thread safety.
        try:
            with fitz.open(pdf_path_str) as doc:
                pix = doc[page_no - 1].get_pixmap(matrix=fitz.Matrix(mat_scale, mat_scale))
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                del pix  # the pixmap doubles this page's footprint until it goes
        except Exception as exc:
            raise PageRenderError(
                stage="rendering a page", path=pdf_path, page=page_no, cause=exc
            ) from exc
        if cache_file is not None:
            try:
                _save_image_atomic(img, cache_file)
            except Exception as exc:
                raise PageRenderError(
                    stage="caching a rendered page",
                    path=cache_file,
                    page=page_no,
                    cause=exc,
                ) from exc
        return page_no, img, "repaired" if stale else "rendered"

    counts = {"cached": 0, "rendered": 0, "repaired": 0}
    remaining = iter(target_pages)

    with (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress,
        ThreadPoolExecutor(max_workers=n_workers) as pool,
    ):
        task = progress.add_task(
            f"Pages at {dpi} DPI · {n_workers} workers…", total=len(target_pages)
        )
        # Keep exactly n_workers pages in flight: submitting the whole list up
        # front is what made peak memory a function of page count (GH #59).
        window: deque[Future[tuple[int, Image.Image, str]]] = deque(
            pool.submit(_page, p) for p in islice(remaining, n_workers)
        )
        try:
            while window:
                page_no, img, status = window.popleft().result()
                for nxt in islice(remaining, 1):
                    window.append(pool.submit(_page, nxt))
                counts[status] += 1
                progress.advance(task)
                yield page_no, img
        finally:
            # An abandoned generator (an exception in the consumer) must not
            # leave the pool rendering pages nobody will ever collect.
            for pending in window:
                pending.cancel()

    if counts["repaired"]:
        console.print(
            f"[yellow]Page render cache:[/] {counts['repaired']} cached page(s) in "
            f"{cache_dir} were unreadable (likely an interrupted render) "
            "— discarded and re-rendered."
        )
    console.print(
        f"[green]Rendered[/] {counts['rendered'] + counts['repaired']} pages "
        f"({counts['cached']} from cache)"
    )


def crop_figure(
    page_image: Image.Image,
    region: FigureRegion,
    *,
    padding_pct: float = 0.02,
    max_right: float = 1.0,
    min_left: float = 0.0,
    max_bottom: float = 1.0,
    min_top: float = 0.0,
) -> Image.Image:
    """Crop a figure region from a rendered page image.

    Textract bounding boxes are normalised (0..1), so we scale to pixel
    coordinates. Adds a small padding to avoid cutting off edges.

    max_right / min_left / max_bottom / min_top (all normalised) cap the
    padded crop edges so they don't bleed into adjacent figures.
    """
    w, h = page_image.size

    left = max(0, int(max(min_left, region.left - padding_pct) * w))
    top = max(0, int(max(min_top, region.top - padding_pct) * h))
    right = min(w, int(min(max_right, region.left + region.width + padding_pct) * w))
    bottom = min(h, int(min(max_bottom, region.top + region.height + padding_pct) * h))

    return page_image.crop((left, top, right, bottom))


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------


def _compute_adjacent_crop_caps(
    regions: list[FigureRegion],
    padding_pct: float,
    vert_overlap_thresh: float = 0.30,
) -> dict[str, dict[str, float]]:
    """Return per-region crop-edge caps that prevent padding from bleeding into neighbours.

    The bleed problem is NOT about bbox overlap — docling's bboxes for adjacent-
    column figures typically have a small gap (~1 %) that is narrower than the
    default padding (2 %). crop_figure then adds padding on both sides, bridging
    the gap and capturing a sliver of the neighbouring figure.

    Fix: for each pair of figures on the same page with significant vertical
    co-occurrence, cap the right-edge crop of the left figure at the right
    figure's left bbox edge, and the left-edge crop of the right figure at the
    left figure's right bbox edge. The bboxes themselves are never modified.

    Cases:
    - Gap >= padding_pct: padding fits comfortably — no cap needed.
    - 0 <= gap < padding_pct: cap so padding stops exactly at neighbour's edge.
    - gap < 0 (actual bbox overlap): skip; don't guess, leave it to the vision model.

    Returns {block_id: {'max_right': x, 'min_left': x}} for affected regions.
    """
    by_page: dict[int, list[FigureRegion]] = defaultdict(list)
    for r in regions:
        by_page[r.page].append(r)

    caps: dict[str, dict[str, float]] = {}

    for page_regions in by_page.values():
        if len(page_regions) < 2:
            continue

        for a in page_regions:
            a_right = a.left + a.width
            for b in page_regions:
                if a is b or a.left >= b.left:
                    continue  # only A-left-of-B pairs

                # Require significant vertical co-occurrence
                a_top, a_bot = a.top, a.top + a.height
                b_top, b_bot = b.top, b.top + b.height
                vert_overlap = max(0.0, min(a_bot, b_bot) - max(a_top, b_top))
                min_h = min(a.height, b.height)
                if min_h <= 0 or vert_overlap / min_h < vert_overlap_thresh:
                    continue

                b_left = b.left
                gap = b_left - a_right

                if gap < 0:
                    # Actual bbox overlap — we can't safely pick a split point;
                    # leave both untouched and let the vision model handle the sliver.
                    console.print(
                        f"[dim]Column bleed:[/] page {a.page} {a.block_id}↔{b.block_id} "
                        f"bbox overlap {-gap:.3f} — leaving for vision model"
                    )
                    continue

                if gap >= padding_pct:
                    continue  # padding fits in the gap, no action needed

                # Gap exists but is smaller than padding — cap at the neighbour's edge.
                # This is safe: we only prevent over-padding, never cut actual content.
                caps.setdefault(a.block_id, {})
                caps[a.block_id]["max_right"] = min(caps[a.block_id].get("max_right", 1.0), b_left)
                caps.setdefault(b.block_id, {})
                caps[b.block_id]["min_left"] = max(caps[b.block_id].get("min_left", 0.0), a_right)
                console.print(
                    f"[yellow]Column cap:[/] page {a.page} "
                    f"{a.block_id}↔{b.block_id} gap={gap:.4f} < padding {padding_pct:.3f}; "
                    f"capping right at {b_left:.4f}, left at {a_right:.4f}"
                )

    return caps


def extract_figures_from_regions(
    pdf_path: Path,
    regions: list[FigureRegion],
    doc_id: str,
    *,
    output_dir: Path | None = None,
    dpi: int = 300,
    image_format: str = "png",
    padding_pct: float = 0.02,
    force: bool = False,
) -> FigureManifest:
    """Crop and save figure regions to disk. Regions may come from Textract or Docling.

    ``force`` is the ingest's --force: it empties this document's page render
    cache first, so a re-ingest genuinely redoes every step instead of
    resurrecting whatever the previous run left behind.
    """
    if output_dir is None:
        from datasheet_rag.config import get_settings

        output_dir = get_settings().figures_dir / doc_id

    output_dir.mkdir(parents=True, exist_ok=True)

    if not regions:
        console.print("[yellow]No figures/formulas detected in this document.[/]")
        return FigureManifest(doc_id=doc_id, source_pdf=str(pdf_path))

    n_figs = sum(1 for r in regions if r.kind == "figure")
    n_formulas = sum(1 for r in regions if r.kind == "formula")
    console.print(
        f"[blue]Found {len(regions)} regions[/] ({n_figs} figures, {n_formulas} formulas) "
        f"across pages {sorted(set(r.page for r in regions))}"
    )

    # Regions grouped by the page they sit on, keeping each one's position in
    # the caller's list: a page is cropped the moment it is rendered and then
    # dropped, so the manifest is reassembled into region order at the end.
    pending: dict[int, list[tuple[int, FigureRegion]]] = defaultdict(list)
    for i, region in enumerate(regions):
        pending[region.page].append((i, region))
    needed_pages = sorted(pending)

    # Derive a per-document render cache directory so re-ingests of the same
    # document (e.g. after layout fixes) skip re-rendering pages that haven't
    # changed. Falls back to None (no cache) when settings are unavailable
    # (e.g. in unit tests that don't configure RAG_HOME).
    render_cache_dir: Path | None = None
    try:
        from datasheet_rag.config import get_settings

        render_cache_dir = get_settings().rag_home / "page_render_cache" / doc_id
    except Exception:
        pass

    crop_caps = _compute_adjacent_crop_caps(regions, padding_pct)
    if crop_caps:
        console.print(f"[yellow]Column caps:[/] padding capped on {len(crop_caps)} regions")

    manifest = FigureManifest(doc_id=doc_id, source_pdf=str(pdf_path))
    extracted: list[tuple[int, ExtractedFigure]] = []

    # Streamed, one page at a time: holding every rendered page until the crop
    # loop started is what OOM-killed the server on long documents (GH #59).
    page_stream = iter_pdf_pages(
        pdf_path,
        dpi=dpi,
        pages=needed_pages,
        cache_dir=render_cache_dir,
        refresh_cache=force,
    )
    for page_no, page_img in page_stream:
        try:
            for i, region in pending.pop(page_no, []):
                caps = crop_caps.get(region.block_id, {})
                cropped = crop_figure(
                    page_img,
                    region,
                    padding_pct=padding_pct,
                    max_right=caps.get("max_right", 1.0),
                    min_left=caps.get("min_left", 0.0),
                )
                prefix = "formula" if region.kind == "formula" else "fig"
                filename = f"p{region.page:03d}_{prefix}{i:03d}.{image_format}"
                image_path = output_dir / filename
                _save_image_atomic(cropped, image_path, image_format)

                extracted.append(
                    (
                        i,
                        ExtractedFigure(
                            region=region,
                            image_path=image_path,
                            width_px=cropped.width,
                            height_px=cropped.height,
                        ),
                    )
                )
                cropped.close()
        finally:
            page_img.close()  # free it before the next page is pulled in

    for page_no in sorted(pending):
        console.print(
            f"[red]Warning:[/] page {page_no} not rendered, "
            f"skipping {len(pending[page_no])} region(s)"
        )

    manifest.figures = [fig for _, fig in sorted(extracted, key=lambda pair: pair[0])]
    console.print(f"[green]Extracted {len(manifest.figures)} regions[/] → {output_dir}")
    return manifest


def extract_figures(
    pdf_path: Path,
    blocks: list[dict[str, Any]],
    doc_id: str,
    *,
    output_dir: Path | None = None,
    dpi: int = 300,
    image_format: str = "png",
    padding_pct: float = 0.02,
    force: bool = False,
) -> FigureManifest:
    """Extract figures from a PDF using Textract layout blocks (Textract path)."""
    regions = find_figure_regions(blocks)
    return extract_figures_from_regions(
        pdf_path,
        regions,
        doc_id,
        output_dir=output_dir,
        dpi=dpi,
        image_format=image_format,
        padding_pct=padding_pct,
        force=force,
    )


# ---------------------------------------------------------------------------
# S3 upload for extracted figures
# ---------------------------------------------------------------------------


def upload_figures_to_s3(
    manifest: FigureManifest,
    *,
    s3_prefix: str = "figures/",
) -> FigureManifest:
    """Upload all extracted figure images to S3 and update the manifest with S3 keys."""
    from datasheet_rag.aws import s3_client
    from datasheet_rag.config import get_settings

    settings = get_settings()
    bucket = settings.require_s3_bucket()
    client = s3_client()

    for fig in track(manifest.figures, description="Uploading figures to S3…"):
        if fig.image_path is None or not fig.image_path.is_file():
            continue

        s3_key = f"{s3_prefix}{manifest.doc_id}/{fig.image_path.name}"
        suffix = fig.image_path.suffix.lower()
        content_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix, "image/png")

        client.upload_file(
            Filename=str(fig.image_path),
            Bucket=bucket,
            Key=s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        fig.s3_key = s3_key

    uploaded = sum(1 for f in manifest.figures if f.s3_key)
    console.print(
        f"[green]Uploaded {uploaded} figures[/] → s3://{bucket}/{s3_prefix}{manifest.doc_id}/"
    )
    return manifest
