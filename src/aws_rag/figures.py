"""Figure extraction from PDFs using Textract layout bounding boxes.

Renders PDF pages to images, then crops each LAYOUT_FIGURE region
using Textract's bounding box geometry. Supports optional padding,
S3 upload, and generates a manifest for downstream multi-modal embedding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pdf2image import convert_from_path
from PIL import Image
from rich.console import Console
from rich.progress import track

console = Console()


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
    from aws_rag.textract import layout_reading_order

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


def _collect_text_from_block(
    block: dict[str, Any], id_map: dict[str, dict[str, Any]]
) -> str:
    """Recursively collect text from a block and its children."""
    if "Text" in block:
        return block["Text"]

    child_ids = [
        rel["Ids"]
        for rel in block.get("Relationships", [])
        if rel["Type"] == "CHILD"
    ]
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
    fig_bottom = (
        fig_block.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0)
        + fig_block.get("Geometry", {}).get("BoundingBox", {}).get("Height", 0)
    )

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


def render_pdf_pages(
    pdf_path: Path,
    *,
    dpi: int = 300,
    pages: list[int] | None = None,
) -> dict[int, Image.Image]:
    """Render PDF pages to PIL Images. Returns {page_number: Image}.

    Page numbers are 1-indexed to match Textract conventions.

    Pages are rendered one at a time rather than as a single batch. On
    documents where figures are scattered across thousands of pages,
    rendering the whole min(pages)..max(pages) span in one
    convert_from_path() call would materialise every page in that span as
    an in-memory PIL Image simultaneously (~40 MB/page at 300 DPI) — for a
    2,000+ page datasheet that is enough to exhaust system RAM and crash
    the machine. Rendering page-by-page bounds memory to a single page and
    gives per-page progress instead of one long, silent blocking call.
    """
    if not pages:
        console.print(f"[blue]Rendering PDF[/] at {dpi} DPI …")
        images = convert_from_path(str(pdf_path), dpi=dpi)
        result = {i + 1: img for i, img in enumerate(images)}
        console.print(f"[green]Rendered[/] {len(result)} pages")
        return result

    result: dict[int, Image.Image] = {}
    for page in track(sorted(set(pages)), description=f"Rendering PDF at {dpi} DPI…"):
        page_images = convert_from_path(
            str(pdf_path), dpi=dpi, first_page=page, last_page=page
        )
        if page_images:
            result[page] = page_images[0]
    console.print(f"[green]Rendered[/] {len(result)} pages")
    return result


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
    from collections import defaultdict

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
                caps[a.block_id]["max_right"] = min(
                    caps[a.block_id].get("max_right", 1.0), b_left
                )
                caps.setdefault(b.block_id, {})
                caps[b.block_id]["min_left"] = max(
                    caps[b.block_id].get("min_left", 0.0), a_right
                )
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
) -> FigureManifest:
    """Crop and save figure regions to disk. Regions may come from Textract or Docling."""
    if output_dir is None:
        from aws_rag.config import get_settings
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

    needed_pages = sorted(set(r.page for r in regions))
    page_images = render_pdf_pages(pdf_path, dpi=dpi, pages=needed_pages)

    crop_caps = _compute_adjacent_crop_caps(regions, padding_pct)
    if crop_caps:
        console.print(f"[yellow]Column caps:[/] padding capped on {len(crop_caps)} regions")

    manifest = FigureManifest(doc_id=doc_id, source_pdf=str(pdf_path))

    for i, region in enumerate(track(regions, description="Cropping figures…")):
        page_img = page_images.get(region.page)
        if page_img is None:
            console.print(f"[red]Warning:[/] page {region.page} not rendered, skipping")
            continue

        caps = crop_caps.get(region.block_id, {})
        cropped = crop_figure(
            page_img, region, padding_pct=padding_pct,
            max_right=caps.get("max_right", 1.0),
            min_left=caps.get("min_left", 0.0),
        )
        prefix = "formula" if region.kind == "formula" else "fig"
        filename = f"p{region.page:03d}_{prefix}{i:03d}.{image_format}"
        image_path = output_dir / filename
        cropped.save(str(image_path), format=image_format.upper())

        manifest.figures.append(ExtractedFigure(
            region=region,
            image_path=image_path,
            width_px=cropped.width,
            height_px=cropped.height,
        ))

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
) -> FigureManifest:
    """Extract figures from a PDF using Textract layout blocks (Textract path)."""
    regions = find_figure_regions(blocks)
    return extract_figures_from_regions(
        pdf_path, regions, doc_id,
        output_dir=output_dir, dpi=dpi,
        image_format=image_format, padding_pct=padding_pct,
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
    from aws_rag.aws import s3_client
    from aws_rag.config import get_settings

    settings = get_settings()
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
            Bucket=settings.s3_bucket,
            Key=s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        fig.s3_key = s3_key

    uploaded = sum(1 for f in manifest.figures if f.s3_key)
    console.print(
        f"[green]Uploaded {uploaded} figures[/] → "
        f"s3://{settings.s3_bucket}/{s3_prefix}{manifest.doc_id}/"
    )
    return manifest
