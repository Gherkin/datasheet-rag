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
    """A detected figure region from Textract LAYOUT_FIGURE blocks."""

    block_id: str
    page: int  # 1-indexed
    # Textract bounding box (normalised 0..1)
    left: float
    top: float
    width: float
    height: float
    # Nearby text context (populated during extraction)
    caption: str = ""
    preceding_text: str = ""
    section_header: str = ""


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
    """
    console.print(f"[blue]Rendering PDF[/] at {dpi} DPI …")

    kwargs: dict[str, Any] = {"dpi": dpi}
    if pages:
        # pdf2image uses 1-indexed pages
        kwargs["first_page"] = min(pages)
        kwargs["last_page"] = max(pages)

    images = convert_from_path(str(pdf_path), **kwargs)

    start_page = min(pages) if pages else 1
    result = {start_page + i: img for i, img in enumerate(images)}
    console.print(f"[green]Rendered[/] {len(result)} pages")
    return result


def crop_figure(
    page_image: Image.Image,
    region: FigureRegion,
    *,
    padding_pct: float = 0.02,
) -> Image.Image:
    """Crop a figure region from a rendered page image.

    Textract bounding boxes are normalised (0..1), so we scale to pixel
    coordinates. Adds a small padding to avoid cutting off edges.
    """
    w, h = page_image.size

    left = max(0, int((region.left - padding_pct) * w))
    top = max(0, int((region.top - padding_pct) * h))
    right = min(w, int((region.left + region.width + padding_pct) * w))
    bottom = min(h, int((region.top + region.height + padding_pct) * h))

    return page_image.crop((left, top, right, bottom))


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------


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
    """Extract all figures from a PDF using Textract layout blocks.

    1. Find LAYOUT_FIGURE regions in the blocks
    2. Render only the necessary PDF pages
    3. Crop each figure with padding
    4. Save to output_dir and build manifest

    Returns a FigureManifest with paths and metadata for each figure.
    """
    if output_dir is None:
        output_dir = Path("output").resolve() / "figures" / doc_id

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find figure regions
    regions = find_figure_regions(blocks)
    if not regions:
        console.print("[yellow]No figures detected in this document.[/]")
        return FigureManifest(doc_id=doc_id, source_pdf=str(pdf_path))

    console.print(f"[blue]Found {len(regions)} figures[/] across pages "
                  f"{sorted(set(r.page for r in regions))}")

    # Step 2: Render only the pages that contain figures
    needed_pages = sorted(set(r.page for r in regions))
    page_images = render_pdf_pages(pdf_path, dpi=dpi, pages=needed_pages)

    # Step 3: Crop and save each figure
    manifest = FigureManifest(doc_id=doc_id, source_pdf=str(pdf_path))

    for i, region in enumerate(track(regions, description="Cropping figures…")):
        page_img = page_images.get(region.page)
        if page_img is None:
            console.print(f"[red]Warning:[/] page {region.page} not rendered, skipping figure")
            continue

        cropped = crop_figure(page_img, region, padding_pct=padding_pct)

        filename = f"p{region.page:03d}_fig{i:03d}.{image_format}"
        image_path = output_dir / filename
        cropped.save(str(image_path), format=image_format.upper())

        extracted = ExtractedFigure(
            region=region,
            image_path=image_path,
            width_px=cropped.width,
            height_px=cropped.height,
        )
        manifest.figures.append(extracted)

    console.print(f"[green]Extracted {len(manifest.figures)} figures[/] → {output_dir}")
    return manifest


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
