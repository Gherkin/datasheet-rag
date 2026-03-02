"""CLI for the RAG pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from aws_rag.config import get_settings

console = Console()


@click.group()
@click.option("--bucket", envvar="RAG_S3_BUCKET", default=None, help="Override S3 bucket name.")
def cli(bucket: str | None) -> None:
    """AWS RAG Pipeline — electronics datasheet ingestion."""
    if bucket:
        import os
        os.environ["RAG_S3_BUCKET"] = bucket


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("pdf_paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--doc-id", default=None, help="Explicit document ID (default: content hash).")
def upload(pdf_paths: tuple[Path, ...], doc_id: str | None) -> None:
    """Upload one or more PDFs to S3."""
    from aws_rag.storage import upload_pdf

    for pdf_path in pdf_paths:
        if not pdf_path.suffix.lower() == ".pdf":
            console.print(f"[red]Skipping non-PDF:[/] {pdf_path}")
            continue
        did, key = upload_pdf(pdf_path, doc_id=doc_id)
        console.print(f"  doc_id = {did}")
        console.print(f"  s3_key = {key}")
        console.print()


# ---------------------------------------------------------------------------
# Analyze (Textract)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("target", type=str)
@click.option(
    "--mode",
    type=click.Choice(["sync", "async"]),
    default="async",
    help="sync = local single-page PDF, async = S3 multi-page.",
)
@click.option("--wait/--no-wait", default=True, help="Wait for async job to complete.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def analyze(target: str, mode: str, wait: bool, output: Path | None) -> None:
    """Run Textract analysis on a PDF.

    TARGET is either a local file path (sync mode) or a doc_id (async mode).
    """
    from aws_rag.textract import (
        analyze_document_sync,
        get_job_results,
        save_blocks,
        start_analysis,
        wait_for_job,
    )

    settings = get_settings()

    if mode == "sync":
        pdf_path = Path(target)
        if not pdf_path.is_file():
            raise click.BadParameter(f"File not found: {pdf_path}")
        response = analyze_document_sync(pdf_path)
        blocks = response.get("Blocks", [])
    else:
        # async — target is a doc_id; we need to find its S3 key
        from aws_rag.storage import list_documents

        docs = list_documents()
        match = [d for d in docs if d["doc_id"] == target]
        if not match:
            raise click.BadParameter(
                f"doc_id '{target}' not found. Upload the PDF first with `rag upload`."
            )
        # Find the actual PDF key under the prefix
        from aws_rag.aws import s3_client

        client = s3_client()
        prefix = match[0]["prefix"]
        resp = client.list_objects_v2(Bucket=settings.s3_bucket, Prefix=prefix)
        pdf_keys = [
            obj["Key"]
            for obj in resp.get("Contents", [])
            if obj["Key"].lower().endswith(".pdf")
        ]
        if not pdf_keys:
            raise click.ClickException(f"No PDF found under s3://{settings.s3_bucket}/{prefix}")

        s3_key = pdf_keys[0]
        job_id = start_analysis(target, s3_key)

        if not wait:
            console.print(f"Job ID: {job_id}")
            console.print("Use `rag job-status` to check progress.")
            return

        status = wait_for_job(job_id)
        if status != "SUCCEEDED":
            raise click.ClickException(f"Textract job failed with status: {status}")

        blocks = get_job_results(job_id)

    # Save output
    if output is None:
        output = settings.output_dir / f"{target.replace('/', '_')}_blocks.json"

    save_blocks(blocks, output)


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------


@cli.command("list")
def list_docs() -> None:
    """List uploaded documents."""
    from aws_rag.storage import list_documents

    docs = list_documents()
    if not docs:
        console.print("[yellow]No documents found.[/]")
        return

    table = Table(title="Uploaded Documents")
    table.add_column("doc_id", style="cyan")
    table.add_column("S3 Prefix")

    for doc in docs:
        table.add_row(doc["doc_id"], doc["prefix"])

    console.print(table)


# ---------------------------------------------------------------------------
# Extract text (from saved Textract JSON)
# ---------------------------------------------------------------------------


@cli.command("extract-text")
@click.argument("blocks_json", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def extract_text(blocks_json: Path, output: Path | None) -> None:
    """Extract readable text from Textract blocks JSON, preserving layout order."""
    from aws_rag.textract import build_text_from_layout

    with open(blocks_json) as f:
        blocks = json.load(f)

    text = build_text_from_layout(blocks)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        console.print(f"[green]Text saved to[/] {output}")
    else:
        console.print(text)


# ---------------------------------------------------------------------------
# Inspect layout (debug helper)
# ---------------------------------------------------------------------------


@cli.command("inspect-layout")
@click.argument("blocks_json", type=click.Path(exists=True, path_type=Path))
def inspect_layout(blocks_json: Path) -> None:
    """Show a summary of Textract block types and layout structure."""
    from aws_rag.textract import extract_layout_elements

    with open(blocks_json) as f:
        blocks = json.load(f)

    by_type = extract_layout_elements(blocks)

    table = Table(title="Textract Block Types")
    table.add_column("Block Type", style="cyan")
    table.add_column("Count", justify="right")

    for bt, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        table.add_row(bt, str(len(items)))

    console.print(table)

    # Show layout hierarchy if present
    layout_blocks = [b for b in blocks if b.get("BlockType", "").startswith("LAYOUT_")]
    if layout_blocks:
        console.print(f"\n[bold]Layout blocks ({len(layout_blocks)}):[/]")
        for lb in layout_blocks[:30]:
            bt = lb["BlockType"]
            page = lb.get("Page", "?")
            top = lb.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0)
            text_preview = ""
            if "Text" in lb:
                text_preview = lb["Text"][:80]
            console.print(f"  p{page} {bt:30s} top={top:.3f}  {text_preview}")

        if len(layout_blocks) > 30:
            console.print(f"  … and {len(layout_blocks) - 30} more")


# ---------------------------------------------------------------------------
# Extract figures
# ---------------------------------------------------------------------------


@cli.command("extract-figures")
@click.argument("blocks_json", type=click.Path(exists=True, path_type=Path))
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--doc-id", default=None, help="Document ID (inferred from blocks filename if omitted).")
@click.option("--dpi", default=300, type=int, help="Render DPI for PDF pages.")
@click.option("--format", "image_format", default="png", type=click.Choice(["png", "jpg", "webp"]))
@click.option("--padding", default=0.02, type=float, help="Padding around figures (fraction of page).")
@click.option("--upload/--no-upload", default=False, help="Upload figures to S3 after extraction.")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None)
def extract_figures_cmd(
    blocks_json: Path,
    pdf_path: Path,
    doc_id: str | None,
    dpi: int,
    image_format: str,
    padding: float,
    upload: bool,
    output_dir: Path | None,
) -> None:
    """Extract figure images from a PDF using Textract layout detection.

    Crops each LAYOUT_FIGURE region and saves as individual images.
    Generates a manifest JSON with metadata, captions, and context.
    """
    from aws_rag.figures import extract_figures, upload_figures_to_s3

    with open(blocks_json) as f:
        blocks = json.load(f)

    # Infer doc_id from filename if not provided
    if doc_id is None:
        doc_id = blocks_json.stem.replace("_blocks", "")

    manifest = extract_figures(
        pdf_path=pdf_path,
        blocks=blocks,
        doc_id=doc_id,
        output_dir=output_dir,
        dpi=dpi,
        image_format=image_format,
        padding_pct=padding,
    )

    if upload and manifest.figures:
        manifest = upload_figures_to_s3(manifest)

    # Save manifest
    settings = get_settings()
    manifest_dir = output_dir or settings.output_dir / "figures" / doc_id
    manifest.save(manifest_dir / "manifest.json")

    # Summary table
    if manifest.figures:
        table = Table(title=f"Extracted Figures ({len(manifest.figures)})")
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Page")
        table.add_column("Size")
        table.add_column("Caption")
        table.add_column("Section")

        for i, fig in enumerate(manifest.figures):
            table.add_row(
                str(i),
                str(fig.region.page),
                f"{fig.width_px}×{fig.height_px}",
                (fig.region.caption[:60] + "…") if len(fig.region.caption) > 60 else fig.region.caption or "—",
                fig.region.section_header[:40] or "—",
            )

        console.print(table)
