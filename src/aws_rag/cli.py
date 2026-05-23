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


# ---------------------------------------------------------------------------
# Chunk (multi-scale chunking pipeline)
# ---------------------------------------------------------------------------


@cli.command("chunk")
@click.argument("blocks_json", type=click.Path(exists=True, path_type=Path))
@click.option("--doc-id", default=None, help="Document ID (inferred from filename if omitted).")
@click.option(
    "--figures-manifest",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to figure manifest JSON from extract-figures.",
)
@click.option("--micro-tokens", default=128, type=int, help="Max tokens per MICRO chunk.")
@click.option("--meso-tokens", default=512, type=int, help="Max tokens per MESO chunk.")
@click.option(
    "--summarizer",
    type=click.Choice(["extractive", "abstractive"]),
    default="extractive",
    help="Summarization mode for MACRO chunks.",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def chunk_cmd(
    blocks_json: Path,
    doc_id: str | None,
    figures_manifest: Path | None,
    micro_tokens: int,
    meso_tokens: int,
    summarizer: str,
    output: Path | None,
) -> None:
    """Run the multi-scale chunking pipeline on Textract blocks.

    Produces a hierarchical chunk graph at three levels (MACRO/MESO/MICRO)
    with navigation links, context enrichment, and chapter summaries.
    """
    from aws_rag.chunking.pipeline import run_chunking_pipeline, save_chunk_graph
    from aws_rag.chunking.splitter import SplitterConfig

    with open(blocks_json) as f:
        blocks = json.load(f)

    if doc_id is None:
        doc_id = blocks_json.stem.replace("_blocks", "")

    # Load figure manifest if provided
    figure_manifest = None
    if figures_manifest:
        with open(figures_manifest) as f:
            figure_manifest = json.load(f)

    config = SplitterConfig(
        micro_max_tokens=micro_tokens,
        meso_max_tokens=meso_tokens,
    )

    graph = run_chunking_pipeline(
        blocks,
        doc_id=doc_id,
        figure_manifest=figure_manifest,
        config=config,
        summarizer_mode=summarizer,
    )

    # Save
    settings = get_settings()
    if output is None:
        output = settings.output_dir / f"{doc_id}_chunks.json"

    save_chunk_graph(graph, output)

    # Display summary
    stats = graph.stats()
    table = Table(title="Chunk Graph Summary")
    table.add_column("Level", style="cyan")
    table.add_column("Count", justify="right")

    for level_name, count in stats["by_level"].items():
        table.add_row(level_name, str(count))
    table.add_row("TOTAL", str(stats["total_chunks"]), style="bold")

    console.print(table)

    # Show MACRO chunks (chapter summaries)
    from aws_rag.models.chunk import ChunkLevel

    macros = graph.by_level(ChunkLevel.MACRO)
    if macros:
        console.print("\n[bold]Chapter Summaries:[/]")
        for m in macros:
            console.print(f"\n[cyan]{m.metadata.chapter_title}[/] (pages {m.metadata.page_numbers})")
            if m.text:
                preview = m.text[:300] + "…" if len(m.text) > 300 else m.text
                console.print(f"  {preview}")
            else:
                console.print("  [yellow](no summary generated)[/]")


# ---------------------------------------------------------------------------
# Embed (Bedrock Titan + SQLite store)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("chunks_json", type=click.Path(exists=True, path_type=Path))
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None,
              help="SQLite DB path. Defaults to settings.sqlite_db_path.")
@click.option("--project-id", default=None, help="Project ID to attach to every chunk.")
@click.option("--group", "group_name", default=None, help="Group name to attach to every chunk.")
@click.option("--verbose/--quiet", default=True, help="Print per-batch progress.")
@click.option("--dry-run", is_flag=True, help="Embed but do not write to the store.")
def embed(
    chunks_json: Path,
    db_path: Path | None,
    project_id: str | None,
    group_name: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Embed a chunk graph (produced by `rag chunk`) and store it in SQLite."""
    from aws_rag.chunking.pipeline import load_chunk_graph
    from aws_rag.embedding import BedrockEmbedder, embed_chunk_graph
    from aws_rag.store import connect, insert_chunk_graph

    console.print(f"Loading chunk graph from [cyan]{chunks_json}[/]…")
    graph = load_chunk_graph(chunks_json)
    stats = graph.stats()
    console.print(f"  {stats['total_chunks']} chunks "
                  f"(MACRO {stats['by_level']['MACRO']}, "
                  f"MESO {stats['by_level']['MESO']}, "
                  f"MICRO {stats['by_level']['MICRO']})")

    console.print("Embedding with Bedrock Titan v2…")
    embedder = BedrockEmbedder(verbose=verbose)
    vectors = embed_chunk_graph(graph, embedder=embedder)
    s = embedder.stats()
    console.print(
        f"  [green]Embedded[/] {len(vectors)} chunks · "
        f"{s['total_tokens_in']} input tokens · "
        f"{s['total_errors']} errors"
    )

    if dry_run:
        console.print("[yellow]Dry run — not writing to the store.[/]")
        return

    settings = get_settings()
    target = db_path or settings.sqlite_db_path
    console.print(f"Writing to [cyan]{target}[/]…")
    conn = connect(target)
    inserted = insert_chunk_graph(
        conn, graph, vectors=vectors,
        project_id=project_id, group_name=group_name,
    )
    conn.commit()
    conn.close()
    console.print(f"[green]Inserted[/] {inserted} chunks.")


# ---------------------------------------------------------------------------
# Search (hybrid / vector / keyword)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("query", type=str)
@click.option("--mode", type=click.Choice(["hybrid", "vector", "keyword"]),
              default="hybrid", help="Retrieval mode.")
@click.option("-k", "top_k", default=10, type=int, help="Number of results.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None,
              help="SQLite DB path. Defaults to settings.sqlite_db_path.")
@click.option("--project-id", default=None)
@click.option("--group", "group_name", default=None)
@click.option("--doc-id", "doc_ids", multiple=True, help="Restrict to one or more doc IDs.")
@click.option("--level", type=click.Choice(["macro", "meso", "micro"]),
              default=None, help="Restrict to a single zoom level.")
@click.option("--show-context/--no-show-context", default=False,
              help="Show context_text (full embedding-ready blob) instead of raw text.")
def search(
    query: str,
    mode: str,
    top_k: int,
    db_path: Path | None,
    project_id: str | None,
    group_name: str | None,
    doc_ids: tuple[str, ...],
    level: str | None,
    show_context: bool,
) -> None:
    """Search the local SQLite store with hybrid / vector / keyword retrieval."""
    from aws_rag.models.chunk import ChunkLevel
    from aws_rag.store import (
        SearchFilters,
        connect,
        hybrid_search,
        keyword_search,
        vector_search,
    )

    settings = get_settings()
    target = db_path or settings.sqlite_db_path
    conn = connect(target)

    level_enum = None
    if level:
        level_enum = {"macro": ChunkLevel.MACRO, "meso": ChunkLevel.MESO,
                      "micro": ChunkLevel.MICRO}[level]

    filters = SearchFilters(
        doc_ids=list(doc_ids) if doc_ids else None,
        project_id=project_id,
        group_name=group_name,
        level=level_enum,
    )

    if mode in ("vector", "hybrid"):
        from aws_rag.embedding import BedrockEmbedder

        embedder = BedrockEmbedder()
        query_vec = embedder.embed_one(query)
    else:
        query_vec = None

    if mode == "vector":
        results = vector_search(conn, query_vec, k=top_k, filters=filters)  # type: ignore[arg-type]
    elif mode == "keyword":
        results = keyword_search(conn, query, k=top_k, filters=filters)
    else:
        results = hybrid_search(conn, query_vec, query, k=top_k, filters=filters)  # type: ignore[arg-type]

    if not results:
        console.print("[yellow]No results.[/]")
        return

    table = Table(title=f"{mode.title()} search · {len(results)} results")
    table.add_column("#", justify="right", style="dim")
    table.add_column("score", justify="right", style="cyan")
    table.add_column("level", style="magenta")
    table.add_column("doc", style="dim")
    table.add_column("section")
    table.add_column("preview")

    for i, r in enumerate(results, 1):
        body = r.chunk.context_text if show_context else r.chunk.text
        preview = body[:140].replace("\n", " ") + ("…" if len(body) > 140 else "")
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            r.chunk.level.name,
            r.chunk.doc_id[:10],
            (r.chunk.metadata.section_title or r.chunk.metadata.chapter_title or "")[:40],
            preview,
        )

    console.print(table)
    conn.close()


# ---------------------------------------------------------------------------
# Document metadata (sidecar — separate from chunks, no re-ingest required)
# ---------------------------------------------------------------------------


@cli.group()
def metadata() -> None:
    """Manage the doc-level metadata sidecar (project, mpn, manufacturer, …)."""


@metadata.command("set")
@click.argument("doc_id", type=str)
@click.option("--project-id", default=None)
@click.option("--group", "group_name", default=None)
@click.option("--mpn", default=None, help="Manufacturer part number, e.g. STM32H743VIT6.")
@click.option("--manufacturer", default=None)
@click.option("--subsystem", default=None, help="e.g. power, rf, mcu.")
@click.option("--doc-type", default=None,
              help="datasheet | reference-manual | errata | app-note | …")
@click.option("--tag", "tags", multiple=True, help="Repeatable --tag flag.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--apply-to-chunks/--no-apply-to-chunks", default=True,
              help="Propagate project_id and group_name into the chunks table.")
def metadata_set(
    doc_id: str,
    project_id: str | None,
    group_name: str | None,
    mpn: str | None,
    manufacturer: str | None,
    subsystem: str | None,
    doc_type: str | None,
    tags: tuple[str, ...],
    db_path: Path | None,
    apply_to_chunks: bool,
) -> None:
    """Upsert document metadata. Only fields you pass are updated."""
    from aws_rag.store import apply_metadata_to_chunks, connect, set_metadata

    settings = get_settings()
    target = db_path or settings.sqlite_db_path
    conn = connect(target)

    meta = set_metadata(
        conn, doc_id,
        project_id=project_id, group_name=group_name,
        mpn=mpn, manufacturer=manufacturer, subsystem=subsystem,
        doc_type=doc_type,
        tags=list(tags) if tags else None,
    )
    conn.commit()
    console.print(f"[green]Saved metadata for[/] {doc_id}")
    console.print(meta.model_dump_json(indent=2, exclude_none=True))

    if apply_to_chunks:
        updated = apply_metadata_to_chunks(conn, doc_id)
        conn.commit()
        console.print(f"  Propagated to {updated} chunk rows.")

    conn.close()


@metadata.command("get")
@click.argument("doc_id", type=str)
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
def metadata_get(doc_id: str, db_path: Path | None) -> None:
    """Show the sidecar metadata row for a document."""
    from aws_rag.store import connect, get_metadata

    settings = get_settings()
    target = db_path or settings.sqlite_db_path
    conn = connect(target)
    meta = get_metadata(conn, doc_id)
    if meta is None:
        console.print(f"[yellow]No metadata recorded for[/] {doc_id}")
        return
    console.print(meta.model_dump_json(indent=2, exclude_none=True))
    conn.close()


@metadata.command("list")
@click.option("--project-id", default=None)
@click.option("--group", "group_name", default=None)
@click.option("--mpn", default=None)
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
def metadata_list(
    project_id: str | None,
    group_name: str | None,
    mpn: str | None,
    db_path: Path | None,
) -> None:
    """List documents in the sidecar, optionally filtered."""
    from aws_rag.store import connect, list_docs

    settings = get_settings()
    target = db_path or settings.sqlite_db_path
    conn = connect(target)
    docs = list_docs(conn, project_id=project_id, group_name=group_name, mpn=mpn)

    if not docs:
        console.print("[yellow]No documents match.[/]")
        return

    table = Table(title=f"Documents ({len(docs)})")
    table.add_column("doc_id", style="cyan")
    table.add_column("project")
    table.add_column("group")
    table.add_column("mpn")
    table.add_column("manufacturer")
    table.add_column("subsystem")

    for d in docs:
        table.add_row(
            d.doc_id[:14], d.project_id or "—",
            d.group_name or "—", d.mpn or "—",
            d.manufacturer or "—", d.subsystem or "—",
        )
    console.print(table)
    conn.close()
