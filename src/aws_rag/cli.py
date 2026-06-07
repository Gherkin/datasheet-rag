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
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
def list_docs(db_path: Path | None) -> None:
    """List uploaded documents."""
    from aws_rag.storage import list_documents
    from aws_rag.store import connect, get_doc_titles

    docs = list_documents()
    if not docs:
        console.print("[yellow]No documents found.[/]")
        return

    settings = get_settings()
    target = db_path or settings.sqlite_db_path
    conn = connect(target)
    titles = get_doc_titles(conn)
    conn.close()

    table = Table(title="Uploaded Documents")
    table.add_column("doc_id", style="cyan")
    table.add_column("title")
    table.add_column("S3 Prefix")

    for doc in docs:
        doc_id = doc["doc_id"]
        table.add_row(doc_id, titles.get(doc_id, "—"), doc["prefix"])

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

    settings = get_settings()
    target = db_path or settings.sqlite_db_path

    # Fold any previously-generated figure descriptions into context_text so
    # vectors include them. The JSON file never carries descriptions back.
    if target.exists():
        _conn = connect(target)
        try:
            undescribed_ids = [
                c.id for c in graph.chunks.values() if not c.figure_description
            ]
            if undescribed_ids:
                placeholders = ",".join("?" * len(undescribed_ids))
                rows = _conn.execute(
                    f"SELECT id, figure_description FROM chunks "
                    f"WHERE id IN ({placeholders}) AND figure_description IS NOT NULL",
                    undescribed_ids,
                ).fetchall()
                if rows:
                    console.print(
                        f"  Merging [green]{len(rows)}[/] stored figure "
                        f"descriptions into context_text before embedding…"
                    )
                for row in rows:
                    chunk = graph.chunks.get(row["id"])
                    if chunk:
                        desc = row["figure_description"]
                        chunk.figure_description = desc
                        tag = f"Description: {desc}"
                        if tag not in (chunk.context_text or ""):
                            chunk.context_text = (chunk.context_text or chunk.text) + "\n" + tag
        finally:
            _conn.close()

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
# Figures (list / inspect figure chunks in the store)
# ---------------------------------------------------------------------------


@cli.command("list-figures")
@click.option("--doc-id", default=None)
@click.option("--project-id", default=None)
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--missing-description-only", is_flag=True,
              help="Only show figure chunks whose figure_description is empty.")
def list_figures_cmd(
    doc_id: str | None,
    project_id: str | None,
    db_path: Path | None,
    missing_description_only: bool,
) -> None:
    """List figure chunks in the store (those with a usable image)."""
    from aws_rag.store import connect, list_figure_chunks

    settings = get_settings()
    target = db_path or settings.sqlite_db_path
    conn = connect(target)
    figs = list_figure_chunks(conn, doc_id=doc_id, project_id=project_id)
    if missing_description_only:
        figs = [c for c in figs if not c.figure_description]

    if not figs:
        console.print("[yellow]No figure chunks match.[/]")
        return

    table = Table(title=f"Figure chunks ({len(figs)})")
    table.add_column("chunk_id", style="cyan")
    table.add_column("doc")
    table.add_column("page")
    table.add_column("section")
    table.add_column("caption")
    table.add_column("desc?", justify="center")
    table.add_column("source", style="dim")

    for c in figs:
        pages = c.metadata.page_numbers
        page = (str(pages[0]) if len(pages) == 1
                else f"{pages[0]}-{pages[-1]}" if pages else "")
        src = "local" if c.figure_image_path else ("s3" if c.figure_s3_key else "—")
        table.add_row(
            c.id[-14:],
            c.doc_id[:10],
            page,
            (c.metadata.section_title or "")[:30],
            (c.figure_caption or "")[:40],
            "[green]Y[/]" if c.figure_description else "[red]N[/]",
            src,
        )
    console.print(table)
    conn.close()


# ---------------------------------------------------------------------------
# describe-figures (Bedrock Claude vision → figure_description)
# ---------------------------------------------------------------------------


@cli.command("describe-figures")
@click.option("--doc-id", default=None, help="Restrict to a single document.")
@click.option("--project-id", default=None, help="Restrict to a single project.")
@click.option("--missing-only/--all", default=True,
              help="Skip figures that already have a description (default on).")
@click.option("--limit", default=None, type=int,
              help="Stop after this many figures (cost guard).")
@click.option("--model", "model_id", default=None,
              help="Override settings.description_model_id for this run.")
@click.option("--dry-run", is_flag=True,
              help="Generate descriptions and print them but do not persist.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--verbose/--quiet", default=True)
def describe_figures_cmd(
    doc_id: str | None,
    project_id: str | None,
    missing_only: bool,
    limit: int | None,
    model_id: str | None,
    dry_run: bool,
    db_path: Path | None,
    verbose: bool,
) -> None:
    """Generate vision-LLM descriptions for figure chunks and persist them.

    Walks `chunks WHERE layout_type='figure'` (optionally filtered by
    doc/project, skipping those that already have a description), sends
    each image + caption + neighbour text to Bedrock Claude vision, and
    folds the response into the chunk row + context_text.

    After running, re-embed the affected document so the new
    descriptions show up in vector search:

        rag describe-figures --doc-id <doc>
        rag chunk output/<doc>_blocks.json --figures-manifest ...   # if needed
        rag embed output/<doc>_chunks.json --project-id <p>
    """
    from aws_rag.description import FigureDescriber, describe_figures_in_store
    from aws_rag.store import connect

    settings = get_settings()
    target = db_path or settings.sqlite_db_path
    conn = connect(target)

    describer = FigureDescriber(model_id=model_id, verbose=verbose)
    console.print(
        f"Describing figures with [cyan]{describer.model_id}[/] "
        f"(concurrency={describer.max_concurrency}, max_tokens={describer.max_tokens})"
    )

    descriptions = describe_figures_in_store(
        conn,
        doc_id=doc_id,
        project_id=project_id,
        missing_only=missing_only,
        limit=limit,
        describer=describer,
        dry_run=dry_run,
    )

    s = describer.stats()
    console.print(
        f"  [green]{len(descriptions)}[/] descriptions · "
        f"in={s['total_input_tokens']} tok · "
        f"out={s['total_output_tokens']} tok · "
        f"errors={s['total_errors']}"
    )

    if dry_run and descriptions:
        console.print("\n[yellow]Dry run — not persisted.[/]\n")
        for chunk_id, desc in descriptions.items():
            console.print(f"[cyan]{chunk_id}[/]")
            console.print(f"  {desc}\n")
    elif descriptions:
        console.print("[green]Descriptions written to chunks + context_text.[/]")
        console.print(
            "  Re-run `rag embed <chunks.json>` (or implement an "
            "updated-only re-embed) to refresh the vectors."
        )

    conn.close()


# ---------------------------------------------------------------------------
# Ingest (full pipeline: upload → analyze → extract-figures → chunk → embed)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--doc-id", default=None, help="Explicit document ID (default: content hash).")
@click.option("--project-id", default=None, help="Project ID attached to all chunks.")
@click.option("--group", "group_name", default=None, help="Group name attached to all chunks.")
@click.option("--mpn", default=None, help="Manufacturer part number, e.g. STM32H743VIT6.")
@click.option("--manufacturer", default=None)
@click.option("--subsystem", default=None, help="e.g. power, rf, mcu.")
@click.option("--doc-type", default=None,
              help="datasheet | reference-manual | errata | app-note | …")
@click.option("--tag", "tags", multiple=True, help="Repeatable --tag flag.")
@click.option("--skip-figures", is_flag=True, help="Skip figure extraction and description steps.")
@click.option("--skip-describe", is_flag=True, help="Skip AI figure description (but still extract).")
@click.option("--dpi", default=300, type=int, help="Render DPI for figure extraction.")
@click.option("--micro-tokens", default=128, type=int, help="Max tokens per MICRO chunk.")
@click.option("--meso-tokens", default=512, type=int, help="Max tokens per MESO chunk.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None,
              help="SQLite DB path. Defaults to settings.sqlite_db_path.")
@click.option("--dry-run", is_flag=True, help="Run all steps but do not write to the store.")
@click.option("--force", is_flag=True, help="Ignore cached blocks/chunks and redo all steps.")
@click.option(
    "--backend",
    type=click.Choice(["auto", "docling", "textract"], case_sensitive=False),
    default="docling",
    help=(
        "Layout extraction backend. "
        "'docling' (default) handles native PDFs for free and fails verbosely "
        "on scanned PDFs, telling you to pass --backend textract if you want "
        "to pay for AWS OCR. "
        "'auto' detects native vs scanned and silently routes scanned PDFs to "
        "Textract. "
        "'textract' forces AWS OCR for any PDF."
    ),
)
@click.option(
    "--accurate-tables",
    is_flag=True,
    default=False,
    help=(
        "Use TableFormer ACCURATE mode for table structure (Docling backend only). "
        "Default is FAST, which is 44% faster with negligible quality loss for RAG. "
        "Use ACCURATE when precise cell-boundary detection matters."
    ),
)
def ingest(
    pdf_path: Path,
    doc_id: str | None,
    project_id: str | None,
    group_name: str | None,
    mpn: str | None,
    manufacturer: str | None,
    subsystem: str | None,
    doc_type: str | None,
    tags: tuple[str, ...],
    skip_figures: bool,
    skip_describe: bool,
    dpi: int,
    micro_tokens: int,
    meso_tokens: int,
    db_path: Path | None,
    dry_run: bool,
    force: bool,
    backend: str,
    accurate_tables: bool,
) -> None:
    """Full ingestion pipeline: analyse → figures → chunk → embed.

    Defaults to Docling (free, fast, handles tables/formulas/figures on
    native PDFs) and fails verbosely on scanned PDFs rather than silently
    incurring AWS Textract OCR costs. Pass --backend textract to OCR a
    scanned PDF, or --backend auto to route automatically between the two.
    Intermediate artefacts (blocks/chunk graph) are cached in output_dir;
    pass --force to ignore the cache and redo all steps.
    """
    import time

    from aws_rag.chunking.pipeline import (
        load_chunk_graph,
        run_chunking_pipeline,
        run_chunking_pipeline_from_outline,
        save_chunk_graph,
    )
    from aws_rag.chunking.splitter import SplitterConfig
    from aws_rag.embedding import BedrockEmbedder, embed_chunk_graph
    from aws_rag.figures import extract_figures, extract_figures_from_regions
    from aws_rag.store import apply_metadata_to_chunks, connect, insert_chunk_graph, set_metadata

    settings = get_settings()
    t0 = time.monotonic()
    step_n = 0

    def _step(label: str) -> None:
        nonlocal step_n
        step_n += 1
        console.rule(f"[bold cyan]Step {step_n} — {label}[/]")

    # ── 1. Detect backend ────────────────────────────────────────────────────
    _step("Detect PDF type")
    if backend in ("auto", "docling"):
        from aws_rag.docling_parser import is_native_pdf
        native = is_native_pdf(pdf_path)
        if native:
            resolved_backend = "docling"
            console.print("  Native PDF detected → using [cyan]docling[/] backend")
        elif backend == "auto":
            resolved_backend = "textract"
            console.print("  Scanned PDF detected → using [cyan]textract[/] backend")
        else:
            raise click.ClickException(
                f"{pdf_path.name} looks like a scanned PDF — Docling needs a "
                "native text layer and cannot OCR it.\n"
                "  Re-run with --backend textract to use AWS Textract OCR "
                "instead (this incurs AWS costs), or --backend auto to route "
                "automatically based on PDF type."
            )
    else:
        resolved_backend = backend
        console.print(f"  Backend forced to [cyan]{resolved_backend}[/]")

    # ── 2a. Docling path (native PDFs) ───────────────────────────────────────
    if resolved_backend == "docling":
        from aws_rag.docling_parser import content_hash, convert_pdf

        did = doc_id or content_hash(pdf_path)
        console.print(f"  doc_id = [cyan]{did}[/]")

        chunks_path = settings.output_dir / f"{did}_chunks.json"
        if chunks_path.exists() and not force:
            _step("Multi-scale chunking")
            console.print(f"  [yellow]Resuming — loading cached chunk graph[/] → [cyan]{chunks_path}[/]")
            graph = load_chunk_graph(chunks_path)
            stats = graph.stats()
            console.print(
                f"  {stats['total_chunks']} chunks "
                f"(MACRO {stats['by_level']['MACRO']}, "
                f"MESO {stats['by_level']['MESO']}, "
                f"MICRO {stats['by_level']['MICRO']}) (cached)"
            )
        else:
            _step("Docling layout analysis")
            outline, figure_regions = convert_pdf(pdf_path, doc_id=did, accurate_tables=accurate_tables)
            summary = outline.summary()
            console.print(
                f"  {summary['top_level_sections']} chapters, "
                f"{summary['total_sections']} sections, "
                f"{summary['total_elements']} elements "
                f"({summary['elements_by_type'].get('formula', 0)} formulas, "
                f"{summary['elements_by_type'].get('table', 0)} tables, "
                f"{summary['elements_by_type'].get('figure', 0)} figures)"
            )

            figure_manifest_dict = None
            if not skip_figures:
                _step("Extract figures & formulas")
                figures_out = settings.output_dir / "figures" / did
                manifest = extract_figures_from_regions(
                    pdf_path=pdf_path,
                    regions=figure_regions,
                    doc_id=did,
                    output_dir=figures_out,
                    dpi=dpi,
                    image_format="png",
                    padding_pct=0.02,
                )
                manifest_path = figures_out / "manifest.json"
                manifest.save(manifest_path)
                figure_manifest_dict = manifest.to_dict()
                console.print(f"  {len(manifest.figures)} regions → [cyan]{manifest_path}[/]")

            _step("Multi-scale chunking")
            config = SplitterConfig(micro_max_tokens=micro_tokens, meso_max_tokens=meso_tokens)
            graph = run_chunking_pipeline_from_outline(
                outline,
                figure_manifest=figure_manifest_dict,
                config=config,
                summarizer_mode="extractive",
            )
            save_chunk_graph(graph, chunks_path)
            stats = graph.stats()
            console.print(
                f"  {stats['total_chunks']} chunks "
                f"(MACRO {stats['by_level']['MACRO']}, "
                f"MESO {stats['by_level']['MESO']}, "
                f"MICRO {stats['by_level']['MICRO']}) → [cyan]{chunks_path}[/]"
            )

    # ── 2b. Textract path (scanned PDFs) ─────────────────────────────────────
    else:
        from aws_rag.storage import upload_pdf
        from aws_rag.textract import (
            get_job_results,
            load_blocks,
            save_blocks,
            start_analysis,
            wait_for_job,
        )

        _step("Upload PDF to S3")
        did, s3_key = upload_pdf(pdf_path, doc_id=doc_id)
        console.print(f"  doc_id = [cyan]{did}[/]")
        console.print(f"  s3_key = {s3_key}")

        _step("Textract layout analysis (OCR)")
        blocks_path = settings.output_dir / f"{did}_blocks.json"
        if blocks_path.exists() and not force:
            console.print(f"  [yellow]Resuming — loading cached blocks[/] → [cyan]{blocks_path}[/]")
            blocks = load_blocks(blocks_path)
            console.print(f"  {len(blocks)} blocks (cached)")
        else:
            job_id = start_analysis(did, s3_key)
            console.print(f"  job_id = {job_id}  (waiting…)")
            status = wait_for_job(job_id)
            if status != "SUCCEEDED":
                raise click.ClickException(f"Textract job failed with status: {status}")
            blocks = get_job_results(job_id)
            save_blocks(blocks, blocks_path)
            console.print(f"  {len(blocks)} blocks → [cyan]{blocks_path}[/]")

        figure_manifest_dict = None
        if not skip_figures:
            _step("Extract figures")
            figures_out = settings.output_dir / "figures" / did
            manifest = extract_figures(
                pdf_path=pdf_path,
                blocks=blocks,
                doc_id=did,
                output_dir=figures_out,
                dpi=dpi,
                image_format="png",
                padding_pct=0.02,
            )
            manifest_path = figures_out / "manifest.json"
            manifest.save(manifest_path)
            figure_manifest_dict = manifest.to_dict()
            console.print(f"  {len(manifest.figures)} figures → [cyan]{manifest_path}[/]")

        chunks_path = settings.output_dir / f"{did}_chunks.json"
        if chunks_path.exists() and not force:
            _step("Multi-scale chunking")
            console.print(f"  [yellow]Resuming — loading cached chunk graph[/] → [cyan]{chunks_path}[/]")
            graph = load_chunk_graph(chunks_path)
            stats = graph.stats()
            console.print(
                f"  {stats['total_chunks']} chunks "
                f"(MACRO {stats['by_level']['MACRO']}, "
                f"MESO {stats['by_level']['MESO']}, "
                f"MICRO {stats['by_level']['MICRO']}) (cached)"
            )
        else:
            _step("Multi-scale chunking")
            config = SplitterConfig(micro_max_tokens=micro_tokens, meso_max_tokens=meso_tokens)
            graph = run_chunking_pipeline(
                blocks,
                doc_id=did,
                figure_manifest=figure_manifest_dict,
                config=config,
                summarizer_mode="extractive",
            )
            save_chunk_graph(graph, chunks_path)
            stats = graph.stats()
            console.print(
                f"  {stats['total_chunks']} chunks "
                f"(MACRO {stats['by_level']['MACRO']}, "
                f"MESO {stats['by_level']['MESO']}, "
                f"MICRO {stats['by_level']['MICRO']}) → [cyan]{chunks_path}[/]"
            )

    # ── 5. Describe figures (optional) ───────────────────────────────────────
    if not skip_figures and not skip_describe:
        _step("AI figure descriptions")
        # Need chunks in the store first so describe-figures can find them.
        # We do a provisional write, then describe, then the embed step below
        # will merge the descriptions into context_text.
        if not dry_run:
            from aws_rag.description import FigureDescriber, describe_figures_in_store

            target = db_path or settings.sqlite_db_path
            conn = connect(target)
            embedder_tmp = BedrockEmbedder(verbose=False)
            vectors_tmp = embed_chunk_graph(graph, embedder=embedder_tmp)
            inserted_tmp = insert_chunk_graph(
                conn, graph, vectors=vectors_tmp,
                project_id=project_id, group_name=group_name,
            )
            conn.commit()
            console.print(f"  Provisional write of {inserted_tmp} chunks for description pass.")

            describer = FigureDescriber(verbose=True)
            descriptions = describe_figures_in_store(
                conn, doc_id=did, project_id=project_id,
                missing_only=True, describer=describer, dry_run=False,
            )
            s = describer.stats()
            console.print(
                f"  [green]{len(descriptions)}[/] descriptions · "
                f"in={s['total_input_tokens']} tok · out={s['total_output_tokens']} tok"
            )
            conn.close()

            # Merge stored descriptions back into graph so the final embed
            # includes them in context_text.
            conn2 = connect(target)
            for row in conn2.execute(
                "SELECT id, figure_description FROM chunks WHERE doc_id=? AND figure_description IS NOT NULL",
                (did,),
            ).fetchall():
                chunk = graph.chunks.get(row["id"])
                if chunk:
                    chunk.figure_description = row["figure_description"]
                    tag = f"Description: {row['figure_description']}"
                    if tag not in (chunk.context_text or ""):
                        chunk.context_text = (chunk.context_text or chunk.text) + "\n" + tag
            conn2.close()
        else:
            console.print("  [yellow]Dry run — skipping description and provisional embed.[/]")

    # ── 6. Embed + store ─────────────────────────────────────────────────────
    _step("Embed & store")
    console.print("  Embedding with Bedrock Titan v2…")
    embedder = BedrockEmbedder(verbose=True)
    vectors = embed_chunk_graph(graph, embedder=embedder)
    es = embedder.stats()
    console.print(
        f"  [green]Embedded[/] {len(vectors)} chunks · "
        f"{es['total_tokens_in']} input tokens · "
        f"{es['total_errors']} errors"
    )

    if dry_run:
        console.print("[yellow]Dry run — not writing to the store.[/]")
    else:
        target = db_path or settings.sqlite_db_path
        conn = connect(target)
        inserted = insert_chunk_graph(
            conn, graph, vectors=vectors,
            project_id=project_id, group_name=group_name,
        )
        conn.commit()
        console.print(f"  [green]Upserted[/] {inserted} chunks → [cyan]{target}[/]")

        # Metadata sidecar
        any_meta = any([project_id, group_name, mpn, manufacturer, subsystem, doc_type, tags])
        if any_meta:
            set_metadata(
                conn, did,
                project_id=project_id, group_name=group_name,
                mpn=mpn, manufacturer=manufacturer,
                subsystem=subsystem, doc_type=doc_type,
                tags=list(tags) if tags else None,
            )
            apply_metadata_to_chunks(conn, did)
            conn.commit()
            console.print("  Metadata sidecar saved.")

        conn.close()

    elapsed = time.monotonic() - t0
    console.rule(f"[bold green]Done[/] — {elapsed:.0f}s")
    console.print(f"  doc_id = [cyan]{did}[/]")


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


# ---------------------------------------------------------------------------
# Eval (retrieval-layer evaluation)
# ---------------------------------------------------------------------------


@cli.group("eval")
def eval_group() -> None:
    """Retrieval-layer evaluation: golden set, metrics, ablations."""


_CAT_ORDER = ["identifier", "conceptual", "figure", "table_spec", "synthesis", "overall"]


def _render_report_table(report: object) -> None:
    """Print one RunReport's per-category metrics."""
    from aws_rag.eval.harness import RunReport

    assert isinstance(report, RunReport)
    hk = report.config.k
    table = Table(
        title=f"Retrieval eval · {report.config.describe()} "
        f"(hit@k = strict lineage; pg@{hk} = loose page upper bound)"
    )
    table.add_column("category", style="magenta")
    table.add_column("n", justify="right", style="dim")
    for k in report.config.ks:
        table.add_column(f"hit@{k}", justify="right", style="cyan")
    table.add_column(f"pg@{hk}", justify="right", style="dim")
    table.add_column("MRR", justify="right", style="green")
    table.add_column("nDCG", justify="right", style="green")

    for cat in _CAT_ORDER:
        m = report.by_category.get(cat)
        if m is None or m.n == 0:
            continue
        row = [cat, str(m.n)]
        row += [f"{m.hit_rate_at_k.get(k, 0.0):.2f}" for k in report.config.ks]
        row += [f"{m.hit_rate_at_k_loose.get(hk, 0.0):.2f}"]
        row += [f"{m.mrr:.3f}", f"{m.ndcg:.3f}"]
        table.add_row(*row, end_section=(cat == "overall"))
    console.print(table)


def _render_matrix_table(reports: list, headline_k: int) -> None:
    """Print a comparison across configs: overall + per-category hit@k."""
    from aws_rag.eval.dataset import CATEGORIES

    table = Table(title=f"Ablation comparison · hit@{headline_k} by category")
    table.add_column("config", style="yellow")
    table.add_column("overall", justify="right", style="cyan")
    table.add_column("MRR", justify="right", style="green")
    table.add_column("nDCG", justify="right", style="green")
    for cat in CATEGORIES:
        table.add_column(cat[:5], justify="right")

    for rep in reports:
        overall = rep.by_category.get("overall")
        if overall is None:
            continue
        row = [
            rep.config.describe(),
            f"{overall.hit_rate_at_k.get(headline_k, 0.0):.2f}",
            f"{overall.mrr:.3f}",
            f"{overall.ndcg:.3f}",
        ]
        for cat in CATEGORIES:
            m = rep.by_category.get(cat)
            row.append(f"{m.hit_rate_at_k.get(headline_k, 0.0):.2f}" if m else "—")
        table.add_row(*row)
    console.print(table)


@eval_group.command("generate")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None,
              help="SQLite DB path. Defaults to settings.sqlite_db_path.")
@click.option("--per-category", default=4, type=int, help="Items to generate per category.")
@click.option("--doc-id", default=None, help="Restrict sampling to one document.")
@click.option("--project-id", default=None, help="Restrict sampling to one project.")
@click.option("--model", "model_id", default=None, help="Bedrock model ID for generation.")
@click.option("--seed", default=0, type=int, help="Sampling seed (reproducible).")
@click.option("--output", "-o", "out_path", type=click.Path(path_type=Path),
              default=Path("eval/golden.jsonl"), help="Output JSONL path.")
@click.option("--append", is_flag=True, help="Append to the output file instead of overwriting.")
@click.option("--verbose/--quiet", default=True)
def eval_generate(
    db_path: Path | None,
    per_category: int,
    doc_id: str | None,
    project_id: str | None,
    model_id: str | None,
    seed: int,
    out_path: Path,
    append: bool,
    verbose: bool,
) -> None:
    """Generate a reviewable golden set from the corpus (LLM-assisted)."""
    from aws_rag.eval.generate import generate_golden_set
    from aws_rag.store import connect

    settings = get_settings()
    conn = connect(db_path or settings.sqlite_db_path)
    eval_set = generate_golden_set(
        conn,
        per_category=per_category,
        doc_id=doc_id,
        project_id=project_id,
        model_id=model_id,
        seed=seed,
        verbose=verbose,
    )
    conn.close()

    if append:
        eval_set.append_jsonl(out_path)
    else:
        eval_set.save(out_path)
    console.print(
        f"[green]Wrote[/] {len(eval_set)} items to {out_path} "
        f"(source='auto' — review before trusting as ground truth)."
    )


@eval_group.command("run")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--set", "set_path", type=click.Path(exists=True, path_type=Path),
              default=Path("eval/golden.jsonl"), help="Golden set JSONL.")
@click.option("--mode", type=click.Choice(["hybrid", "vector", "keyword"]), default="hybrid")
@click.option("-k", "top_k", default=5, type=int, help="Headline k (nDCG cutoff).")
@click.option("--level", type=click.Choice(["macro", "meso", "micro"]), default=None)
@click.option("--rrf-k", default=60, type=int)
@click.option("--vector-weight", default=1.0, type=float)
@click.option("--keyword-weight", default=1.0, type=float)
@click.option("--trace", "trace_path", type=click.Path(path_type=Path), default=None,
              help="Append per-query JSONL traces here.")
@click.option("--json-out", type=click.Path(path_type=Path), default=None,
              help="Write the full report JSON here.")
def eval_run(
    db_path: Path | None,
    set_path: Path,
    mode: str,
    top_k: int,
    level: str | None,
    rrf_k: int,
    vector_weight: float,
    keyword_weight: float,
    trace_path: Path | None,
    json_out: Path | None,
) -> None:
    """Run the golden set through one search config and print metrics."""
    from aws_rag.eval.dataset import EvalSet
    from aws_rag.eval.harness import RunConfig, run_eval
    from aws_rag.store import connect

    settings = get_settings()
    conn = connect(db_path or settings.sqlite_db_path)
    eval_set = EvalSet.load(set_path)

    embedder = None
    if mode in ("vector", "hybrid"):
        from aws_rag.embedding import BedrockEmbedder

        embedder = BedrockEmbedder()

    config = RunConfig(
        mode=mode,  # type: ignore[arg-type]
        k=top_k,
        level=level,  # type: ignore[arg-type]
        rrf_k=rrf_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
    report = run_eval(conn, eval_set, config, embedder=embedder, trace_path=trace_path)
    conn.close()

    _render_report_table(report)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Report JSON →[/] {json_out}")


@eval_group.command("ablate")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--set", "set_path", type=click.Path(exists=True, path_type=Path),
              default=Path("eval/golden.jsonl"))
@click.option("-k", "top_k", default=5, type=int, help="Headline k for the comparison.")
@click.option("--trace", "trace_path", type=click.Path(path_type=Path), default=None)
@click.option("--json-out", type=click.Path(path_type=Path), default=None)
@click.option("--index-ablation", type=click.Choice(["context-vs-raw", "figure-desc"]),
              default=None, help="Heavy re-embedding ablation (incurs Bedrock cost).")
@click.option("--variant-db", type=click.Path(path_type=Path),
              default=Path("test-project/output/rag-variant.sqlite"),
              help="Where to build the variant store for an index ablation.")
@click.option("--limit", default=None, type=int, help="Cap chunks re-embedded (index ablation).")
@click.option("--verbose/--quiet", default=True)
def eval_ablate(
    db_path: Path | None,
    set_path: Path,
    top_k: int,
    trace_path: Path | None,
    json_out: Path | None,
    index_ablation: str | None,
    variant_db: Path,
    limit: int | None,
    verbose: bool,
) -> None:
    """Run the ablation matrix and print which concepts move the needle."""
    from aws_rag.embedding import BedrockEmbedder
    from aws_rag.eval.ablation import (
        build_variant_store,
        default_matrix,
        run_matrix,
    )
    from aws_rag.eval.dataset import EvalSet
    from aws_rag.eval.harness import RunConfig, run_eval
    from aws_rag.store import connect

    settings = get_settings()
    conn = connect(db_path or settings.sqlite_db_path)
    eval_set = EvalSet.load(set_path)
    embedder = BedrockEmbedder()

    if index_ablation is None:
        reports = run_matrix(
            conn, eval_set, default_matrix(base_k=top_k),
            embedder=embedder, trace_path=trace_path,
        )
        conn.close()
        _render_matrix_table(reports, headline_k=top_k)
        if json_out:
            _dump_reports_json(reports, json_out)
        return

    # Index ablation: baseline (current store) vs variant (re-embedded).
    variant = "raw_text" if index_ablation == "context-vs-raw" else "no_figure_desc"
    console.print(
        f"[yellow]Index ablation[/] '{index_ablation}': building variant store "
        f"(variant={variant}) — this re-embeds and incurs Bedrock cost."
    )
    variant_conn = build_variant_store(
        conn, variant_db, variant, embedder,  # type: ignore[arg-type]
        limit=limit, verbose=verbose,
    )

    base_cfg = RunConfig(mode="hybrid", k=top_k, label="baseline (context_text)")
    var_label = "raw text" if variant == "raw_text" else "no figure desc"
    var_cfg = RunConfig(mode="hybrid", k=top_k, label=f"variant ({var_label})")

    base_report = run_eval(conn, eval_set, base_cfg, embedder=embedder, trace_path=trace_path)
    var_report = run_eval(variant_conn, eval_set, var_cfg, embedder=embedder, trace_path=trace_path)
    conn.close()
    variant_conn.close()

    reports = [base_report, var_report]
    _render_matrix_table(reports, headline_k=top_k)
    if json_out:
        _dump_reports_json(reports, json_out)


def _dump_reports_json(reports: list, path: Path) -> None:
    import json as _json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _json.dumps([r.model_dump() for r in reports], indent=2, default=str),
        encoding="utf-8",
    )
    console.print(f"[green]Reports JSON →[/] {path}")


@eval_group.command("review")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--set", "set_path", type=click.Path(exists=True, path_type=Path),
              default=Path("eval/golden.jsonl"), help="Golden set JSONL to review.")
@click.option("--port", default=0, type=int, help="Port (0 = pick a free one).")
@click.option("-k", "top_k", default=5, type=int, help="Retrieval results to preview per item.")
@click.option("--open/--no-open", "open_browser", default=True,
              help="Open the review page in a browser.")
def eval_review(
    db_path: Path | None,
    set_path: Path,
    port: int,
    top_k: int,
    open_browser: bool,
) -> None:
    """Hand-review the golden set in a local web app (PDF page + page/chunk labels)."""
    from aws_rag.eval.review import serve

    serve(set_path, db_path=db_path, port=port, k=top_k, open_browser=open_browser)
