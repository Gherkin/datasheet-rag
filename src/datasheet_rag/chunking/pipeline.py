"""Orchestrates the full chunking pipeline.

Ties together: layout_parser → splitter → linker → context → summarizer
into a single callable pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from datasheet_rag.chunking.context import enrich_context
from datasheet_rag.chunking.layout_parser import DocumentOutline, parse_textract_blocks
from datasheet_rag.chunking.linker import link_chunks
from datasheet_rag.chunking.splitter import SplitterConfig, split_document
from datasheet_rag.chunking.summarizer import AbstractiveSummarizer, ExtractiveSummarizer
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import ChunkGraph

console = Console()


def run_chunking_pipeline_from_outline(
    outline: DocumentOutline,
    *,
    figure_manifest: dict[str, Any] | None = None,
    config: SplitterConfig | None = None,
    summarizer_mode: str = "extractive",
    llm_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
) -> ChunkGraph:
    """Run the chunking pipeline starting from a pre-built DocumentOutline.

    Accepts outlines produced by either the Textract parser or the Docling parser.
    Steps: split → link → context → summarize → re-enrich context.
    """
    console.print("[blue]Step 1:[/] Splitting into multi-scale chunks…")
    graph = split_document(outline, config=config, figure_manifest=figure_manifest)
    stats = graph.stats()
    console.print(
        f"  MACRO: {stats['by_level']['MACRO']}, "
        f"MESO: {stats['by_level']['MESO']}, "
        f"MICRO: {stats['by_level']['MICRO']}"
    )

    console.print("[blue]Step 2:[/] Linking navigation graph…")
    graph = link_chunks(graph)

    console.print("[blue]Step 3:[/] Enriching context strings…")
    graph = enrich_context(graph)

    console.print(f"[blue]Step 4:[/] Summarizing chapters ({summarizer_mode})…")
    if summarizer_mode == "abstractive":
        summarizer = AbstractiveSummarizer(
            model_id=llm_model_id,
            region=get_settings().aws_region,
        )
        graph = summarizer.summarize_graph(graph)
    else:
        summarizer_ext = ExtractiveSummarizer()
        graph = summarizer_ext.summarize_graph(graph)

    console.print("[blue]Step 5:[/] Finalizing context for macro chunks…")
    graph = enrich_context(graph)

    console.print("[green]Chunking pipeline complete.[/]")
    return graph


def run_chunking_pipeline(
    blocks: list[dict[str, Any]],
    *,
    doc_id: str,
    figure_manifest: dict[str, Any] | None = None,
    config: SplitterConfig | None = None,
    summarizer_mode: str = "extractive",
    llm_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
) -> ChunkGraph:
    """Run the full chunking pipeline from raw Textract blocks.

    Parses blocks → DocumentOutline, then delegates to
    run_chunking_pipeline_from_outline for the generic steps.
    """
    console.print("[blue]Parsing Textract layout…[/]")
    outline = parse_textract_blocks(blocks, doc_id=doc_id)
    summary = outline.summary()
    console.print(
        f"  {summary['top_level_sections']} chapters, "
        f"{summary['total_sections']} sections, "
        f"{summary['total_elements']} elements"
    )
    return run_chunking_pipeline_from_outline(
        outline,
        figure_manifest=figure_manifest,
        config=config,
        summarizer_mode=summarizer_mode,
        llm_model_id=llm_model_id,
    )


def save_chunk_graph(graph: ChunkGraph, path: Path) -> Path:
    """Save a ChunkGraph to JSON for inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize without embeddings (they'd make the file huge)
    data = {
        "doc_id": graph.doc_id,
        "stats": graph.stats(),
        "chunks": {},
    }
    for chunk_id, chunk in graph.chunks.items():
        chunk_dict = chunk.model_dump(
            exclude={"content_embedding", "context_embedding"}
        )
        data["chunks"][chunk_id] = chunk_dict

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    console.print(f"[green]Chunk graph saved[/] → {path}")
    return path


def load_chunk_graph(path: Path) -> ChunkGraph:
    """Load a ChunkGraph from JSON."""
    from datasheet_rag.models.chunk import Chunk

    with open(path) as f:
        data = json.load(f)

    graph = ChunkGraph(doc_id=data["doc_id"])
    for chunk_id, chunk_dict in data["chunks"].items():
        chunk = Chunk(**chunk_dict)
        graph.add(chunk)

    return graph
