"""Orchestrates the full chunking pipeline.

Ties together: layout_parser → splitter → linker → context → summarizer
into a single callable pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from aws_rag.chunking.context import enrich_context
from aws_rag.chunking.layout_parser import DocumentOutline, parse_textract_blocks
from aws_rag.chunking.linker import link_chunks
from aws_rag.chunking.splitter import SplitterConfig, split_document
from aws_rag.chunking.summarizer import AbstractiveSummarizer, ExtractiveSummarizer
from aws_rag.config import get_settings
from aws_rag.models.chunk import ChunkGraph

console = Console()


def run_chunking_pipeline(
    blocks: list[dict[str, Any]],
    *,
    doc_id: str,
    figure_manifest: dict[str, Any] | None = None,
    config: SplitterConfig | None = None,
    summarizer_mode: str = "extractive",  # "extractive" or "abstractive"
    llm_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
) -> ChunkGraph:
    """Run the full chunking pipeline and return a ChunkGraph.

    Steps:
    1. Parse Textract blocks → DocumentOutline
    2. Split into multi-scale chunks (MICRO, MESO, MACRO placeholder)
    3. Link chunks (prev/next, figure neighbors)
    4. Enrich context strings
    5. Summarize MACRO chunks (extractive or LLM-based)
    6. Re-enrich MACRO context after summarization

    Returns the fully wired ChunkGraph.
    """
    # Step 1: Parse layout
    console.print("[blue]Step 1:[/] Parsing layout structure…")
    outline = parse_textract_blocks(blocks, doc_id=doc_id)
    summary = outline.summary()
    console.print(
        f"  {summary['top_level_sections']} chapters, "
        f"{summary['total_sections']} sections, "
        f"{summary['total_elements']} elements"
    )

    # Step 2: Split into chunks
    console.print("[blue]Step 2:[/] Splitting into multi-scale chunks…")
    graph = split_document(outline, config=config, figure_manifest=figure_manifest)
    stats = graph.stats()
    console.print(
        f"  MACRO: {stats['by_level']['MACRO']}, "
        f"MESO: {stats['by_level']['MESO']}, "
        f"MICRO: {stats['by_level']['MICRO']}"
    )

    # Step 3: Link chunks
    console.print("[blue]Step 3:[/] Linking navigation graph…")
    graph = link_chunks(graph)

    # Step 4: Enrich context
    console.print("[blue]Step 4:[/] Enriching context strings…")
    graph = enrich_context(graph)

    # Step 5: Summarize MACRO chunks
    console.print(f"[blue]Step 5:[/] Summarizing chapters ({summarizer_mode})…")
    if summarizer_mode == "abstractive":
        summarizer = AbstractiveSummarizer(
            model_id=llm_model_id,
            region=get_settings().aws_region,
        )
        graph = summarizer.summarize_graph(graph)
    else:
        summarizer_ext = ExtractiveSummarizer()
        graph = summarizer_ext.summarize_graph(graph)

    # Step 6: Re-enrich MACRO context after summarization
    console.print("[blue]Step 6:[/] Finalizing context for macro chunks…")
    graph = enrich_context(graph)

    console.print("[green]Chunking pipeline complete.[/]")
    return graph


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
    from aws_rag.models.chunk import Chunk

    with open(path) as f:
        data = json.load(f)

    graph = ChunkGraph(doc_id=data["doc_id"])
    for chunk_id, chunk_dict in data["chunks"].items():
        chunk = Chunk(**chunk_dict)
        graph.add(chunk)

    return graph
