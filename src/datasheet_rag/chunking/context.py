"""Context enrichment for chunks.

Before embedding, each chunk's `context_text` field is populated with a
context-enriched version of the text. This steers the embedding model to
place the chunk in the right semantic neighbourhood.

The context string includes:
  - Document title
  - Chapter title
  - Section title
  - Layout type hint (table, figure, etc.)
  - For figures: caption and description
  - For tables: table title and column headers

This produces two usable fields per chunk:
  - text        — raw content (shown to the user in retrieval results)
  - context_text — enriched content (used for embedding)
"""

from __future__ import annotations

from datasheet_rag.models.chunk import Chunk, ChunkGraph, LayoutType


def enrich_context(graph: ChunkGraph) -> ChunkGraph:
    """Populate context_text and context_string for all chunks in the graph.

    Modifies the graph in place and returns it.
    """
    for chunk in graph.chunks.values():
        chunk.metadata.context_string = _build_context_string(chunk, graph)
        chunk.context_text = _build_context_text(chunk, graph)
    return graph


def _build_context_string(chunk: Chunk, graph: ChunkGraph) -> str:
    """Build the context prefix for a chunk.

    This is a structured breadcrumb that encodes the chunk's position
    in the document hierarchy.
    """
    parts: list[str] = []

    meta = chunk.metadata

    if meta.doc_title:
        parts.append(f"Document: {meta.doc_title}")

    if meta.chapter_title and meta.chapter_title != meta.doc_title:
        parts.append(f"Chapter: {meta.chapter_title}")

    if meta.section_title and meta.section_title != meta.chapter_title:
        parts.append(f"Section: {meta.section_title}")

    # Add layout type hint for non-text chunks
    if meta.layout_type == LayoutType.TABLE:
        parts.append("Content type: Table")
    elif meta.layout_type == LayoutType.FIGURE:
        parts.append("Content type: Figure/Diagram")
    elif meta.layout_type == LayoutType.KEY_VALUE:
        parts.append("Content type: Specification/Key-Value")
    elif meta.layout_type == LayoutType.LIST:
        parts.append("Content type: List")

    # Add page reference
    if meta.page_numbers:
        if len(meta.page_numbers) == 1:
            parts.append(f"Page: {meta.page_numbers[0]}")
        else:
            parts.append(f"Pages: {meta.page_numbers[0]}-{meta.page_numbers[-1]}")

    return " > ".join(parts) if parts else ""


def _build_context_text(chunk: Chunk, graph: ChunkGraph) -> str:
    """Build the full context-enriched text for embedding.

    Structure:
      [context_string]
      ---
      [figure caption / table title if applicable]
      [chunk text]
    """
    parts: list[str] = []

    # Context prefix
    ctx = chunk.metadata.context_string
    if ctx:
        parts.append(ctx)
        parts.append("---")

    # Figure-specific context
    if chunk.metadata.layout_type == LayoutType.FIGURE:
        if chunk.figure_caption:
            parts.append(f"Caption: {chunk.figure_caption}")
        if chunk.figure_description:
            parts.append(f"Description: {chunk.figure_description}")
        # Include surrounding text context from neighbors
        neighbor_context = _get_neighbor_context(chunk, graph)
        if neighbor_context:
            parts.append(f"Surrounding context: {neighbor_context}")

    # Main text
    if chunk.text:
        parts.append(chunk.text)

    return "\n".join(parts)


def _get_neighbor_context(chunk: Chunk, graph: ChunkGraph, max_chars: int = 300) -> str:
    """Get text from neighboring chunks for context (especially for figures)."""
    texts: list[str] = []

    if chunk.prev_id:
        prev = graph.get(chunk.prev_id)
        if prev and prev.text:
            texts.append(prev.text[:max_chars])

    if chunk.next_id:
        nxt = graph.get(chunk.next_id)
        if nxt and nxt.text:
            texts.append(nxt.text[:max_chars])

    return " [...] ".join(texts) if texts else ""


# ---------------------------------------------------------------------------
# MACRO chunk context (for summarized chunks)
# ---------------------------------------------------------------------------


def build_macro_context(
    chunk: Chunk, graph: ChunkGraph, *, max_subsection_chars: int = 2000
) -> str:
    """Build context text for a MACRO chunk after summarization.

    Called by the summarizer after it fills in the macro chunk's text
    with a concentrated summary.

    ``max_subsection_chars`` bounds the "Subsections: ..." line — large
    documents can have thousands of MESO children, and joining every
    section title verbatim can blow past embedding-model input limits
    (e.g. Titan's 50,000-char ``inputText`` cap).
    """
    parts: list[str] = []

    ctx = _build_context_string(chunk, graph)
    if ctx:
        parts.append(ctx)
        parts.append("---")

    # List child section titles for navigation context, capped so this
    # can't dwarf the rest of the context on documents with many sections.
    children = graph.children_of(chunk.id)
    if children:
        section_titles = []
        for child in children:
            title = child.metadata.section_title
            if title and title not in section_titles:
                section_titles.append(title)
        if section_titles:
            kept: list[str] = []
            total = 0
            omitted = 0
            for title in section_titles:
                if total + len(title) + 3 > max_subsection_chars:
                    omitted += 1
                    continue
                kept.append(title)
                total += len(title) + 3
            line = "Subsections: " + " | ".join(kept)
            if omitted:
                line += f" | … and {omitted} more"
            parts.append(line)
            parts.append("---")

    # The summary text
    if chunk.text:
        parts.append(chunk.text)

    return "\n".join(parts)
