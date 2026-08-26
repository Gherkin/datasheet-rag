"""Wire up navigation links between chunks in the ChunkGraph.

After the splitter creates chunks at all levels, the linker adds:
- prev_id / next_id   — sequential reading order within the same level
- parent_id / children_ids — already set by splitter, verified here
- chapter_root_id     — already set by splitter, verified here
- Figure location links — figures reference their position in the text flow
"""

from __future__ import annotations

from datasheet_rag.models.chunk import Chunk, ChunkGraph, ChunkLevel


def link_chunks(graph: ChunkGraph) -> ChunkGraph:
    """Add sequential (prev/next) links and verify parent/child links.

    Modifies the graph in place and returns it.
    """
    _link_sequential(graph, ChunkLevel.MACRO)
    _link_sequential(graph, ChunkLevel.MESO)
    _link_sequential(graph, ChunkLevel.MICRO)
    _link_figures_to_neighbors(graph)
    _verify_links(graph)
    return graph


def _link_sequential(graph: ChunkGraph, level: ChunkLevel) -> None:
    """Link chunks at the same level in reading order via prev/next."""
    chunks = graph.by_level(level)
    # Sort by the chunk index (extracted from ID: doc_id:L{level}:{index})
    chunks.sort(key=lambda c: _chunk_sort_key(c))

    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk.prev_id = chunks[i - 1].id
        if i < len(chunks) - 1:
            chunk.next_id = chunks[i + 1].id


def _link_figures_to_neighbors(graph: ChunkGraph) -> None:
    """For figure chunks, store references to adjacent text chunks.

    This lets the ReAct agent understand where a figure sits in the
    document flow — what text comes before and after it.
    """
    micro_chunks = graph.by_level(ChunkLevel.MICRO)
    micro_chunks.sort(key=lambda c: _chunk_sort_key(c))

    for i, chunk in enumerate(micro_chunks):
        if chunk.metadata.layout_type.value != "figure":
            continue

        # Find nearest non-figure text chunk before
        for j in range(i - 1, -1, -1):
            neighbor = micro_chunks[j]
            if neighbor.metadata.layout_type.value != "figure":
                # Store as prev_id if not already set
                if not chunk.prev_id:
                    chunk.prev_id = neighbor.id
                break

        # Find nearest non-figure text chunk after
        for j in range(i + 1, len(micro_chunks)):
            neighbor = micro_chunks[j]
            if neighbor.metadata.layout_type.value != "figure":
                if not chunk.next_id:
                    chunk.next_id = neighbor.id
                break


def _verify_links(graph: ChunkGraph) -> None:
    """Verify that all links point to existing chunks. Remove broken links."""
    for chunk in graph.chunks.values():
        if chunk.parent_id and chunk.parent_id not in graph.chunks:
            chunk.parent_id = None
        if chunk.prev_id and chunk.prev_id not in graph.chunks:
            chunk.prev_id = None
        if chunk.next_id and chunk.next_id not in graph.chunks:
            chunk.next_id = None
        if chunk.chapter_root_id and chunk.chapter_root_id not in graph.chunks:
            chunk.chapter_root_id = None
        chunk.children_ids = [cid for cid in chunk.children_ids if cid in graph.chunks]


def _chunk_sort_key(chunk: Chunk) -> tuple[int, int]:
    """Extract (level, index) from a chunk ID for sorting."""
    # ID format: doc_id:L{level}:{index}
    parts = chunk.id.rsplit(":", 2)
    try:
        level = int(parts[-2].replace("L", ""))
        index = int(parts[-1])
    except (ValueError, IndexError):
        level = chunk.level.value
        index = 0
    return (level, index)
