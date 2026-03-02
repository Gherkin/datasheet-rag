"""Data models for the hierarchical chunking system."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ChunkLevel(int, Enum):
    """Zoom level for multi-scale chunking."""

    MACRO = 0    # Chapter / full-section summaries — ~2000 tokens
    MESO = 1     # Subsection chunks — ~512 tokens
    MICRO = 2    # Paragraph / table / figure level — ~128 tokens


class LayoutType(str, Enum):
    """Type of layout element the chunk originated from."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    KEY_VALUE = "key_value"
    HEADER = "header"
    LIST = "list"
    MIXED = "mixed"


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk for context-aware embedding."""

    doc_id: str
    doc_title: str = ""
    chapter_title: str = ""
    section_title: str = ""
    page_numbers: list[int] = Field(default_factory=list)
    layout_type: LayoutType = LayoutType.TEXT

    # Contextual summary prepended during embedding
    # e.g. "Chapter: Power Supply Specifications > Section: Thermal Characteristics"
    context_string: str = ""

    # Arbitrary key-value pairs from FORMS extraction
    form_fields: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A single chunk in the hierarchical structure.

    Chunks form a tree (parent/child for zoom) overlaid with a linked list
    (prev/next for sequential reading) and cross-links to chapter roots.
    """

    id: str = Field(description="Unique chunk ID (doc_id:level:index)")
    doc_id: str
    level: ChunkLevel

    # Content
    text: str = Field(description="Raw chunk text")
    context_text: str = Field(
        default="",
        description="Context-enriched text used for embedding (context_string + text)",
    )
    token_count: int = 0

    # Metadata
    metadata: ChunkMetadata

    # Navigation links (IDs of other chunks)
    parent_id: str | None = None        # Zoom out
    children_ids: list[str] = Field(default_factory=list)  # Zoom in
    prev_id: str | None = None          # Previous in reading order (same level)
    next_id: str | None = None          # Next in reading order (same level)
    chapter_root_id: str | None = None  # Jump to chapter start (level 0)

    # Figure / multi-modal references
    figure_image_path: str | None = None     # Local path to cropped figure image
    figure_s3_key: str | None = None         # S3 key for the figure image
    figure_caption: str | None = None        # Detected or nearby caption text
    figure_description: str | None = None    # LLM-generated description for text embedding

    # Concept links (populated in Phase 3)
    concept_ids: list[str] = Field(default_factory=list)

    # Embedding vectors (populated during embedding)
    content_embedding: list[float] | None = None
    context_embedding: list[float] | None = None


class ChunkGraph(BaseModel):
    """The full chunk graph for a document — serializable for debugging."""

    doc_id: str
    chunks: dict[str, Chunk] = Field(default_factory=dict, description="chunk_id → Chunk")

    def add(self, chunk: Chunk) -> None:
        self.chunks[chunk.id] = chunk

    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def children_of(self, chunk_id: str) -> list[Chunk]:
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return []
        return [self.chunks[cid] for cid in chunk.children_ids if cid in self.chunks]

    def siblings_of(self, chunk_id: str) -> list[Chunk]:
        chunk = self.chunks.get(chunk_id)
        if not chunk or not chunk.parent_id:
            return []
        parent = self.chunks.get(chunk.parent_id)
        if not parent:
            return []
        return [
            self.chunks[cid]
            for cid in parent.children_ids
            if cid in self.chunks and cid != chunk_id
        ]

    def by_level(self, level: ChunkLevel) -> list[Chunk]:
        return [c for c in self.chunks.values() if c.level == level]

    def stats(self) -> dict[str, Any]:
        return {
            "total_chunks": len(self.chunks),
            "by_level": {
                level.name: len(self.by_level(level)) for level in ChunkLevel
            },
        }
