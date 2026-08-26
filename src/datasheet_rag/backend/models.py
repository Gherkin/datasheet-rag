"""Transport DTOs for the backend boundary.

These small pydantic models are the wire format shared between the
``RemoteBackend`` HTTP client and the FastAPI server. Where a richer model
already exists (``Chunk``, ``SearchResult``, ``DocMetadata``,
``SearchFilters``) we reuse it directly — these DTOs only cover the shapes
that don't have a home yet (figure bytes, stats rollups, ingest payloads).
"""

from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel, Field


class DocSummary(BaseModel):
    """A document listing row: sidecar metadata + a couple derived fields."""

    doc_id: str
    project_id: str | None = None
    group_name: str | None = None
    mpn: str | None = None
    manufacturer: str | None = None
    subsystem: str | None = None
    doc_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    doc_title: str | None = None
    page_count: int | None = None


class IngestedDoc(BaseModel):
    """One row of ``get_ingested_docs`` — a doc that's actually in the store."""

    doc_id: str
    doc_title: str | None = None
    chunk_count: int = 0
    page_count: int | None = None
    ingested_at: str | None = None


class StatsResult(BaseModel):
    """Chunk-count rollup for a scope.

    ``fts_missing`` is the one field that is *not* scoped: the keyword index
    covers the whole store, so its health is reported store-wide however the
    counts above were filtered. ``None`` means the backend did not report it
    (an older server), which is not the same as zero.
    """

    total_chunks: int = 0
    by_level: dict[str, int] = Field(default_factory=dict)
    project_id: str | None = None
    doc_id: str | None = None
    fts_missing: int | None = None


class FigureCitation(BaseModel):
    doc_id: str
    page: str = ""
    section: str = ""
    chapter: str = ""


class FigureBytes(BaseModel):
    """A figure's image plus the text needed to cite it.

    ``image_b64`` carries the raw image base64-encoded so the whole thing
    round-trips as JSON. Use :meth:`image_bytes` to get the decoded bytes.
    """

    chunk_id: str
    doc_id: str
    image_b64: str
    format: str = "png"
    local_path: str | None = None
    caption: str = ""
    description: str = ""
    citation: FigureCitation

    @classmethod
    def from_bytes(
        cls,
        *,
        chunk_id: str,
        doc_id: str,
        image_bytes: bytes,
        fmt: str,
        local_path: str | None,
        caption: str,
        description: str,
        citation: FigureCitation,
    ) -> FigureBytes:
        return cls(
            chunk_id=chunk_id,
            doc_id=doc_id,
            image_b64=base64.b64encode(image_bytes).decode(),
            format=fmt,
            local_path=local_path,
            caption=caption,
            description=description,
            citation=citation,
        )

    def image_bytes(self) -> bytes:
        return base64.b64decode(self.image_b64)


class MetadataPatch(BaseModel):
    """Partial metadata update — mirrors ``store.metadata.set_metadata`` kwargs.

    ``None`` means "leave the existing value alone" (matching the store's
    merge semantics). Pass an empty string / ``[]`` to clear a field.
    """

    project_id: str | None = None
    group_name: str | None = None
    mpn: str | None = None
    manufacturer: str | None = None
    subsystem: str | None = None
    doc_type: str | None = None
    tags: list[str] | None = None
    attributes: dict[str, Any] | None = None

    def kwargs(self) -> dict[str, Any]:
        """Return only the set (non-None) fields as set_metadata kwargs."""
        return self.model_dump(exclude_none=True)

    def is_empty(self) -> bool:
        return not self.kwargs()


class IngestResult(BaseModel):
    """Outcome of an ingest call."""

    doc_id: str
    inserted: int = 0
    # Stale chunks deleted because the new graph no longer carries their ids
    # (GH #44). Worth reporting: nobody asked for those rows to go.
    pruned: int = 0
    described: int = 0
    title: str | None = None
