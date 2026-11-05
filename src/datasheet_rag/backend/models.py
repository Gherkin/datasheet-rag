"""Transport DTOs for the backend boundary.

These small pydantic models are the wire format shared between the
``RemoteBackend`` HTTP client and the FastAPI server. Where a richer model
already exists (``Chunk``, ``SearchResult``, ``DocMetadata``,
``SearchFilters``) we reuse it directly — these DTOs only cover the shapes
that don't have a home yet (figure bytes, stats rollups, ingest payloads).
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datasheet_rag.models.chunk import Chunk
from datasheet_rag.store.projects import check_project_change


class DocSummary(BaseModel):
    """A document listing row: sidecar metadata + a couple derived fields."""

    doc_id: str
    #: Every project the document belongs to, sorted.
    projects: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    #: Document-scoped field values, e.g. ``{"mpn": ["INA226"], "acme-id": "D-17"}``.
    fields: dict[str, str | list[str]] = Field(default_factory=dict)
    #: Project-scoped field values per project, e.g.
    #: ``{"netdaq": {"subsystem": "power"}}``.
    project_fields: dict[str, dict[str, str | list[str]]] = Field(default_factory=dict)
    doc_title: str | None = None
    page_count: int | None = None


class DocPage(BaseModel):
    """One page of ``list_documents``, in ``doc_id`` order.

    ``total`` counts every document the filters match, not just this page's,
    so a caller can tell a short page from the end of the list (GH #42).
    """

    documents: list[DocSummary] = Field(default_factory=list)
    total: int = 0


class ChunkPage(BaseModel):
    """One page of a chunk's children, in document order.

    ``total`` counts every child, not just this page's (GH #42).
    """

    chunks: list[Chunk] = Field(default_factory=list)
    total: int = 0


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
    # Neighbouring chunk text, shipped alongside the pixels so a client that
    # runs the vision model itself (RAG_COMPUTE=client) gets the whole prompt
    # in one round trip. Empty from a server too old to send it.
    surrounding_text: str = ""

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
        surrounding_text: str = "",
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
            surrounding_text=surrounding_text,
        )

    def image_bytes(self) -> bytes:
        return base64.b64decode(self.image_b64)


class MetadataPatch(BaseModel):
    """Partial metadata update — mirrors ``store.metadata.set_metadata`` kwargs.

    ``None`` means "leave the existing value alone" (matching the store's
    merge semantics). Pass an empty string / ``[]`` to clear a field.
    ``add_projects`` / ``remove_projects`` join / leave the named projects
    and leave the document's other projects alone; a project in both, or an
    empty name, is rejected here.
    In ``fields`` each named field is replaced; a ``None`` value removes it.
    ``project`` names the project the project-scoped fields in ``fields``
    (``subsystem``) are written in; it is required when there are any.
    The pipeline's bookkeeping (title provenance, title hints, the re-embed
    marker) is not part of a patch. Unknown names are rejected, not ignored,
    so a value sent under a name the patch does not have is never dropped.
    """

    model_config = ConfigDict(extra="forbid")

    add_projects: list[str] | None = None
    remove_projects: list[str] | None = None
    tags: list[str] | None = None
    fields: dict[str, str | list[str] | None] | None = None
    project: str | None = None

    @model_validator(mode="after")
    def _check_projects(self) -> MetadataPatch:
        add, remove = check_project_change(self.add_projects, self.remove_projects)
        self.add_projects = add or None
        self.remove_projects = remove or None
        return self

    def add_project(self, project_id: str | None) -> None:
        """Add ``project_id`` to ``add_projects`` (no-op for ``None``/blank)."""
        if not project_id or not project_id.strip():
            return
        add, remove = check_project_change(
            [*(self.add_projects or []), project_id], self.remove_projects
        )
        self.add_projects = add or None
        self.remove_projects = remove or None

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


class TitleContext(BaseModel):
    """Everything a title inference needs from the store, minus the model call.

    Split out so the LLM can run on a client while the store stays remote
    (``RAG_COMPUTE=client``, GH #43): reading page-1 text and the provenance is
    cheap SQL that belongs wherever the database is, and only the inference
    itself is worth moving.
    """

    doc_id: str
    first_page_text: str = ""
    title_source: str = ""
    running_header: str | None = None
    pdf_meta_title: str | None = None


class ChunkVectors(BaseModel):
    """Embedding vectors for a chunk graph, in the store's own wire format.

    Vectors are float32 in sqlite, and a datasheet's worth of them is tens of
    megabytes as JSON numbers. Packing them as one base64 float32 buffer keeps
    a client-side embed (``RAG_COMPUTE=client``) roughly a quarter of that
    size, and lands them in exactly the layout ``vec0`` expects.
    """

    dim: int
    ids: list[str]
    data: str  # base64 of len(ids) * dim little-endian float32 values

    @classmethod
    def from_mapping(cls, vectors: Mapping[str, Sequence[float]]) -> ChunkVectors:
        import numpy as np

        ids = list(vectors)
        if not ids:
            return cls(dim=0, ids=[], data="")
        matrix = np.asarray([vectors[i] for i in ids], dtype=np.float32)
        return cls(
            dim=int(matrix.shape[1]),
            ids=ids,
            data=base64.b64encode(matrix.tobytes()).decode(),
        )

    def to_mapping(self) -> dict[str, list[float]]:
        import numpy as np

        if not self.ids:
            return {}
        flat = np.frombuffer(base64.b64decode(self.data), dtype=np.float32)
        expected = len(self.ids) * self.dim
        if flat.size != expected:
            raise ValueError(
                f"vector payload is {flat.size} float32 values, expected "
                f"{expected} ({len(self.ids)} chunks x {self.dim} dimensions)"
            )
        matrix = flat.reshape(len(self.ids), self.dim)
        return {cid: matrix[i].tolist() for i, cid in enumerate(self.ids)}
