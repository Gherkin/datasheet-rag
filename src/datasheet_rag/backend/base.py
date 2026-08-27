"""The ``RagBackend`` boundary shared by the CLI, the MCP server and the
HTTP server.

Both the CLI and the MCP server consume a ``RagBackend`` instead of touching
``store/`` + the embedder directly. Two implementations exist:

* :class:`~datasheet_rag.backend.local.LocalBackend` — wraps the local sqlite store
  and an in-process embedder (today's behaviour).
* :class:`~datasheet_rag.backend.remote.RemoteBackend` — an HTTP client to the
  FastAPI server, which itself runs a ``LocalBackend``.

The method set is intentionally semantic (search/navigate/ingest), not raw
SQL, so ``RemoteBackend`` can serve every operation over JSON.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from datasheet_rag.backend.models import (
    DocSummary,
    FigureBytes,
    IngestedDoc,
    IngestResult,
    MetadataPatch,
    StatsResult,
    TitleContext,
)
from datasheet_rag.ingest_pipeline import ProgressCallback
from datasheet_rag.models.chunk import Chunk, ChunkGraph
from datasheet_rag.store import DocMetadata, SearchFilters, SearchResult, TitleSource

SearchMode = Literal["hybrid", "vector", "keyword"]

# chunk_id -> (image_bytes, extension) for figures uploaded during ingest.
FigureUploads = Mapping[str, tuple[bytes, str]]


class RagBackend(ABC):
    """Semantic store operations, served either locally or over HTTP."""

    # ---- search (the backend embeds query text) ----
    @abstractmethod
    def search(
        self,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        k: int = 10,
        filters: SearchFilters | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> list[SearchResult]:
        """Search the store, embedding ``query`` unless a vector is supplied.

        ``query_vector`` short-circuits the embedding step for the vector and
        hybrid modes — the caller has already embedded the text itself. That is
        how a client with a GPU searches a store held by a GPU-less server
        (``RAG_COMPUTE=client``, GH #43); the text is still sent, since hybrid
        keyword matching needs it.
        """

    # ---- chunk reads ----
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Chunk | None: ...

    @abstractmethod
    def get_children(self, chunk_id: str) -> list[Chunk]: ...

    @abstractmethod
    def count_chunks(self, *, doc_id: str | None = None, project_id: str | None = None) -> int: ...

    # ---- documents / titles ----
    @abstractmethod
    def list_documents(
        self,
        *,
        project_id: str | None = None,
        group_name: str | None = None,
        mpn: str | None = None,
        manufacturer: str | None = None,
    ) -> list[DocSummary]: ...

    @abstractmethod
    def get_ingested_docs(self, *, project_id: str | None = None) -> list[IngestedDoc]: ...

    @abstractmethod
    def get_doc_titles(self) -> dict[str, str]: ...

    @abstractmethod
    def set_doc_title(
        self, doc_id: str, title: str, *, source: TitleSource = "manual", force: bool = False
    ) -> int:
        """Set a document's title, recording where it came from.

        ``source`` decides whether the write lands: a title ranking below the
        provenance already on record is refused and returns 0 (see
        :mod:`datasheet_rag.store.metadata`). ``force`` overrules that.
        """

    @abstractmethod
    def resolve_doc_id(self, doc_id: str) -> str: ...

    # ---- metadata sidecar ----
    @abstractmethod
    def get_metadata(self, doc_id: str) -> DocMetadata | None: ...

    @abstractmethod
    def set_metadata(self, doc_id: str, patch: MetadataPatch) -> DocMetadata: ...

    @abstractmethod
    def list_docs(
        self,
        *,
        project_id: str | None = None,
        group_name: str | None = None,
        mpn: str | None = None,
    ) -> list[DocMetadata]: ...

    @abstractmethod
    def apply_metadata_to_chunks(self, doc_id: str) -> int: ...

    # ---- stats ----
    @abstractmethod
    def stats(self, *, project_id: str | None = None, doc_id: str | None = None) -> StatsResult: ...

    # ---- figures ----
    @abstractmethod
    def list_figure_chunks(
        self,
        *,
        doc_id: str | None = None,
        project_id: str | None = None,
        only_with_image: bool = True,
    ) -> list[Chunk]: ...

    @abstractmethod
    def get_figure_bytes(self, chunk_id: str) -> FigureBytes: ...

    @abstractmethod
    def update_figure_description(
        self, chunk_id: str, description: str, *, update_context_text: bool = True
    ) -> bool: ...

    @abstractmethod
    def describe_figures(
        self,
        *,
        doc_id: str | None = None,
        project_id: str | None = None,
        missing_only: bool = True,
        limit: int | None = None,
        model_id: str | None = None,
        dry_run: bool = False,
    ) -> tuple[dict[str, str], dict[str, int]]:
        """Generate vision-LLM descriptions for figure chunks (server-side).

        Returns ``(descriptions, stats)`` where descriptions maps chunk_id to
        the generated text and stats carries token/error counts.
        """

    # ---- titles (text LLM, server-side) ----
    @abstractmethod
    def infer_title(
        self,
        doc_id: str,
        *,
        model_id: str | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> str | None:
        """Infer and (unless dry_run) backfill a document's title.

        Returns None without writing when the stored title outranks an
        inferred one (i.e. a human set it), unless ``force``.
        """

    @abstractmethod
    def get_title_context(self, doc_id: str) -> TitleContext:
        """Return the store-side inputs for a title inference, without inferring.

        Page-1 text, the ingest-time hints and the current title provenance —
        the cheap half of :func:`~datasheet_rag.titling.infer_and_backfill_title`,
        so a client can run the expensive half itself (GH #43).
        """

    # ---- source PDF ----
    @abstractmethod
    def get_pdf_bytes(self, doc_id: str) -> bytes: ...

    # ---- ingestion (write) ----
    @abstractmethod
    def ingest_chunk_graph(
        self,
        graph: ChunkGraph,
        *,
        figures: FigureUploads | None = None,
        project_id: str | None = None,
        group_name: str | None = None,
        metadata: MetadataPatch | None = None,
        embed: bool = True,
        describe_figures: bool = False,
        infer_title: bool = False,
        title_hints: dict[str, str] | None = None,
        vectors: Mapping[str, Sequence[float]] | None = None,
        inferred_title: str | None = None,
        source_pdf: Path | None = None,
    ) -> IngestResult:
        """Store a parsed graph, embedding / describing / titling as asked.

        ``vectors`` supplies embeddings the caller already computed, which
        makes ``embed`` a no-op — the client-side compute path (GH #43) uses it
        so the store never has to load an embedding model. ``inferred_title``
        does the same for the title LLM: the caller supplies the answer and the
        store just records it, provenance checks included.

        ``source_pdf`` is the file the graph was parsed from, for a caller that
        parsed it somewhere other than where the store lives. A store that
        already holds the PDF ignores it; a remote one uploads it, so the
        document it now has chunks for is one it can also serve the source of
        (GH #43). Whether the parse ran here is not the backend's caller's
        business to encode — passing the path it parsed is.
        """

    @abstractmethod
    def ingest_pdf(
        self,
        pdf_path: Path,
        *,
        doc_id: str | None = None,
        project_id: str | None = None,
        group_name: str | None = None,
        metadata: MetadataPatch | None = None,
        backend: str = "docling",
        skip_figures: bool = False,
        upload_figures: bool = False,
        skip_describe: bool = False,
        infer_title: bool = False,
        dpi: int = 300,
        micro_tokens: int = 128,
        meso_tokens: int = 512,
        accurate_tables: bool | None = None,
        force: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IngestResult:
        """Ingest a raw PDF end-to-end: parse → figures → chunk → embed → store.

        ``LocalBackend`` runs the whole pipeline in-process; ``RemoteBackend``
        streams the PDF to the server, which runs it there — so a thin client
        needs no Docling/torch/Textract stack. ``progress`` receives
        :class:`~datasheet_rag.ingest_pipeline.ProgressEvent` s as the parse advances.
        """

    @abstractmethod
    def delete_doc(self, doc_id: str) -> int: ...

    # ---- lifecycle ----
    def close(self) -> None:  # pragma: no cover - optional override
        """Release any held resources (HTTP session, sqlite conn)."""


class FigureUnavailableError(ValueError):
    """A figure chunk exists, but its image cannot be served.

    Distinct from "unknown chunk" and from a transient read failure: the row
    is real, it is a figure, and there is simply no image behind it — a
    document ingested with figures skipped, or a store restored without its
    figures directory. Callers are expected to degrade gracefully (say so and
    move on) rather than retry, which is the whole point of typing it (GH #41).
    """


class RagServerError(RuntimeError):
    """Raised by RemoteBackend when the server returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"RAG server error {status_code}: {detail}")


class RemoteIngestError(RagServerError):
    """The server accepted the ingest, then the pipeline failed mid-stream.

    Reported as an SSE ``error`` event rather than an HTTP status, so the
    base class's "RAG server error 500" framing only buries the detail that
    matters — which is the server-side exception message itself (GH #40).
    Subclasses RagServerError so existing ``except`` clauses still catch it.
    """

    def __init__(self, detail: str):
        self.status_code = 500
        self.detail = detail
        RuntimeError.__init__(self, f"Server-side ingest failed: {detail}")
