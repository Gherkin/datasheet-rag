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
    ChunkPage,
    DocPage,
    DocSummary,
    DocumentRevisions,
    FigureBytes,
    IngestedDoc,
    IngestLink,
    IngestResult,
    MetadataPatch,
    StatsResult,
    TitleContext,
)
from datasheet_rag.ingest_pipeline import ProgressCallback
from datasheet_rag.models.chunk import Chunk, ChunkGraph
from datasheet_rag.store import DocMetadata, FieldSpec, SearchFilters, SearchResult, TitleSource
from datasheet_rag.store.fields import FieldScope, FieldShape

SearchMode = Literal["hybrid", "vector", "keyword"]

#: How many hits one search ranks. Every page of a query is cut from this
#: same ranking, so pages neither repeat nor skip hits (GH #42); the store's
#: over-fetch grows with ``k``, so ranking the top ``offset + k`` instead
#: would reorder hits between pages. Paging ends here.
SEARCH_POOL = 200

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
        offset: int = 0,
        filters: SearchFilters | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> list[SearchResult]:
        """Search the store, embedding ``query`` unless a vector is supplied.

        Returns hits ``offset`` to ``offset + k`` of the query's ranking of
        the best :data:`SEARCH_POOL` hits, so nothing past the pool.

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
    def get_children(
        self, chunk_id: str, *, limit: int | None = None, offset: int = 0
    ) -> ChunkPage:
        """One page of ``chunk_id``'s children, in document order.

        ``limit=None`` returns every child from ``offset`` on. A parent can
        have hundreds of children, so the MCP layer always passes a limit
        (GH #42).
        """

    @abstractmethod
    def count_chunks(self, *, rev_id: str | None = None, project_id: str | None = None) -> int: ...

    # ---- documents / titles ----
    @abstractmethod
    def list_documents(
        self,
        *,
        project_id: str | None = None,
        fields: dict[str, str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        all_revisions: bool = False,
    ) -> DocPage:
        """One page of the matching documents, in ``rev_id`` order.

        One row per document, showing its newest revision; ``all_revisions``
        gives one row per revision instead (GH #124).

        ``project_id`` keeps documents that belong to that project, among
        any others they belong to. ``fields`` filters on registered fields
        (AND-ed; a list field matches if any value is equal,
        case-insensitive; a project-scoped field matches in ``project_id``,
        or in any project when none is given). ``limit=None`` returns every
        match from ``offset`` on.
        """

    @abstractmethod
    def get_document(self, rev_id: str) -> DocSummary | None:
        """One document's listing row, or None if the store has no such document.

        A document exists if it has a sidecar row or ingested chunks — the same
        rule :meth:`list_documents` uses — but only this one is looked up, so
        the cost does not grow with the store (GH #86).
        """

    @abstractmethod
    def get_ingested_docs(self, *, project_id: str | None = None) -> list[IngestedDoc]: ...

    @abstractmethod
    def get_doc_titles(self) -> dict[str, str]: ...

    @abstractmethod
    def set_doc_title(
        self, rev_id: str, title: str, *, source: TitleSource = "manual", force: bool = False
    ) -> int:
        """Set a document's title, recording where it came from.

        ``source`` decides whether the write lands: a title ranking below the
        provenance already on record is refused and returns 0 (see
        :mod:`datasheet_rag.store.metadata`). ``force`` overrules that.
        """

    @abstractmethod
    def resolve_rev_id(self, rev_id: str) -> str:
        """A full revision rev_id for a prefix, or for a ``D12`` its newest revision."""

    @abstractmethod
    def resolve_document(self, ref: str) -> str:
        """The document (``"D12"``) a ref names: a ``D12`` itself, or a
        revision's rev_id (prefix ok). Raises ``ValueError`` when unknown."""

    # The two ref resolvers every caller above the store goes through (GH #124).
    def resolve_ref(self, ref: str) -> str:
        """A ref as read and written by metadata: a ``D12`` stays a document
        (after checking it exists), a rev_id prefix becomes the full rev_id.
        Resolving first means a prefix or an unknown id never creates a
        revision row. Raises ``ValueError`` when unknown."""
        from datasheet_rag.store.documents import parse_document_ref

        if parse_document_ref(ref) is not None:
            return self.resolve_document(ref)
        return self.resolve_rev_id(ref)

    def rev_ids_for(self, ref: str, *, all_revisions: bool) -> list[str]:
        """The revisions a ref covers: for a ``D12`` all of them (newest
        first) or only the newest; for a rev_id (prefix ok) that one."""
        from datasheet_rag.store.documents import parse_document_ref

        if all_revisions and parse_document_ref(ref) is not None:
            return [r.rev_id for r in self.list_revisions(ref).revisions]
        return [self.resolve_rev_id(ref)]

    # ---- metadata sidecar ----
    @abstractmethod
    def get_metadata(self, rev_id: str) -> DocMetadata | None:
        """A revision's metadata (full rev_id), or a document's (``D12``)."""

    @abstractmethod
    def set_metadata(self, rev_id: str, patch: MetadataPatch) -> DocMetadata:
        """Write a patch on the document of ``rev_id`` (a full rev_id or a
        ``D12``), or on that one revision with ``patch.this_revision``."""

    @abstractmethod
    def list_docs(
        self,
        *,
        project_id: str | None = None,
        fields: dict[str, str] | None = None,
        all_revisions: bool = False,
    ) -> list[DocMetadata]: ...

    # ---- revisions (GH #124) ----
    @abstractmethod
    def list_revisions(self, ref: str) -> DocumentRevisions:
        """A document's revisions, newest first. ``ref`` is a ``D12`` or a
        revision's full rev_id."""

    @abstractmethod
    def add_revision(self, document: str, rev_id: str) -> DocumentRevisions:
        """Make stored PDF ``rev_id`` the newest revision of ``document``.

        A PDF alone in its own document is merged in (projects and tags
        joined; a field with different values on both is refused). A PDF
        that is one of several revisions elsewhere is refused.
        """

    @abstractmethod
    def move_revision(
        self, rev_id: str, *, above: str | None = None, below: str | None = None
    ) -> DocumentRevisions:
        """Move a revision directly above (newer than) or below a sibling."""

    @abstractmethod
    def unlink_revision(self, rev_id: str) -> DocumentRevisions:
        """Take a revision out into a new document (with a copy of the old
        document's metadata) and return the new document."""

    @abstractmethod
    def get_identity_rule(self) -> str | None:
        """The store's identity rule, or None when it has none."""

    @abstractmethod
    def set_identity_rule(self, rule: str | None) -> str | None:
        """Set (or with None clear) the identity rule; returns it as stored."""

    @abstractmethod
    def plan_ingest_link(
        self,
        *,
        rev_id: str | None = None,
        metadata: MetadataPatch | None = None,
        new_revision_of: str | None = None,
    ) -> IngestLink:
        """Which document an ingest would put a PDF in, without writing.

        Lets a client fail before parsing: an ambiguous rule match or an
        unknown ``new_revision_of`` raises ``ValueError`` here exactly as the
        ingest itself would. ``rev_id`` is the PDF's id when known.
        """

    @abstractmethod
    def set_needs_reembed(self, rev_id: str, value: bool) -> None:
        """Mark (or unmark) a document as needing its chunks re-embedded.

        Pipeline bookkeeping, kept apart from :meth:`set_metadata`: a
        metadata patch cannot set it.
        """

    # ---- metadata fields (GH #120) ----
    @abstractmethod
    def list_fields(self) -> list[FieldSpec]:
        """Every registered metadata field, built-ins first."""

    @abstractmethod
    def register_field(
        self,
        name: str,
        *,
        shape: FieldShape = "single",
        scope: FieldScope = "document",
        description: str = "",
        roles: Sequence[str] = (),
    ) -> FieldSpec:
        """Register a field and return it."""

    @abstractmethod
    def remove_field(self, name: str, *, clear: bool = False) -> int:
        """Unregister a field; returns how many documents held a value.

        Refuses while documents hold values, unless ``clear`` deletes them.
        """

    # ---- stats ----
    @abstractmethod
    def stats(self, *, project_id: str | None = None, rev_id: str | None = None) -> StatsResult: ...

    # ---- figures ----
    @abstractmethod
    def list_figure_chunks(
        self,
        *,
        rev_id: str | None = None,
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
        rev_id: str | None = None,
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
        rev_id: str,
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
    def get_title_context(self, rev_id: str) -> TitleContext:
        """Return the store-side inputs for a title inference, without inferring.

        Page-1 text, the ingest-time hints and the current title provenance —
        the cheap half of :func:`~datasheet_rag.titling.infer_and_backfill_title`,
        so a client can run the expensive half itself (GH #43).
        """

    # ---- source PDF ----
    @abstractmethod
    def get_pdf_bytes(self, rev_id: str) -> bytes: ...

    # ---- ingestion (write) ----
    @abstractmethod
    def ingest_chunk_graph(
        self,
        graph: ChunkGraph,
        *,
        figures: FigureUploads | None = None,
        project_id: str | None = None,
        metadata: MetadataPatch | None = None,
        embed: bool = True,
        describe_figures: bool = False,
        infer_title: bool = False,
        title_hints: dict[str, str] | None = None,
        vectors: Mapping[str, Sequence[float]] | None = None,
        inferred_title: str | None = None,
        source_pdf: Path | None = None,
        new_revision_of: str | None = None,
    ) -> IngestResult:
        """Store a parsed graph, embedding / describing / titling as asked.

        A new PDF becomes the newest revision of ``new_revision_of`` (a
        ``D12`` or a revision's rev_id) when given, else of the one document
        the identity rule matches, else a new document (GH #124). The
        metadata patch is written on that document. ``IngestResult.link``
        says which happened.

        ``project_id`` adds the document to that project; projects it is
        already in are kept (GH #121). Leaving a project is a metadata
        patch's ``remove_projects``. Project-scoped fields in ``metadata``
        are written in ``project_id`` unless the patch names its own
        ``project`` (GH #123).

        ``vectors`` supplies embeddings the caller already computed, which
        makes ``embed`` a no-op — the client-side compute path (GH #43) uses it
        so the store never has to load an embedding model. ``inferred_title``
        does the same for the title LLM: the caller supplies the answer and the
        store just records it, provenance checks included.

        ``title_hints`` are the hints the parse found. A dict, even an empty
        one, replaces the stored hints whole. ``None`` means no parse computed
        them (a cached chunk graph saved without hints, a Textract parse,
        `rag repair embeddings`), and the stored hints are kept.

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
        rev_id: str | None = None,
        project_id: str | None = None,
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
        new_revision_of: str | None = None,
    ) -> IngestResult:
        """Ingest a raw PDF end-to-end: parse → figures → chunk → embed → store.

        ``LocalBackend`` runs the whole pipeline in-process; ``RemoteBackend``
        streams the PDF to the server, which runs it there — so a thin client
        needs no Docling/torch/Textract stack. ``progress`` receives
        :class:`~datasheet_rag.ingest_pipeline.ProgressEvent` s as the parse advances.
        """

    @abstractmethod
    def delete_doc(self, rev_id: str) -> int:
        """Delete one revision; its document goes with its last revision."""

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


class FigureNotFoundError(ValueError):
    """No figure chunk by that id: the chunk is unknown, or not a figure.

    Typed so callers can tell this expected answer apart from other
    ``ValueError``s — JSON, pydantic and base64 decode errors all subclass it
    and are real faults, not "no image here" (GH #70).
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
