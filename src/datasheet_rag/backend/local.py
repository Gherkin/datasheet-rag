"""Local backend — wraps the sqlite store and an in-process embedder.

This is the historical code path (open ``rag.sqlite`` directly, build an
embedder from settings) packaged behind the :class:`RagBackend` interface.
The FastAPI server also runs one of these per process to serve remote
clients.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from datasheet_rag.backend.base import (
    SEARCH_POOL,
    FigureNotFoundError,
    FigureUnavailableError,
    FigureUploads,
    RagBackend,
    SearchMode,
)
from datasheet_rag.backend.models import (
    ChunkPage,
    DocPage,
    DocSummary,
    DocumentRevisions,
    FigureBytes,
    FigureCitation,
    IngestedDoc,
    IngestLink,
    IngestResult,
    MetadataPatch,
    RevisionSummary,
    StatsResult,
    TitleContext,
)
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import IMAGE_LAYOUT_TYPES, Chunk, ChunkGraph, ChunkLevel
from datasheet_rag.store import (
    DocMetadata,
    FieldSpec,
    SearchFilters,
    SearchResult,
    TitleSource,
    connect,
    count_chunks,
    delete_doc,
    delete_metadata,
    figure_source_available,
    fts_status,
    get_chunk,
    get_doc_titles,
    get_ingested_docs,
    get_metadata,
    hybrid_search,
    insert_chunk_graph,
    keyword_search,
    list_docs,
    list_fields,
    list_figure_chunks,
    register_field,
    remove_field,
    resolve_figure_path,
    resolve_rev_id,
    set_doc_title,
    set_metadata,
    stored_embedding_dim,
    to_relative_figure_path,
    update_figure_description,
    vector_search,
)
from datasheet_rag.store import documents as revs
from datasheet_rag.store.fields import (
    FieldScope,
    FieldShape,
    check_filter_fields,
    field_scopes,
    normalize_field_patch,
)
from datasheet_rag.store.identity import get_identity_rule, match_documents, set_identity_rule
from datasheet_rag.store.metadata import (
    check_write,
    join_projects,
    set_needs_reembed,
    set_title_hints,
)
from datasheet_rag.store.projects import project_scope_sql

if TYPE_CHECKING:
    from datasheet_rag.ingest_pipeline import ProgressCallback

logger = logging.getLogger("datasheet_rag.backend")


class LocalBackend(RagBackend):
    """Direct sqlite + embedder backend."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        conn: sqlite3.Connection | None = None,
        embedder: Any | None = None,
    ):
        settings = get_settings()
        self._db_path = str(db_path or settings.sqlite_db_path)
        # `conn` / `embedder` allow injecting an already-open connection or a
        # stub embedder (used by tests and by the MCP impls' legacy kwargs).
        self._conn: sqlite3.Connection | None = conn
        self._conn_lock = Lock()
        self._embedder: Any | None = embedder
        self._embedder_lock = Lock()
        # Serialise writes — one shared connection, WAL handles readers.
        self._write_lock = Lock()

    # -- lazy resources ------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        with self._conn_lock:
            if self._conn is None:
                self._conn = connect(self._db_path)
            return self._conn

    def _get_embedder(self) -> Any:
        with self._embedder_lock:
            if self._embedder is None:
                from datasheet_rag.embedding import get_embedder

                self._embedder = get_embedder()
            return self._embedder

    @property
    def conn(self) -> sqlite3.Connection:
        """The underlying sqlite connection (lazily opened).

        Exposed so the server's control plane (API keys + audit log) can read
        and write the auth/audit tables on the same connection.
        """
        return self._get_conn()

    @property
    def write_lock(self) -> Lock:
        """The write serialisation lock, shared with control-plane writes."""
        return self._write_lock

    def close(self) -> None:
        with self._conn_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- search --------------------------------------------------------
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
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if filters is not None and filters.fields:
            check_filter_fields((f.name for f in self.list_fields()), filters.fields)
        ranked = self._rank(query, mode=mode, filters=filters, query_vector=query_vector)
        return ranked[offset : offset + k]

    def _rank(
        self,
        query: str,
        *,
        mode: SearchMode,
        filters: SearchFilters | None,
        query_vector: Sequence[float] | None,
    ) -> list[SearchResult]:
        """The query's ranking of the best ``SEARCH_POOL`` hits."""
        k = SEARCH_POOL
        conn = self._get_conn()
        if mode in ("vector", "hybrid"):
            # A caller-supplied vector means the embedding already happened
            # elsewhere (a client with the GPU — GH #43). Check its width
            # here: a wrong-sized vector is a silently wrong result set
            # otherwise, and the fix is a config change, not a retry.
            if query_vector is not None:
                expected = stored_embedding_dim(conn)
                if expected is not None and len(query_vector) != expected:
                    raise ValueError(
                        f"query_vector has {len(query_vector)} dimensions, but this "
                        f"store's vectors are {expected}-dimensional — the client and "
                        f"server are configured with different embedding models."
                    )
                query_vec = list(query_vector)
            else:
                query_vec = self._get_embedder().embed_one(query)
        if mode == "vector":
            return vector_search(conn, query_vec, k=k, filters=filters)
        if mode == "keyword":
            return keyword_search(conn, query, k=k, filters=filters)
        return hybrid_search(conn, query_vec, query, k=k, filters=filters)

    # -- chunk reads ---------------------------------------------------
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return get_chunk(self._get_conn(), chunk_id)

    def get_children(
        self, chunk_id: str, *, limit: int | None = None, offset: int = 0
    ) -> ChunkPage:
        conn = self._get_conn()
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE parent_id = ?", (chunk_id,)
        ).fetchone()["n"]
        # SQLite reads a negative LIMIT as "no limit".
        rows = conn.execute(
            "SELECT id FROM chunks WHERE parent_id = ? ORDER BY rowid LIMIT ? OFFSET ?",
            (chunk_id, -1 if limit is None else limit, offset),
        ).fetchall()
        out: list[Chunk] = []
        for row in rows:
            child = get_chunk(conn, row["id"])
            if child is not None:
                out.append(child)
        return ChunkPage(chunks=out, total=int(total))

    def count_chunks(self, *, rev_id: str | None = None, project_id: str | None = None) -> int:
        return count_chunks(self._get_conn(), rev_id=rev_id, project_id=project_id)

    # -- documents / titles -------------------------------------------
    def _derived_doc_fields(self, rev_id: str) -> tuple[str | None, int | None]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT doc_title FROM chunks WHERE rev_id = ? AND doc_title != '' LIMIT 1",
            (rev_id,),
        ).fetchone()
        title = row["doc_title"] if row else None
        page_row = conn.execute(
            """
            SELECT page_numbers FROM chunks
             WHERE rev_id = ? AND page_numbers IS NOT NULL AND page_numbers != '[]'
             ORDER BY rowid DESC LIMIT 1
            """,
            (rev_id,),
        ).fetchone()
        page_count: int | None = None
        if page_row:
            try:
                pages = json.loads(page_row["page_numbers"])
                if pages:
                    page_count = max(pages)
            except (ValueError, TypeError):
                pass
        return title, page_count

    def list_documents(
        self,
        *,
        project_id: str | None = None,
        fields: dict[str, str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        all_revisions: bool = False,
    ) -> DocPage:
        docs = list_docs(
            self._get_conn(), project_id=project_id, fields=fields, all_revisions=all_revisions
        )
        matched: dict[str, DocMetadata | None] = {d.rev_id: d for d in docs}
        # insert_chunks gives every stored PDF a revision row (GH #124), but a
        # store can still hold chunks without one (a sidecar deleted on its
        # own). Such a PDF is chunked and searchable, so list it too, with its
        # metadata left empty, rather than hide it. A field filter suppresses
        # the fallback: a PDF without metadata cannot match one.
        if not fields:
            known = revs.documents_of(self._get_conn(), None)
            for row in get_ingested_docs(self._get_conn(), project_id=project_id):
                if row["rev_id"] not in known:
                    matched[row["rev_id"]] = None
        # Page before deriving title and page count: those cost two queries
        # per document, and a page should not pay for the whole store.
        ordered = sorted(matched)
        end = None if limit is None else offset + limit
        out = [self._doc_summary(rev_id, matched[rev_id]) for rev_id in ordered[offset:end]]
        return DocPage(documents=out, total=len(ordered))

    def get_document(self, rev_id: str) -> DocSummary | None:
        meta = get_metadata(self._get_conn(), rev_id)
        if meta is None:
            row = (
                self._get_conn()
                .execute("SELECT 1 FROM chunks WHERE rev_id = ? LIMIT 1", (rev_id,))
                .fetchone()
            )
            if row is None:
                return None
            return self._doc_summary(rev_id, None)
        return self._doc_summary(meta.rev_id, meta)

    def _doc_summary(self, rev_id: str, d: DocMetadata | None) -> DocSummary:
        title, page_count = self._derived_doc_fields(rev_id)
        if d is None:
            return DocSummary(rev_id=rev_id, doc_title=title, page_count=page_count)
        return DocSummary(
            rev_id=d.rev_id,
            document_id=d.document_id,
            revision=d.revision,
            revisions=d.revisions,
            overrides=d.overrides,
            projects=d.projects,
            tags=d.tags,
            fields=d.fields,
            project_fields=d.project_fields,
            doc_title=title,
            page_count=page_count,
        )

    def get_ingested_docs(self, *, project_id: str | None = None) -> list[IngestedDoc]:
        rows = get_ingested_docs(self._get_conn(), project_id=project_id)
        return [IngestedDoc(**r) for r in rows]

    def get_doc_titles(self) -> dict[str, str]:
        return get_doc_titles(self._get_conn())

    def set_doc_title(
        self, rev_id: str, title: str, *, source: TitleSource = "manual", force: bool = False
    ) -> int:
        with self._write_lock:
            conn = self._get_conn()
            n = set_doc_title(conn, rev_id, title, source=source, force=force)
            conn.commit()
            return n

    def resolve_rev_id(self, rev_id: str) -> str:
        return resolve_rev_id(self._get_conn(), rev_id)

    def resolve_document(self, ref: str) -> str:
        return revs.format_document_id(self._document_id(ref))

    def _document_id(self, ref: str) -> int:
        """The document a ``D12`` or a revision rev_id (prefix ok) names."""
        conn = self._get_conn()
        if revs.parse_document_ref(ref) is None:
            ref = resolve_rev_id(conn, ref)
        found = revs.lookup(conn, ref).document_id
        if found is None:
            raise revs.RevisionError(f"{ref[:12]} is not a known revision")
        return found

    # -- metadata ------------------------------------------------------
    def get_metadata(self, rev_id: str) -> DocMetadata | None:
        return get_metadata(self._get_conn(), rev_id)

    def set_metadata(self, rev_id: str, patch: MetadataPatch) -> DocMetadata:
        with self._write_lock:
            return set_metadata(self._get_conn(), rev_id, **patch.kwargs())

    def list_docs(
        self,
        *,
        project_id: str | None = None,
        fields: dict[str, str] | None = None,
        all_revisions: bool = False,
    ) -> list[DocMetadata]:
        return list_docs(
            self._get_conn(), project_id=project_id, fields=fields, all_revisions=all_revisions
        )

    # -- revisions -----------------------------------------------------
    def _revisions(self, document_id: int) -> DocumentRevisions:
        conn = self._get_conn()
        ids = revs.revisions_of(conn, document_id)
        if not ids:
            return DocumentRevisions(document_id=revs.format_document_id(document_id))
        metas = {m.rev_id: m for m in (get_metadata(conn, d) for d in ids) if m is not None}
        counts = {
            r[0]: (int(r[1]), r[2], r[3])
            for r in conn.execute(
                "SELECT rev_id, COUNT(*), MIN(created_at), MAX(doc_title) FROM chunks "
                f"WHERE rev_id IN ({','.join('?' * len(ids))}) GROUP BY rev_id",
                ids,
            )
        }
        out = []
        for i, d in enumerate(ids):
            m = metas.get(d)
            n, at, title = counts.get(d, (0, None, None))
            out.append(
                RevisionSummary(
                    rev_id=d,
                    revision=m.revision if m else None,
                    newest=i == 0,
                    doc_title=title,
                    chunk_count=n,
                    ingested_at=at,
                    overrides=m.overrides if m else [],
                )
            )
        return DocumentRevisions(document_id=revs.format_document_id(document_id), revisions=out)

    def list_revisions(self, ref: str) -> DocumentRevisions:
        return self._revisions(self._document_id(ref))

    def add_revision(self, document: str, rev_id: str) -> DocumentRevisions:
        with self._write_lock:
            target = self._document_id(document)
            full = resolve_rev_id(self._get_conn(), rev_id)
            revs.add_revision(self._get_conn(), target, full)
            return self._revisions(target)

    def move_revision(
        self, rev_id: str, *, above: str | None = None, below: str | None = None
    ) -> DocumentRevisions:
        with self._write_lock:
            conn = self._get_conn()
            document_id = revs.move_revision(
                conn,
                resolve_rev_id(conn, rev_id),
                above=resolve_rev_id(conn, above) if above else None,
                below=resolve_rev_id(conn, below) if below else None,
            )
            return self._revisions(document_id)

    def unlink_revision(self, rev_id: str) -> DocumentRevisions:
        with self._write_lock:
            conn = self._get_conn()
            return self._revisions(revs.unlink_revision(conn, resolve_rev_id(conn, rev_id)))

    def get_identity_rule(self) -> str | None:
        rule = get_identity_rule(self._get_conn())
        return str(rule) if rule is not None else None

    def set_identity_rule(self, rule: str | None) -> str | None:
        with self._write_lock:
            stored = set_identity_rule(self._get_conn(), rule)
        return str(stored) if stored is not None else None

    def plan_ingest_link(
        self,
        *,
        rev_id: str | None = None,
        metadata: MetadataPatch | None = None,
        new_revision_of: str | None = None,
    ) -> IngestLink:
        return self._plan_link(rev_id, metadata, new_revision_of)

    def _plan_link(
        self,
        rev_id: str | None,
        metadata: MetadataPatch | None,
        new_revision_of: str | None,
    ) -> IngestLink:
        """Decide the document a PDF goes into (see :meth:`ingest_chunk_graph`)."""
        conn = self._get_conn()
        existing = revs.document_of(conn, rev_id) if rev_id else None
        if existing is not None:
            ref = revs.format_document_id(existing)
            wanted = self._document_id(new_revision_of) if new_revision_of is not None else None
            if wanted is not None and wanted != existing:
                short = (rev_id or "")[:12]
                add = f"rag revisions add {revs.format_document_id(wanted)} {short}"
                # add refuses a revision that has siblings: unlink it first.
                if len(revs.revisions_of(conn, existing)) > 1:
                    add = f"rag revisions unlink {short}, then {add}"
                raise revs.RevisionError(
                    f"{short} is already stored, as a revision of {ref}; move it with {add}"
                )
            return IngestLink(action="kept", document_id=ref)
        if new_revision_of is not None:
            target = revs.format_document_id(self._document_id(new_revision_of))
            return IngestLink(action="explicit", document_id=target)
        rule = get_identity_rule(conn)
        if rule is None or metadata is None or not metadata.fields:
            return IngestLink(action="new")
        scopes = field_scopes(conn)
        values = {
            name: v
            for name, v in normalize_field_patch(conn, metadata.fields).items()
            if v and scopes.get(name) == "document"
        }
        matches = match_documents(conn, rule, values)
        if len(matches) > 1:
            raise revs.RevisionError(
                f"the identity rule ({rule}) matches "
                + " and ".join(
                    f"{revs.format_document_id(m.document_id)} ({', '.join(m.fields)})"
                    for m in matches
                )
                + "; nothing was ingested. Name the document with --new-revision-of, "
                "or make the field values tell them apart."
            )
        if matches:
            return IngestLink(
                action="rule",
                document_id=revs.format_document_id(matches[0].document_id),
                fields=matches[0].fields,
            )
        return IngestLink(action="new")

    def set_needs_reembed(self, rev_id: str, value: bool) -> None:
        with self._write_lock:
            set_needs_reembed(self._get_conn(), rev_id, value)

    # -- metadata fields -----------------------------------------------
    def list_fields(self) -> list[FieldSpec]:
        return list_fields(self._get_conn())

    def register_field(
        self,
        name: str,
        *,
        shape: FieldShape = "single",
        scope: FieldScope = "document",
        description: str = "",
        roles: Sequence[str] = (),
    ) -> FieldSpec:
        with self._write_lock:
            return register_field(
                self._get_conn(),
                name,
                shape=shape,
                scope=scope,
                description=description,
                roles=roles,
            )

    def remove_field(self, name: str, *, clear: bool = False) -> int:
        with self._write_lock:
            return remove_field(self._get_conn(), name, clear=clear)

    # -- stats ---------------------------------------------------------
    def stats(self, *, project_id: str | None = None, rev_id: str | None = None) -> StatsResult:
        conn = self._get_conn()
        total = count_chunks(conn, rev_id=rev_id, project_id=project_id)
        where: list[str] = []
        params: list[Any] = []
        if rev_id:
            where.append("rev_id = ?")
            params.append(rev_id)
        if project_id:
            clause, clause_params = project_scope_sql(project_id, rev_id_column="rev_id")
            where.append(clause)
            params.extend(clause_params)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        by_level: dict[str, int] = {}
        for row in conn.execute(
            f"SELECT level, COUNT(*) AS c FROM chunks{where_sql} GROUP BY level",
            params,
        ).fetchall():
            try:
                name = ChunkLevel(int(row["level"])).name
            except ValueError:
                name = str(row["level"])
            by_level[name] = int(row["c"])
        # Store-wide, deliberately unscoped: a desynced keyword index breaks
        # search for every project, and this rollup is where someone looks
        # when results seem off (GH #23).
        return StatsResult(
            total_chunks=total,
            by_level=by_level,
            project_id=project_id,
            rev_id=rev_id,
            fts_missing=fts_status(conn).missing,
        )

    # -- figures -------------------------------------------------------
    def list_figure_chunks(
        self,
        *,
        rev_id: str | None = None,
        project_id: str | None = None,
        only_with_image: bool = True,
    ) -> list[Chunk]:
        chunks = list_figure_chunks(
            self._get_conn(),
            rev_id=rev_id,
            project_id=project_id,
            only_with_image=only_with_image,
        )
        if only_with_image:
            # The SQL filter only proves the column is set; hydration checked
            # whether the file is really there. Honour the stricter answer so
            # callers that intend to *read* the bytes (describe, MCP) never
            # get a chunk that cannot be served.
            chunks = [c for c in chunks if c.figure_available]
        return chunks

    def _read_figure_image(self, chunk: Chunk) -> tuple[bytes, str, Path | None]:
        """Read a figure's bytes from disk (figure_image_path) or S3."""
        path = resolve_figure_path(chunk.figure_image_path)
        if path is not None:
            if path.is_file():
                fmt = path.suffix.lstrip(".").lower() or "png"
                return path.read_bytes(), fmt, path
        if chunk.figure_s3_key:
            from datasheet_rag.aws import s3_client

            settings = get_settings()
            client = s3_client()
            resp = client.get_object(Bucket=settings.require_s3_bucket(), Key=chunk.figure_s3_key)
            data = resp["Body"].read()
            ext = Path(chunk.figure_s3_key).suffix.lstrip(".").lower() or "png"
            return data, ext, None
        raise FigureUnavailableError(
            f"chunk {chunk.id} has no usable figure source — "
            f"figure_image_path is missing locally and figure_s3_key is unset."
        )

    def get_figure_bytes(self, chunk_id: str) -> FigureBytes:
        from datasheet_rag.description.describer import surrounding_text_for

        chunk = get_chunk(self._get_conn(), chunk_id)
        if chunk is None:
            raise FigureNotFoundError(f"unknown chunk_id: {chunk_id}")
        if chunk.metadata.layout_type not in IMAGE_LAYOUT_TYPES:
            raise FigureNotFoundError(
                f"chunk {chunk_id} is not a figure (layout_type={chunk.metadata.layout_type.value})"
            )
        image_bytes, fmt, resolved = self._read_figure_image(chunk)
        pages = chunk.metadata.page_numbers
        page = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}" if pages else ""
        return FigureBytes.from_bytes(
            chunk_id=chunk.id,
            rev_id=chunk.rev_id,
            image_bytes=image_bytes,
            fmt=fmt,
            local_path=str(resolved) if resolved else None,
            caption=chunk.figure_caption or "",
            description=chunk.figure_description or "",
            citation=FigureCitation(
                rev_id=chunk.rev_id,
                page=page,
                section=chunk.metadata.section_title or "",
                chapter=chunk.metadata.chapter_title or "",
            ),
            # Shipped with the pixels so a client running the vision model
            # itself needs one round trip per figure, not three (GH #43).
            surrounding_text=surrounding_text_for(self._get_conn(), chunk),
        )

    def update_figure_description(
        self, chunk_id: str, description: str, *, update_context_text: bool = True
    ) -> bool:
        with self._write_lock:
            return update_figure_description(
                self._get_conn(),
                chunk_id,
                description,
                update_context_text=update_context_text,
            )

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
        from datasheet_rag.description import FigureDescriber, describe_figures_in_store

        describer = FigureDescriber(model_id=model_id, verbose=False)
        with self._write_lock:
            descriptions = describe_figures_in_store(
                self._get_conn(),
                rev_id=rev_id,
                project_id=project_id,
                missing_only=missing_only,
                limit=limit,
                describer=describer,
                dry_run=dry_run,
            )
        return descriptions, describer.stats()

    def get_title_context(self, rev_id: str) -> TitleContext:
        from datasheet_rag.titling import first_page_text

        conn = self._get_conn()
        md = get_metadata(conn, rev_id) or DocMetadata(rev_id=rev_id)
        return TitleContext(
            rev_id=rev_id,
            first_page_text=first_page_text(conn, rev_id),
            title_source=md.title_source,
            running_header=md.running_header,
            pdf_meta_title=md.pdf_meta_title,
        )

    def infer_title(
        self,
        rev_id: str,
        *,
        model_id: str | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> str | None:
        from datasheet_rag.titling import TitleInferer, infer_and_backfill_title

        inferer = TitleInferer(model_id=model_id)
        with self._write_lock:
            return infer_and_backfill_title(
                self._get_conn(),
                rev_id,
                inferer=inferer,
                dry_run=dry_run,
                force=force,
            )

    # -- source PDF ----------------------------------------------------
    def get_pdf_bytes(self, rev_id: str) -> bytes:
        from datasheet_rag import pdf_viewer

        return pdf_viewer.load_pdf_bytes(rev_id)

    def _assert_vector_dims(self, vectors: Mapping[str, Sequence[float]]) -> None:
        """Reject caller-supplied vectors whose width is not the store's.

        The same check ``search`` runs on ``query_vector``, on the write side:
        without it a mismatched vector reaches ``vec0`` and surfaces as an
        unhandled 500 with the real reason buried in the audit row, when the
        cause is a config difference the client can act on.
        """
        expected = stored_embedding_dim(self._get_conn())
        if expected is None:
            return
        for chunk_id, vec in vectors.items():
            if len(vec) != expected:
                raise ValueError(
                    f"vector for chunk {chunk_id} has {len(vec)} dimensions, but "
                    f"this store's vectors are {expected}-dimensional — the client "
                    f"and server are configured with different embedding models."
                )

    # -- ingestion -----------------------------------------------------
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
        # source_pdf is accepted and ignored: this store is on the same
        # machine that parsed, and save_pdf_locally already put the file in
        # pdf_dir on the way through. Only RemoteBackend has a copy to make.
        del source_pdf
        from datasheet_rag.embedding import embed_chunk_graph

        # Vectors handed in were computed by the caller (RAG_COMPUTE=client,
        # GH #43), so there is nothing left to embed here — and no embedding
        # model to load, which is the whole point on a GPU-less host.
        if vectors is not None:
            if describe_figures:
                raise ValueError(
                    "vectors= and describe_figures= are mutually exclusive: a "
                    "description written here would change the text the "
                    "supplied vectors were computed from. Describe first, then "
                    "embed, then send both."
                )
            self._assert_vector_dims(vectors)
            embed = False

        did = graph.rev_id
        settings = get_settings()

        # 1. Land uploaded figure bytes locally and rewrite host-local paths.
        if figures:
            figures_out = settings.figures_dir / did
            figures_out.mkdir(parents=True, exist_ok=True)
            for chunk_id, (img_bytes, ext) in figures.items():
                ext = (ext or "png").lstrip(".")
                dest = figures_out / f"{chunk_id}.{ext}"
                dest.write_bytes(img_bytes)
                chunk = graph.chunks.get(chunk_id)
                if chunk is not None:
                    # Store relative to figures_dir so the DB stays portable.
                    chunk.figure_image_path = to_relative_figure_path(dest)

        # 1b. Drop any figure path that does not resolve *here*. A remote
        #     client ships host-local paths for crops it could not upload
        #     (deleted, or never made); storing one verbatim mints a row that
        #     search would advertise and `get_figure` could never serve — the
        #     exact broken contract in GH #41. Blanking it lets the upsert's
        #     COALESCE keep whatever good source the row already had.
        dropped = 0
        for chunk in graph.chunks.values():
            if chunk.metadata.layout_type not in IMAGE_LAYOUT_TYPES:
                continue
            if chunk.figure_image_path and not figure_source_available(
                chunk.figure_image_path, chunk.figure_s3_key
            ):
                chunk.figure_image_path = None
                dropped += 1
        if dropped:
            logger.warning(
                "ingest %s: dropped %d figure path(s) that do not resolve on "
                "this host — those chunks keep any previously stored image "
                "and are otherwise served without one",
                did,
                dropped,
            )

        with self._write_lock:
            conn = self._get_conn()
            # Build the sidecar patch now: a project both added and removed
            # raises here, before any chunk is written. Ingest only ever adds
            # a project (GH #121): a document already in others stays in them.
            # Project-scoped fields go to the ingest project unless the patch
            # names one (GH #123).
            patch = (metadata or MetadataPatch()).model_copy(deep=True)
            patch.add_project(project_id)
            if patch.project is None and project_id and project_id.strip():
                patch.project = project_id.strip()
            # Decide which document the PDF goes into (GH #124): the one named
            # by new_revision_of, else the one the identity rule matches, else
            # a new one. An ambiguous match raises here, before any write.
            link = self._plan_link(did, patch, new_revision_of)
            target: int | None = None
            if link.action in ("explicit", "rule"):
                target = revs.parse_document_ref(link.document_id or "")
                assert target is not None
            # Reject an unregistered or mis-shaped field, or a project-scoped
            # one with no project or in a project the document will not be
            # in, before any chunk is written, not after the whole ingest has
            # run (GH #120, #123). It checks against the target document, so
            # nothing is written before the chunks: no write is open while
            # the embed runs.
            check_write(
                conn,
                did,
                add_projects=patch.add_projects,
                remove_projects=patch.remove_projects,
                fields=patch.fields,
                project=patch.project,
                this_revision=bool(patch.this_revision),
                revision=patch.revision,
                into=target,
            )

            def place_revision() -> None:
                # Uncommitted: it lands with the chunk insert that follows. A
                # second call (after the provisional insert) finds the row.
                if target is not None:
                    revs.ensure_revision(conn, did, document_id=target)

            # The projects are joined right before each chunk insert, which
            # commits them with the chunks: no stored state has chunks without
            # their projects (a stopped describe) or projects without chunks
            # (a failed embed).
            join = patch.add_projects or []
            described = 0
            # The provisional insert below prunes too, so the final insert
            # finds nothing left to drop — count both or the describe path
            # would always report 0.
            pruned = 0

            # 2. Optional figure descriptions (vision LLM). Needs the chunks
            #    in the store first, so do a provisional embed+insert, run the
            #    describer, then merge descriptions back into context_text.
            if describe_figures:
                from datasheet_rag.description import (
                    FigureDescriber,
                    apply_description_to_chunk,
                    describe_figures_in_store,
                )

                vectors_tmp = (
                    embed_chunk_graph(graph, embedder=self._get_embedder()) if embed else vectors
                )
                place_revision()
                join_projects(conn, did, join)
                pruned += insert_chunk_graph(conn, graph, vectors=vectors_tmp).pruned
                conn.commit()
                descriptions = describe_figures_in_store(
                    conn,
                    rev_id=did,
                    missing_only=True,
                    describer=FigureDescriber(verbose=False),
                    dry_run=False,
                )
                described = len(descriptions)
                for row in conn.execute(
                    "SELECT id, figure_description FROM chunks "
                    "WHERE rev_id=? AND figure_description IS NOT NULL",
                    (did,),
                ).fetchall():
                    chunk = graph.chunks.get(row["id"])
                    if chunk:
                        apply_description_to_chunk(chunk, row["figure_description"])

            # 3. Final embed + insert.
            if embed:
                vectors = embed_chunk_graph(graph, embedder=self._get_embedder())
            # insert_chunk_graph prunes: this graph is the whole document,
            # so rows it does not carry are leftovers from a previous chunking
            # of the same doc — a re-chunk that changes the count shifts every
            # positional id after it and would otherwise strand the tail in
            # search forever (GH #44). The upsert still preserves what a fresh
            # graph does not carry (descriptions, a curated title, a relinked
            # figure path) for every id that survives, which is why this is not
            # a delete-then-insert.
            place_revision()
            join_projects(conn, did, join)
            stats = insert_chunk_graph(conn, graph, vectors=vectors)
            pruned += stats.pruned
            conn.commit()

            # 4. Metadata sidecar (the patch built above).
            if not patch.is_empty():
                set_metadata(conn, did, **patch.kwargs())
            # A dict (even {}) is what the parse found and replaces the stored
            # hints whole. None means no parse computed them: keep the stored.
            if title_hints is not None:
                set_title_hints(
                    conn,
                    did,
                    pdf_meta_title=title_hints.get("pdf_meta_title"),
                    running_header=title_hints.get("running_header"),
                )

        # 5. Title. Either the caller already inferred one (client-side
        #    compute) and we only record it, or we infer it here.
        title = None
        current = self.get_doc_titles().get(did) if (infer_title or inferred_title) else None
        if inferred_title:
            if current in (None, "", "—") and self.set_doc_title(
                did, inferred_title, source="inferred"
            ):
                title = inferred_title
        elif infer_title:
            if current in (None, "", "—"):
                from datasheet_rag.titling import infer_and_backfill_title

                title = infer_and_backfill_title(self._get_conn(), did)

        document = revs.document_of(self._get_conn(), did)
        document_ref = revs.format_document_id(document) if document is not None else ""
        return IngestResult(
            rev_id=did,
            document_id=document_ref,
            link=link.model_copy(update={"document_id": document_ref}),
            inserted=stats.inserted,
            pruned=pruned,
            described=described,
            title=title,
        )

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
        # Local: run the parse pipeline in-process, then embed + store. The
        # figures are already cropped to disk under figures_dir and the graph
        # carries their paths, so ingest_chunk_graph needs no figure uploads.
        from datasheet_rag.ingest_pipeline import parse_pdf_to_graph

        parsed = parse_pdf_to_graph(
            pdf_path,
            rev_id=rev_id,
            backend=backend,
            skip_figures=skip_figures,
            upload_figures=upload_figures,
            dpi=dpi,
            micro_tokens=micro_tokens,
            meso_tokens=meso_tokens,
            accurate_tables=accurate_tables,
            force=force,
            progress=progress,
        )
        do_describe = not skip_figures and not skip_describe
        return self.ingest_chunk_graph(
            parsed.graph,
            project_id=project_id,
            metadata=metadata,
            embed=True,
            describe_figures=do_describe,
            infer_title=infer_title,
            title_hints=parsed.title_hints,
            new_revision_of=new_revision_of,
        )

    def delete_doc(self, rev_id: str) -> int:
        from datasheet_rag.delete import purge_local_files, purge_s3_objects

        with self._write_lock:
            conn = self._get_conn()
            n = delete_doc(conn, rev_id)
            delete_metadata(conn, rev_id)

        purge_local_files(rev_id)
        purge_s3_objects(rev_id)
        return n
