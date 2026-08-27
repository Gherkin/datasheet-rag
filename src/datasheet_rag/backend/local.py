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
    FigureUnavailableError,
    FigureUploads,
    RagBackend,
    SearchMode,
)
from datasheet_rag.backend.models import (
    DocSummary,
    FigureBytes,
    FigureCitation,
    IngestedDoc,
    IngestResult,
    MetadataPatch,
    StatsResult,
    TitleContext,
)
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkGraph, ChunkLevel, LayoutType
from datasheet_rag.store import (
    DocMetadata,
    SearchFilters,
    SearchResult,
    TitleSource,
    apply_metadata_to_chunks,
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
    list_figure_chunks,
    resolve_doc_id,
    resolve_figure_path,
    set_doc_title,
    set_metadata,
    stored_embedding_dim,
    to_relative_figure_path,
    update_figure_description,
    vector_search,
)

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
        filters: SearchFilters | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
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

    def get_children(self, chunk_id: str) -> list[Chunk]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id FROM chunks WHERE parent_id = ? ORDER BY rowid",
            (chunk_id,),
        ).fetchall()
        out: list[Chunk] = []
        for row in rows:
            child = get_chunk(conn, row["id"])
            if child is not None:
                out.append(child)
        return out

    def count_chunks(self, *, doc_id: str | None = None, project_id: str | None = None) -> int:
        return count_chunks(self._get_conn(), doc_id=doc_id, project_id=project_id)

    # -- documents / titles -------------------------------------------
    def _derived_doc_fields(self, doc_id: str) -> tuple[str | None, int | None]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT doc_title FROM chunks WHERE doc_id = ? AND doc_title != '' LIMIT 1",
            (doc_id,),
        ).fetchone()
        title = row["doc_title"] if row else None
        page_row = conn.execute(
            """
            SELECT page_numbers FROM chunks
             WHERE doc_id = ? AND page_numbers IS NOT NULL AND page_numbers != '[]'
             ORDER BY rowid DESC LIMIT 1
            """,
            (doc_id,),
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
        group_name: str | None = None,
        mpn: str | None = None,
        manufacturer: str | None = None,
    ) -> list[DocSummary]:
        docs = list_docs(
            self._get_conn(),
            project_id=project_id,
            group_name=group_name,
            mpn=mpn,
        )
        if manufacturer is not None:
            docs = [d for d in docs if d.manufacturer == manufacturer]
        out: list[DocSummary] = []
        for d in docs:
            title, page_count = self._derived_doc_fields(d.doc_id)
            out.append(
                DocSummary(
                    doc_id=d.doc_id,
                    project_id=d.project_id,
                    group_name=d.group_name,
                    mpn=d.mpn,
                    manufacturer=d.manufacturer,
                    subsystem=d.subsystem,
                    doc_type=d.doc_type,
                    tags=d.tags,
                    doc_title=title,
                    page_count=page_count,
                )
            )

        # The sidecar row is optional: it is written by the metadata-tagging
        # step, which ingestion does not require. A document that skipped it
        # is still chunked, searchable and reachable by every other read path,
        # so list it too — with the sidecar fields left null — rather than
        # reporting an empty catalogue for a store that plainly has documents
        # in it. Filters on group, mpn or manufacturer suppress the fallback:
        # those fields live nowhere but the sidecar, so a document without one
        # cannot match them.
        if group_name is None and mpn is None and manufacturer is None:
            listed = {d.doc_id for d in out}
            for row in get_ingested_docs(self._get_conn(), project_id=project_id):
                doc_id = row["doc_id"]
                if doc_id in listed:
                    continue
                title, page_count = self._derived_doc_fields(doc_id)
                out.append(DocSummary(doc_id=doc_id, doc_title=title, page_count=page_count))
            out.sort(key=lambda d: d.doc_id)
        return out

    def get_ingested_docs(self, *, project_id: str | None = None) -> list[IngestedDoc]:
        rows = get_ingested_docs(self._get_conn(), project_id=project_id)
        return [IngestedDoc(**r) for r in rows]

    def get_doc_titles(self) -> dict[str, str]:
        return get_doc_titles(self._get_conn())

    def set_doc_title(
        self, doc_id: str, title: str, *, source: TitleSource = "manual", force: bool = False
    ) -> int:
        with self._write_lock:
            conn = self._get_conn()
            n = set_doc_title(conn, doc_id, title, source=source, force=force)
            conn.commit()
            return n

    def resolve_doc_id(self, doc_id: str) -> str:
        return resolve_doc_id(self._get_conn(), doc_id)

    # -- metadata ------------------------------------------------------
    def get_metadata(self, doc_id: str) -> DocMetadata | None:
        return get_metadata(self._get_conn(), doc_id)

    def set_metadata(self, doc_id: str, patch: MetadataPatch) -> DocMetadata:
        with self._write_lock:
            return set_metadata(self._get_conn(), doc_id, **patch.kwargs())

    def list_docs(
        self,
        *,
        project_id: str | None = None,
        group_name: str | None = None,
        mpn: str | None = None,
    ) -> list[DocMetadata]:
        return list_docs(self._get_conn(), project_id=project_id, group_name=group_name, mpn=mpn)

    def apply_metadata_to_chunks(self, doc_id: str) -> int:
        with self._write_lock:
            return apply_metadata_to_chunks(self._get_conn(), doc_id)

    # -- stats ---------------------------------------------------------
    def stats(self, *, project_id: str | None = None, doc_id: str | None = None) -> StatsResult:
        conn = self._get_conn()
        total = count_chunks(conn, doc_id=doc_id, project_id=project_id)
        where: list[str] = []
        params: list[Any] = []
        if doc_id:
            where.append("doc_id = ?")
            params.append(doc_id)
        if project_id:
            where.append("project_id = ?")
            params.append(project_id)
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
            doc_id=doc_id,
            fts_missing=fts_status(conn).missing,
        )

    # -- figures -------------------------------------------------------
    def list_figure_chunks(
        self,
        *,
        doc_id: str | None = None,
        project_id: str | None = None,
        only_with_image: bool = True,
    ) -> list[Chunk]:
        chunks = list_figure_chunks(
            self._get_conn(),
            doc_id=doc_id,
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
            raise ValueError(f"unknown chunk_id: {chunk_id}")
        if chunk.metadata.layout_type != LayoutType.FIGURE:
            raise ValueError(
                f"chunk {chunk_id} is not a figure (layout_type={chunk.metadata.layout_type.value})"
            )
        image_bytes, fmt, resolved = self._read_figure_image(chunk)
        pages = chunk.metadata.page_numbers
        page = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}" if pages else ""
        return FigureBytes.from_bytes(
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            image_bytes=image_bytes,
            fmt=fmt,
            local_path=str(resolved) if resolved else None,
            caption=chunk.figure_caption or "",
            description=chunk.figure_description or "",
            citation=FigureCitation(
                doc_id=chunk.doc_id,
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
        doc_id: str | None = None,
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
                doc_id=doc_id,
                project_id=project_id,
                missing_only=missing_only,
                limit=limit,
                describer=describer,
                dry_run=dry_run,
            )
        return descriptions, describer.stats()

    def get_title_context(self, doc_id: str) -> TitleContext:
        from datasheet_rag.store.metadata import get_title_source
        from datasheet_rag.titling import first_page_text

        conn = self._get_conn()
        md = get_metadata(conn, doc_id)
        return TitleContext(
            doc_id=doc_id,
            first_page_text=first_page_text(conn, doc_id),
            attributes=dict(md.attributes) if md is not None else {},
            title_source=get_title_source(conn, doc_id),
        )

    def infer_title(
        self,
        doc_id: str,
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
                doc_id,
                inferer=inferer,
                dry_run=dry_run,
                force=force,
            )

    # -- source PDF ----------------------------------------------------
    def get_pdf_bytes(self, doc_id: str) -> bytes:
        from datasheet_rag import pdf_viewer

        return pdf_viewer.load_pdf_bytes(doc_id)

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
        group_name: str | None = None,
        metadata: MetadataPatch | None = None,
        embed: bool = True,
        describe_figures: bool = False,
        infer_title: bool = False,
        title_hints: dict[str, str] | None = None,
        vectors: Mapping[str, Sequence[float]] | None = None,
        inferred_title: str | None = None,
    ) -> IngestResult:
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

        did = graph.doc_id
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
            if chunk.metadata.layout_type != LayoutType.FIGURE:
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
                pruned += insert_chunk_graph(
                    conn,
                    graph,
                    vectors=vectors_tmp,
                    project_id=project_id,
                    group_name=group_name,
                ).pruned
                conn.commit()
                descriptions = describe_figures_in_store(
                    conn,
                    doc_id=did,
                    project_id=project_id,
                    missing_only=True,
                    describer=FigureDescriber(verbose=False),
                    dry_run=False,
                )
                described = len(descriptions)
                for row in conn.execute(
                    "SELECT id, figure_description FROM chunks "
                    "WHERE doc_id=? AND figure_description IS NOT NULL",
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
            stats = insert_chunk_graph(
                conn,
                graph,
                vectors=vectors,
                project_id=project_id,
                group_name=group_name,
            )
            pruned += stats.pruned
            conn.commit()

            # 4. Metadata sidecar.
            patch = metadata or MetadataPatch()
            if project_id is not None:
                patch.project_id = patch.project_id or project_id
            if group_name is not None:
                patch.group_name = patch.group_name or group_name
            if not patch.is_empty():
                set_metadata(conn, did, **patch.kwargs())
                apply_metadata_to_chunks(conn, did)
                conn.commit()
            if title_hints:
                set_metadata(conn, did, attributes=dict(title_hints))
                conn.commit()

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

        return IngestResult(
            doc_id=did,
            inserted=stats.inserted,
            pruned=pruned,
            described=described,
            title=title,
        )

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
        # Local: run the parse pipeline in-process, then embed + store. The
        # figures are already cropped to disk under figures_dir and the graph
        # carries their paths, so ingest_chunk_graph needs no figure uploads.
        from datasheet_rag.ingest_pipeline import parse_pdf_to_graph

        parsed = parse_pdf_to_graph(
            pdf_path,
            doc_id=doc_id,
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
            group_name=group_name,
            metadata=metadata,
            embed=True,
            describe_figures=do_describe,
            infer_title=infer_title,
            title_hints=parsed.title_hints or None,
        )

    def delete_doc(self, doc_id: str) -> int:
        from datasheet_rag.delete import purge_local_files, purge_s3_objects

        with self._write_lock:
            conn = self._get_conn()
            n = delete_doc(conn, doc_id)
            delete_metadata(conn, doc_id)

        purge_local_files(doc_id)
        purge_s3_objects(doc_id)
        return n
