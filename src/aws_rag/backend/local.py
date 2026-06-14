"""Local backend — wraps the sqlite store and an in-process embedder.

This is the historical code path (open ``rag.sqlite`` directly, build an
embedder from settings) packaged behind the :class:`RagBackend` interface.
The FastAPI server also runs one of these per process to serve remote
clients.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

from aws_rag.backend.base import FigureUploads, RagBackend, SearchMode
from aws_rag.backend.models import (
    DocSummary,
    FigureBytes,
    FigureCitation,
    IngestedDoc,
    IngestResult,
    MetadataPatch,
    StatsResult,
)
from aws_rag.config import get_settings
from aws_rag.models.chunk import Chunk, ChunkGraph, ChunkLevel, LayoutType
from aws_rag.store import (
    DocMetadata,
    SearchFilters,
    SearchResult,
    apply_metadata_to_chunks,
    connect,
    count_chunks,
    delete_doc,
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
    to_relative_figure_path,
    update_figure_description,
    vector_search,
)


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
                from aws_rag.embedding import get_embedder

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
    ) -> list[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        conn = self._get_conn()
        if mode in ("vector", "hybrid"):
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

    def count_chunks(
        self, *, doc_id: str | None = None, project_id: str | None = None
    ) -> int:
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
        return out

    def get_ingested_docs(
        self, *, project_id: str | None = None
    ) -> list[IngestedDoc]:
        rows = get_ingested_docs(self._get_conn(), project_id=project_id)
        return [IngestedDoc(**r) for r in rows]

    def get_doc_titles(self) -> dict[str, str]:
        return get_doc_titles(self._get_conn())

    def set_doc_title(self, doc_id: str, title: str) -> int:
        with self._write_lock:
            conn = self._get_conn()
            n = set_doc_title(conn, doc_id, title)
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
        return list_docs(
            self._get_conn(), project_id=project_id, group_name=group_name, mpn=mpn
        )

    def apply_metadata_to_chunks(self, doc_id: str) -> int:
        with self._write_lock:
            return apply_metadata_to_chunks(self._get_conn(), doc_id)

    # -- stats ---------------------------------------------------------
    def stats(
        self, *, project_id: str | None = None, doc_id: str | None = None
    ) -> StatsResult:
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
        return StatsResult(
            total_chunks=total,
            by_level=by_level,
            project_id=project_id,
            doc_id=doc_id,
        )

    # -- figures -------------------------------------------------------
    def list_figure_chunks(
        self,
        *,
        doc_id: str | None = None,
        project_id: str | None = None,
        only_with_image: bool = True,
    ) -> list[Chunk]:
        return list_figure_chunks(
            self._get_conn(),
            doc_id=doc_id,
            project_id=project_id,
            only_with_image=only_with_image,
        )

    def _read_figure_image(self, chunk: Chunk) -> tuple[bytes, str, Path | None]:
        """Read a figure's bytes from disk (figure_image_path) or S3."""
        path = resolve_figure_path(chunk.figure_image_path)
        if path is not None:
            if path.is_file():
                fmt = path.suffix.lstrip(".").lower() or "png"
                return path.read_bytes(), fmt, path
        if chunk.figure_s3_key:
            from aws_rag.aws import s3_client

            settings = get_settings()
            client = s3_client()
            resp = client.get_object(
                Bucket=settings.s3_bucket, Key=chunk.figure_s3_key
            )
            data = resp["Body"].read()
            ext = Path(chunk.figure_s3_key).suffix.lstrip(".").lower() or "png"
            return data, ext, None
        raise ValueError(
            f"chunk {chunk.id} has no usable figure source — "
            f"figure_image_path is missing locally and figure_s3_key is unset."
        )

    def get_figure_bytes(self, chunk_id: str) -> FigureBytes:
        chunk = get_chunk(self._get_conn(), chunk_id)
        if chunk is None:
            raise ValueError(f"unknown chunk_id: {chunk_id}")
        if chunk.metadata.layout_type != LayoutType.FIGURE:
            raise ValueError(
                f"chunk {chunk_id} is not a figure "
                f"(layout_type={chunk.metadata.layout_type.value})"
            )
        image_bytes, fmt, resolved = self._read_figure_image(chunk)
        pages = chunk.metadata.page_numbers
        page = (
            str(pages[0])
            if len(pages) == 1
            else f"{pages[0]}-{pages[-1]}"
            if pages
            else ""
        )
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
        from aws_rag.description import FigureDescriber, describe_figures_in_store

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

    def infer_title(
        self, doc_id: str, *, model_id: str | None = None, dry_run: bool = False
    ) -> str | None:
        from aws_rag.titling import TitleInferer, infer_and_backfill_title

        inferer = TitleInferer(model_id=model_id)
        with self._write_lock:
            return infer_and_backfill_title(
                self._get_conn(), doc_id, inferer=inferer, dry_run=dry_run
            )

    # -- source PDF ----------------------------------------------------
    def get_pdf_bytes(self, doc_id: str) -> bytes:
        from aws_rag import pdf_viewer

        return pdf_viewer.load_pdf_bytes(doc_id)

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
    ) -> IngestResult:
        from aws_rag.embedding import embed_chunk_graph

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

        with self._write_lock:
            conn = self._get_conn()
            described = 0

            # 2. Optional figure descriptions (vision LLM). Needs the chunks
            #    in the store first, so do a provisional embed+insert, run the
            #    describer, then merge descriptions back into context_text.
            if describe_figures:
                from aws_rag.description import (
                    FigureDescriber,
                    describe_figures_in_store,
                )

                embedder_tmp = self._get_embedder() if embed else None
                vectors_tmp = (
                    embed_chunk_graph(graph, embedder=embedder_tmp) if embed else None
                )
                insert_chunk_graph(
                    conn,
                    graph,
                    vectors=vectors_tmp,
                    project_id=project_id,
                    group_name=group_name,
                )
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
                        chunk.figure_description = row["figure_description"]
                        tag = f"Description: {row['figure_description']}"
                        if tag not in (chunk.context_text or ""):
                            chunk.context_text = (
                                chunk.context_text or chunk.text
                            ) + "\n" + tag

            # 3. Final embed + insert.
            vectors = None
            if embed:
                vectors = embed_chunk_graph(graph, embedder=self._get_embedder())
            inserted = insert_chunk_graph(
                conn,
                graph,
                vectors=vectors,
                project_id=project_id,
                group_name=group_name,
            )
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

        # 5. Title inference (own transactions inside).
        title = None
        if infer_title:
            current = self.get_doc_titles().get(did)
            if current in (None, "", "—"):
                from aws_rag.titling import infer_and_backfill_title

                title = infer_and_backfill_title(self._get_conn(), did)

        return IngestResult(
            doc_id=did, inserted=inserted, described=described, title=title
        )

    def delete_doc(self, doc_id: str) -> int:
        with self._write_lock:
            return delete_doc(self._get_conn(), doc_id)
