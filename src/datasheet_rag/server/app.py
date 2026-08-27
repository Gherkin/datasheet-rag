"""FastAPI app factory + all routes.

Routes map 1:1 to ``RagBackend`` methods. Request/response bodies reuse the
existing pydantic models (``Chunk``, ``SearchResult``, ``DocMetadata``,
``SearchFilters``) and the small DTOs in ``datasheet_rag.backend.models``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from datasheet_rag.backend.base import FigureUnavailableError, RagServerError
from datasheet_rag.backend.local import LocalBackend
from datasheet_rag.backend.models import (
    ChunkVectors,
    MetadataPatch,
)
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkGraph
from datasheet_rag.server.audit import audit
from datasheet_rag.server.deps import (
    get_backend,
    require_admin,
    require_ingest,
    require_read,
)
from datasheet_rag.store import (
    SearchFilters,
    SearchResult,
    TitleSource,
    create_api_key,
    list_api_keys,
    list_audit,
    revoke_api_key,
    stored_embedding_dim,
)

if TYPE_CHECKING:
    # Runtime import stays inside the ingest endpoint: pulling the pipeline in
    # at module scope would drag docling (and torch behind it) into every
    # server process, including ones that only ever serve reads.
    from datasheet_rag.ingest_pipeline import ProgressEvent

# Vectors are always None post-retrieval; never ship them.
_CHUNK_EXCLUDE = {"content_embedding", "context_embedding"}


def _chunk_json(chunk: Chunk) -> dict[str, Any]:
    return chunk.model_dump(mode="json", exclude=_CHUNK_EXCLUDE)


def _result_json(result: SearchResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "score": result.score,
        "match_source": result.match_source,
        "chunk": _chunk_json(result.chunk),
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    """Encode one Server-Sent Event (single JSON data line)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---- request bodies ------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    k: int = 10
    filters: SearchFilters | None = None
    # Set by a client that embeds its own queries (RAG_COMPUTE=client, GH #43).
    # The text is still sent — hybrid keyword matching needs it.
    query_vector: list[float] | None = None


class TitleBody(BaseModel):
    # Validated at the boundary rather than left to the store: an unknown
    # source ranks the same as "auto" and would clear the precedence guard,
    # so the write only fails after the title has already been overwritten.
    title: str
    source: TitleSource = "manual"
    force: bool = False


class FigureDescriptionBody(BaseModel):
    description: str
    update_context_text: bool = True


class DescribeFiguresBody(BaseModel):
    doc_id: str | None = None
    project_id: str | None = None
    missing_only: bool = True
    limit: int | None = None
    model_id: str | None = None
    dry_run: bool = False


class InferTitleBody(BaseModel):
    model_id: str | None = None
    dry_run: bool = False
    force: bool = False


class CreateKeyBody(BaseModel):
    label: str
    scopes: list[str] = ["ingest"]


def build_app() -> FastAPI:
    # The MCP endpoint is built before the app so its session manager can be
    # folded into the app's lifespan — it must be running before /mcp can
    # take a request. It serves /mcp and /mcp/<project_id>, the trailing
    # segment scoping the tools to a project the way a `.rag.toml` scopes the
    # stdio server. Bound to the server's own LocalBackend, never the
    # config-resolved one, so the server cannot call itself (GH #39).
    mcp_mount = None
    if get_settings().server_mcp_enabled:
        from datasheet_rag.server.mcp_mount import build_mcp_mount

        mcp_mount = build_mcp_mount(get_backend())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if mcp_mount is None:
            yield
            return
        async with mcp_mount.lifespan():
            yield

    app = FastAPI(
        title="datasheet-rag server",
        description="HTTP API over the shared RAG store.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Installed before CORS so CORS ends up the outer layer and covers /mcp
    # too — middleware added later wraps middleware added earlier.
    if mcp_mount is not None:
        mcp_mount.install(app)

    origins = get_settings().cors_origins_list()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Per-route scope gates. Read endpoints take `read`; cost/write endpoints
    # take `ingest`; key management takes `admin` (each implies the lower).
    read_dep = [require_read]
    ingest_dep = [require_ingest]
    admin_dep = [require_admin]
    dep = read_dep  # default for plain read routes

    @app.exception_handler(RagServerError)
    async def _rag_err(_: Any, exc: RagServerError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code or 500, content={"detail": exc.detail})

    @app.exception_handler(FigureUnavailableError)
    async def _fig_unavailable(_: Any, exc: FigureUnavailableError) -> JSONResponse:
        # 404, not 400: the request was well-formed, the image simply is not
        # there. The `code` lets RemoteBackend re-raise the typed error so the
        # MCP layer can answer softly instead of looking like a bug (GH #41).
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc), "code": "figure_unavailable"},
        )

    @app.exception_handler(ValueError)
    async def _value_err(_: Any, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # -- health (unauthenticated) ----------------------------------------
    @app.get("/health")
    def health(be: LocalBackend = Depends(get_backend)) -> dict[str, Any]:
        # Unauthenticated — keep the body minimal (no filesystem paths or
        # other internal detail that could aid an attacker).
        s = get_settings()
        # The store's own width, not this process's config: that is what a
        # client-supplied vector actually has to match (GH #43). Falls back to
        # the setting for a database too old to record one — or for a store
        # this process cannot read right now. /health is a liveness probe
        # before it is anything else: a locked or missing database must not
        # turn it into a 500 and take the container down with it.
        try:
            dims = stored_embedding_dim(be.conn) or s.embedding_dimensions
        except Exception:
            dims = s.embedding_dimensions
        return {
            "status": "ok",
            "embedding_backend": s.embedding_backend,
            "embedding_dimensions": dims,
            # Model identity, so a client that embeds its own vectors can check
            # they will be comparable with the store's (GH #43). Not a secret:
            # it is a public model name, and a client cannot use the store
            # correctly without it.
            "embedding_model": (
                s.local_embedding_model if s.embedding_backend == "local" else s.embedding_model_id
            ),
        }

    # -- search ----------------------------------------------------------
    @app.post("/search", dependencies=dep)
    def search(req: SearchRequest, be: LocalBackend = Depends(get_backend)) -> dict[str, Any]:
        results = be.search(
            req.query,
            mode=req.mode,  # type: ignore[arg-type]
            k=req.k,
            filters=req.filters,
            query_vector=req.query_vector,
        )
        return {"results": [_result_json(r) for r in results]}

    # -- chunks ----------------------------------------------------------
    @app.get("/chunks/count", dependencies=dep)
    def chunks_count(
        doc_id: str | None = None,
        project_id: str | None = None,
        be: LocalBackend = Depends(get_backend),
    ) -> dict[str, Any]:
        return {"count": be.count_chunks(doc_id=doc_id, project_id=project_id)}

    @app.get("/chunks/{chunk_id}/children", dependencies=dep)
    def chunk_children(chunk_id: str, be: LocalBackend = Depends(get_backend)) -> dict[str, Any]:
        return {"chunks": [_chunk_json(c) for c in be.get_children(chunk_id)]}

    @app.get("/chunks/{chunk_id}", dependencies=dep)
    def chunk(chunk_id: str, be: LocalBackend = Depends(get_backend)) -> Response:
        c = be.get_chunk(chunk_id)
        if c is None:
            return Response(status_code=204)
        return JSONResponse(_chunk_json(c))

    # -- documents -------------------------------------------------------
    @app.get("/documents/ingested", dependencies=dep)
    def documents_ingested(
        project_id: str | None = None, be: LocalBackend = Depends(get_backend)
    ) -> dict[str, Any]:
        docs = be.get_ingested_docs(project_id=project_id)
        return {"documents": [d.model_dump(mode="json") for d in docs]}

    @app.get("/documents/titles", dependencies=dep)
    def documents_titles(be: LocalBackend = Depends(get_backend)) -> dict[str, Any]:
        return {"titles": be.get_doc_titles()}

    @app.get("/documents/resolve/{doc_id}", dependencies=dep)
    def documents_resolve(doc_id: str, be: LocalBackend = Depends(get_backend)) -> dict[str, Any]:
        return {"doc_id": be.resolve_doc_id(doc_id)}

    @app.get("/documents/{doc_id}/pdf", dependencies=dep)
    def document_pdf(doc_id: str, be: LocalBackend = Depends(get_backend)) -> Response:
        try:
            data = be.get_pdf_bytes(doc_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=data, media_type="application/pdf")

    @app.put("/documents/{doc_id}/pdf", dependencies=ingest_dep)
    async def document_put_pdf(
        doc_id: str,
        payload: UploadFile = File(...),
        be: LocalBackend = Depends(get_backend),
    ) -> dict[str, Any]:
        """Store a source PDF for a document ingested elsewhere (GH #43).

        Under ``RAG_COMPUTE=client`` the parse happens on the client, so the
        PDF never passes through ``/ingest-pdf``. Without this the store would
        hold chunks whose source document nothing else on the network can
        fetch, and ``rag show`` / ``show_pdf`` would 404 for every other user.
        """
        from datasheet_rag.storage import save_pdf_bytes

        save_pdf_bytes(await payload.read(), doc_id)
        return {"stored": True, "doc_id": doc_id}

    @app.get("/documents/{doc_id}/metadata", dependencies=dep)
    def document_metadata(doc_id: str, be: LocalBackend = Depends(get_backend)) -> Response:
        md = be.get_metadata(doc_id)
        if md is None:
            return Response(status_code=204)
        return JSONResponse(md.model_dump(mode="json"))

    @app.patch("/documents/{doc_id}/metadata", dependencies=ingest_dep)
    def document_set_metadata(
        doc_id: str, patch: MetadataPatch, be: LocalBackend = Depends(get_backend)
    ) -> dict[str, Any]:
        return be.set_metadata(doc_id, patch).model_dump(mode="json")

    @app.put("/documents/{doc_id}/title", dependencies=ingest_dep)
    def document_set_title(
        doc_id: str, body: TitleBody, be: LocalBackend = Depends(get_backend)
    ) -> dict[str, Any]:
        return {
            "updated": be.set_doc_title(doc_id, body.title, source=body.source, force=body.force)
        }

    @app.post("/documents/{doc_id}/apply-metadata", dependencies=ingest_dep)
    def document_apply_metadata(
        doc_id: str, be: LocalBackend = Depends(get_backend)
    ) -> dict[str, Any]:
        return {"updated": be.apply_metadata_to_chunks(doc_id)}

    @app.delete("/documents/{doc_id}", dependencies=ingest_dep)
    def document_delete(
        doc_id: str, request: Request, be: LocalBackend = Depends(get_backend)
    ) -> dict[str, Any]:
        try:
            deleted = be.delete_doc(doc_id)
        except Exception as exc:
            audit(
                request,
                be,
                action="delete_doc",
                status="error",
                doc_id=doc_id,
                error=str(exc),
            )
            raise
        audit(
            request,
            be,
            action="delete_doc",
            status="ok",
            doc_id=doc_id,
            detail={"deleted": deleted},
        )
        return {"deleted": deleted}

    @app.get("/documents", dependencies=dep)
    def documents(
        project_id: str | None = None,
        group_name: str | None = None,
        mpn: str | None = None,
        manufacturer: str | None = None,
        be: LocalBackend = Depends(get_backend),
    ) -> dict[str, Any]:
        docs = be.list_documents(
            project_id=project_id,
            group_name=group_name,
            mpn=mpn,
            manufacturer=manufacturer,
        )
        return {"documents": [d.model_dump(mode="json") for d in docs]}

    # -- metadata listing ------------------------------------------------
    @app.get("/metadata", dependencies=dep)
    def metadata(
        project_id: str | None = None,
        group_name: str | None = None,
        mpn: str | None = None,
        be: LocalBackend = Depends(get_backend),
    ) -> dict[str, Any]:
        docs = be.list_docs(project_id=project_id, group_name=group_name, mpn=mpn)
        return {"documents": [d.model_dump(mode="json") for d in docs]}

    # -- stats -----------------------------------------------------------
    @app.get("/stats", dependencies=dep)
    def stats(
        project_id: str | None = None,
        doc_id: str | None = None,
        be: LocalBackend = Depends(get_backend),
    ) -> dict[str, Any]:
        return be.stats(project_id=project_id, doc_id=doc_id).model_dump(mode="json")

    # -- figures ---------------------------------------------------------
    @app.get("/figures", dependencies=dep)
    def figures(
        doc_id: str | None = None,
        project_id: str | None = None,
        only_with_image: bool = True,
        be: LocalBackend = Depends(get_backend),
    ) -> dict[str, Any]:
        chunks = be.list_figure_chunks(
            doc_id=doc_id, project_id=project_id, only_with_image=only_with_image
        )
        return {"chunks": [_chunk_json(c) for c in chunks]}

    @app.get("/figures/{chunk_id}/bytes", dependencies=dep)
    def figure_bytes(chunk_id: str, be: LocalBackend = Depends(get_backend)) -> dict[str, Any]:
        return be.get_figure_bytes(chunk_id).model_dump(mode="json")

    @app.put("/figures/{chunk_id}/description", dependencies=ingest_dep)
    def figure_description(
        chunk_id: str,
        body: FigureDescriptionBody,
        be: LocalBackend = Depends(get_backend),
    ) -> dict[str, Any]:
        updated = be.update_figure_description(
            chunk_id, body.description, update_context_text=body.update_context_text
        )
        return {"updated": updated}

    @app.post("/figures/describe", dependencies=ingest_dep)
    def figures_describe(
        body: DescribeFiguresBody,
        request: Request,
        be: LocalBackend = Depends(get_backend),
    ) -> dict[str, Any]:
        try:
            descriptions, stats = be.describe_figures(
                doc_id=body.doc_id,
                project_id=body.project_id,
                missing_only=body.missing_only,
                limit=body.limit,
                model_id=body.model_id,
                dry_run=body.dry_run,
            )
        except Exception as exc:
            audit(
                request,
                be,
                action="describe_figures",
                status="error",
                doc_id=body.doc_id,
                project_id=body.project_id,
                error=str(exc),
            )
            raise
        audit(
            request,
            be,
            action="describe_figures",
            status="ok",
            doc_id=body.doc_id,
            project_id=body.project_id,
            detail=stats,
        )
        return {"descriptions": descriptions, "stats": stats}

    @app.get("/documents/{doc_id}/title-context", dependencies=dep)
    def document_title_context(
        doc_id: str, be: LocalBackend = Depends(get_backend)
    ) -> dict[str, Any]:
        """The store-side half of a title inference, for a client that runs the LLM."""
        return be.get_title_context(doc_id).model_dump(mode="json")

    @app.post("/documents/{doc_id}/infer-title", dependencies=ingest_dep)
    def document_infer_title(
        doc_id: str,
        body: InferTitleBody,
        request: Request,
        be: LocalBackend = Depends(get_backend),
    ) -> dict[str, Any]:
        try:
            title = be.infer_title(
                doc_id,
                model_id=body.model_id,
                dry_run=body.dry_run,
                force=body.force,
            )
        except Exception as exc:
            audit(
                request,
                be,
                action="infer_title",
                status="error",
                doc_id=doc_id,
                error=str(exc),
            )
            raise
        audit(
            request,
            be,
            action="infer_title",
            status="ok",
            doc_id=doc_id,
            detail={"title": title, "dry_run": body.dry_run},
        )
        return {"title": title}

    # -- ingestion -------------------------------------------------------
    @app.post("/ingest", dependencies=ingest_dep)
    async def ingest(
        request: Request,
        payload: UploadFile = File(...),
        figures: list[UploadFile] = File(default=[]),
        be: LocalBackend = Depends(get_backend),
    ) -> dict[str, Any]:
        data = json.loads((await payload.read()).decode())
        graph = ChunkGraph.model_validate(data["graph"])
        fig_uploads: dict[str, tuple[bytes, str]] = {}
        for f in figures:
            name = f.filename or ""
            chunk_id, _, ext = name.rpartition(".")
            fig_uploads[chunk_id] = (await f.read(), ext or "png")
        metadata = MetadataPatch.model_validate(data["metadata"]) if data.get("metadata") else None
        # A client that embedded for itself sends the vectors along (GH #43);
        # unpacking them here is what keeps this server off an embedding model.
        vectors = (
            ChunkVectors.model_validate(data["vectors"]).to_mapping()
            if data.get("vectors")
            else None
        )
        try:
            result = be.ingest_chunk_graph(
                graph,
                figures=fig_uploads or None,
                project_id=data.get("project_id"),
                group_name=data.get("group_name"),
                metadata=metadata,
                embed=data.get("embed", True),
                describe_figures=data.get("describe_figures", False),
                infer_title=data.get("infer_title", False),
                title_hints=data.get("title_hints"),
                vectors=vectors,
                inferred_title=data.get("inferred_title"),
            )
        except Exception as exc:
            audit(
                request,
                be,
                action="ingest",
                status="error",
                project_id=data.get("project_id"),
                error=str(exc),
            )
            raise
        audit(
            request,
            be,
            action="ingest",
            status="ok",
            doc_id=result.doc_id,
            project_id=data.get("project_id"),
            detail={
                "inserted": result.inserted,
                "pruned": result.pruned,
                "described": result.described,
                "embed": data.get("embed", True),
                "client_vectors": len(vectors) if vectors else 0,
                "figures": len(fig_uploads),
            },
        )
        return result.model_dump(mode="json")

    @app.post("/ingest-pdf", dependencies=ingest_dep)
    async def ingest_pdf(
        request: Request,
        payload: UploadFile = File(...),
        options: str = Form("{}"),
        be: LocalBackend = Depends(get_backend),
    ) -> StreamingResponse:
        """Raw-PDF ingest: the server runs the full parse pipeline (GH #16).

        The client uploads just the PDF plus an ``options`` JSON blob; the
        server detects the PDF type, runs Docling/Textract, crops figures,
        chunks, embeds, describes and stores — streaming progress back as
        Server-Sent Events, then a final ``result`` (or ``error``) event.
        """
        import asyncio
        import tempfile

        from datasheet_rag.ingest_pipeline import parse_pdf_to_graph

        opts = json.loads(options) if options else {}
        pdf_bytes = await payload.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_bytes)
        tmp.close()
        tmp_path = Path(tmp.name)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        max_step = {"n": 0}

        def on_progress(ev: ProgressEvent) -> None:
            max_step["n"] = max(max_step["n"], ev.step)
            loop.call_soon_threadsafe(queue.put_nowait, ("progress", ev.to_dict()))

        def worker() -> None:
            try:
                skip_figures = bool(opts.get("skip_figures", False))
                skip_describe = bool(opts.get("skip_describe", False))
                parsed = parse_pdf_to_graph(
                    tmp_path,
                    doc_id=opts.get("doc_id"),
                    backend=opts.get("backend", "docling"),
                    skip_figures=skip_figures,
                    upload_figures=bool(opts.get("upload_figures", False)),
                    dpi=int(opts.get("dpi", 300)),
                    micro_tokens=int(opts.get("micro_tokens", 128)),
                    meso_tokens=int(opts.get("meso_tokens", 512)),
                    accurate_tables=opts.get("accurate_tables"),
                    force=bool(opts.get("force", False)),
                    progress=on_progress,
                )
                embed_step = {"kind": "step", "text": "Embed & store", "step": max_step["n"] + 1}
                loop.call_soon_threadsafe(queue.put_nowait, ("progress", embed_step))
                metadata = (
                    MetadataPatch.model_validate(opts["metadata"]) if opts.get("metadata") else None
                )
                result = be.ingest_chunk_graph(
                    parsed.graph,
                    project_id=opts.get("project_id"),
                    group_name=opts.get("group_name"),
                    metadata=metadata,
                    embed=True,
                    describe_figures=not skip_figures and not skip_describe,
                    infer_title=bool(opts.get("infer_title", False)),
                    title_hints=parsed.title_hints or None,
                )
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("result", result.model_dump(mode="json"))
                )
            except Exception as exc:  # surfaced to the client as an error event
                loop.call_soon_threadsafe(queue.put_nowait, ("error", {"detail": str(exc)}))
            finally:
                # Empty payload rather than None: the sentinel is recognised
                # by its kind, and keeping one payload type off the queue
                # saves every reader a None check it would never hit.
                loop.call_soon_threadsafe(queue.put_nowait, ("__done__", {}))

        async def event_gen() -> AsyncIterator[str]:
            fut = loop.run_in_executor(None, worker)
            status = "ok"
            result_doc: dict[str, Any] | None = None
            error_detail: str | None = None
            try:
                while True:
                    kind, data = await queue.get()
                    if kind == "__done__":
                        break
                    if kind == "result":
                        result_doc = data
                        yield _sse("result", data)
                    elif kind == "error":
                        status = "error"
                        error_detail = data.get("detail")
                        yield _sse("error", data)
                    else:
                        yield _sse("progress", data)
            finally:
                await fut
                tmp_path.unlink(missing_ok=True)
                if status == "ok" and result_doc is not None:
                    audit(
                        request,
                        be,
                        action="ingest",
                        status="ok",
                        doc_id=result_doc.get("doc_id"),
                        project_id=opts.get("project_id"),
                        detail={
                            "inserted": result_doc.get("inserted"),
                            "described": result_doc.get("described"),
                            "raw_pdf": True,
                        },
                    )
                else:
                    audit(
                        request,
                        be,
                        action="ingest",
                        status="error",
                        project_id=opts.get("project_id"),
                        error=error_detail,
                    )

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # -- admin: API key management ---------------------------------------
    @app.post("/admin/keys", dependencies=admin_dep)
    def admin_create_key(
        body: CreateKeyBody, be: LocalBackend = Depends(get_backend)
    ) -> dict[str, Any]:
        with be.write_lock:
            rec, token = create_api_key(be.conn, label=body.label, scopes=body.scopes)
        # Plaintext token returned exactly once; never stored or shown again.
        return {"id": rec.id, "label": rec.label, "scopes": rec.scopes, "token": token}

    @app.get("/admin/keys", dependencies=admin_dep)
    def admin_list_keys(be: LocalBackend = Depends(get_backend)) -> dict[str, Any]:
        keys = list_api_keys(be.conn)
        return {
            "keys": [
                {
                    "id": k.id,
                    "label": k.label,
                    "scopes": k.scopes,
                    "created_at": k.created_at,
                    "revoked_at": k.revoked_at,
                }
                for k in keys
            ]
        }

    @app.delete("/admin/keys/{key_id}", dependencies=admin_dep)
    def admin_revoke_key(key_id: str, be: LocalBackend = Depends(get_backend)) -> dict[str, Any]:
        with be.write_lock:
            revoked = revoke_api_key(be.conn, key_id)
        if not revoked:
            raise HTTPException(status_code=404, detail="key not found or already revoked")
        return {"revoked": key_id}

    @app.get("/audit", dependencies=admin_dep)
    def audit_list(
        doc_id: str | None = None,
        since: str | None = None,
        limit: int = 200,
        be: LocalBackend = Depends(get_backend),
    ) -> dict[str, Any]:
        return {"entries": list_audit(be.conn, doc_id=doc_id, since=since, limit=limit)}

    return app
