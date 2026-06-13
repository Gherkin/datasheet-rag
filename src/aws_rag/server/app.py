"""FastAPI app factory + all routes.

Routes map 1:1 to ``RagBackend`` methods. Request/response bodies reuse the
existing pydantic models (``Chunk``, ``SearchResult``, ``DocMetadata``,
``SearchFilters``) and the small DTOs in ``aws_rag.backend.models``.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from aws_rag.backend.base import RagServerError
from aws_rag.backend.local import LocalBackend
from aws_rag.backend.models import (
    MetadataPatch,
)
from aws_rag.config import get_settings
from aws_rag.models.chunk import Chunk, ChunkGraph
from aws_rag.server.deps import get_backend, require_token
from aws_rag.store import SearchFilters, SearchResult

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


# ---- request bodies ------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    k: int = 10
    filters: SearchFilters | None = None


class TitleBody(BaseModel):
    title: str


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


def build_app() -> FastAPI:
    app = FastAPI(
        title="aws-rag server",
        description="HTTP API over the shared RAG store.",
        version="0.1.0",
    )

    dep = [Depends(require_token)]

    @app.exception_handler(RagServerError)
    async def _rag_err(_: Any, exc: RagServerError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code or 500, content={"detail": exc.detail}
        )

    @app.exception_handler(ValueError)
    async def _value_err(_: Any, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # -- health (unauthenticated) ----------------------------------------
    @app.get("/health")
    def health() -> dict[str, Any]:
        s = get_settings()
        return {
            "status": "ok",
            "db_path": str(s.sqlite_db_path),
            "embedding_backend": s.embedding_backend,
            "embedding_dimensions": s.embedding_dimensions,
        }

    # -- search ----------------------------------------------------------
    @app.post("/search", dependencies=dep)
    def search(req: SearchRequest, be: LocalBackend = Depends(get_backend)) -> dict:
        results = be.search(req.query, mode=req.mode, k=req.k, filters=req.filters)  # type: ignore[arg-type]
        return {"results": [_result_json(r) for r in results]}

    # -- chunks ----------------------------------------------------------
    @app.get("/chunks/count", dependencies=dep)
    def chunks_count(
        doc_id: str | None = None,
        project_id: str | None = None,
        be: LocalBackend = Depends(get_backend),
    ) -> dict:
        return {"count": be.count_chunks(doc_id=doc_id, project_id=project_id)}

    @app.get("/chunks/{chunk_id}/children", dependencies=dep)
    def chunk_children(chunk_id: str, be: LocalBackend = Depends(get_backend)) -> dict:
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
    ) -> dict:
        docs = be.get_ingested_docs(project_id=project_id)
        return {"documents": [d.model_dump(mode="json") for d in docs]}

    @app.get("/documents/titles", dependencies=dep)
    def documents_titles(be: LocalBackend = Depends(get_backend)) -> dict:
        return {"titles": be.get_doc_titles()}

    @app.get("/documents/resolve/{doc_id}", dependencies=dep)
    def documents_resolve(doc_id: str, be: LocalBackend = Depends(get_backend)) -> dict:
        return {"doc_id": be.resolve_doc_id(doc_id)}

    @app.get("/documents/{doc_id}/pdf", dependencies=dep)
    def document_pdf(doc_id: str, be: LocalBackend = Depends(get_backend)) -> Response:
        try:
            data = be.get_pdf_bytes(doc_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=data, media_type="application/pdf")

    @app.get("/documents/{doc_id}/metadata", dependencies=dep)
    def document_metadata(
        doc_id: str, be: LocalBackend = Depends(get_backend)
    ) -> Response:
        md = be.get_metadata(doc_id)
        if md is None:
            return Response(status_code=204)
        return JSONResponse(md.model_dump(mode="json"))

    @app.patch("/documents/{doc_id}/metadata", dependencies=dep)
    def document_set_metadata(
        doc_id: str, patch: MetadataPatch, be: LocalBackend = Depends(get_backend)
    ) -> dict:
        return be.set_metadata(doc_id, patch).model_dump(mode="json")

    @app.put("/documents/{doc_id}/title", dependencies=dep)
    def document_set_title(
        doc_id: str, body: TitleBody, be: LocalBackend = Depends(get_backend)
    ) -> dict:
        return {"updated": be.set_doc_title(doc_id, body.title)}

    @app.post("/documents/{doc_id}/apply-metadata", dependencies=dep)
    def document_apply_metadata(
        doc_id: str, be: LocalBackend = Depends(get_backend)
    ) -> dict:
        return {"updated": be.apply_metadata_to_chunks(doc_id)}

    @app.delete("/documents/{doc_id}", dependencies=dep)
    def document_delete(doc_id: str, be: LocalBackend = Depends(get_backend)) -> dict:
        return {"deleted": be.delete_doc(doc_id)}

    @app.get("/documents", dependencies=dep)
    def documents(
        project_id: str | None = None,
        group_name: str | None = None,
        mpn: str | None = None,
        manufacturer: str | None = None,
        be: LocalBackend = Depends(get_backend),
    ) -> dict:
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
    ) -> dict:
        docs = be.list_docs(project_id=project_id, group_name=group_name, mpn=mpn)
        return {"documents": [d.model_dump(mode="json") for d in docs]}

    # -- stats -----------------------------------------------------------
    @app.get("/stats", dependencies=dep)
    def stats(
        project_id: str | None = None,
        doc_id: str | None = None,
        be: LocalBackend = Depends(get_backend),
    ) -> dict:
        return be.stats(project_id=project_id, doc_id=doc_id).model_dump(mode="json")

    # -- figures ---------------------------------------------------------
    @app.get("/figures", dependencies=dep)
    def figures(
        doc_id: str | None = None,
        project_id: str | None = None,
        only_with_image: bool = True,
        be: LocalBackend = Depends(get_backend),
    ) -> dict:
        chunks = be.list_figure_chunks(
            doc_id=doc_id, project_id=project_id, only_with_image=only_with_image
        )
        return {"chunks": [_chunk_json(c) for c in chunks]}

    @app.get("/figures/{chunk_id}/bytes", dependencies=dep)
    def figure_bytes(chunk_id: str, be: LocalBackend = Depends(get_backend)) -> dict:
        return be.get_figure_bytes(chunk_id).model_dump(mode="json")

    @app.put("/figures/{chunk_id}/description", dependencies=dep)
    def figure_description(
        chunk_id: str,
        body: FigureDescriptionBody,
        be: LocalBackend = Depends(get_backend),
    ) -> dict:
        updated = be.update_figure_description(
            chunk_id, body.description, update_context_text=body.update_context_text
        )
        return {"updated": updated}

    @app.post("/figures/describe", dependencies=dep)
    def figures_describe(
        body: DescribeFiguresBody, be: LocalBackend = Depends(get_backend)
    ) -> dict:
        descriptions, stats = be.describe_figures(
            doc_id=body.doc_id,
            project_id=body.project_id,
            missing_only=body.missing_only,
            limit=body.limit,
            model_id=body.model_id,
            dry_run=body.dry_run,
        )
        return {"descriptions": descriptions, "stats": stats}

    @app.post("/documents/{doc_id}/infer-title", dependencies=dep)
    def document_infer_title(
        doc_id: str, body: InferTitleBody, be: LocalBackend = Depends(get_backend)
    ) -> dict:
        return {
            "title": be.infer_title(
                doc_id, model_id=body.model_id, dry_run=body.dry_run
            )
        }

    # -- ingestion -------------------------------------------------------
    @app.post("/ingest", dependencies=dep)
    async def ingest(
        payload: UploadFile = File(...),
        figures: list[UploadFile] = File(default=[]),
        be: LocalBackend = Depends(get_backend),
    ) -> dict:
        data = json.loads((await payload.read()).decode())
        graph = ChunkGraph.model_validate(data["graph"])
        fig_uploads: dict[str, tuple[bytes, str]] = {}
        for f in figures:
            name = f.filename or ""
            chunk_id, _, ext = name.rpartition(".")
            fig_uploads[chunk_id] = (await f.read(), ext or "png")
        metadata = (
            MetadataPatch.model_validate(data["metadata"])
            if data.get("metadata")
            else None
        )
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
        )
        return result.model_dump(mode="json")

    return app
