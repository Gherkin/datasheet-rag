"""Remote backend — an HTTP client to the FastAPI RAG server.

Every method maps to one request. Models round-trip as JSON via pydantic
``model_dump(mode="json")`` / ``model_validate``; figure and PDF bytes ride
base64 (figures) or raw streaming (PDF). The server embeds query text, so
this client never loads an embedding model.
"""

from __future__ import annotations

import json
from typing import Any

from datasheet_rag.backend.base import (
    FigureUnavailableError,
    FigureUploads,
    RagBackend,
    RagServerError,
    SearchMode,
)
from datasheet_rag.backend.models import (
    DocSummary,
    FigureBytes,
    IngestedDoc,
    IngestResult,
    MetadataPatch,
    StatsResult,
)
from datasheet_rag.models.chunk import Chunk, ChunkGraph
from datasheet_rag.store import DocMetadata, SearchFilters, SearchResult

_EXCLUDE_VECS = {"content_embedding", "context_embedding"}


class RemoteBackend(RagBackend):
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 120.0,
    ):
        import httpx

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
        )

    def close(self) -> None:
        self._client.close()

    # -- helpers -------------------------------------------------------
    def _request(self, method: str, path: str, **kw: Any) -> Any:
        import httpx

        try:
            resp = self._client.request(method, path, **kw)
        except httpx.HTTPError as exc:  # connection refused, timeout, etc.
            raise RagServerError(0, str(exc)) from exc
        if resp.status_code >= 400:
            detail = resp.text
            code = ""
            try:
                body = resp.json()
                detail = body.get("detail", detail)
                code = body.get("code", "")
            except Exception:
                pass
            # Preserve the server's typed "there is no image" answer instead
            # of flattening it into a generic transport error (GH #41).
            if code == "figure_unavailable":
                raise FigureUnavailableError(detail)
            raise RagServerError(resp.status_code, detail)
        return resp

    def _json(self, method: str, path: str, **kw: Any) -> Any:
        return self._request(method, path, **kw).json()

    # -- search --------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        mode: SearchMode = "hybrid",
        k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        body = {
            "query": query,
            "mode": mode,
            "k": k,
            "filters": filters.model_dump(mode="json") if filters else None,
        }
        data = self._json("POST", "/search", json=body)
        return [SearchResult.model_validate(r) for r in data["results"]]

    # -- chunk reads ---------------------------------------------------
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        resp = self._request("GET", f"/chunks/{chunk_id}")
        if resp.status_code == 204:
            return None
        return Chunk.model_validate(resp.json())

    def get_children(self, chunk_id: str) -> list[Chunk]:
        data = self._json("GET", f"/chunks/{chunk_id}/children")
        return [Chunk.model_validate(c) for c in data["chunks"]]

    def count_chunks(
        self, *, doc_id: str | None = None, project_id: str | None = None
    ) -> int:
        params = _drop_none(doc_id=doc_id, project_id=project_id)
        return int(self._json("GET", "/chunks/count", params=params)["count"])

    # -- documents -----------------------------------------------------
    def list_documents(
        self,
        *,
        project_id: str | None = None,
        group_name: str | None = None,
        mpn: str | None = None,
        manufacturer: str | None = None,
    ) -> list[DocSummary]:
        params = _drop_none(
            project_id=project_id,
            group_name=group_name,
            mpn=mpn,
            manufacturer=manufacturer,
        )
        data = self._json("GET", "/documents", params=params)
        return [DocSummary.model_validate(d) for d in data["documents"]]

    def get_ingested_docs(
        self, *, project_id: str | None = None
    ) -> list[IngestedDoc]:
        params = _drop_none(project_id=project_id)
        data = self._json("GET", "/documents/ingested", params=params)
        return [IngestedDoc.model_validate(d) for d in data["documents"]]

    def get_doc_titles(self) -> dict[str, str]:
        return self._json("GET", "/documents/titles")["titles"]

    def set_doc_title(self, doc_id: str, title: str) -> int:
        data = self._json("PUT", f"/documents/{doc_id}/title", json={"title": title})
        return int(data["updated"])

    def resolve_doc_id(self, doc_id: str) -> str:
        return self._json("GET", f"/documents/resolve/{doc_id}")["doc_id"]

    # -- metadata ------------------------------------------------------
    def get_metadata(self, doc_id: str) -> DocMetadata | None:
        resp = self._request("GET", f"/documents/{doc_id}/metadata")
        if resp.status_code == 204:
            return None
        return DocMetadata.model_validate(resp.json())

    def set_metadata(self, doc_id: str, patch: MetadataPatch) -> DocMetadata:
        data = self._json(
            "PATCH",
            f"/documents/{doc_id}/metadata",
            json=patch.model_dump(mode="json"),
        )
        return DocMetadata.model_validate(data)

    def list_docs(
        self,
        *,
        project_id: str | None = None,
        group_name: str | None = None,
        mpn: str | None = None,
    ) -> list[DocMetadata]:
        params = _drop_none(project_id=project_id, group_name=group_name, mpn=mpn)
        data = self._json("GET", "/metadata", params=params)
        return [DocMetadata.model_validate(d) for d in data["documents"]]

    def apply_metadata_to_chunks(self, doc_id: str) -> int:
        return int(
            self._json("POST", f"/documents/{doc_id}/apply-metadata")["updated"]
        )

    # -- stats ---------------------------------------------------------
    def stats(
        self, *, project_id: str | None = None, doc_id: str | None = None
    ) -> StatsResult:
        params = _drop_none(project_id=project_id, doc_id=doc_id)
        return StatsResult.model_validate(self._json("GET", "/stats", params=params))

    # -- figures -------------------------------------------------------
    def list_figure_chunks(
        self,
        *,
        doc_id: str | None = None,
        project_id: str | None = None,
        only_with_image: bool = True,
    ) -> list[Chunk]:
        params = _drop_none(
            doc_id=doc_id,
            project_id=project_id,
            only_with_image=only_with_image,
        )
        data = self._json("GET", "/figures", params=params)
        return [Chunk.model_validate(c) for c in data["chunks"]]

    def get_figure_bytes(self, chunk_id: str) -> FigureBytes:
        return FigureBytes.model_validate(
            self._json("GET", f"/figures/{chunk_id}/bytes")
        )

    def update_figure_description(
        self, chunk_id: str, description: str, *, update_context_text: bool = True
    ) -> bool:
        data = self._json(
            "PUT",
            f"/figures/{chunk_id}/description",
            json={
                "description": description,
                "update_context_text": update_context_text,
            },
        )
        return bool(data["updated"])

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
        data = self._json(
            "POST",
            "/figures/describe",
            json={
                "doc_id": doc_id,
                "project_id": project_id,
                "missing_only": missing_only,
                "limit": limit,
                "model_id": model_id,
                "dry_run": dry_run,
            },
        )
        return data["descriptions"], data["stats"]

    def infer_title(
        self,
        doc_id: str,
        *,
        model_id: str | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> str | None:
        data = self._json(
            "POST",
            f"/documents/{doc_id}/infer-title",
            json={"model_id": model_id, "dry_run": dry_run, "force": force},
        )
        return data["title"]

    # -- source PDF ----------------------------------------------------
    def get_pdf_bytes(self, doc_id: str) -> bytes:
        return self._request("GET", f"/documents/{doc_id}/pdf").content

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
        payload = {
            "graph": graph.model_dump(mode="json", exclude=_chunk_excludes(graph)),
            "project_id": project_id,
            "group_name": group_name,
            "metadata": metadata.model_dump(mode="json") if metadata else None,
            "embed": embed,
            "describe_figures": describe_figures,
            "infer_title": infer_title,
            "title_hints": title_hints,
        }
        files: list[tuple[str, tuple[str, bytes, str]]] = [
            ("payload", ("payload.json", json.dumps(payload).encode(), "application/json"))
        ]
        for chunk_id, (img_bytes, ext) in (figures or {}).items():
            ext = (ext or "png").lstrip(".")
            files.append(
                ("figures", (f"{chunk_id}.{ext}", img_bytes, f"image/{ext}"))
            )
        data = self._json("POST", "/ingest", files=files)
        return IngestResult.model_validate(data)

    def ingest_pdf(
        self,
        pdf_path,
        *,
        doc_id=None,
        project_id=None,
        group_name=None,
        metadata=None,
        backend="docling",
        skip_figures=False,
        upload_figures=False,
        skip_describe=False,
        infer_title=False,
        dpi=300,
        micro_tokens=128,
        meso_tokens=512,
        accurate_tables=None,
        force=False,
        progress=None,
    ) -> IngestResult:
        # Stream the raw PDF up; the server runs parse → figures → chunk →
        # embed → store and pushes progress back as Server-Sent Events. We
        # forward each progress event to `progress` and return the final
        # result. No read timeout: the parse can run for minutes, and the SSE
        # progress stream keeps the connection alive meanwhile.
        from pathlib import Path

        import httpx

        from datasheet_rag.ingest_pipeline import ProgressEvent

        pdf_path = Path(pdf_path)
        options = {
            "doc_id": doc_id,
            "project_id": project_id,
            "group_name": group_name,
            "metadata": metadata.model_dump(mode="json") if metadata else None,
            "backend": backend,
            "skip_figures": skip_figures,
            "upload_figures": upload_figures,
            "skip_describe": skip_describe,
            "infer_title": infer_title,
            "dpi": dpi,
            "micro_tokens": micro_tokens,
            "meso_tokens": meso_tokens,
            "accurate_tables": accurate_tables,
            "force": force,
        }
        files = {
            "payload": (pdf_path.name, pdf_path.read_bytes(), "application/pdf"),
        }
        data = {"options": json.dumps(options)}

        result: IngestResult | None = None
        try:
            with self._client.stream(
                "POST", "/ingest-pdf", files=files, data=data, timeout=None
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    detail = resp.text
                    try:
                        detail = resp.json().get("detail", detail)
                    except Exception:
                        pass
                    raise RagServerError(resp.status_code, detail)
                for event, payload in _iter_sse(resp.iter_lines()):
                    if event == "progress":
                        if progress:
                            progress(ProgressEvent.from_dict(payload))
                    elif event == "result":
                        result = IngestResult.model_validate(payload)
                    elif event == "error":
                        raise RagServerError(500, payload.get("detail", "ingest failed"))
        except httpx.HTTPError as exc:
            raise RagServerError(0, str(exc)) from exc

        if result is None:
            raise RagServerError(0, "server closed the ingest stream without a result")
        return result

    def delete_doc(self, doc_id: str) -> int:
        return int(self._json("DELETE", f"/documents/{doc_id}")["deleted"])


def _iter_sse(lines: Any):
    """Yield ``(event, data)`` pairs from an SSE line stream.

    Minimal parser for the subset the server emits: one ``event:`` and one
    JSON ``data:`` line per event, separated by blank lines.
    """
    event = "message"
    data_lines: list[str] = []
    for line in lines:
        if line == "":
            if data_lines:
                yield event, json.loads("\n".join(data_lines))
            event = "message"
            data_lines = []
            continue
        if line.startswith(":"):  # comment / keepalive
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:  # stream ended without a trailing blank line
        yield event, json.loads("\n".join(data_lines))


def _drop_none(**kw: Any) -> dict[str, Any]:
    return {k: v for k, v in kw.items() if v is not None}


def _chunk_excludes(graph: ChunkGraph) -> dict[str, Any]:
    """Exclude embedding vectors from each chunk in the serialized graph."""
    return {"chunks": {cid: _EXCLUDE_VECS for cid in graph.chunks}}
