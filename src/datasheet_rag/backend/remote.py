"""Remote backend — an HTTP client to the FastAPI RAG server.

Every method maps to one request. Models round-trip as JSON via pydantic
``model_dump(mode="json")`` / ``model_validate``; figure and PDF bytes ride
base64 (figures) or raw streaming (PDF).

Where the *models* run is a setting (``RAG_COMPUTE``, GH #43):

* ``server`` (default) — the client ships text, images and PDFs and the
  server embeds, describes and titles. A thin client needs no torch.
* ``client`` — this process runs every model and the server is only a
  vector store: query vectors, chunk vectors, figure descriptions and
  inferred titles all arrive precomputed. That is the setup for a GPU-less
  server host with a GPU workstation in front of it.

The branch lives here rather than in the CLI so both the CLI and the MCP
server inherit it without knowing about it, and ``RagBackend`` stays a
"what", not a "where".
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

from datasheet_rag.backend.base import (
    FigureUnavailableError,
    FigureUploads,
    RagBackend,
    RagServerError,
    RemoteIngestError,
    SearchMode,
)
from datasheet_rag.backend.models import (
    ChunkVectors,
    DocSummary,
    FigureBytes,
    IngestedDoc,
    IngestResult,
    MetadataPatch,
    StatsResult,
    TitleContext,
)
from datasheet_rag.models.chunk import Chunk, ChunkGraph

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    import httpx

    from datasheet_rag.ingest_pipeline import ProgressCallback
from datasheet_rag.store import DocMetadata, SearchFilters, SearchResult

_EXCLUDE_VECS = {"content_embedding", "context_embedding"}


class RemoteBackend(RagBackend):
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 120.0,
        compute: str = "server",
    ):
        import httpx

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers)
        self._compute = compute
        self._embedder: Any | None = None
        self._embedding_checked = False

    def close(self) -> None:
        self._client.close()

    # -- client-side compute -------------------------------------------
    @property
    def client_compute(self) -> bool:
        """True when this process runs the models instead of the server."""
        return self._compute == "client"

    def _get_embedder(self) -> Any:
        """Build (once) the in-process embedder, after checking it fits the store.

        The check is the important half. Vectors from two different embedding
        models are not comparable, so a client embedding with the wrong one
        does not fail — it quietly writes rows that never match, and quietly
        searches with a query the index cannot answer. ``/health`` reports what
        the store expects, and a mismatch stops the run here.
        """
        if self._embedder is None:
            self._assert_embedding_compatible()
            from datasheet_rag.embedding import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def _has_usable_title(self, graph: ChunkGraph) -> bool:
        """Whether this document already has a title worth keeping.

        Checks the graph the parser just produced, then the store — a
        re-ingest can carry no title while the stored document has a curated
        one, and inferring over that would be a call thrown away.
        """
        blank = (None, "", "—")
        if any(c.metadata.doc_title not in blank for c in graph.chunks.values()):
            return True
        try:
            return self.get_doc_titles().get(graph.doc_id) not in blank
        except RagServerError:
            return False

    def _assert_embedding_compatible(self) -> None:
        from datasheet_rag.config import get_settings

        if self._embedding_checked:
            return
        try:
            health = self._json("GET", "/health")
        except RagServerError:
            # Health is unauthenticated and cheap; if it cannot be reached the
            # real call is about to fail with a better message anyway. Counts
            # as checked: there is nothing more to learn by asking again.
            self._embedding_checked = True
            return
        settings = get_settings()
        # Note the flag is set only once the check *passes* — a mismatch has to
        # keep raising, not pass silently on the second call in the process.
        remote_dims = health.get("embedding_dimensions")
        if remote_dims is not None and int(remote_dims) != settings.embedding_dimensions:
            raise RagServerError(
                0,
                f"RAG_COMPUTE=client, but this machine embeds to "
                f"{settings.embedding_dimensions} dimensions and the server's store "
                f"holds {remote_dims}-dimensional vectors. Set "
                f"RAG_EMBEDDING_DIMENSIONS (and the matching model) to the "
                f"server's, or set RAG_COMPUTE=server to let it embed.",
            )
        remote_model = health.get("embedding_model")
        local_model = (
            settings.local_embedding_model
            if settings.embedding_backend == "local"
            else settings.embedding_model_id
        )
        if remote_model and remote_model != local_model:
            # Same width, different model: still incomparable, but we cannot
            # prove the ids mean different weights (an Ollama tag vs an HF repo
            # id for the same model, say), so warn rather than refuse.
            print(
                f"rag: warning — RAG_COMPUTE=client embeds with {local_model!r} but "
                f"the server's store was built with {remote_model!r}. Vectors from "
                f"different models do not match; searches may return nothing useful.",
                file=sys.stderr,
            )
        self._embedding_checked = True

    # -- helpers -------------------------------------------------------
    def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
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
        query_vector: Sequence[float] | None = None,
    ) -> list[SearchResult]:
        # Keyword search needs no vector at all; the other two modes get one
        # from here rather than from the server when compute is client-side.
        if query_vector is None and self.client_compute and mode != "keyword":
            query_vector = self._get_embedder().embed_one(query)
        body = {
            "query": query,
            "mode": mode,
            "k": k,
            "filters": filters.model_dump(mode="json") if filters else None,
            "query_vector": list(query_vector) if query_vector is not None else None,
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

    def count_chunks(self, *, doc_id: str | None = None, project_id: str | None = None) -> int:
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

    def get_ingested_docs(self, *, project_id: str | None = None) -> list[IngestedDoc]:
        params = _drop_none(project_id=project_id)
        data = self._json("GET", "/documents/ingested", params=params)
        return [IngestedDoc.model_validate(d) for d in data["documents"]]

    def get_doc_titles(self) -> dict[str, str]:
        titles: dict[str, str] = self._json("GET", "/documents/titles")["titles"]
        return titles

    def set_doc_title(
        self, doc_id: str, title: str, *, source: str = "manual", force: bool = False
    ) -> int:
        data = self._json(
            "PUT",
            f"/documents/{doc_id}/title",
            json={"title": title, "source": source, "force": force},
        )
        return int(data["updated"])

    def resolve_doc_id(self, doc_id: str) -> str:
        resolved: str = self._json("GET", f"/documents/resolve/{doc_id}")["doc_id"]
        return resolved

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
        return int(self._json("POST", f"/documents/{doc_id}/apply-metadata")["updated"])

    # -- stats ---------------------------------------------------------
    def stats(self, *, project_id: str | None = None, doc_id: str | None = None) -> StatsResult:
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
        return FigureBytes.model_validate(self._json("GET", f"/figures/{chunk_id}/bytes"))

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
        if self.client_compute:
            # Run the vision model here, reading each figure's pixels and
            # neighbour text off the server one request at a time (GH #43).
            from datasheet_rag.description import (
                FigureDescriber,
                describe_figures_via_backend,
            )

            describer = FigureDescriber(model_id=model_id, verbose=False)
            descriptions = describe_figures_via_backend(
                self,
                doc_id=doc_id,
                project_id=project_id,
                missing_only=missing_only,
                limit=limit,
                describer=describer,
                dry_run=dry_run,
            )
            return descriptions, describer.stats()
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
        if self.client_compute:
            from datasheet_rag.titling import TitleInferer, infer_title_via_backend

            return infer_title_via_backend(
                self,
                doc_id,
                inferer=TitleInferer(model_id=model_id),
                dry_run=dry_run,
                force=force,
            )
        data = self._json(
            "POST",
            f"/documents/{doc_id}/infer-title",
            json={"model_id": model_id, "dry_run": dry_run, "force": force},
        )
        title: str | None = data["title"]
        return title

    def get_title_context(self, doc_id: str) -> TitleContext:
        return TitleContext.model_validate(self._json("GET", f"/documents/{doc_id}/title-context"))

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
        vectors: Mapping[str, Sequence[float]] | None = None,
        inferred_title: str | None = None,
    ) -> IngestResult:
        described_here = 0
        if self.client_compute:
            # Run every model here, in the order the text depends on: a figure
            # description becomes part of the chunk's context_text, so it has
            # to land before the embedding is taken (GH #43).
            if describe_figures:
                from datasheet_rag.description import describe_figures_in_graph

                described_here = len(describe_figures_in_graph(graph))
                describe_figures = False
            if infer_title and inferred_title is None:
                from datasheet_rag.titling import infer_title_from_graph

                # Match the server's own guard before spending the call: it
                # only infers for a document that has no usable title, and the
                # answer would be discarded on arrival otherwise.
                if not self._has_usable_title(graph):
                    inferred_title = infer_title_from_graph(graph, title_hints=title_hints)
                infer_title = False
            if embed and vectors is None:
                from datasheet_rag.embedding import embed_chunk_graph

                vectors = embed_chunk_graph(graph, embedder=self._get_embedder())
            if vectors is not None:
                embed = False

        payload = {
            "graph": graph.model_dump(mode="json", exclude=_chunk_excludes(graph)),
            "project_id": project_id,
            "group_name": group_name,
            "metadata": metadata.model_dump(mode="json") if metadata else None,
            "embed": embed,
            "describe_figures": describe_figures,
            "infer_title": infer_title,
            "title_hints": title_hints,
            "inferred_title": inferred_title,
            # Packed as base64 float32 rather than JSON numbers: a datasheet's
            # vectors are tens of megabytes spelled out in decimal.
            "vectors": (
                ChunkVectors.from_mapping(vectors).model_dump(mode="json")
                if vectors is not None
                else None
            ),
        }
        files: list[tuple[str, tuple[str, bytes, str]]] = [
            ("payload", ("payload.json", json.dumps(payload).encode(), "application/json"))
        ]
        for chunk_id, (img_bytes, ext) in (figures or {}).items():
            ext = (ext or "png").lstrip(".")
            files.append(("figures", (f"{chunk_id}.{ext}", img_bytes, f"image/{ext}")))
        data = self._json("POST", "/ingest", files=files)
        result = IngestResult.model_validate(data)
        if described_here:
            # The server described nothing — it only stored what arrived — so
            # report what this process actually did.
            result.described = described_here
        return result

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
        # Stream the raw PDF up; the server runs parse → figures → chunk →
        # embed → store and pushes progress back as Server-Sent Events. We
        # forward each progress event to `progress` and return the final
        # result. No read timeout: the parse can run for minutes, and the SSE
        # progress stream keeps the connection alive meanwhile.
        from pathlib import Path

        import httpx

        from datasheet_rag.ingest_pipeline import ProgressEvent

        pdf_path = Path(pdf_path)

        if self.client_compute:
            # Nothing about this document should touch the server's CPU: parse
            # here, describe/title/embed here, and upload the finished graph
            # plus its crops (GH #43).
            from datasheet_rag.ingest_pipeline import (
                collect_figure_uploads,
                parse_pdf_to_graph,
            )

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
            uploads, _missing = ({}, []) if skip_figures else collect_figure_uploads(parsed.graph)
            return self.ingest_chunk_graph(
                parsed.graph,
                figures=uploads or None,
                project_id=project_id,
                group_name=group_name,
                metadata=metadata,
                embed=True,
                describe_figures=not skip_figures and not skip_describe,
                infer_title=infer_title,
                title_hints=parsed.title_hints or None,
            )

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
                        raise RemoteIngestError(payload.get("detail", "ingest failed"))
        except httpx.HTTPError as exc:
            raise RagServerError(0, str(exc)) from exc

        if result is None:
            raise RagServerError(0, "server closed the ingest stream without a result")
        return result

    def delete_doc(self, doc_id: str) -> int:
        return int(self._json("DELETE", f"/documents/{doc_id}")["deleted"])


def _iter_sse(lines: Any) -> Iterator[tuple[str, dict[str, Any]]]:
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
