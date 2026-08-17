"""Reusable parse → figures → chunk pipeline, shared by the CLI and the server.

This is the parse half of ``rag ingest``: detect the PDF type, run Docling
layout analysis (native PDFs) or Textract OCR (scanned PDFs), crop figures,
and build the multi-scale :class:`~datasheet_rag.models.chunk.ChunkGraph`. It stops
short of embedding / figure-description / storage — those cross the
``RagBackend`` boundary (local sqlite or the remote server) and are driven by
``ingest_chunk_graph``.

It used to live inline in ``cli.py``'s ``_ingest_one``. It was pulled out so
the FastAPI server can run the exact same pipeline server-side for the
raw-PDF upload endpoint (``POST /ingest-pdf``) — letting a thin client ship
just the PDF instead of carrying the whole Docling/torch stack. See GH #16.

Progress is reported through a callback rather than hardcoded ``console``
prints, so the CLI can render Rich rules locally while the server bridges the
same events onto a Server-Sent-Events stream.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from datasheet_rag.models.chunk import ChunkGraph


@dataclass
class ProgressEvent:
    """A single progress signal from the pipeline.

    ``kind`` is ``"step"`` for a new numbered stage (rendered as a rule) or
    ``"detail"`` for an indented line under the current step. ``text`` may
    carry Rich markup — the CLI/remote console callback renders it; the SSE
    transport ships it verbatim.
    """

    kind: str  # "step" | "detail"
    text: str
    step: int = 0

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "step": self.step}

    @classmethod
    def from_dict(cls, d: dict) -> ProgressEvent:
        return cls(kind=d["kind"], text=d["text"], step=int(d.get("step", 0)))


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class ParseResult:
    """Output of :func:`parse_pdf_to_graph` — everything the store step needs."""

    graph: ChunkGraph
    doc_id: str
    resolved_backend: str
    title_hints: dict[str, str] = field(default_factory=dict)
    figure_count: int = 0


class OcrRequiredError(Exception):
    """Raised when a scanned PDF needs paid Textract OCR but it was disallowed.

    Used by the cost-estimation path: it wants an OCR line item without
    actually paying for OCR, so it passes ``allow_ocr=False`` and catches this
    to estimate from the page count instead.
    """

    def __init__(self, doc_id: str, pages: int):
        self.doc_id = doc_id
        self.pages = pages
        super().__init__(f"scanned PDF needs Textract OCR ({pages} pages)")


class ScannedPdfError(Exception):
    """Raised when Docling is forced on a scanned PDF (no text layer to parse)."""


def parse_pdf_to_graph(
    pdf_path: Path,
    *,
    doc_id: str | None = None,
    backend: str = "docling",
    skip_figures: bool = False,
    upload_figures: bool = False,
    dpi: int = 300,
    micro_tokens: int = 128,
    meso_tokens: int = 512,
    accurate_tables: bool | None = None,
    force: bool = False,
    allow_ocr: bool = True,
    progress: ProgressCallback | None = None,
) -> ParseResult:
    """Parse a PDF into a chunk graph, caching intermediate artefacts.

    Runs the same steps as ``rag ingest`` up to (but not including) embed /
    store: detect backend, Docling/Textract layout analysis, figure cropping,
    and multi-scale chunking. Intermediate artefacts (``{doc_id}_outline.json``,
    ``{doc_id}_blocks.json``, ``{doc_id}_chunks.json``) are cached under
    ``settings.output_dir`` and reused unless ``force`` is set.

    ``backend`` is ``"docling"`` | ``"textract"`` | ``"auto"``. ``"docling"``
    raises :class:`ScannedPdfError` on a scanned PDF rather than silently
    incurring Textract cost; ``"auto"`` routes scanned PDFs to Textract.

    With ``allow_ocr=False`` a scanned PDF with no cached OCR blocks raises
    :class:`OcrRequiredError` instead of paying for OCR (for cost estimation).
    """
    from datasheet_rag.chunking.pipeline import (
        load_chunk_graph,
        run_chunking_pipeline,
        run_chunking_pipeline_from_outline,
        save_chunk_graph,
    )
    from datasheet_rag.chunking.splitter import SplitterConfig
    from datasheet_rag.config import get_settings
    from datasheet_rag.docling_parser import content_hash
    from datasheet_rag.figures import (
        extract_figures,
        extract_figures_from_regions,
        upload_figures_to_s3,
    )
    from datasheet_rag.storage import save_pdf_locally

    settings = get_settings()
    if accurate_tables is None:
        accurate_tables = settings.table_structure_mode == "accurate"

    step_n = 0

    def _step(label: str) -> None:
        nonlocal step_n
        step_n += 1
        if progress:
            progress(ProgressEvent(kind="step", text=label, step=step_n))

    def _detail(text: str) -> None:
        if progress:
            progress(ProgressEvent(kind="detail", text=text, step=step_n))

    running_header = ""
    pdf_meta_title = ""
    figure_count = 0

    # ── 1. Detect backend ────────────────────────────────────────────────
    _step("Detect PDF type")
    if backend in ("auto", "docling"):
        from datasheet_rag.docling_parser import is_native_pdf

        native = is_native_pdf(pdf_path)
        if native:
            resolved_backend = "docling"
            _detail("Native PDF detected → using [cyan]docling[/] backend")
        elif backend == "auto":
            resolved_backend = "textract"
            _detail("Scanned PDF detected → using [cyan]textract[/] backend")
        else:
            raise ScannedPdfError(
                f"{pdf_path.name} looks like a scanned PDF — Docling needs a "
                "native text layer and cannot OCR it.\n"
                "  Re-run with --backend textract to use AWS Textract OCR "
                "instead (this incurs AWS costs), or --backend auto to route "
                "automatically based on PDF type."
            )
    else:
        resolved_backend = backend
        _detail(f"Backend forced to [cyan]{resolved_backend}[/]")

    # ── 2a. Docling path (native PDFs) ───────────────────────────────────
    if resolved_backend == "docling":
        from datasheet_rag.chunking.layout_parser import DocumentOutline
        from datasheet_rag.docling_parser import convert_pdf
        from datasheet_rag.figures import FigureRegion

        did = doc_id or content_hash(pdf_path)
        _detail(f"doc_id = [cyan]{did}[/]")
        save_pdf_locally(pdf_path, did)

        chunks_path = settings.output_dir / f"{did}_chunks.json"
        if chunks_path.exists() and not force:
            _step("Multi-scale chunking")
            _detail(f"[yellow]Resuming — loading cached chunk graph[/] → [cyan]{chunks_path}[/]")
            graph = load_chunk_graph(chunks_path)
        else:
            _step("Docling layout analysis")
            outline_path = settings.output_dir / f"{did}_outline.json"
            if outline_path.exists() and not force:
                _detail(
                    f"[yellow]Resuming — loading cached layout analysis[/] → "
                    f"[cyan]{outline_path}[/]"
                )
                with open(outline_path) as f:
                    cached = json.load(f)
                outline = DocumentOutline.from_dict(cached["outline"])
                figure_regions = [FigureRegion(**r) for r in cached["figure_regions"]]
            else:
                outline, figure_regions = convert_pdf(
                    pdf_path, doc_id=did, accurate_tables=accurate_tables
                )
                outline_path.parent.mkdir(parents=True, exist_ok=True)
                with open(outline_path, "w") as f:
                    json.dump(
                        {
                            "outline": outline.to_dict(),
                            "figure_regions": [dataclasses.asdict(r) for r in figure_regions],
                        },
                        f,
                    )
                _detail(f"Layout analysis cached → [cyan]{outline_path}[/]")

            running_header = outline.running_header
            pdf_meta_title = outline.pdf_meta_title
            summary = outline.summary()
            _detail(
                f"{summary['top_level_sections']} chapters, "
                f"{summary['total_sections']} sections, "
                f"{summary['total_elements']} elements "
                f"({summary['elements_by_type'].get('formula', 0)} formulas, "
                f"{summary['elements_by_type'].get('table', 0)} tables, "
                f"{summary['elements_by_type'].get('figure', 0)} figures)"
            )

            figure_manifest_dict = None
            if not skip_figures:
                _step("Extract figures & formulas")
                figures_out = settings.figures_dir / did
                manifest = extract_figures_from_regions(
                    pdf_path=pdf_path,
                    regions=figure_regions,
                    doc_id=did,
                    output_dir=figures_out,
                    dpi=dpi,
                    image_format="png",
                    padding_pct=0.02,
                )
                if upload_figures and manifest.figures:
                    manifest = upload_figures_to_s3(manifest)
                manifest_path = figures_out / "manifest.json"
                manifest.save(manifest_path)
                figure_manifest_dict = manifest.to_dict()
                figure_count = len(manifest.figures)
                _detail(f"{len(manifest.figures)} regions → [cyan]{manifest_path}[/]")

            _step("Multi-scale chunking")
            config = SplitterConfig(micro_max_tokens=micro_tokens, meso_max_tokens=meso_tokens)
            graph = run_chunking_pipeline_from_outline(
                outline,
                figure_manifest=figure_manifest_dict,
                config=config,
                summarizer_mode="extractive",
            )
            save_chunk_graph(graph, chunks_path)

    # ── 2b. Textract path (scanned PDFs) ─────────────────────────────────
    else:
        from datasheet_rag.textract import (
            get_job_results,
            load_blocks,
            save_blocks,
            start_analysis,
            wait_for_job,
        )

        blocks_path_probe = None
        did_probe = doc_id or content_hash(pdf_path)
        blocks_path_probe = settings.output_dir / f"{did_probe}_blocks.json"
        have_cached_blocks = blocks_path_probe.exists() and not force

        if not allow_ocr and not have_cached_blocks:
            from datasheet_rag.costs import pdf_page_count

            raise OcrRequiredError(did_probe, pdf_page_count(pdf_path))

        if have_cached_blocks:
            did = did_probe
            blocks_path = blocks_path_probe
            _step("Textract layout analysis (OCR)")
            _detail(f"doc_id = [cyan]{did}[/]")
            _detail(f"[yellow]Resuming — loading cached blocks[/] → [cyan]{blocks_path}[/]")
            blocks = load_blocks(blocks_path)
            _detail(f"{len(blocks)} blocks (cached)")
        else:
            from datasheet_rag.storage import upload_pdf

            _step("Upload PDF to S3")
            did, s3_key = upload_pdf(pdf_path, doc_id=doc_id)
            _detail(f"doc_id = [cyan]{did}[/]")
            _detail(f"s3_key = {s3_key}")

            blocks_path = settings.output_dir / f"{did}_blocks.json"
            _step("Textract layout analysis (OCR)")
            job_id = start_analysis(did, s3_key)
            _detail(f"job_id = {job_id}  (waiting…)")
            status = wait_for_job(job_id)
            if status != "SUCCEEDED":
                raise RuntimeError(f"Textract job failed with status: {status}")
            blocks = get_job_results(job_id)
            save_blocks(blocks, blocks_path)
            _detail(f"{len(blocks)} blocks → [cyan]{blocks_path}[/]")

        save_pdf_locally(pdf_path, did)

        figure_manifest_dict = None
        if not skip_figures:
            _step("Extract figures")
            figures_out = settings.figures_dir / did
            manifest = extract_figures(
                pdf_path=pdf_path,
                blocks=blocks,
                doc_id=did,
                output_dir=figures_out,
                dpi=dpi,
                image_format="png",
                padding_pct=0.02,
            )
            if upload_figures and manifest.figures:
                manifest = upload_figures_to_s3(manifest)
            manifest_path = figures_out / "manifest.json"
            manifest.save(manifest_path)
            figure_manifest_dict = manifest.to_dict()
            figure_count = len(manifest.figures)
            _detail(f"{len(manifest.figures)} figures → [cyan]{manifest_path}[/]")

        chunks_path = settings.output_dir / f"{did}_chunks.json"
        if chunks_path.exists() and not force:
            _step("Multi-scale chunking")
            _detail(f"[yellow]Resuming — loading cached chunk graph[/] → [cyan]{chunks_path}[/]")
            graph = load_chunk_graph(chunks_path)
        else:
            _step("Multi-scale chunking")
            config = SplitterConfig(micro_max_tokens=micro_tokens, meso_max_tokens=meso_tokens)
            graph = run_chunking_pipeline(
                blocks,
                doc_id=did,
                figure_manifest=figure_manifest_dict,
                config=config,
                summarizer_mode="extractive",
            )
            save_chunk_graph(graph, chunks_path)

    stats = graph.stats()
    _detail(
        f"{stats['total_chunks']} chunks "
        f"(MACRO {stats['by_level']['MACRO']}, "
        f"MESO {stats['by_level']['MESO']}, "
        f"MICRO {stats['by_level']['MICRO']})"
    )

    if figure_count == 0:
        figure_count = sum(1 for c in graph.chunks.values() if c.figure_image_path)

    title_hints: dict[str, str] = {}
    if running_header:
        title_hints["running_header"] = running_header
    if pdf_meta_title:
        title_hints["pdf_meta_title"] = pdf_meta_title

    return ParseResult(
        graph=graph,
        doc_id=did,
        resolved_backend=resolved_backend,
        title_hints=title_hints,
        figure_count=figure_count,
    )
