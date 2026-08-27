"""Vision-LLM figure description for the RAG pipeline.

The describer takes a figure chunk (image + caption + surrounding text)
and asks a vision-capable Claude model on Bedrock to produce a 2-3
sentence description tuned for retrieval — naming any visible part
numbers, register names, signal names, or block names, but staying
faithful to what is actually in the image.

Cost notes (approximate, as of writing — verify in your console):

* Claude 3 Haiku (vision)       — ``$0.25 / 1M`` input, ``$1.25 / 1M`` output.
  Per figure: roughly ``$0.0005 – $0.001`` including the image-token
  surcharge and ~200 output tokens.
* Claude 3.5 Sonnet v2 (vision) — ``$3 / 1M`` in, ``$15 / 1M`` out
  (~10× the cost; better on dense diagrams).

The describer uses botocore adaptive retry mode for throttling and a
tenacity wrapper for ``ModelErrorException`` (transient Bedrock internal
errors), matching the :class:`datasheet_rag.embedding.BedrockEmbedder` style.
"""

from __future__ import annotations

import base64
import concurrent.futures as _cf
import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rich.console import Console
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from datasheet_rag.aws import s3_client
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkGraph, LayoutType
from datasheet_rag.store import resolve_figure_path

console = Console()

_MAX_INVOKE_ATTEMPTS = 4
_INVOKE_WAIT = wait_exponential(multiplier=1, min=2, max=8)

# Per-figure retry in the concurrent describe path: a transient failure
# (timeout, throttle, transient model/HTTP 5xx error, empty response) on one
# figure shouldn't silently drop it. Retried with linear backoff before the
# figure is finally skipped. Permanent errors (missing image, non-figure) are
# not retried.
_DESCRIBE_MAX_ATTEMPTS = 3
_DESCRIBE_RETRY_WAIT = 1.5  # seconds, scaled by attempt number


def _is_transient_model_error(exc: BaseException) -> bool:
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return False
    return bool(resp.get("Error", {}).get("Code") == "ModelErrorException")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You describe technical figures from electronics datasheets, "
    "reference manuals, and application notes. Your descriptions are "
    "consumed by a retrieval system that searches over them, so be "
    "concrete and specific: name visible part numbers, register names, "
    "signal names, pin names, block names, and any axis labels or "
    "numeric ranges. Do not speculate about anything not visible in "
    "the image. Keep the description to 2-3 sentences."
)


def _build_user_blocks(
    *,
    image_bytes: bytes,
    image_format: str,
    caption: str,
    section_context: str,
    surrounding_text: str,
) -> list[dict[str, Any]]:
    """Build the user-message ``content`` array for the Anthropic messages API."""
    media_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(image_format.lower(), "image/png")

    text_parts: list[str] = []
    if section_context:
        text_parts.append(f"Section context: {section_context}")
    if caption:
        text_parts.append(f"Caption: {caption}")
    if surrounding_text:
        text_parts.append(f"Surrounding text: {surrounding_text}")
    text_parts.append(
        "Write a 2-3 sentence description of the figure for a retrieval index. "
        "Be specific about what is visible — block / signal / register names, "
        "labels, and any visible numeric values."
    )

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(image_bytes).decode("ascii"),
            },
        },
        {"type": "text", "text": "\n\n".join(text_parts)},
    ]


# ---------------------------------------------------------------------------
# Image loading (local first, S3 fallback)
# ---------------------------------------------------------------------------


def _load_figure_bytes(chunk: Chunk) -> tuple[bytes, str]:
    """Read a figure chunk's image, returning ``(bytes, format)``.

    Prefers ``figure_image_path`` (local file). Falls back to S3 via
    ``figure_s3_key``. Raises a clear error if neither resolves.
    """
    path = resolve_figure_path(chunk.figure_image_path)
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(
                f"figure_image_path on chunk {chunk.id} points to a missing file: {path}"
            )
        fmt = path.suffix.lstrip(".").lower() or "png"
        return path.read_bytes(), fmt

    if chunk.figure_s3_key:
        settings = get_settings()
        resp = s3_client().get_object(Bucket=settings.require_s3_bucket(), Key=chunk.figure_s3_key)
        data = resp["Body"].read()
        fmt = Path(chunk.figure_s3_key).suffix.lstrip(".").lower() or "png"
        return data, fmt

    raise ValueError(
        f"chunk {chunk.id} has no figure_image_path or figure_s3_key — nothing to describe."
    )


# ---------------------------------------------------------------------------
# Surrounding context fetch from the store
# ---------------------------------------------------------------------------


_NEIGHBOR_CHAR_LIMIT = 400


def surrounding_text_for(
    conn: sqlite3.Connection,
    chunk: Chunk,
    *,
    char_limit: int = _NEIGHBOR_CHAR_LIMIT,
) -> str:
    """Concat the text of the chunk's prev and next siblings (trimmed)."""
    # TODO: skip siblings that are just the figure's own caption — Textract
    # emits the caption as its own chunk, so it reappears here as redundant
    # surrounding text (e.g. next sibling == chunk.figure_caption).
    fragments: list[str] = []
    for nid in (chunk.prev_id, chunk.next_id):
        if not nid:
            continue
        row = conn.execute("SELECT text FROM chunks WHERE id = ?", (nid,)).fetchone()
        if row and row["text"]:
            fragments.append(row["text"][:char_limit])
    return " [...] ".join(fragments)


# ---------------------------------------------------------------------------
# Figure sources — where a describer reads pixels and neighbour text from
# ---------------------------------------------------------------------------
#
# The vision call is the expensive part; fetching its inputs is not. Splitting
# them apart is what lets the model run somewhere other than where the store
# lives (GH #43): a client with a GPU can describe figures held by a GPU-less
# server, reading each image over HTTP, and the ingest path can describe a
# freshly parsed graph before it is uploaded at all.


@dataclass(frozen=True)
class FigureInputs:
    """Everything the vision call needs about one figure, fetched together."""

    image_bytes: bytes
    image_format: str
    surrounding_text: str = ""


class FigureSource(Protocol):
    """Supplies the inputs a figure description needs, one chunk at a time."""

    def inputs(self, chunk: Chunk) -> FigureInputs: ...


class StoreFigureSource:
    """Read from a local sqlite store: crops on disk (or S3), neighbours in SQL."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def inputs(self, chunk: Chunk) -> FigureInputs:
        image_bytes, fmt = _load_figure_bytes(chunk)
        return FigureInputs(image_bytes, fmt, surrounding_text_for(self._conn, chunk))


class GraphFigureSource:
    """Read from an in-memory :class:`ChunkGraph` — the ingest-time source.

    Both inputs are already on this machine right after a parse: the crops sit
    under ``figures_dir`` and the neighbours are in the graph. Using this
    avoids the provisional insert the store-backed path needs before it can
    see either.
    """

    def __init__(self, graph: ChunkGraph):
        self._graph = graph

    def inputs(self, chunk: Chunk) -> FigureInputs:
        image_bytes, fmt = _load_figure_bytes(chunk)
        fragments: list[str] = []
        for nid in (chunk.prev_id, chunk.next_id):
            neighbour = self._graph.chunks.get(nid) if nid else None
            if neighbour is not None and neighbour.text:
                fragments.append(neighbour.text[:_NEIGHBOR_CHAR_LIMIT])
        return FigureInputs(image_bytes, fmt, " [...] ".join(fragments))


class BackendFigureSource:
    """Read from a :class:`~datasheet_rag.backend.base.RagBackend`.

    ``get_figure_bytes`` returns the image *and* the neighbour text in one
    response, so describing N figures held by a remote server costs N round
    trips rather than 3N.
    """

    def __init__(self, backend: Any):
        self._backend = backend

    def inputs(self, chunk: Chunk) -> FigureInputs:
        fig = self._backend.get_figure_bytes(chunk.id)
        return FigureInputs(fig.image_bytes(), fig.format, fig.surrounding_text or "")


def _as_source(source: sqlite3.Connection | FigureSource) -> FigureSource:
    """Accept a bare sqlite connection where a :class:`FigureSource` is expected."""
    if isinstance(source, sqlite3.Connection):
        return StoreFigureSource(source)
    return source


# ---------------------------------------------------------------------------
# The describer
# ---------------------------------------------------------------------------


class FigureDescriber:
    """Wrap Bedrock Claude vision for figure → description.

    Concurrent batching follows the :class:`BedrockEmbedder` pattern.
    Use :meth:`describe_chunk_in_context` for the common path; it loads
    image bytes + neighbour text from the store for you.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        max_tokens: int | None = None,
        region: str | None = None,
        profile: str | None = None,
        client: Any | None = None,
        max_concurrency: int | None = None,
        verbose: bool = False,
    ) -> None:
        settings = get_settings()
        self.model_id = model_id or settings.description_model_id
        self.max_tokens = max_tokens or settings.description_max_tokens
        self.max_concurrency = max_concurrency or settings.description_concurrency
        self.verbose = verbose
        self.region = region or settings.aws_region

        self.client: Any = client

        self._total_invocations = 0
        self._total_errors = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _get_client(self) -> Any:
        if self.client is None:
            from datasheet_rag.local_models import get_chat_client

            self.client = get_chat_client(kind="vision", region=self.region)
        return self.client

    # ---- public ---------------------------------------------------------

    def describe_one(
        self,
        *,
        image_bytes: bytes,
        image_format: str,
        caption: str = "",
        section_context: str = "",
        surrounding_text: str = "",
    ) -> str:
        """Send one figure to Bedrock and return the description text."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": _SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": _build_user_blocks(
                        image_bytes=image_bytes,
                        image_format=image_format,
                        caption=caption,
                        section_context=section_context,
                        surrounding_text=surrounding_text,
                    ),
                }
            ],
        }
        return self._invoke(body)

    def describe_chunk_in_context(
        self,
        chunk: Chunk,
        source: sqlite3.Connection | FigureSource,
    ) -> str:
        """Convenience: pull image + neighbours from *source*, call the model.

        *source* is a :class:`FigureSource` — a local store, an in-memory
        graph, or a remote backend. A bare sqlite connection is accepted too
        and wrapped in a :class:`StoreFigureSource`.
        """
        if chunk.metadata.layout_type != LayoutType.FIGURE:
            raise ValueError(
                f"chunk {chunk.id} is not a figure (layout_type={chunk.metadata.layout_type.value})"
            )
        figure = _as_source(source).inputs(chunk)
        # TODO: dedupe consecutive identical levels — when chapter_title ==
        # section_title this emits "X > X" in the prompt. Collapse repeats.
        section_context = " > ".join(
            p for p in (chunk.metadata.chapter_title, chunk.metadata.section_title) if p
        )
        return self.describe_one(
            image_bytes=figure.image_bytes,
            image_format=figure.image_format,
            caption=chunk.figure_caption or "",
            section_context=section_context,
            surrounding_text=figure.surrounding_text,
        )

    def describe_chunks(
        self,
        chunks: Iterable[Chunk],
        source: sqlite3.Connection | FigureSource,
    ) -> dict[str, str]:
        """Describe many chunks concurrently. Returns ``{chunk_id: description}``.

        Each figure is retried up to ``_DESCRIBE_MAX_ATTEMPTS`` times on a
        transient failure (timeout/throttle/5xx) before being skipped; the
        dict only contains successes. Use :meth:`stats` for the failure count.
        """
        targets = [c for c in chunks if c.metadata.layout_type == LayoutType.FIGURE]
        if not targets:
            return {}
        figures = _as_source(source)

        # Resolve the client (and its credential chain) here, single-threaded —
        # concurrent first-use from the worker pool below races through
        # botocore's AssumeRoleProvider and can trip its spurious
        # "Infinite loop in credential configuration detected" check.
        self._get_client()

        results: dict[str, str] = {}

        def _one(c: Chunk) -> tuple[str, str | None]:
            for attempt in range(1, _DESCRIBE_MAX_ATTEMPTS + 1):
                try:
                    return c.id, self.describe_chunk_in_context(c, figures)
                except (FileNotFoundError, ValueError) as e:
                    # Permanent (missing image / not a figure) — don't retry.
                    console.print(f"[red]describe failed[/] for {c.id}: {e}")
                    return c.id, None
                except Exception as e:
                    if attempt < _DESCRIBE_MAX_ATTEMPTS:
                        if self.verbose:
                            console.print(
                                f"[yellow]describe retry[/] {c.id} "
                                f"({attempt}/{_DESCRIBE_MAX_ATTEMPTS}): {e}"
                            )
                        time.sleep(_DESCRIBE_RETRY_WAIT * attempt)
                        continue
                    console.print(
                        f"[red]describe failed[/] for {c.id} after "
                        f"{_DESCRIBE_MAX_ATTEMPTS} attempts: {e}"
                    )
                    return c.id, None
            return c.id, None  # unreachable; satisfies the type checker

        with _cf.ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            for chunk_id, desc in pool.map(_one, targets):
                if desc is not None:
                    results[chunk_id] = desc

        if self.verbose:
            console.print(
                f"[cyan]described[/] {len(results)}/{len(targets)} figures · "
                f"in={self._total_input_tokens} out={self._total_output_tokens} "
                f"errors={self._total_errors}"
            )
        return results

    def stats(self) -> dict[str, int]:
        return {
            "total_invocations": self._total_invocations,
            "total_errors": self._total_errors,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }

    # ---- private --------------------------------------------------------

    def _invoke(self, body: dict[str, Any]) -> str:
        self._get_client()
        try:
            for attempt in Retrying(
                retry=retry_if_exception(_is_transient_model_error),
                stop=stop_after_attempt(_MAX_INVOKE_ATTEMPTS),
                wait=_INVOKE_WAIT,
                reraise=True,
            ):
                with attempt:
                    response = self.client.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(body),
                        contentType="application/json",
                        accept="application/json",
                    )
        except Exception:
            self._total_errors += 1
            raise
        self._total_invocations += 1

        payload = json.loads(response["body"].read())
        usage = payload.get("usage", {}) or {}
        self._total_input_tokens += int(usage.get("input_tokens", 0))
        self._total_output_tokens += int(usage.get("output_tokens", 0))

        content = payload.get("content", [])
        # The messages API returns a list of blocks; concatenate text blocks.
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(t for t in text_parts if t).strip()


# ---------------------------------------------------------------------------
# Orchestrator: walk the store, describe missing figures, persist
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Planning: which figures actually need a vision call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DescriptionPlan:
    """Which figures to describe, and where each answer has to be copied.

    One image can back more than one chunk — a MESO wrapping a single figure
    carries its MICRO's image (see ``chunking.splitter``). Describing each of
    them would pay the vision call twice for identical pixels, so the plan
    picks one representative per image and records the siblings its answer
    fans out to.
    """

    to_describe: list[Chunk]
    fanout: dict[str, list[str]]  # representative chunk id → sibling ids
    reuse: dict[str, str]  # chunk id → description a sibling already had


def plan_figure_descriptions(
    chunks: Iterable[Chunk],
    *,
    missing_only: bool = True,
    limit: int | None = None,
) -> DescriptionPlan:
    """Group figure chunks by image and decide which ones need a model call.

    Pure list logic over chunks the caller already has, so it is shared by
    every place a description run can start from: a local sqlite store, a
    remote backend, or an in-memory graph mid-ingest.
    """
    groups: dict[str, list[Chunk]] = {}
    for c in chunks:
        key = c.figure_image_path or c.figure_s3_key or c.id
        groups.setdefault(key, []).append(c)

    reuse: dict[str, str] = {}
    to_describe: list[Chunk] = []
    fanout: dict[str, list[str]] = {}

    for members in groups.values():
        needing = [c for c in members if not c.figure_description] if missing_only else members
        if not needing:
            continue
        # Only a missing_only run may take the shortcut of copying a
        # description a sibling already has — an explicit --all run is asking
        # for a fresh one.
        existing = next((c.figure_description for c in members if c.figure_description), None)
        if existing and missing_only:
            for c in needing:
                reuse[c.id] = existing
            continue
        rep = max(members, key=lambda c: int(c.level))
        to_describe.append(rep)
        fanout[rep.id] = [c.id for c in members if c.id != rep.id]

    if limit is not None:
        dropped = {c.id for c in to_describe[limit:]}
        to_describe = to_describe[:limit]
        for rep_id in dropped:
            fanout.pop(rep_id, None)

    return DescriptionPlan(to_describe=to_describe, fanout=fanout, reuse=reuse)


def _run_description_plan(
    plan: DescriptionPlan,
    source: FigureSource,
    *,
    describer: FigureDescriber | None = None,
) -> dict[str, str]:
    """Execute a plan against *source*, fanning each answer out to its siblings."""
    descriptions: dict[str, str] = dict(plan.reuse)
    if plan.to_describe:
        describer = describer or FigureDescriber()
        fresh = describer.describe_chunks(plan.to_describe, source)
        for rep_id, desc in fresh.items():
            descriptions[rep_id] = desc
            for sibling_id in plan.fanout.get(rep_id, []):
                descriptions[sibling_id] = desc
    return descriptions


def describe_figures_in_store(
    conn: sqlite3.Connection,
    *,
    doc_id: str | None = None,
    project_id: str | None = None,
    missing_only: bool = True,
    limit: int | None = None,
    describer: FigureDescriber | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Walk figure chunks, generate descriptions, persist them.

    Parameters
    ----------
    missing_only:
        Skip chunks that already have a ``figure_description``.
    limit:
        Stop after describing this many figures (handy for cost-bounded runs).
    dry_run:
        Generate descriptions but do not write them back.

    Returns ``{chunk_id: description}`` for the figures that were described.
    """
    from datasheet_rag.store import list_figure_chunks, update_figure_description

    targets = list_figure_chunks(conn, doc_id=doc_id, project_id=project_id)
    # A row can name an image that is not on this host (a store restored
    # without its figures). Reading it would fail per chunk; skip it instead.
    targets = [c for c in targets if c.figure_available]

    descriptions = _run_description_plan(
        plan_figure_descriptions(targets, missing_only=missing_only, limit=limit),
        StoreFigureSource(conn),
        describer=describer,
    )
    if not descriptions:
        return {}

    if not dry_run:
        for chunk_id, desc in descriptions.items():
            update_figure_description(conn, chunk_id, desc)

    return descriptions


def describe_figures_via_backend(
    backend: Any,
    *,
    doc_id: str | None = None,
    project_id: str | None = None,
    missing_only: bool = True,
    limit: int | None = None,
    describer: FigureDescriber | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """:func:`describe_figures_in_store`, driven entirely through a backend.

    Same contract, but every store access goes through
    :class:`~datasheet_rag.backend.base.RagBackend` methods instead of SQL, so
    the vision model runs in *this* process against a store that may live on
    another host (``RAG_COMPUTE=client``, GH #43).
    """
    targets = backend.list_figure_chunks(doc_id=doc_id, project_id=project_id, only_with_image=True)
    descriptions = _run_description_plan(
        plan_figure_descriptions(targets, missing_only=missing_only, limit=limit),
        BackendFigureSource(backend),
        describer=describer,
    )
    if not dry_run:
        for chunk_id, desc in descriptions.items():
            backend.update_figure_description(chunk_id, desc)
    return descriptions


def describe_figures_in_graph(
    graph: ChunkGraph,
    *,
    missing_only: bool = True,
    limit: int | None = None,
    describer: FigureDescriber | None = None,
) -> dict[str, str]:
    """Describe a freshly parsed graph's figures in memory, before it is stored.

    The crops are already on this machine and the neighbours are in the graph,
    so nothing has to be inserted first. Descriptions are folded into each
    chunk's ``figure_description`` and ``context_text`` in place — the same
    shape ``ingest_chunk_graph`` would otherwise have to do server-side after
    a provisional insert.

    Returns ``{chunk_id: description}`` for the chunks that were updated.
    """
    from datasheet_rag.store import figure_source_available

    targets = [
        c
        for c in graph.chunks.values()
        if c.metadata.layout_type == LayoutType.FIGURE
        and figure_source_available(c.figure_image_path, c.figure_s3_key)
    ]
    descriptions = _run_description_plan(
        plan_figure_descriptions(targets, missing_only=missing_only, limit=limit),
        GraphFigureSource(graph),
        describer=describer,
    )
    for chunk_id, desc in descriptions.items():
        apply_description_to_chunk(graph.chunks[chunk_id], desc)
    return descriptions


def apply_description_to_chunk(chunk: Chunk, description: str) -> None:
    """Fold *description* into a chunk's row field and its embedded context text.

    Mirrors what ``store.update_figure_description`` does to a stored row, for
    a chunk that is still in memory.
    """
    chunk.figure_description = description
    tag = f"Description: {description}"
    if tag not in (chunk.context_text or ""):
        chunk.context_text = (chunk.context_text or chunk.text) + "\n" + tag
