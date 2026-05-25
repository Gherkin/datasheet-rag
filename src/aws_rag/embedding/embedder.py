"""Bedrock Titan v2 text embedding.

This module wraps the AWS Bedrock runtime ``invoke_model`` API for the
``amazon.titan-embed-text-v2:0`` model.

Responsibilities
----------------
* Build a ``bedrock-runtime`` client with sane timeouts and adaptive
  retries (matches the per-region throttling behavior Bedrock applies on
  Titan embeddings).
* Embed one string, many strings (concurrently), or every chunk in a
  :class:`ChunkGraph`.
* Track lightweight metrics (``total_tokens_in``, ``total_invocations``,
  ``total_errors``) for cost accounting.

What gets embedded
------------------
For chunk-level helpers, :attr:`Chunk.context_text` is used — that field
is populated by the chunking pipeline and already contains the contextual
prefix (``"Chapter: ... > Section: ..."``) that the RAG retriever
benefits from. We fall back to :attr:`Chunk.text` if ``context_text`` is
empty and warn; chunks where both are empty are skipped with a warning.

Retry strategy
--------------
We rely on botocore's built-in *adaptive* retry mode (configured below
in :func:`_bedrock_runtime_client`). Adaptive mode handles
``ThrottlingException`` with token-bucket-based backoff, which is the
right thing for the bursty bedrock invoke pattern.

In addition, we layer a tenacity retry specifically for
``ModelErrorException`` (Bedrock HTTP 500 "unexpected error during
processing"), which botocore does not automatically retry. Up to
``_MAX_INVOKE_ATTEMPTS`` attempts are made with exponential backoff
(2 → 4 → 8 seconds). Any other exception escapes immediately.

Cost
----
Informational: Titan v2 input embeddings are priced at $0.02 per 1M
input tokens (eu-west-1, May 2026). The ``stats()`` method returns
``total_tokens_in`` so a caller can compute the cost of a run. This is a
hard-coded reference; we do not auto-fetch pricing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config
from rich.console import Console
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from aws_rag.config import get_settings
from aws_rag.models.chunk import Chunk, ChunkGraph

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

console = Console()


# Titan v2 supports exactly these output dimensions. Anything else will
# silently fail at invoke time; we validate up front so callers get a
# clear error message.
_TITAN_V2_VALID_DIMS: frozenset[int] = frozenset({256, 512, 1024})

# Marker prefix used to detect Titan family model IDs so we only attach
# Titan-specific request fields (``dimensions``, ``normalize``) for those.
_TITAN_MODEL_PREFIX = "amazon.titan-embed"

# Cost reference, see module docstring.
TITAN_V2_USD_PER_1M_INPUT_TOKENS = 0.02

# Retry parameters for ModelErrorException (transient Bedrock internal error).
_MAX_INVOKE_ATTEMPTS = 4
_INVOKE_WAIT = wait_exponential(multiplier=1, min=2, max=8)


def _is_transient_model_error(exc: BaseException) -> bool:
    """True for Bedrock ModelErrorException — a transient internal error."""
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return False
    return resp.get("Error", {}).get("Code") == "ModelErrorException"


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def _bedrock_runtime_client(
    *,
    region: str | None = None,
    profile: str | None = None,
) -> "BedrockRuntimeClient":
    """Build a configured ``bedrock-runtime`` client.

    Lives here (rather than in :mod:`aws_rag.aws`) to keep this track
    self-contained. Once another caller needs it, it should migrate over.
    """
    settings = get_settings()
    session_kwargs: dict[str, str] = {
        "region_name": region or settings.aws_region,
    }
    effective_profile = profile if profile is not None else settings.aws_profile
    if effective_profile:
        session_kwargs["profile_name"] = effective_profile
    session = boto3.Session(**session_kwargs)

    # Adaptive retries handle ThrottlingException with token-bucket
    # backoff. 60s timeouts are generous for what is normally a sub-second
    # call but allow headroom under throttling.
    config = Config(
        connect_timeout=60,
        read_timeout=60,
        retries={"max_attempts": 5, "mode": "adaptive"},
    )
    return session.client("bedrock-runtime", config=config)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class BedrockEmbedder:
    """Thin wrapper around the Bedrock runtime client for Titan v2.

    Holds a single client and a small counter dict. Thread-safe for
    concurrent ``invoke_model`` calls (boto3 low-level clients are).
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        dimensions: int | None = None,
        normalize: bool | None = None,
        region: str | None = None,
        profile: str | None = None,
        client: Any | None = None,
        max_concurrency: int | None = None,
        verbose: bool = False,
    ) -> None:
        settings = get_settings()

        self.model_id: str = model_id if model_id is not None else settings.embedding_model_id
        self.dimensions: int = (
            dimensions if dimensions is not None else settings.embedding_dimensions
        )
        self.normalize: bool = (
            normalize if normalize is not None else settings.embedding_normalize
        )
        self.max_concurrency: int = (
            max_concurrency if max_concurrency is not None else settings.embedding_batch_size
        )
        self.verbose: bool = verbose

        # --- Validate model + dimensions ---------------------------------
        self._is_titan = self.model_id.startswith(_TITAN_MODEL_PREFIX)
        if not self._is_titan:
            console.print(
                f"[yellow]warning[/]: model_id {self.model_id!r} is not a Titan "
                "embedding model — Titan-specific params (dimensions, normalize) "
                "will be omitted from the request body."
            )
        else:
            if self.dimensions not in _TITAN_V2_VALID_DIMS:
                raise ValueError(
                    f"Titan v2 dimensions must be one of "
                    f"{sorted(_TITAN_V2_VALID_DIMS)}, got {self.dimensions}."
                )

        if self.max_concurrency < 1:
            raise ValueError(
                f"max_concurrency must be >= 1, got {self.max_concurrency}."
            )

        # --- Client ------------------------------------------------------
        if client is not None:
            self.client = client
        else:
            self.client = _bedrock_runtime_client(region=region, profile=profile)

        # --- Metrics -----------------------------------------------------
        self.total_tokens_in: int = 0
        self.total_invocations: int = 0
        self.total_errors: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Raises :class:`ValueError` on empty input."""
        if not text:
            raise ValueError("embed_one() requires a non-empty string.")
        return self._invoke_one(text)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many texts concurrently, preserving input order.

        Empty input → returns ``[]`` without contacting Bedrock. An empty
        string inside the sequence raises :class:`ValueError` (fail loud
        rather than silently writing a zero vector).
        """
        if len(texts) == 0:
            return []

        for i, t in enumerate(texts):
            if not t:
                raise ValueError(
                    f"embed_texts(): text at index {i} is empty — refusing "
                    "to embed an empty string."
                )

        # ThreadPoolExecutor.map preserves order across the iterable.
        workers = min(self.max_concurrency, len(texts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            vectors = list(pool.map(self._invoke_one, texts))

        if self.verbose:
            console.print(
                f"[green]embed_texts[/]: embedded {len(texts)} texts, "
                f"{self.total_tokens_in} cumulative input tokens."
            )
        return vectors

    def embed_chunks(self, chunks: Iterable[Chunk]) -> dict[str, list[float]]:
        """Embed a sequence of :class:`Chunk` objects.

        Uses :attr:`Chunk.context_text`; falls back to :attr:`Chunk.text`
        if context is empty (with a console warning). Chunks where both
        are empty are skipped with a warning.

        Returns a mapping ``chunk_id -> embedding`` in chunk-iteration
        order (insertion-ordered dict).
        """
        chunks_list = list(chunks)
        if not chunks_list:
            return {}

        # Materialize the (id, text) pairs in order, recording skips.
        ids_in_order: list[str] = []
        texts_in_order: list[str] = []
        for c in chunks_list:
            payload = c.context_text
            if not payload:
                if c.text:
                    console.print(
                        f"[yellow]warning[/]: chunk {c.id} has empty "
                        "context_text — falling back to raw text."
                    )
                    payload = c.text
                else:
                    console.print(
                        f"[yellow]warning[/]: chunk {c.id} has empty "
                        "context_text AND text — skipping."
                    )
                    continue
            ids_in_order.append(c.id)
            texts_in_order.append(payload)

        if not ids_in_order:
            return {}

        vectors = self.embed_texts(texts_in_order)
        return dict(zip(ids_in_order, vectors, strict=True))

    def stats(self) -> dict[str, int]:
        """Return a snapshot of the metric counters."""
        return {
            "total_tokens_in": self.total_tokens_in,
            "total_invocations": self.total_invocations,
            "total_errors": self.total_errors,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_request_body(self, text: str) -> dict[str, Any]:
        """Construct the JSON request body for one invocation.

        For Titan v2 we send ``inputText``, ``dimensions`` and
        ``normalize``. For non-Titan models we send only ``inputText``;
        the caller is responsible for supplying any model-specific knobs
        via a custom subclass.
        """
        body: dict[str, Any] = {"inputText": text}
        if self._is_titan:
            body["dimensions"] = self.dimensions
            body["normalize"] = self.normalize
        return body

    def _invoke_one(self, text: str) -> list[float]:
        """Single Bedrock ``invoke_model`` call. Updates metrics in place.

        Botocore handles ``ThrottlingException`` / ``ServiceUnavailable``
        via the adaptive retry config set on the client. ``ModelErrorException``
        (transient Bedrock internal error) is retried up to
        ``_MAX_INVOKE_ATTEMPTS`` times with exponential backoff via tenacity.
        Anything else is treated as a hard error.
        """
        body = self._build_request_body(text)
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
            self.total_errors += 1
            raise

        try:
            raw = response["body"].read()
            payload = json.loads(raw)
            vector = payload["embedding"]
            if not isinstance(vector, list):
                raise TypeError(
                    f"Bedrock returned non-list embedding: {type(vector).__name__}"
                )
            # Token count is informational; older models may omit it.
            tokens_in = payload.get("inputTextTokenCount", 0)
        except Exception:
            self.total_errors += 1
            raise

        self.total_tokens_in += int(tokens_in)
        self.total_invocations += 1
        # Cast every element to float — boto returns plain ints when the
        # quantized output happens to be exact (rare, but defensive).
        return [float(x) for x in vector]


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------


def embed_chunk_graph(
    graph: ChunkGraph,
    embedder: BedrockEmbedder | None = None,
) -> dict[str, list[float]]:
    """Embed every chunk in a :class:`ChunkGraph`.

    If ``embedder`` is None, a default :class:`BedrockEmbedder` is built
    using settings from :mod:`aws_rag.config`.
    """
    if embedder is None:
        embedder = BedrockEmbedder()
    return embedder.embed_chunks(graph.chunks.values())


def embed_texts(
    texts: Sequence[str],
    embedder: BedrockEmbedder | None = None,
) -> list[list[float]]:
    """Module-level convenience for one-shot embedding of a text list."""
    if embedder is None:
        embedder = BedrockEmbedder()
    return embedder.embed_texts(texts)
