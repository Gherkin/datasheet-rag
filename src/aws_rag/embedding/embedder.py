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
import math
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

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


class _TransientOllamaError(RuntimeError):
    """A retryable Ollama failure (HTTP 5xx other than the NaN bug)."""


def _is_transient_ollama_error(exc: BaseException) -> bool:
    return isinstance(exc, _TransientOllamaError)


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
    # Lazy import: this module also hosts the Ollama embedding path, so a
    # fully-local install (no `aws` extra) must be able to import it. boto3 is
    # only pulled in when a Bedrock embedding backend is actually used.
    try:
        import boto3
        from botocore.config import Config
    except ModuleNotFoundError as exc:  # pragma: no cover - guidance path
        raise ModuleNotFoundError(
            "The Bedrock embedding backend was selected but boto3 is not "
            "installed. Install the AWS extra:  pip install 'aws-rag[aws]'"
        ) from exc

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
# Shared chunk-embedding logic
# ---------------------------------------------------------------------------


class _ChunkEmbeddingMixin:
    """``embed_chunks`` shared by every embedder (Bedrock or local).

    Depends only on the concrete embedder providing ``embed_texts``.
    """

    verbose: bool

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError  # pragma: no cover - interface, overridden

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


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class BedrockEmbedder(_ChunkEmbeddingMixin):
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
# Local (Ollama) embedder
# ---------------------------------------------------------------------------


class OllamaEmbedder(_ChunkEmbeddingMixin):
    """Local text embedding via an Ollama server.

    Mirrors :class:`BedrockEmbedder`'s public surface (``embed_one``,
    ``embed_texts``, ``embed_chunks``, ``stats``) so it is interchangeable.
    Each text is one ``POST /api/embed`` call (truncate=True), batched
    concurrently. Use a robust model such as ``mxbai-embed-large`` here — not
    ``bge-m3``, whose llama.cpp F16 path emits NaN on some inputs (HTTP 500).

    The first returned vector's length is checked against
    ``settings.embedding_dimensions`` and a mismatch fails loud — that value
    is baked into the sqlite-vec table, so a silently-wrong dimension would
    only surface as an opaque insert error later.
    """

    _TIMEOUT_SECONDS = 120.0

    def __init__(
        self,
        *,
        model: str | None = None,
        dimensions: int | None = None,
        host: str | None = None,
        max_concurrency: int | None = None,
        verbose: bool = False,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.local_embedding_model
        self.dimensions = (
            dimensions if dimensions is not None else settings.embedding_dimensions
        )
        self.host = (host or settings.ollama_host).rstrip("/")
        self.max_concurrency = (
            max_concurrency if max_concurrency is not None else settings.embedding_batch_size
        )
        if self.max_concurrency < 1:
            raise ValueError(
                f"max_concurrency must be >= 1, got {self.max_concurrency}."
            )
        self.verbose = verbose

        self.total_tokens_in: int = 0  # Ollama embeddings don't report tokens.
        self.total_invocations: int = 0
        self.total_errors: int = 0

    def embed_one(self, text: str) -> list[float]:
        if not text:
            raise ValueError("embed_one() requires a non-empty string.")
        return self._invoke_one(text)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if len(texts) == 0:
            return []
        for i, t in enumerate(texts):
            if not t:
                raise ValueError(
                    f"embed_texts(): text at index {i} is empty — refusing "
                    "to embed an empty string."
                )
        workers = min(self.max_concurrency, len(texts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            vectors = list(pool.map(self._invoke_one, texts))
        if self.verbose:
            console.print(
                f"[green]embed_texts[/]: embedded {len(texts)} texts via "
                f"Ollama model {self.model!r}."
            )
        return vectors

    def stats(self) -> dict[str, int]:
        return {
            "total_tokens_in": self.total_tokens_in,
            "total_invocations": self.total_invocations,
            "total_errors": self.total_errors,
        }

    def _invoke_one(self, text: str) -> list[float]:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "The local embedding backend needs the 'httpx' package "
                "(part of the base install). Reinstall:  pip install aws-rag"
            ) from exc
        def _post() -> list[float] | None:
            # Use the newer /api/embed endpoint with truncate=True so inputs
            # longer than the model's context window are truncated rather than
            # rejected with a hard HTTP 500 (the legacy /api/embeddings has no
            # truncate option — mxbai-embed-large's 512-token window otherwise
            # fails on long context_text chunks).
            resp = httpx.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": text, "truncate": True},
                timeout=self._TIMEOUT_SECONDS,
            )
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("error", "") or resp.text
                except Exception:
                    detail = resp.text
                # bge-m3 on Ollama deterministically emits NaN embeddings for
                # certain token sequences (e.g. our "\n---\n" separator) -> 500.
                # Retrying won't help; fail with an actionable message rather
                # than an opaque 500 traceback.
                if "NaN" in detail:
                    raise ValueError(
                        f"Ollama model {self.model!r} produced a NaN embedding "
                        f"(HTTP {resp.status_code}: {detail!r}). Known "
                        "bge-m3/llama.cpp F16 bug — use mxbai-embed-large via "
                        "Ollama, or serve bge-m3 in-process (transformers)."
                    )
                if resp.status_code >= 500:
                    raise _TransientOllamaError(
                        f"Ollama HTTP {resp.status_code}: {detail!r}"
                    )
                resp.raise_for_status()
            # /api/embed returns {"embeddings": [[...]]} (one row per input).
            rows = resp.json().get("embeddings") or []
            return rows[0] if rows else None

        try:
            for attempt in Retrying(
                retry=retry_if_exception(_is_transient_ollama_error),
                stop=stop_after_attempt(_MAX_INVOKE_ATTEMPTS),
                wait=_INVOKE_WAIT,
                reraise=True,
            ):
                with attempt:
                    vector = _post()
        except Exception:
            self.total_errors += 1
            raise

        if not isinstance(vector, list) or not vector:
            self.total_errors += 1
            raise TypeError(
                f"Ollama returned no embedding for model {self.model!r} "
                "(is the model pulled and an embedding model?)."
            )
        if len(vector) != self.dimensions:
            self.total_errors += 1
            raise ValueError(
                f"Ollama model {self.model!r} returned a {len(vector)}-dim "
                f"vector but embedding_dimensions={self.dimensions}. Set "
                "RAG_EMBEDDING_DIMENSIONS to match the model (and re-create "
                "the DB), or pick a model with the configured dimension."
            )
        if any(math.isnan(x) or math.isinf(x) for x in vector):
            self.total_errors += 1
            raise ValueError(
                f"Ollama model {self.model!r} returned a non-finite (NaN/inf) "
                "embedding — refusing to store it. Try mxbai-embed-large."
            )

        self.total_invocations += 1
        return [float(x) for x in vector]


# ---------------------------------------------------------------------------
# Local (sentence-transformers) embedder — in-process, GPU via torch
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder(_ChunkEmbeddingMixin):
    """Local text embedding via sentence-transformers (HuggingFace + PyTorch).

    In-process (no server): loads a HuggingFace model with ``transformers`` and
    runs it on the GPU through ``torch``. Unlike the Ollama path this uses the
    full-precision reference implementation, so it is numerically robust —
    notably it does *not* hit llama.cpp's F16 NaN bug that ``bge-m3`` triggers
    on certain inputs, and it honours the model's full context window
    (truncating longer inputs rather than erroring).

    The model is loaded lazily on first use (the first call also downloads the
    weights from the HF Hub into ``~/.cache/huggingface`` if not cached).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        dimensions: int | None = None,
        normalize: bool | None = None,
        device: str | None = None,
        max_concurrency: int | None = None,
        verbose: bool = False,
    ) -> None:
        settings = get_settings()
        self.model_name = model or settings.local_embedding_model
        self.dimensions = (
            dimensions if dimensions is not None else settings.embedding_dimensions
        )
        self.normalize = normalize if normalize is not None else settings.embedding_normalize
        self.device = device
        # batch_size for the GPU forward pass (reuses the embedding batch knob).
        self.batch_size = (
            max_concurrency if max_concurrency is not None else settings.embedding_batch_size
        )
        self.verbose = verbose
        self._model: Any = None

        self.total_tokens_in: int = 0  # not tracked for the local path
        self.total_invocations: int = 0
        self.total_errors: int = 0

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "The huggingface embedding runtime needs the "
                    "'sentence-transformers' package. Install the extra:  "
                    "pip install 'aws-rag[local-hf]'"
                ) from exc
            device = self.device
            if device is None:
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            if self.verbose:
                console.print(
                    f"[cyan]loading[/] {self.model_name!r} on {device} "
                    "(first run downloads weights from the HF Hub)…"
                )
            self._model = SentenceTransformer(self.model_name, device=device)
            # Method renamed in newer sentence-transformers; support both.
            get_dim = getattr(self._model, "get_embedding_dimension", None) or (
                self._model.get_sentence_embedding_dimension
            )
            actual = get_dim()
            if actual != self.dimensions:
                raise ValueError(
                    f"sentence-transformers model {self.model_name!r} produces "
                    f"{actual}-dim vectors but embedding_dimensions={self.dimensions}. "
                    "Set RAG_EMBEDDING_DIMENSIONS to match (and re-create the DB)."
                )
        return self._model

    def embed_one(self, text: str) -> list[float]:
        if not text:
            raise ValueError("embed_one() requires a non-empty string.")
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if len(texts) == 0:
            return []
        for i, t in enumerate(texts):
            if not t:
                raise ValueError(
                    f"embed_texts(): text at index {i} is empty — refusing "
                    "to embed an empty string."
                )
        model = self._get_model()
        try:
            arr = model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception:
            self.total_errors += 1
            raise
        self.total_invocations += len(texts)
        if self.verbose:
            console.print(
                f"[green]embed_texts[/]: embedded {len(texts)} texts via "
                f"sentence-transformers {self.model_name!r}."
            )
        return [[float(x) for x in row] for row in arr]

    def stats(self) -> dict[str, int]:
        return {
            "total_tokens_in": self.total_tokens_in,
            "total_invocations": self.total_invocations,
            "total_errors": self.total_errors,
        }


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


# A structural type covering every embedder; callers only need these methods.
Embedder = BedrockEmbedder | OllamaEmbedder | SentenceTransformerEmbedder


def get_embedder(**kwargs: Any) -> Embedder:
    """Return the embedder for the configured backend.

    ``embedding_backend='bedrock'`` (default) → :class:`BedrockEmbedder`.
    ``embedding_backend='local'`` selects a local runtime via
    ``local_embedding_runtime``: ``'sentence-transformers'`` (default,
    :class:`SentenceTransformerEmbedder`) or ``'ollama'``
    (:class:`OllamaEmbedder`).

    ``kwargs`` (e.g. ``verbose=True``) are forwarded to whichever embedder is
    built; constructor kwargs the chosen backend doesn't accept are dropped so
    call sites can pass backend-agnostic flags.
    """
    settings = get_settings()
    if settings.embedding_backend == "local":
        if settings.local_embedding_runtime == "ollama":
            cls: type[Embedder] = OllamaEmbedder
        else:
            cls = SentenceTransformerEmbedder
    else:
        cls = BedrockEmbedder
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**filtered)


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------


def embed_chunk_graph(
    graph: ChunkGraph,
    embedder: Embedder | None = None,
) -> dict[str, list[float]]:
    """Embed every chunk in a :class:`ChunkGraph`.

    If ``embedder`` is None, one is built from settings via
    :func:`get_embedder` (Bedrock or local depending on ``embedding_backend``).
    """
    if embedder is None:
        embedder = get_embedder()
    return embedder.embed_chunks(graph.chunks.values())


def embed_texts(
    texts: Sequence[str],
    embedder: Embedder | None = None,
) -> list[list[float]]:
    """Module-level convenience for one-shot embedding of a text list."""
    if embedder is None:
        embedder = get_embedder()
    return embedder.embed_texts(texts)
