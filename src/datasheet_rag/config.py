"""Configuration management."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_rag_home() -> Path:
    """The local RAG store root: db + PDFs + figures + caches all live here.

    Resolved eagerly (outside the ``Settings`` instance) so it can seed the
    layered ``env_file`` tuple below — a global ``<rag_home>/config.env`` is
    itself a settings *source*, so its location can't depend on a fully
    constructed ``Settings``.
    """
    return Path(os.environ.get("RAG_HOME", str(Path.home() / ".rag"))).expanduser()


_RAG_HOME_DEFAULT = _default_rag_home()

# Storage paths that default to a fixed name under ``rag_home``. Kept as data
# so the field validator below can resolve any of them from its field name.
_STORAGE_DEFAULTS = {
    "sqlite_db_path": "rag.sqlite",
    "pdf_dir": "pdfs",
    "figures_dir": "figures",
    "output_dir": "cache",
}


class Settings(BaseSettings):
    """Application settings loaded from environment / env files.

    Env files are layered: the global ``<rag_home>/config.env`` (machine-wide
    defaults — AWS region/profile, S3 bucket, model IDs) loads first, then a
    cwd-local ``.env`` overrides it for one-off/per-checkout experiments.
    """

    model_config = SettingsConfigDict(
        env_file=(_RAG_HOME_DEFAULT / "config.env", ".env"),
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )

    # RAG home — local store root for db, PDFs, figures, and caches
    rag_home: Path = Field(
        default=_RAG_HOME_DEFAULT,
        alias="RAG_HOME",
        description=(
            "Root directory for the local RAG store: sqlite db, source PDFs, "
            "cropped figures, and ingestion caches all live under here by "
            "default (see sqlite_db_path / pdf_dir / figures_dir / output_dir). "
            "Defaults to ~/.rag — one shared store for every project, scoped "
            "by project_id metadata rather than split across directories."
        ),
    )

    # ------------------------------------------------------------------
    # Remote server — when set, the CLI and MCP talk to a shared RAG
    # server over HTTP instead of opening the local sqlite file. This is
    # how multiple developers share one corpus (server runs in Docker and
    # owns the db + embedder). Unset = local mode (the historical default).
    # ------------------------------------------------------------------
    server_url: str | None = Field(
        default=None,
        alias="RAG_SERVER_URL",
        description=(
            "Base URL of a remote RAG server (e.g. http://rag.internal:8080). "
            "When set, the CLI and MCP operate against it over HTTP instead of "
            "the local sqlite file — the server owns the database and the "
            "embedder. Unset = local mode (uses sqlite_db_path directly)."
        ),
    )
    server_token: str | None = Field(
        default=None,
        alias="RAG_SERVER_TOKEN",
        description=(
            "Bearer token sent by the client to the remote server (read or "
            "ingest key). On the server side this is the legacy alias for "
            "RAG_SERVER_READ_TOKEN (shared read token) when the latter is unset."
        ),
    )
    server_read_token: str | None = Field(
        default=None,
        alias="RAG_SERVER_READ_TOKEN",
        description=(
            "Server-side shared read token. When set, read/search endpoints "
            "require 'Authorization: Bearer <token>'. Ingest/admin always "
            "require a per-client API key regardless. Unset (and no API keys) "
            "= open mode (trusted-LAN dev)."
        ),
    )
    server_token_file: Path | None = Field(
        default=None,
        alias="RAG_SERVER_TOKEN_FILE",
        description=(
            "Optional path to a file holding the shared read token (e.g. a "
            "Docker/K8s secret mount). Read at startup; takes precedence over "
            "RAG_SERVER_READ_TOKEN/RAG_SERVER_TOKEN when present."
        ),
    )
    server_cors_origins: str | None = Field(
        default=None,
        alias="RAG_SERVER_CORS_ORIGINS",
        description=(
            "Comma-separated allowlist of browser origins permitted via CORS. "
            "Empty/unset = no cross-origin access (server-to-server CLI/MCP "
            "traffic is unaffected). Never use '*' with credentials."
        ),
    )
    server_timeout: float = Field(
        default=120.0,
        alias="RAG_SERVER_TIMEOUT",
        description=(
            "HTTP timeout in seconds for remote backend calls. Generous by "
            "default because ingest requests embed server-side and can be slow."
        ),
    )
    compute: Literal["server", "client"] = Field(
        default="server",
        description=(
            "Where model work runs in remote mode: 'server' (default — the "
            "server parses, embeds, describes figures and infers titles, so "
            "this client needs no models) or 'client' (this machine does all "
            "of it and the server is only a vector store). Set it to 'client' "
            "when the server host has no GPU and your workstation does. "
            "Ignored in local mode, where there is no server to defer to. "
            "Client-side embedding only makes sense if this machine's "
            "embedding model matches the one the corpus was built with — the "
            "client checks the server's /health and refuses on a dimension "
            "mismatch (see RAG_EMBEDDING_DIMENSIONS)."
        ),
    )

    # ------------------------------------------------------------------
    # MCP over HTTP — the server mounts the MCP tool surface at /mcp so
    # clients can point straight at it, with no local `rag-mcp` process
    # to install and configure (GH #39).
    # ------------------------------------------------------------------
    server_mcp_enabled: bool = Field(
        default=True,
        alias="RAG_SERVER_MCP_ENABLED",
        description=(
            "Mount the MCP endpoint at /mcp (and /mcp/<project_id>). Set false "
            "to serve the plain REST API only. The endpoint honours the same "
            "read token / API keys as every other read route."
        ),
    )
    mcp_allowed_hosts: str | None = Field(
        default=None,
        alias="RAG_MCP_ALLOWED_HOSTS",
        description=(
            "Comma-separated allowlist of Host header values accepted at /mcp "
            "(DNS-rebinding protection). Entries are exact ('rag.internal:8080') "
            "or port-wildcarded ('rag.internal:*'); there is no '*' catch-all. "
            "Unset = protection disabled, which is the sane default for a server "
            "reached by many names and through proxies — the bearer token, not "
            "the Host header, is what actually guards this endpoint."
        ),
    )
    mcp_allowed_origins: str | None = Field(
        default=None,
        alias="RAG_MCP_ALLOWED_ORIGINS",
        description=(
            "Comma-separated allowlist of browser Origin values accepted at "
            "/mcp. Only consulted when RAG_MCP_ALLOWED_HOSTS is set. Requests "
            "with no Origin header (every non-browser MCP client) always pass."
        ),
    )

    # AWS
    aws_region: str = Field(default="eu-west-1", alias="AWS_REGION")
    aws_profile: str | None = Field(default=None, alias="AWS_PROFILE")

    # ------------------------------------------------------------------
    # Model backend selection — per capability, each independently 'local'
    # (default; run on this machine, no per-call AWS cost) or 'bedrock' (AWS).
    #
    # Three capabilities, split so they can be mixed (e.g. the recommended
    # hybrid: local embeddings + local text, but Bedrock for figure/vision
    # since 7-8B local VLMs that fit a consumer GPU trail Claude on complex
    # diagrams):
    #   embedding_backend — text embeddings
    #   text_backend      — title inference, chunk summaries, eval generation
    #   vision_backend    — figure descriptions, table-structure repair
    #
    # NOTE: switching embedding_backend makes existing stored vectors
    # incompatible — re-embed the corpus on a DB created with the matching
    # embedding_dimensions (vectors from different models are not comparable).
    #
    # The "local" defaults use the in-process HuggingFace runtimes, which need:
    #   pip install 'datasheet-rag[local-hf]'
    # (The Ollama runtimes need only the base install — httpx ships there.)
    # ------------------------------------------------------------------
    embedding_backend: Literal["bedrock", "local"] = Field(
        default="local",
        description="Embeddings backend: 'local' (default) or 'bedrock' (Titan v2).",
    )
    text_backend: Literal["bedrock", "local"] = Field(
        default="local",
        description=(
            "Backend for text LLM calls (titling, summaries, eval): 'local' "
            "(default) or 'bedrock' (Claude)."
        ),
    )
    vision_backend: Literal["bedrock", "local"] = Field(
        default="local",
        description=(
            "Backend for vision LLM calls (figure descriptions, table-structure "
            "repair): 'local' (default) or 'bedrock' (Claude). On a small GPU, "
            "'bedrock' is recommended for quality (the hybrid setup)."
        ),
    )

    # Local runtimes — consulted only when the matching *_backend is 'local'.
    # Each local capability can run via 'huggingface' (in-process, PyTorch/CUDA;
    # full precision, robust, fits whatever VRAM allows and supports any HF
    # model) or 'ollama' (the Ollama server; lighter Python deps, GGUF/quantized,
    # but its memory estimation can force large vision models onto CPU).
    ollama_host: str = Field(
        default="http://localhost:11434",
        alias="RAG_OLLAMA_HOST",
        description="Base URL of the Ollama HTTP server.",
    )
    local_embedding_runtime: Literal["huggingface", "ollama"] = Field(
        default="huggingface",
        description=(
            "Local embeddings runtime. 'huggingface' (default) = "
            "sentence-transformers in-process (robust; the only reliable way to "
            "run bge-m3 — Ollama's F16 path emits NaN on some inputs). 'ollama' "
            "= Ollama server (use mxbai-embed-large there, not bge-m3)."
        ),
    )
    local_text_runtime: Literal["huggingface", "ollama"] = Field(
        default="ollama",
        description=(
            "Local text-LLM runtime. 'ollama' (default; light, qwen2.5:7b) or "
            "'huggingface' (in-process transformers causal LM)."
        ),
    )
    local_vision_runtime: Literal["huggingface", "ollama"] = Field(
        default="huggingface",
        description=(
            "Local vision-LLM runtime. 'huggingface' (default; in-process "
            "transformers VLM with precise VRAM control — the only way large "
            "VLMs fit a consumer GPU) or 'ollama'."
        ),
    )

    local_embedding_model: str = Field(
        default="BAAI/bge-m3",
        description=(
            "Local embedding model; output dim MUST equal embedding_dimensions "
            "(bge-m3 / mxbai-embed-large = 1024; nomic-embed-text = 768). Id "
            "format follows local_embedding_runtime: a HuggingFace repo id for "
            "'huggingface' (default 'BAAI/bge-m3') or an Ollama tag for 'ollama' "
            "(e.g. 'mxbai-embed-large' — not bge-m3, which hits the F16 bug)."
        ),
    )
    local_text_model: str = Field(
        default="qwen2.5:7b",
        description=(
            "Local text model. Id format follows local_text_runtime: an Ollama "
            "tag (default 'qwen2.5:7b') or a HuggingFace repo id (e.g. "
            "'Qwen/Qwen2.5-7B-Instruct')."
        ),
    )
    local_vision_model: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        description=(
            "Local vision model. Id format follows local_vision_runtime: a "
            "HuggingFace repo id (default 'Qwen/Qwen2.5-VL-3B-Instruct'; use "
            "...-7B/-32B/-72B on bigger GPUs) or an Ollama tag (e.g. "
            "'qwen2.5vl:3b'). NOTE: 7-8B VLMs that fit a 12GB GPU trail Claude "
            "on complex diagrams — Haiku-parity needs ~32B (~24GB VRAM)."
        ),
    )
    local_hf_load_4bit: bool = Field(
        default=False,
        description=(
            "Load huggingface vision/text models in 4-bit (bitsandbytes) to fit "
            "larger models on limited VRAM. Off by default (a 3B VLM fits 12GB "
            "in fp16); enable for 7B+ on a 12-24GB GPU."
        ),
    )

    # S3 (opt-in remote storage — see s3_bucket description)
    s3_bucket: str | None = Field(
        default=None,
        description=(
            "S3 bucket for optional remote storage. Required only when using "
            "the Textract backend (which reads PDFs from S3) or when "
            "explicitly uploading PDFs/figures with --upload flags. The "
            "default local-only workflow (Docling backend, no --upload) "
            "never touches S3."
        ),
        alias="RAG_S3_BUCKET",
    )
    s3_pdf_prefix: str = "raw-pdfs/"
    s3_textract_prefix: str = "textract-output/"

    # Textract
    textract_role_arn: str | None = Field(
        default=None,
        description="IAM role ARN for Textract async jobs (needed for S3 notifications)",
    )
    textract_features: list[str] = Field(
        default=["TABLES", "FORMS", "LAYOUT"],
        description="Textract AnalyzeDocument feature types",
    )

    # Local storage — all derived from rag_home unless explicitly overridden
    output_dir: Path = Field(
        default=None,  # resolved by _resolve_storage_path
        validate_default=True,
        description=(
            "Cache for intermediate ingestion artefacts (blocks/chunk-graph "
            "JSON, figure manifests) — safe to delete; regenerated on demand "
            "(or with --force). Defaults to ``<rag_home>/cache``."
        ),
    )
    sqlite_db_path: Path = Field(
        default=None,  # resolved by _resolve_storage_path
        validate_default=True,
        description=(
            "Path to the SQLite database file holding chunks + vectors. "
            "Defaults to ``<rag_home>/rag.sqlite`` — one shared db across "
            "every project. Override with --db / RAG_SQLITE_DB_PATH for "
            "tests or intentionally-isolated databases."
        ),
    )
    pdf_dir: Path = Field(
        default=None,  # resolved by _resolve_storage_path
        validate_default=True,
        description=(
            "Directory holding source PDFs, named ``<doc_id>.pdf`` (doc_id "
            "is a content hash, so the filename doubles as the lookup key — "
            "no scanning required). Defaults to ``<rag_home>/pdfs``."
        ),
    )
    figures_dir: Path = Field(
        default=None,  # resolved by _resolve_storage_path
        validate_default=True,
        description=(
            "Directory holding cropped figure images, one subdirectory per "
            "doc_id. Defaults to ``<rag_home>/figures``."
        ),
    )

    @field_validator(*_STORAGE_DEFAULTS, mode="before")
    @classmethod
    def _resolve_storage_path(cls, v: Any, info: ValidationInfo) -> Any:
        """Fill an unset storage path from ``rag_home`` so every command — CLI
        or MCP — shares one local store with zero configuration.

        Unset covers three spellings: absent, an explicit ``None``, and a blank
        string (Claude Desktop's .mcpb config substitutes ``${user_config.X}``
        with an empty string when X is left blank in the UI, which would
        otherwise turn ``figures_dir`` into ``Path('.')``).

        Resolving here rather than in an after-validator is what lets the four
        fields be declared as plain ``Path``: callers never have to re-prove
        that a path the model guarantees is set.
        """
        if v is not None and not (isinstance(v, str) and not v.strip()):
            return v
        # rag_home is declared first, so it is already in info.data — unless it
        # failed its own validation, in which case fall back to the default
        # and let pydantic report the real error.
        home = info.data.get("rag_home", _RAG_HOME_DEFAULT)
        assert info.field_name is not None
        return Path(home) / _STORAGE_DEFAULTS[info.field_name]

    # Bedrock embedding
    embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="Bedrock model ID for text embeddings.",
    )
    embedding_dimensions: int = Field(
        default=1024,
        description="Output dimension (Titan v2 supports 256/512/1024).",
    )
    embedding_normalize: bool = Field(
        default=True,
        description="Whether Titan should L2-normalize output vectors.",
    )
    embedding_batch_size: int = Field(
        default=16,
        description="Number of texts to embed concurrently per batch.",
    )

    # Docling table structure recognition
    table_structure_mode: Literal["fast", "accurate"] = Field(
        default="fast",
        description=(
            "Docling TableFormer mode for table structure recognition. "
            "'fast' (default) is roughly 2.4x faster and good enough for most "
            "tables, but is known to misparse complex multi-level headers — "
            "producing garbled/duplicated header cells (we detect and drop "
            "those at chunking time, see docling_parser._detect_garbled_header). "
            "'accurate' uses slower, more precise cell-boundary detection; "
            "empirically it still doesn't fully fix complex garbled headers "
            "(it can produce a *different* garbled parse), so treat it as a "
            "quality knob, not a guaranteed fix. Override per-ingest with "
            "--accurate-tables/--fast-tables on `rag ingest`."
        ),
    )

    # Figure description (Bedrock Claude vision)
    description_model_id: str = Field(
        default="eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        description=(
            "Bedrock model ID for figure description generation. "
            "Claude Haiku 4.5 is the cheapest vision-capable Claude on Bedrock. "
            "Point this at a Sonnet inference profile for higher-quality "
            "diagram interpretation at higher cost."
        ),
    )
    description_max_tokens: int = Field(
        default=400,
        description="Max output tokens per figure description.",
    )
    description_concurrency: int = Field(
        default=4,
        description="Concurrent vision API calls. Lower than embeddings — "
        "vision is slower and rate-limits are tighter.",
    )

    # Table-structure repair (Bedrock Claude vision — see
    # docs/table-structure-repair/{problem,plan}.md). Distinct from
    # description_model_id: re-deriving a table's row/column/header structure
    # from a crop + the existing (suspect-structure, trustworthy-text) cells
    # is a harder structural-reasoning task than describing a figure, so a
    # stronger model than figure-description Haiku is recommended — set this
    # explicitly to a vision-capable Claude on Bedrock (e.g. a Sonnet
    # inference-profile ARN for your account/region). Falls back to
    # description_model_id if unset, but that default is tuned for cheap
    # figure descriptions, not table-structure inference.
    table_repair_model_id: str | None = Field(
        default=None,
        description=(
            "Bedrock model ID for LLM-assisted table-structure repair "
            "(Stage 3 of docs/table-structure-repair/plan.md). Recommended: "
            "Claude Sonnet — stronger structural reasoning than the Haiku "
            "used for figure descriptions, which matters for re-deriving "
            "row/column/header layout from a table crop. Falls back to "
            "description_model_id if unset (not recommended for production "
            "use — that default is tuned for cheap figure descriptions)."
        ),
    )
    table_repair_max_tokens: int = Field(
        default=4000,
        description=(
            "Max output tokens per table-repair call. Higher than figure "
            "descriptions — the response is structured per-cell data "
            "(row/col/span/header for every origin cell), not 2-3 sentences."
        ),
    )
    table_repair_concurrency: int = Field(
        default=2,
        description="Concurrent table-repair API calls. Lower than figure "
        "description — larger payloads (image + full cell list), "
        "slower responses, and this is an opt-in maintenance "
        "path (rag repair tables), not an ingest hot-path knob.",
    )

    @field_validator(
        "default_project_id",
        "server_url",
        "server_token",
        "s3_bucket",
        mode="before",
    )
    @classmethod
    def _blank_string_means_unset(cls, v: Any) -> Any:
        """Treat empty/whitespace env values as 'use the field default'.

        Claude Desktop's .mcpb config substitutes ``${user_config.X}`` with an
        empty string when X is left blank in the UI, which would otherwise
        clobber sensible defaults. The storage paths need the same treatment
        and get it in :meth:`_resolve_storage_path`.
        """
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    def require_s3_bucket(self) -> str:
        """Return ``s3_bucket``, or explain that it has to be configured.

        S3 is opt-in (see the field description), so every code path that
        actually reaches S3 has to establish that a bucket exists. Doing it
        here gives one actionable message instead of boto3's
        ``ParamValidationError: Invalid type for parameter Bucket``.
        """
        if not self.s3_bucket:
            raise RuntimeError(
                "This operation needs S3, but no bucket is configured — set "
                "RAG_S3_BUCKET. S3 is only required for the Textract backend "
                "and for explicit --upload flags; the default local workflow "
                "(Docling, local storage) never touches it."
            )
        return self.s3_bucket

    def effective_read_token(self) -> str | None:
        """Resolve the shared read token: file mount > read_token > legacy token."""
        if self.server_token_file is not None:
            try:
                tok = Path(self.server_token_file).read_text().strip()
            except OSError:
                tok = ""
            if tok:
                return tok
        return self.server_read_token or self.server_token

    def client_side_compute(self) -> bool:
        """True when this process should run the models itself, not the server.

        Only ever true in remote mode: in local mode every model call already
        happens here, and ``compute`` has nothing to redirect.
        """
        return bool(self.server_url) and self.compute == "client"

    def cors_origins_list(self) -> list[str]:
        """Parse the CORS allowlist into a list (empty when unset)."""
        if not self.server_cors_origins:
            return []
        return [o.strip() for o in self.server_cors_origins.split(",") if o.strip()]

    def mcp_allowed_hosts_list(self) -> list[str]:
        """Parse the /mcp Host allowlist into a list (empty when unset)."""
        if not self.mcp_allowed_hosts:
            return []
        return [h.strip() for h in self.mcp_allowed_hosts.split(",") if h.strip()]

    def mcp_allowed_origins_list(self) -> list[str]:
        """Parse the /mcp Origin allowlist into a list (empty when unset)."""
        if not self.mcp_allowed_origins:
            return []
        return [o.strip() for o in self.mcp_allowed_origins.split(",") if o.strip()]

    # MCP server scoping
    default_project_id: str | None = Field(
        default=None,
        description=(
            "Default project_id the MCP server scopes searches to. "
            "Tool callers may override per-call. Set per project via env "
            "RAG_DEFAULT_PROJECT_ID in the Claude Code .mcp.json."
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
