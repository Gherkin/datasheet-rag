"""Configuration management."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
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

    # AWS
    aws_region: str = Field(default="eu-west-1", alias="AWS_REGION")
    aws_profile: str | None = Field(default=None, alias="AWS_PROFILE")

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
    output_dir: Path | None = Field(
        default=None,
        description=(
            "Cache for intermediate ingestion artefacts (blocks/chunk-graph "
            "JSON, figure manifests) — safe to delete; regenerated on demand "
            "(or with --force). Defaults to ``<rag_home>/cache``."
        ),
    )
    sqlite_db_path: Path | None = Field(
        default=None,
        description=(
            "Path to the SQLite database file holding chunks + vectors. "
            "Defaults to ``<rag_home>/rag.sqlite`` — one shared db across "
            "every project. Override with --db / RAG_SQLITE_DB_PATH for "
            "tests or intentionally-isolated databases."
        ),
    )
    pdf_dir: Path | None = Field(
        default=None,
        description=(
            "Directory holding source PDFs, named ``<doc_id>.pdf`` (doc_id "
            "is a content hash, so the filename doubles as the lookup key — "
            "no scanning required). Defaults to ``<rag_home>/pdfs``."
        ),
    )
    figures_dir: Path | None = Field(
        default=None,
        description=(
            "Directory holding cropped figure images, one subdirectory per "
            "doc_id. Defaults to ``<rag_home>/figures``."
        ),
    )

    @model_validator(mode="after")
    def _derive_storage_paths(self) -> Settings:
        """Fill unset storage paths from ``rag_home`` so every command — CLI
        or MCP — shares one local store with zero configuration."""
        if self.sqlite_db_path is None:
            self.sqlite_db_path = self.rag_home / "rag.sqlite"
        if self.pdf_dir is None:
            self.pdf_dir = self.rag_home / "pdfs"
        if self.figures_dir is None:
            self.figures_dir = self.rag_home / "figures"
        if self.output_dir is None:
            self.output_dir = self.rag_home / "cache"
        return self

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
            "Claude 3.5 Haiku is the cheapest vision-capable Claude on Bedrock. "
            "Use anthropic.claud-3-5-sonnet for higher-quality "
            "diagram interpretation at ~10× cost."
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

    @field_validator(
        "figures_dir", "pdf_dir", "output_dir", "sqlite_db_path", "default_project_id",
        mode="before",
    )
    @classmethod
    def _blank_string_means_unset(cls, v: Any) -> Any:
        """Treat empty/whitespace env values as 'use the field default'.

        Claude Desktop's .mcpb config substitutes ``${user_config.X}`` with
        an empty string when X is left blank in the UI, which would
        otherwise clobber sensible defaults (e.g. turn ``figures_dir`` into
        ``Path('.')``).
        """
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

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
    return Settings()  # type: ignore[call-arg]
