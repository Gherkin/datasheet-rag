"""Configuration management."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )

    # AWS
    aws_region: str = Field(default="eu-west-1", alias="AWS_REGION")
    aws_profile: str | None = Field(default=None, alias="AWS_PROFILE")

    # S3
    s3_bucket: str = Field(description="S3 bucket for PDF storage and Textract output")
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

    # Local output (for debugging / offline work)
    output_dir: Path = Path("output")

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

    # SQLite store
    sqlite_db_path: Path = Field(
        default=Path("output/rag.sqlite"),
        description="Path to the SQLite database file holding chunks + vectors.",
    )

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
