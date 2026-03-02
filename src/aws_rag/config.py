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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()  # type: ignore[call-arg]
