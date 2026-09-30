"""Shared AWS client factory."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from datasheet_rag.config import get_settings

if TYPE_CHECKING:
    from boto3.session import Session
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_textract import TextractClient


@lru_cache(maxsize=1)
def _session() -> Session:
    # Imported lazily so modules that merely reference AWS clients can be
    # imported without boto3 installed (the `aws` extra). boto3 is only
    # required once an AWS-backed client is actually built at runtime.
    try:
        import boto3
    except ModuleNotFoundError as exc:  # pragma: no cover - guidance path
        raise ModuleNotFoundError(
            "An AWS backend (Bedrock/Textract/S3) was selected but boto3 is not "
            "installed. Install the AWS extra:  pip install 'datasheet-rag[aws]'"
        ) from exc

    settings = get_settings()
    return boto3.Session(
        region_name=settings.aws_region,
        profile_name=settings.aws_profile or None,
    )


def bedrock_runtime_client(
    *,
    region: str | None = None,
    profile: str | None = None,
    read_timeout: int = 60,
    max_attempts: int = 5,
) -> BedrockRuntimeClient:
    """Build a configured ``bedrock-runtime`` client.

    Adaptive retries handle ThrottlingException with token-bucket backoff.
    60 s is generous for an embedding or a short completion; an agent turn
    that writes thousands of tokens needs a longer ``read_timeout``.
    """
    # Lazy import: embedding and local-model modules import this, and a
    # fully-local install (no `aws` extra) must be able to import them.
    try:
        import boto3
        from botocore.config import Config
    except ModuleNotFoundError as exc:  # pragma: no cover - guidance path
        raise ModuleNotFoundError(
            "A Bedrock backend was selected but boto3 is not installed. "
            "Install the AWS extra:  pip install 'datasheet-rag[aws]'"
        ) from exc

    settings = get_settings()
    effective_profile = profile if profile is not None else settings.aws_profile
    session = boto3.Session(
        region_name=region or settings.aws_region,
        profile_name=effective_profile or None,
    )
    config = Config(
        connect_timeout=60,
        read_timeout=read_timeout,
        retries={"max_attempts": max_attempts, "mode": "adaptive"},
    )
    return session.client("bedrock-runtime", config=config)


def s3_client() -> S3Client:
    return _session().client("s3")


def textract_client() -> TextractClient:
    return _session().client("textract")
