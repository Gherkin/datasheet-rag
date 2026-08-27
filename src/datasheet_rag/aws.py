"""Shared AWS client factory."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from datasheet_rag.config import get_settings

if TYPE_CHECKING:
    from boto3.session import Session
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


def s3_client() -> S3Client:
    return _session().client("s3")


def textract_client() -> TextractClient:
    return _session().client("textract")
