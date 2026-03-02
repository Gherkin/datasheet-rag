"""S3 operations for PDF upload and Textract output retrieval."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from aws_rag.aws import s3_client
from aws_rag.config import get_settings

if TYPE_CHECKING:
    pass

console = Console()


def _content_hash(path: Path) -> str:
    """Return SHA-256 hex digest of a file (used as de-duplication key)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_pdf(pdf_path: Path, *, doc_id: str | None = None) -> tuple[str, str]:
    """Upload a PDF to S3 and return (doc_id, s3_key).

    If *doc_id* is not supplied, a content-hash is used so the same file
    is never uploaded twice.
    """
    settings = get_settings()
    client = s3_client()

    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    if doc_id is None:
        doc_id = _content_hash(pdf_path)

    s3_key = f"{settings.s3_pdf_prefix}{doc_id}/{pdf_path.name}"

    # Check if already uploaded
    try:
        client.head_object(Bucket=settings.s3_bucket, Key=s3_key)
        console.print(f"[yellow]Already exists:[/] s3://{settings.s3_bucket}/{s3_key}")
        return doc_id, s3_key
    except client.exceptions.ClientError:
        pass

    console.print(f"[blue]Uploading[/] {pdf_path.name} → s3://{settings.s3_bucket}/{s3_key}")
    client.upload_file(
        Filename=str(pdf_path),
        Bucket=settings.s3_bucket,
        Key=s3_key,
        ExtraArgs={"ContentType": "application/pdf"},
    )
    console.print(f"[green]Uploaded[/] doc_id={doc_id}")
    return doc_id, s3_key


def download_json(s3_key: str, dest: Path) -> Path:
    """Download a Textract JSON result from S3 to a local file."""
    settings = get_settings()
    client = s3_client()
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(Bucket=settings.s3_bucket, Key=s3_key, Filename=str(dest))
    return dest


def list_documents() -> list[dict[str, str]]:
    """List all uploaded document prefixes in the PDF folder."""
    settings = get_settings()
    client = s3_client()
    paginator = client.get_paginator("list_objects_v2")
    docs: dict[str, str] = {}  # doc_id → s3_key of first object

    for page in paginator.paginate(
        Bucket=settings.s3_bucket,
        Prefix=settings.s3_pdf_prefix,
        Delimiter="/",
    ):
        for prefix_info in page.get("CommonPrefixes", []):
            prefix = prefix_info["Prefix"]
            doc_id = prefix.rstrip("/").rsplit("/", 1)[-1]
            docs[doc_id] = prefix

    return [{"doc_id": k, "prefix": v} for k, v in sorted(docs.items())]
