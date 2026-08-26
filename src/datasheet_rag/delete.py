"""Full-document deletion: local filesystem artifacts and, when configured,
matching S3 objects.

Every artifact is keyed by ``doc_id`` alone (see :mod:`datasheet_rag.storage`,
:mod:`datasheet_rag.figures`), so paths and S3 prefixes are derived rather than
looked up. Called from :meth:`datasheet_rag.backend.local.LocalBackend.delete_doc`
after the sqlite rows are gone, so it runs both for the CLI's local backend
and for the FastAPI server (which also wraps a ``LocalBackend``).
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from datasheet_rag.config import get_settings

if TYPE_CHECKING:
    from mypy_boto3_s3.type_defs import ObjectIdentifierTypeDef


def purge_local_files(doc_id: str) -> list[str]:
    """Remove every local on-disk artifact for *doc_id*.

    Covers the source PDF, the figure-crop directory, and the pipeline's
    cache artifacts (blocks/outline/chunks JSON, rendered-page cache).
    Returns the paths actually removed, for CLI reporting.
    """
    settings = get_settings()
    removed: list[str] = []

    pdf_path = settings.pdf_dir / f"{doc_id}.pdf"
    if pdf_path.is_file():
        pdf_path.unlink()
        removed.append(str(pdf_path))

    figures_dir = settings.figures_dir / doc_id
    if figures_dir.is_dir():
        shutil.rmtree(figures_dir)
        removed.append(str(figures_dir))

    for suffix in ("_blocks.json", "_outline.json", "_chunks.json"):
        cache_path = settings.output_dir / f"{doc_id}{suffix}"
        if cache_path.is_file():
            cache_path.unlink()
            removed.append(str(cache_path))

    render_cache_dir = settings.rag_home / "page_render_cache" / doc_id
    if render_cache_dir.is_dir():
        shutil.rmtree(render_cache_dir)
        removed.append(str(render_cache_dir))

    return removed


def purge_s3_objects(doc_id: str) -> int:
    """Delete every S3 object for *doc_id* (raw PDF + figure uploads).

    No-ops (returns 0) when no bucket is configured — S3 is opt-in, see
    ``Settings.s3_bucket``. Returns the number of objects deleted.
    """
    settings = get_settings()
    if not settings.s3_bucket:
        return 0

    from datasheet_rag.aws import s3_client

    client = s3_client()
    deleted = 0
    for prefix in (f"{settings.s3_pdf_prefix}{doc_id}/", f"figures/{doc_id}/"):
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix):
            keys: list[ObjectIdentifierTypeDef] = [
                {"Key": obj["Key"]} for obj in page.get("Contents", [])
            ]
            if not keys:
                continue
            client.delete_objects(Bucket=settings.s3_bucket, Delete={"Objects": keys})
            deleted += len(keys)

    return deleted
