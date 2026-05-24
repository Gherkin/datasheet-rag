"""Resolve and render source PDFs by ``doc_id``.

``doc_id`` is the SHA-256 content hash assigned at upload (see
``aws_rag.storage.upload_pdf``). This module turns that id back into PDF
bytes — trying the in-process cache, then S3, then a local filesystem
scan — and renders individual pages to PNG via poppler/pdf2image.

It is intentionally standalone (no MCP / FastMCP imports) so the eval
review tool can depend on it without dragging in the server. The MCP
server keeps its own equivalent; the duplication is small and keeps the
two import graphs independent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock

from aws_rag.config import get_settings

_pdf_cache_lock = Lock()
_pdf_cache: dict[str, bytes] = {}


def _project_root() -> Path:
    """Project root — the parent of the ``output/`` dir holding rag.sqlite."""
    settings = get_settings()
    return Path(settings.sqlite_db_path).resolve().parent.parent


def load_pdf_bytes(doc_id: str) -> bytes:
    """Return raw PDF bytes for ``doc_id`` (cache → S3 → local hash scan)."""
    with _pdf_cache_lock:
        cached = _pdf_cache.get(doc_id)
    if cached is not None:
        return cached

    settings = get_settings()

    # ── S3: s3_pdf_prefix/{doc_id}/*.pdf ──────────────────────────────────
    try:
        from aws_rag.aws import s3_client

        client = s3_client()
        resp = client.list_objects_v2(
            Bucket=settings.s3_bucket,
            Prefix=f"{settings.s3_pdf_prefix}{doc_id}/",
        )
        for obj in resp.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                body = client.get_object(
                    Bucket=settings.s3_bucket, Key=obj["Key"]
                )["Body"].read()
                with _pdf_cache_lock:
                    _pdf_cache[doc_id] = body
                return body
    except Exception:
        pass  # fall through to local scan

    # ── Local filesystem: scan for a .pdf whose content hash matches ──────
    try:
        for pdf_path in _project_root().rglob("*.pdf"):
            try:
                h = hashlib.sha256()
                with open(pdf_path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() == doc_id:
                    body = pdf_path.read_bytes()
                    with _pdf_cache_lock:
                        _pdf_cache[doc_id] = body
                    return body
            except OSError:
                continue
    except Exception:
        pass

    raise FileNotFoundError(
        f"PDF not found for doc_id={doc_id!r}. Check that it was uploaded to "
        "S3 or that the original PDF is accessible under the project directory."
    )


def render_page_png(doc_id: str, page: int, *, dpi: int = 150) -> bytes:
    """Render a single 1-based ``page`` of ``doc_id`` to PNG bytes."""
    import io

    from pdf2image import convert_from_bytes

    pdf_bytes = load_pdf_bytes(doc_id)
    images = convert_from_bytes(pdf_bytes, first_page=page, last_page=page, dpi=dpi)
    if not images:
        raise ValueError(f"page {page} not found in document {doc_id!r}")
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()
