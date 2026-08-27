"""Resolve and render source PDFs by ``doc_id``.

``doc_id`` is the SHA-256 content hash assigned at ingestion (see
``datasheet_rag.storage.save_pdf_locally`` / ``upload_pdf``). This module turns
that id back into PDF bytes — trying the in-process cache, then the local
PDF store, then S3 — and renders individual pages to PNG via
poppler/pdf2image.

It is intentionally standalone (no MCP SDK imports) so the eval
review tool can depend on it without dragging in the server. The MCP
server keeps its own equivalent; the duplication is small and keeps the
two import graphs independent.
"""

from __future__ import annotations

from threading import Lock

from datasheet_rag.config import get_settings

_pdf_cache_lock = Lock()
_pdf_cache: dict[str, bytes] = {}


def load_pdf_bytes(doc_id: str) -> bytes:
    """Return raw PDF bytes for ``doc_id`` (cache → local store → S3).

    ``doc_id`` is the content hash, and ingestion saves the source PDF as
    ``<pdf_dir>/<doc_id>.pdf`` (see ``storage.save_pdf_locally``), so the
    local lookup is a direct path join — no scanning or re-hashing needed.
    S3 (``s3_pdf_prefix/{doc_id}/*.pdf``) is a fallback for stores that were
    explicitly uploaded remotely.
    """
    with _pdf_cache_lock:
        cached = _pdf_cache.get(doc_id)
    if cached is not None:
        return cached

    settings = get_settings()

    local_path = settings.pdf_dir / f"{doc_id}.pdf"
    if local_path.is_file():
        body = local_path.read_bytes()
        with _pdf_cache_lock:
            _pdf_cache[doc_id] = body
        return body

    if settings.s3_bucket:
        try:
            from datasheet_rag.aws import s3_client

            client = s3_client()
            resp = client.list_objects_v2(
                Bucket=settings.s3_bucket,
                Prefix=f"{settings.s3_pdf_prefix}{doc_id}/",
            )
            for obj in resp.get("Contents", []):
                if obj["Key"].lower().endswith(".pdf"):
                    body = client.get_object(Bucket=settings.s3_bucket, Key=obj["Key"])[
                        "Body"
                    ].read()
                    with _pdf_cache_lock:
                        _pdf_cache[doc_id] = body
                    return body
        except Exception:
            pass

    raise FileNotFoundError(
        f"PDF not found for doc_id={doc_id!r}. Expected it at {local_path} "
        "(or in S3 if RAG_S3_BUCKET is configured)."
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
