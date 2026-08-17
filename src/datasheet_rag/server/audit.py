"""Ingest-path audit trail.

Every cost-inducing or destructive action (ingest, figure description, title
inference, delete) records one row in the ``audit_log`` table *and* emits one
structured JSON log line — so the trail survives even if the DB is lost.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import Request

from datasheet_rag.backend.local import LocalBackend
from datasheet_rag.store import record_audit as _store_record_audit

logger = logging.getLogger("datasheet_rag.audit")


def _client_ip(request: Request) -> str | None:
    # The reverse proxy sets X-Forwarded-For; fall back to the socket peer.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


def _key_label(request: Request) -> str | None:
    ctx = getattr(request.state, "key", None)
    return getattr(ctx, "label", None) if ctx is not None else None


def audit(
    request: Request,
    backend: LocalBackend,
    *,
    action: str,
    status: str,
    doc_id: str | None = None,
    project_id: str | None = None,
    detail: dict | None = None,
    error: str | None = None,
) -> None:
    """Write an audit row + structured log line. Never raises."""
    label = _key_label(request)
    ip = _client_ip(request)
    record: dict[str, Any] = {
        "action": action,
        "status": status,
        "key_label": label,
        "client_ip": ip,
        "doc_id": doc_id,
        "project_id": project_id,
        "detail": detail,
        "error": error,
    }
    try:
        logger.info("audit %s", json.dumps(record))
    except Exception:  # logging must never break a request
        pass
    try:
        with backend.write_lock:
            _store_record_audit(
                backend.conn,
                action=action,
                status=status,
                key_label=label,
                client_ip=ip,
                doc_id=doc_id,
                project_id=project_id,
                detail=detail,
                error=error,
            )
    except Exception:  # pragma: no cover - audit is best-effort
        logger.exception("failed to persist audit row")
