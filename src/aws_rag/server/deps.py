"""Server-side dependencies: the LocalBackend singleton and auth.

The backend is built directly as a ``LocalBackend`` (never via
``aws_rag.backend.get_backend``) so the server cannot recurse into a remote
backend even if ``RAG_SERVER_URL`` happens to be set in its environment.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Header, HTTPException

from aws_rag.backend.local import LocalBackend
from aws_rag.config import get_settings


@lru_cache(maxsize=1)
def get_backend() -> LocalBackend:
    """Process-wide LocalBackend (lazy sqlite conn + embedder)."""
    return LocalBackend()


def require_token(authorization: str | None = Header(default=None)) -> None:
    """Optional bearer-token gate.

    No-op unless ``RAG_SERVER_TOKEN`` is set on the server. When set, every
    request must carry ``Authorization: Bearer <token>``.

    TODO(auth): stage 1 ships this single static shared token only. Before
    exposing the server beyond a trusted LAN, replace this with per-user
    credentials / proper auth (and add TLS termination in front).
    """
    expected = get_settings().server_token
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")
