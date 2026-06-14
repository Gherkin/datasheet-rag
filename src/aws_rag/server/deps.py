"""Server-side dependencies: the LocalBackend singleton and auth gates.

The backend is built directly as a ``LocalBackend`` (never via
``aws_rag.backend.get_backend``) so the server cannot recurse into a remote
backend even if ``RAG_SERVER_URL`` happens to be set in its environment.

The scope dependencies live here (rather than in :mod:`aws_rag.server.auth`)
so they can ``Depends(get_backend)`` — that keeps them honouring FastAPI's
``dependency_overrides`` in tests, and shares the route's backend instance.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from fastapi import Depends, Header, HTTPException, Request

from aws_rag.backend.local import LocalBackend
from aws_rag.config import get_settings
from aws_rag.server.auth import (  # noqa: F401  (Scope re-exported)
    KeyContext,
    Scope,
    _bearer,
    auth_enabled,
    resolve_context,
)


@lru_cache(maxsize=1)
def get_backend() -> LocalBackend:
    """Process-wide LocalBackend (lazy sqlite conn + embedder)."""
    return LocalBackend()


def require_scope(scope: Scope) -> Callable:
    """FastAPI dependency factory enforcing that the caller holds ``scope``.

    Stores the resolved :class:`KeyContext` on ``request.state.key`` for the
    audit log. In open mode (no credentials configured) every request passes
    as an anonymous all-scope context.
    """

    def dep(
        request: Request,
        be: LocalBackend = Depends(get_backend),
        authorization: str | None = Header(default=None),
    ) -> KeyContext:
        conn = be.conn
        settings = get_settings()
        token = _bearer(authorization)

        if not auth_enabled(conn, settings):
            ctx = KeyContext.anonymous()
            request.state.key = ctx
            return ctx

        ctx = resolve_context(conn, settings, token)
        if ctx is None:
            raise HTTPException(status_code=401, detail="missing or invalid credentials")
        if not ctx.allows(scope):
            raise HTTPException(
                status_code=403,
                detail=f"this credential lacks the '{scope.name.lower()}' scope",
            )
        request.state.key = ctx
        return ctx

    return dep


# Convenience dependencies for the common scopes.
require_read = Depends(require_scope(Scope.READ))
require_ingest = Depends(require_scope(Scope.INGEST))
require_admin = Depends(require_scope(Scope.ADMIN))
