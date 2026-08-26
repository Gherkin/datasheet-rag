"""Mount the MCP tool surface inside the RAG server's FastAPI app.

Without this, using the tools against a shared server means running a second
process locally — ``rag-mcp`` over stdio, pointed at ``RAG_SERVER_URL``, so
every tool call is an MCP round trip to a process that immediately makes an
HTTP round trip to the server. Mounting the same tools at ``/mcp`` removes
that hop and the install that goes with it: a client configures a URL and a
token and is done (GH #39).

Three things have to be arranged around ``FastMCP`` for this to work:

* **Routing.** ``streamable_http_app()`` already serves at its own
  ``streamable_http_path``. Building it with that set to ``/`` and mounting
  the result at ``/mcp`` puts the endpoint where clients expect it, instead of
  the ``/mcp/mcp`` you get by mounting a default-configured app.
* **Lifespan.** The session manager must be running before it can take a
  request, so :func:`build_app` folds :meth:`McpMount.lifespan` into the
  FastAPI app's own.
* **Auth and scoping.** A mounted ASGI app is opaque to FastAPI's dependency
  system, so ``Depends(require_read)`` cannot reach it. :class:`_Gate` wraps
  the app and does both jobs directly on the ASGI scope: it enforces the same
  read scope every other read route requires, and it turns the
  ``/mcp/<project_id>`` path segment into the project the tool calls resolve
  against.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from datasheet_rag.backend.base import RagBackend
from datasheet_rag.config import get_settings
from datasheet_rag.mcp import server as mcp_server
from datasheet_rag.server.auth import Scope, _bearer, auth_enabled, resolve_context

logger = logging.getLogger("datasheet_rag.server")

# Header alternative to the /mcp/<project_id> path segment, for clients whose
# config format pins the URL but lets you add headers.
PROJECT_HEADER = b"x-rag-project"


def _header(scope: dict[str, Any], name: bytes) -> str | None:
    headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
    for key, value in headers:
        if key.lower() == name:
            return value.decode("latin-1")
    return None


async def _send_json(
    send: Any,
    status: int,
    body: dict[str, Any],
    headers: tuple[tuple[bytes, bytes], ...] = (),
) -> None:
    raw = json.dumps(body).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(raw)).encode()),
            *headers,
        ],
    })
    await send({"type": "http.response.body", "body": raw})


class _Gate:
    """ASGI wrapper enforcing read scope and extracting the request's project.

    Sits between the ``/mcp`` mount and the FastMCP app. Everything it needs
    is on the raw scope, which is what makes it usable where a FastAPI
    dependency is not.
    """

    def __init__(self, app: Any, backend: RagBackend) -> None:
        self.app = app
        self.backend = backend

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not self._authorized(scope):
            await _send_json(
                send,
                401,
                {"detail": "missing or invalid credentials"},
                # A bare challenge: this endpoint takes a static bearer token,
                # so point clients at that rather than at an OAuth flow the
                # server does not implement.
                ((b"www-authenticate", b'Bearer realm="datasheet-rag"'),),
            )
            return

        project, scope = self._scope_project(scope)

        # Both are request-scoped, not process-scoped: this process also
        # answers stdio-shaped assumptions elsewhere (and the tests do), so
        # nothing here may outlive the request that set it.
        scoped_project = mcp_server.request_project.set(project)
        scoped_client = mcp_server.request_local_client.set(False)
        try:
            await self.app(scope, receive, send)
        finally:
            mcp_server.request_project.reset(scoped_project)
            mcp_server.request_local_client.reset(scoped_client)

    @staticmethod
    def _scope_project(scope: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """Split the project off the URL and rewrite the path to the endpoint.

        Starlette's ``Mount`` does not trim the matched prefix out of
        ``scope["path"]``; it records the prefix in ``root_path`` and leaves
        the full path in place for the sub-app to resolve against. So the
        remainder after ``root_path`` is what carries the project — empty or
        ``/`` for the unscoped endpoint, ``/<project_id>`` for a scoped one —
        and the path has to be rewritten back to the bare mount point for the
        single FastMCP route to match either way.
        """
        root = scope.get("root_path", "")
        path = scope["path"]
        remainder = path[len(root) :] if root and path.startswith(root) else path
        project = remainder.strip("/") or _header(scope, PROJECT_HEADER) or None
        endpoint = root + "/"
        return project, {**scope, "path": endpoint, "raw_path": endpoint.encode()}

    def _authorized(self, scope: dict[str, Any]) -> bool:
        """Same gate as ``require_scope(Scope.READ)``, against the ASGI scope.

        Read scope, not ingest: the mounted tools only search and read. Open
        mode (no read token, no API keys) stays open here too, so a
        trusted-LAN server behaves the same on /mcp as on /search.
        """
        settings = get_settings()
        conn = self.backend.conn  # type: ignore[attr-defined]
        if not auth_enabled(conn, settings):
            return True
        ctx = resolve_context(conn, settings, _bearer(_header(scope, b"authorization")))
        return ctx is not None and ctx.allows(Scope.READ)


@dataclass
class McpMount:
    """A built MCP endpoint: the ASGI app to mount plus its lifespan."""

    app: Any
    server: Any

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Run the streamable-HTTP session manager for the app's lifetime."""
        async with self.server.session_manager.run():
            yield


def build_mcp_mount(backend: RagBackend) -> McpMount:
    """Build the MCP endpoint to mount at ``/mcp``, bound to ``backend``.

    The backend is passed in rather than resolved inside the tools, because
    ``datasheet_rag.backend.get_backend`` returns a ``RemoteBackend`` whenever
    ``RAG_SERVER_URL`` is set — which, in a container that also runs a client
    config, would have the server calling itself.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    settings = get_settings()
    allowed_hosts = settings.mcp_allowed_hosts_list()
    security = TransportSecuritySettings(
        # Host-header checking defends a desktop-local server against DNS
        # rebinding by a page in the user's browser. This server is reached
        # by many names and usually through a proxy, and is guarded by a
        # bearer token, so the check is opt-in: enumerate the hosts you serve
        # in RAG_MCP_ALLOWED_HOSTS to turn it on. There is no '*' wildcard —
        # an empty allowlist rejects everything, hence the explicit disable.
        enable_dns_rebinding_protection=bool(allowed_hosts),
        allowed_hosts=allowed_hosts,
        allowed_origins=settings.mcp_allowed_origins_list(),
    )

    server = mcp_server.build_server(
        backend,
        local_client=False,
        # Stateless: no session state survives a call, so any worker can serve
        # any request and a dropped client leaves nothing behind. json_response
        # skips the SSE framing for the same reason — there is nothing to
        # stream back.
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        transport_security=security,
    )
    return McpMount(app=_Gate(server.streamable_http_app(), backend), server=server)
