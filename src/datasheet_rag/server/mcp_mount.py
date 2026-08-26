"""Serve the MCP tool surface from the RAG server's FastAPI app.

Without this, using the tools against a shared server means running a second
process locally — ``rag-mcp`` over stdio, pointed at ``RAG_SERVER_URL``, so
every tool call is an MCP round trip to a process that immediately makes an
HTTP round trip to the server. Serving the same tools at ``/mcp`` removes that
hop and the install that goes with it: a client configures a URL and a token
and is done (GH #39).

Three things have to be arranged around ``FastMCP`` for this to work:

* **Routing.** ``streamable_http_app()`` serves at its own
  ``streamable_http_path``, so it is built with that set to ``/`` and reached
  through :class:`_Dispatch` rather than through a route of its own.
* **Lifespan.** The session manager must be running before it can take a
  request, so :func:`build_app` folds :meth:`McpMount.lifespan` into the
  FastAPI app's own.
* **Auth and scoping.** FastAPI's dependency system cannot reach inside
  another ASGI app, so ``Depends(require_read)`` is no use here.
  :class:`_Dispatch` does both jobs directly on the ASGI scope: it enforces
  the same read scope every other read route requires, and it turns the
  ``/mcp/<project_id>`` path segment into the project the tool calls resolve
  against.

Dispatching in middleware rather than ``app.mount("/mcp", ...)`` is what makes
a bare ``/mcp`` work. A Starlette mount matches only *below* its prefix, so the
bare path falls through to the router's trailing-slash redirect — and since
that is the URL most clients get configured with, every single request would
pay a 307 and a re-POST of its body.
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

#: Where the endpoint lives. A project id may follow as one more segment.
MCP_PREFIX = "/mcp"

#: Header alternative to the ``/mcp/<project_id>`` path segment, for clients
#: whose config format pins the URL but lets you add headers.
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


class _Dispatch:
    """ASGI middleware routing ``/mcp[/<project_id>]`` to the MCP app.

    Everything it needs is on the raw scope, which is what lets it stand in
    for the FastAPI dependencies it cannot use. Anything not addressed to the
    endpoint passes straight through to the REST app.
    """

    def __init__(self, app: Any, mount: McpMount) -> None:
        self.app = app
        self.mount = mount

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        root = scope.get("root_path", "")
        path = scope["path"]
        rel = path[len(root) :] if root and path.startswith(root) else path
        if rel != MCP_PREFIX and not rel.startswith(MCP_PREFIX + "/"):
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

        # Whatever follows the prefix is the project — nothing, a bare slash,
        # or "/<project_id>". The MCP app has a single route, so the path it
        # sees is rewritten down to that regardless.
        project = rel[len(MCP_PREFIX) :].strip("/") or _header(scope, PROJECT_HEADER) or None
        endpoint = root + "/"
        scope = {**scope, "path": endpoint, "raw_path": endpoint.encode()}

        # Both of these are request-scoped, not process-scoped: one process
        # serves every project and every client, so nothing set here may
        # outlive the request that set it.
        scoped_project = mcp_server.request_project.set(project)
        scoped_client = mcp_server.request_local_client.set(False)
        try:
            await self.mount.asgi(scope, receive, send)
        finally:
            mcp_server.request_project.reset(scoped_project)
            mcp_server.request_local_client.reset(scoped_client)

    def _authorized(self, scope: dict[str, Any]) -> bool:
        """Same gate as ``require_scope(Scope.READ)``, against the ASGI scope.

        Read scope, not ingest: the tools served here only search and read.
        Open mode (no read token, no API keys) stays open, so a trusted-LAN
        server behaves the same on /mcp as on /search.
        """
        settings = get_settings()
        conn = self.mount.backend.conn  # type: ignore[attr-defined]
        if not auth_enabled(conn, settings):
            return True
        ctx = resolve_context(conn, settings, _bearer(_header(scope, b"authorization")))
        return ctx is not None and ctx.allows(Scope.READ)


@dataclass
class McpMount:
    """A built MCP endpoint: the ASGI app, its lifespan, and how to install it."""

    server: Any
    asgi: Any
    backend: RagBackend

    def install(self, app: Any) -> None:
        """Put the dispatcher in front of ``app``'s routing."""
        app.add_middleware(_Dispatch, mount=self)

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Run the streamable-HTTP session manager for the app's lifetime."""
        async with self.server.session_manager.run():
            yield


def build_mcp_mount(backend: RagBackend) -> McpMount:
    """Build the MCP endpoint, bound to ``backend``.

    The backend is passed in rather than resolved inside the tools, because
    ``datasheet_rag.backend.get_backend`` returns a ``RemoteBackend`` whenever
    ``RAG_SERVER_URL`` is set — which, in a container that also carries a
    client config, would have the server calling itself.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    settings = get_settings()
    allowed_hosts = settings.mcp_allowed_hosts_list()
    security = TransportSecuritySettings(
        # Host-header checking defends a desktop-local server against DNS
        # rebinding by a page in the user's browser. This server is reached by
        # many names and usually through a proxy, and is guarded by a bearer
        # token, so the check is opt-in: enumerate the hosts you serve in
        # RAG_MCP_ALLOWED_HOSTS to turn it on. There is no '*' wildcard — an
        # empty allowlist rejects everything, hence the explicit disable.
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
    return McpMount(server=server, asgi=server.streamable_http_app(), backend=backend)
