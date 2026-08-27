"""Backend boundary: local sqlite or remote HTTP, chosen from config.

``get_backend()`` is what the CLI and MCP call. When ``RAG_SERVER_URL`` is
set it returns a :class:`RemoteBackend`; otherwise a :class:`LocalBackend`
and a one-line "local mode" notice is emitted to stderr (non-failing).

The server's own dependency wiring builds a ``LocalBackend`` directly and
must never call this factory, so the server can't accidentally recurse to a
remote backend.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from datasheet_rag.backend.base import (
    FigureUnavailableError,
    RagBackend,
    RagServerError,
    RemoteIngestError,
)
from datasheet_rag.backend.local import LocalBackend
from datasheet_rag.backend.models import (
    ChunkVectors,
    DocSummary,
    FigureBytes,
    IngestedDoc,
    IngestResult,
    MetadataPatch,
    StatsResult,
    TitleContext,
)
from datasheet_rag.config import get_settings

__all__ = [
    "ChunkVectors",
    "DocSummary",
    "FigureBytes",
    "FigureUnavailableError",
    "IngestResult",
    "IngestedDoc",
    "LocalBackend",
    "MetadataPatch",
    "RagBackend",
    "RagServerError",
    "RemoteIngestError",
    "StatsResult",
    "TitleContext",
    "backend_mode",
    "compute_mode",
    "emit_client_compute_notice",
    "get_backend",
]

_notice_emitted = False
_compute_notice_emitted = False


def backend_mode() -> str:
    """Return 'remote' when a server URL is configured, else 'local'."""
    return "remote" if get_settings().server_url else "local"


def compute_mode() -> str:
    """Return 'client' when this process runs the models itself, else 'server'.

    Always 'client' in local mode — there is no server to hand work to. In
    remote mode it follows ``RAG_COMPUTE`` (GH #43).
    """
    settings = get_settings()
    if not settings.server_url:
        return "client"
    return settings.compute


def emit_local_notice(*, force: bool = False) -> None:
    """Print the local-mode notice to stderr once per process.

    Hooked into the CLI group callback and the MCP start event so users
    are always reminded when they're hitting the local sqlite file rather
    than a shared server. Non-failing — purely informational.
    """
    global _notice_emitted
    settings = get_settings()
    if settings.server_url:
        return
    if _notice_emitted and not force:
        return
    _notice_emitted = True
    print(
        f"rag: local mode (no RAG_SERVER_URL set) — using {settings.sqlite_db_path}",
        file=sys.stderr,
    )


def emit_client_compute_notice() -> None:
    """Print a one-line reminder that the models are running here, not on the server.

    Remote mode normally means "the server does the heavy lifting", so the
    inverse is worth stating once per process — it explains both the startup
    latency and the GPU load (GH #43).
    """
    global _compute_notice_emitted
    if _compute_notice_emitted or compute_mode() != "client" or backend_mode() != "remote":
        return
    _compute_notice_emitted = True
    print(
        "rag: RAG_COMPUTE=client — parsing, embeddings, figure descriptions and "
        "titles run on this machine; the server only stores the results.",
        file=sys.stderr,
    )


@lru_cache(maxsize=1)
def get_backend() -> RagBackend:
    """Return the configured backend (cached for the process)."""
    settings = get_settings()
    if settings.server_url:
        from datasheet_rag.backend.remote import RemoteBackend

        return RemoteBackend(
            settings.server_url,
            token=settings.server_token,
            timeout=settings.server_timeout,
            compute=settings.compute,
        )
    emit_local_notice()
    return LocalBackend()
