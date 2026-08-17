"""FastAPI server exposing the RAG store over HTTP for remote clients.

The server runs a single :class:`~datasheet_rag.backend.local.LocalBackend` and maps
its methods 1:1 onto JSON/HTTP endpoints. It owns the sqlite database, the
embedder, the figure files and the source PDFs; thin clients
(:class:`~datasheet_rag.backend.remote.RemoteBackend`) send query *text* and the
server embeds it.

A future web UI is expected to live in this same app (mount ``/ui`` and reuse
the JSON routes), so the design deliberately keeps everything behind one
process on one port.
"""

from __future__ import annotations

from datasheet_rag.server.app import build_app

__all__ = ["build_app"]
