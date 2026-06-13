"""Console-script entry point for the RAG HTTP server (``rag-server``)."""

from __future__ import annotations

import os

from aws_rag.server.app import build_app

# Module-level app so `uvicorn aws_rag.server.main:app` works directly.
app = build_app()


def run() -> None:
    """Run the server with uvicorn. Host/port from env, sensible defaults."""
    import uvicorn

    host = os.environ.get("RAG_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("RAG_SERVER_PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    run()
