"""Datasheet RAG Pipeline for electronics datasheets."""

# Single source of truth for the version. `pyproject.toml` reads it from here
# via `[tool.setuptools.dynamic]`, and `mcp/server.py` falls back to it for the
# `initialize` handshake when no installed distribution metadata exists — which
# is the normal case inside the `.mcpb` bundle, where the package is vendored
# onto `sys.path` rather than installed.
__version__ = "0.1.0"
