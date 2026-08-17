"""MCP server exposing the local RAG store to LLM agents (Claude Code etc.).

This package wraps the `store` and `embedding` modules behind the
`mcp` (Model Context Protocol) stdio transport so an agent can search,
navigate, and zoom through datasheet chunks as tool calls.

See :mod:`datasheet_rag.mcp.server` for the tool definitions.
"""

from __future__ import annotations

from datasheet_rag.mcp.server import build_server, main

__all__ = ["build_server", "main"]
