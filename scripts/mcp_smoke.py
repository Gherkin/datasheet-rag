"""Smoke test the rag-mcp server by driving it as a real MCP client.

Spawns ``rag-mcp`` over stdio, lists the tools, then calls a handful of
them and prints concise results. Exits non-zero on the first failure.

Run after ``rag embed`` has populated the SQLite store. Set
``RAG_DEFAULT_PROJECT_ID`` in the environment to scope the calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _short(value: Any, n: int = 220) -> str:
    s = json.dumps(value, default=str)
    return s if len(s) <= n else s[:n] + "…"


def _unwrap(result: Any) -> Any:
    """Pull the structured content out of a CallToolResult, if present.

    FastMCP returns either ``structuredContent`` (dict/list) or a list of
    ``TextContent`` blocks; we try structured first, fall back to parsing
    the first text block as JSON.
    """
    sc = getattr(result, "structuredContent", None)
    if sc is not None:
        # FastMCP wraps list/primitive returns in {"result": ...} because the
        # MCP protocol only allows dict for structuredContent, not bare lists.
        if isinstance(sc, dict) and list(sc.keys()) == ["result"]:
            return sc["result"]
        return sc
    content = getattr(result, "content", None) or []
    if content and hasattr(content[0], "text"):
        try:
            return json.loads(content[0].text)
        except Exception:
            return content[0].text
    return None


async def main() -> int:
    project = os.environ.get("RAG_DEFAULT_PROJECT_ID", "(unscoped)")
    print(f"[mcp-smoke] launching rag-mcp · project={project}")

    params = StdioServerParameters(command="rag-mcp", args=[], env=dict(os.environ))

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # ---- 1. List tools -------------------------------------------
            tools_resp = await session.list_tools()
            tool_names = [t.name for t in tools_resp.tools]
            print(f"[mcp-smoke] tools: {tool_names}")
            for required in ("search", "get_chunk", "navigate", "zoom_in",
                             "zoom_out", "list_documents", "stats"):
                assert required in tool_names, f"missing tool: {required}"

            # ---- 2. stats — sanity-check the store is populated ---------
            stats_result = _unwrap(await session.call_tool("stats", {}))
            print(f"[mcp-smoke] stats: {_short(stats_result)}")
            if not isinstance(stats_result, dict) or stats_result.get("total_chunks", 0) == 0:
                print("[mcp-smoke] WARNING: store is empty — run `rag embed` first.")
                return 2

            # ---- 3. list_documents --------------------------------------
            docs = _unwrap(await session.call_tool("list_documents", {}))
            print(f"[mcp-smoke] docs: {_short(docs)}")

            # ---- 4. search (hybrid) -------------------------------------
            search_result = _unwrap(await session.call_tool(
                "search",
                {"query": "specification", "mode": "hybrid", "k": 3},
            ))
            assert isinstance(search_result, list), f"search returned {type(search_result)}"
            print(f"[mcp-smoke] hybrid search returned {len(search_result)} results")
            if search_result:
                first = search_result[0]
                print(f"            top hit: chunk_id={first['chunk_id']} "
                      f"score={first.get('score')} level={first['level']} "
                      f"page={first['page']} section={first['section'][:40]!r}")

                # ---- 5. get_chunk on the top hit ------------------------
                chunk = _unwrap(await session.call_tool(
                    "get_chunk",
                    {"chunk_id": first["chunk_id"], "include_neighbors": True},
                ))
                print(f"[mcp-smoke] get_chunk(top): text[:80]={chunk['text'][:80]!r}")
                neighbors = chunk.get("neighbors", {})
                nb_summary = {k: (v and v["chunk_id"]) for k, v in neighbors.items()}
                print(f"            neighbors: {nb_summary}")

                # ---- 6. zoom_out to the parent --------------------------
                if first.get("parent_id"):
                    parent = _unwrap(await session.call_tool(
                        "zoom_out", {"chunk_id": first["chunk_id"]},
                    ))
                    if parent:
                        print(f"[mcp-smoke] zoom_out: parent={parent[0]['chunk_id']} "
                              f"level={parent[0]['level']} "
                              f"section={parent[0]['section'][:40]!r}")

                # ---- 7. navigate next -----------------------------------
                if first.get("next_id"):
                    nxt = _unwrap(await session.call_tool(
                        "navigate",
                        {"chunk_id": first["chunk_id"], "direction": "next"},
                    ))
                    if nxt:
                        print(f"[mcp-smoke] navigate(next): {nxt[0]['chunk_id']}")

            # ---- 8. keyword-only search ---------------------------------
            kw_result = _unwrap(await session.call_tool(
                "search",
                {"query": "voltage", "mode": "keyword", "k": 3},
            ))
            print(f"[mcp-smoke] keyword search returned "
                  f"{len(kw_result) if isinstance(kw_result, list) else '?'} results")

            # ---- 9. figure path (only if a figure chunk exists) ---------
            fig_search = _unwrap(await session.call_tool(
                "search",
                {"query": "diagram", "mode": "keyword", "k": 10},
            ))
            fig_hits = [
                r for r in (fig_search or [])
                if isinstance(r, dict) and r.get("has_figure")
            ]
            if fig_hits:
                target = fig_hits[0]
                print(f"[mcp-smoke] figure hit: chunk_id={target['chunk_id']} "
                      f"uri={target.get('figure_uri')}")
                # The get_figure tool returns mixed content blocks (Image
                # + text). Don't _unwrap — inspect the raw result.
                gf = await session.call_tool(
                    "get_figure", {"chunk_id": target["chunk_id"]},
                )
                content = getattr(gf, "content", []) or []
                kinds = [type(b).__name__ for b in content]
                print(f"[mcp-smoke] get_figure returned {len(content)} blocks: {kinds}")
                # Sanity check: at least one block exposes image bytes.
                img_blocks = [b for b in content
                              if getattr(b, "type", None) == "image"
                              or hasattr(b, "data")]
                assert img_blocks, "get_figure did not return any image content"
            else:
                print("[mcp-smoke] no figure chunks in this corpus — skipping "
                      "get_figure (run `rag chunk --figures-manifest …` to ingest figures)")

    print("[mcp-smoke] OK — all tool calls returned without error.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
