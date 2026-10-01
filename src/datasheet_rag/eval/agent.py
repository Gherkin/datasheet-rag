"""One agent answering one question under one condition (GH #93).

The three conditions differ only in the tools the agent gets:

* **A · no tool** — model memory only. The guess rate.
* **B · raw PDF** — the project's PDFs as text: list them, search a PDF's
  text, read pages. Roughly what an agent with Read/Grep on the files has.
* **C · datasheet-rag** — the real MCP tools, listed and called in process
  through :func:`datasheet_rag.mcp.server.build_server`, so the agent sees
  the exact tool descriptions and server instructions a client gets.

Every condition ends the same way: the agent calls ``submit_answer`` with
its answer, an optional single value + unit, and the pages it relied on.
The loop runs on the Bedrock Converse API.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from datasheet_rag.eval.grade import Converse, FinalAnswer

Condition = Literal["A", "B", "C"]
CONDITIONS: tuple[Condition, ...] = ("A", "B", "C")
CONDITION_NAMES: dict[Condition, str] = {
    "A": "no tool",
    "B": "raw PDF",
    "C": "datasheet-rag",
}

#: Model turns before the agent is made to submit what it has.
MAX_TURNS = 30
#: A tool result's text is cut here; the agent is told it was cut.
MAX_RESULT_CHARS = 40_000
#: Pages one read_pages call returns.
MAX_READ_PAGES = 10
#: Lines one find_text call returns.
MAX_FIND_HITS = 50

Block = dict[str, Any]


class ToolSet(Protocol):
    """The tools one condition offers, beside ``submit_answer``."""

    def specs(self) -> list[Block]: ...

    def call(self, name: str, args: dict[str, Any]) -> tuple[list[Block], bool]:
        """Run a tool. Returns Converse content blocks and whether it failed."""
        ...

    def guidance(self) -> str: ...


# ---------------------------------------------------------------------------
# A · no tool
# ---------------------------------------------------------------------------


class NoTools:
    def specs(self) -> list[Block]:
        return []

    def call(self, name: str, args: dict[str, Any]) -> tuple[list[Block], bool]:
        return [{"text": f"unknown tool {name!r}"}], True

    def guidance(self) -> str:
        return (
            "You have no documents and no tools besides submit_answer. Answer "
            "from what you know. Leave citations empty."
        )


# ---------------------------------------------------------------------------
# B · raw PDF
# ---------------------------------------------------------------------------


class PdfText:
    """Per-page text of the project's PDFs, extracted once with pdftotext.

    ``-layout`` keeps table columns apart, which is the best a plain text
    reader gets from a datasheet. Shared across worker threads.
    """

    def __init__(self, load_pdf: Callable[[str], bytes]):
        self._load_pdf = load_pdf
        self._pages: dict[str, list[str]] = {}
        self._lock = Lock()

    def pages(self, doc_id: str) -> list[str]:
        with self._lock:
            cached = self._pages.get(doc_id)
        if cached is not None:
            return cached
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "doc.pdf"
            pdf.write_bytes(self._load_pdf(doc_id))
            out = subprocess.run(
                ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"],
                check=True,
                capture_output=True,
            ).stdout.decode("utf-8", errors="replace")
        # pdftotext ends every page with a form feed.
        pages = out.split("\f")
        if pages and not pages[-1].strip():
            pages.pop()
        with self._lock:
            self._pages[doc_id] = pages
        return pages


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> Block:
    return {
        "toolSpec": {
            "name": name,
            "description": description,
            "inputSchema": {
                "json": {"type": "object", "properties": properties, "required": required}
            },
        }
    }


class PdfTools:
    def __init__(self, documents: list[dict[str, Any]], text: PdfText):
        self._docs = documents
        self._ids = {d["doc_id"] for d in documents}
        self._text = text

    def guidance(self) -> str:
        return (
            "The project's datasheets and application notes are PDF files. "
            "list_pdfs lists them, find_text searches one PDF's text, and "
            "read_pages returns the text of a range of pages. Page numbers are "
            "1-based physical PDF pages."
        )

    def specs(self) -> list[Block]:
        doc = {"type": "string", "description": "A doc_id from list_pdfs."}
        return [
            _tool(
                "list_pdfs",
                "List the project's PDF files: doc_id, title, part number, "
                "manufacturer and page count.",
                {},
                [],
            ),
            _tool(
                "find_text",
                "Search one PDF's text, like grep. Case-insensitive regular "
                f"expression. Returns up to {MAX_FIND_HITS} matching lines, each "
                "with its page number.",
                {"doc_id": doc, "pattern": {"type": "string"}},
                ["doc_id", "pattern"],
            ),
            _tool(
                "read_pages",
                f"Return the text of pages first..last (inclusive, at most "
                f"{MAX_READ_PAGES} pages) of one PDF, extracted with the page "
                "layout kept.",
                {
                    "doc_id": doc,
                    "first": {"type": "integer"},
                    "last": {"type": "integer"},
                },
                ["doc_id", "first", "last"],
            ),
        ]

    def call(self, name: str, args: dict[str, Any]) -> tuple[list[Block], bool]:
        if name == "list_pdfs":
            return [{"text": json.dumps(self._docs, indent=1)}], False
        doc_id = str(args.get("doc_id", ""))
        if doc_id not in self._ids:
            return [{"text": f"no PDF with doc_id {doc_id!r}; call list_pdfs"}], True
        pages = self._text.pages(doc_id)
        if name == "find_text":
            try:
                rx = re.compile(str(args.get("pattern", "")), re.IGNORECASE)
            except re.error as e:
                return [{"text": f"bad pattern: {e}"}], True
            hits: list[str] = []
            total = 0
            for no, page in enumerate(pages, start=1):
                for line in page.splitlines():
                    if rx.search(line):
                        total += 1
                        if len(hits) < MAX_FIND_HITS:
                            hits.append(f"p{no}: {line.strip()}")
            head = f"{total} matching line(s)" + (
                f", first {MAX_FIND_HITS} shown" if total > MAX_FIND_HITS else ""
            )
            return [{"text": "\n".join([head, *hits])}], False
        if name == "read_pages":
            try:
                first, last = int(args["first"]), int(args["last"])
            except (KeyError, TypeError, ValueError):
                return [{"text": "first and last must be integers"}], True
            if not 1 <= first <= last <= len(pages):
                return [{"text": f"pages must lie in 1..{len(pages)}, first <= last"}], True
            last = min(last, first + MAX_READ_PAGES - 1)
            body = "\n".join(f"=== page {no} ===\n{pages[no - 1]}" for no in range(first, last + 1))
            return [{"text": body}], False
        return [{"text": f"unknown tool {name!r}"}], True


# ---------------------------------------------------------------------------
# C · datasheet-rag
# ---------------------------------------------------------------------------


class RagTools:
    """The MCP server's own tools, called in process.

    ``server`` is what :func:`build_server` returns. Its calls run under
    ``project`` the way an HTTP client at ``/mcp/<project>`` would.
    """

    def __init__(self, server: Any, project: str):
        self._server = server
        self._project = project
        self._tools = asyncio.run(server.list_tools())

    def guidance(self) -> str:
        return (
            "The project's datasheets and application notes are indexed in "
            "datasheet-rag; use its tools. Every chunk result carries a doc_id "
            "and a 1-based PDF page.\n\n"
            f"Server instructions:\n{self._server.instructions}"
        )

    def specs(self) -> list[Block]:
        return [
            {
                "toolSpec": {
                    "name": t.name,
                    "description": t.description or t.name,
                    "inputSchema": {"json": t.input_schema},
                }
            }
            for t in self._tools
        ]

    def call(self, name: str, args: dict[str, Any]) -> tuple[list[Block], bool]:
        from datasheet_rag.mcp.server import request_local_client, request_project

        async def run() -> Any:
            request_project.set(self._project)
            request_local_client.set(False)
            return await self._server.call_tool(name, args)

        try:
            result = asyncio.run(run())
        except Exception as e:  # noqa: BLE001 - the agent sees tool errors
            return [{"text": f"{type(e).__name__}: {e}"}], True
        blocks: list[Block] = []
        for c in result.content:
            if c.type == "text":
                blocks.append({"text": c.text})
            elif c.type == "image":
                fmt = c.mime_type.split("/")[-1].replace("jpg", "jpeg")
                blocks.append(
                    {"image": {"format": fmt, "source": {"bytes": base64.b64decode(c.data)}}}
                )
        return blocks or [{"text": "(empty result)"}], bool(result.is_error)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

_SUBMIT = _tool(
    "submit_answer",
    "Hand in your final answer. Call this exactly once, when you are done.",
    {
        "answer": {
            "type": "string",
            "description": "Your full answer, with values, units and the conditions they apply to.",
        },
        "value": {
            "type": "number",
            "description": "When the question asks for one number: that number (the "
            "limit or figure asked for, not a range).",
        },
        "unit": {"type": "string", "description": "The unit of value, e.g. V, mA, kOhm, uF."},
        "not_stated": {
            "type": "boolean",
            "description": "True when the documents do not state the answer.",
        },
        "citations": {
            "type": "array",
            "description": "Every document page the answer rests on.",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "page": {"type": "integer", "description": "1-based PDF page."},
                },
                "required": ["doc_id", "page"],
            },
        },
    },
    ["answer"],
)

_SYSTEM = (
    "You help a hardware engineer who is designing a circuit board. Answer "
    "their question about an electronic part or application note. Be precise: "
    "give values with units and the conditions they apply to. If the source "
    "does not state the answer, say so rather than guess.\n\n"
    "{guidance}\n\n"
    "When you are done, call submit_answer once. Put your full answer in "
    "`answer`. When the question asks for one number, also fill `value` and "
    "`unit`. List in `citations` every document page (doc_id and page) your "
    "answer rests on."
)


class ToolCall(BaseModel):
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    error: bool = False


class AgentRun(BaseModel):
    final: FinalAnswer | None = None
    turns: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    # Input tokens are split as Bedrock bills them: uncached, read from the
    # prompt cache, and written to it. Their sum is the context the model read.
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0
    # Why the loop ended when it did not end on submit_answer.
    stop: str = "submitted"
    # "converse" (this module's loop on Bedrock) or "claude-code" (a
    # `claude -p` run, see datasheet_rag.eval.claude_code) and its version.
    runner: str = "converse"
    runner_version: str | None = None
    # List-price cost as the runner reported it. Set for claude-code runs,
    # whose subscription models are not in costs.CLAUDE_TOKEN_PRICES.
    reported_cost_usd: float | None = None

    @property
    def context_tokens(self) -> int:
        """Every input token the model read, cached or not."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


_CACHE_POINT: Block = {"cachePoint": {"type": "default"}}


def _with_cache_point(messages: list[Block]) -> list[Block]:
    """The conversation with a cache point after its last message.

    Every turn resends the whole conversation. A cache point at its end lets
    the next turn read all of it from the cache (a tenth of the input price)
    and pay full price only for the new tool results. Only the request gets
    the marker: kept in the history, the markers would pile up past the
    four a request may carry.
    """
    *head, last = messages
    return [*head, {**last, "content": [*last["content"], _CACHE_POINT]}]


def _clip(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        text = b.get("text")
        if text is not None and len(text) > MAX_RESULT_CHARS:
            b = {"text": text[:MAX_RESULT_CHARS] + f"\n[... cut at {MAX_RESULT_CHARS} characters]"}
        out.append(b)
    return out


def run_agent(
    client: Converse,
    model_id: str,
    question: str,
    tools: ToolSet,
    *,
    max_turns: int = MAX_TURNS,
    run: AgentRun | None = None,
    temperature: float | None = None,
) -> AgentRun:
    """Let the agent work on ``question`` until it submits an answer.

    After ``max_turns`` model turns, or when the agent stops without
    submitting, it is asked once more with only ``submit_answer`` allowed.
    Progress accumulates on ``run`` when given, so a caller still holds the
    token counts of a run that raises partway.

    ``temperature`` None leaves the model's default, as a real client does.
    An A/B run sets 0: most run-to-run flips come from the agent sampling a
    different path, and they drown small differences between two versions.
    """
    run = run if run is not None else AgentRun()
    inference: Block = {"maxTokens": 8192}
    if temperature is not None:
        inference["temperature"] = temperature
    # The system prompt and the tool list are the same on every turn; cache
    # points after each let every turn after the first read them from cache.
    # A prompt below the model's minimum simply is not cached.
    system = [{"text": _SYSTEM.format(guidance=tools.guidance())}, _CACHE_POINT]
    specs = [*tools.specs(), _SUBMIT, _CACHE_POINT]
    messages: list[Block] = [{"role": "user", "content": [{"text": question}]}]
    t0 = time.perf_counter()
    forced = False
    try:
        while True:
            tool_config: Block = {"tools": specs}
            if forced:
                tool_config["toolChoice"] = {"tool": {"name": "submit_answer"}}
            resp = client.converse(
                modelId=model_id,
                system=system,
                messages=_with_cache_point(messages),
                toolConfig=tool_config,
                inferenceConfig=inference,
            )
            run.turns += 1
            usage = resp.get("usage", {})
            run.input_tokens += int(usage.get("inputTokens", 0))
            run.cache_read_tokens += int(usage.get("cacheReadInputTokens", 0))
            run.cache_write_tokens += int(usage.get("cacheWriteInputTokens", 0))
            run.output_tokens += int(usage.get("outputTokens", 0))
            msg = resp["output"]["message"]
            messages.append(msg)

            uses = [b["toolUse"] for b in msg["content"] if "toolUse" in b]
            submit = next((u for u in uses if u["name"] == "submit_answer"), None)
            if submit is not None:
                try:
                    run.final = FinalAnswer.model_validate(submit["input"])
                except ValidationError:
                    run.final = FinalAnswer(answer=json.dumps(submit["input"]))
                    run.stop = "malformed submit"
                return run
            if forced:
                run.stop = "no submit when forced"
                return run
            if not uses:
                # Stopped without submitting: ask for the answer.
                forced = True
                run.stop = "forced after end_turn"
                messages.append(
                    {"role": "user", "content": [{"text": "Call submit_answer with your answer."}]}
                )
                continue

            results: list[Block] = []
            for u in uses:
                blocks, failed = tools.call(u["name"], u.get("input") or {})
                run.tool_calls.append(
                    ToolCall(name=u["name"], input=u.get("input") or {}, error=failed)
                )
                results.append(
                    {
                        "toolResult": {
                            "toolUseId": u["toolUseId"],
                            "content": _clip(blocks),
                            "status": "error" if failed else "success",
                        }
                    }
                )
            messages.append({"role": "user", "content": results})
            if run.turns >= max_turns:
                forced = True
                run.stop = "forced after max turns"
    finally:
        run.wall_s = time.perf_counter() - t0
