"""Run the answer eval through Claude Code on a subscription (GH #93).

Each agent run and each judge call is one ``claude -p`` process. That
measures what datasheet-rag is used from (Claude Code), and it spends the
subscription's usage limits instead of money. So the budget is the usage
meter, not dollars: every run reports the 5-hour and weekly utilization, and
:class:`Meter` lets the runner stop starting new work at a level the user
picks. A later invocation resumes from the saved records.

Isolation (checked by hand on Claude Code 2.1.286):

* ``--setting-sources ""`` loads no user or project settings, so the user's
  CLAUDE.md, hooks and permission rules stay out.
* ``--strict-mcp-config`` loads only the servers passed with
  ``--mcp-config``: datasheet-rag for C, none for A and B.
* ``--tools`` fixes each condition's built-in tools; ``--json-schema``
  returns the answer and citations as structured output.

``--bare`` would isolate further but reads only an API key, never the
subscription login. ``--safe-mode`` also drops ``--mcp-config`` servers.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from datasheet_rag.eval.agent import AgentRun, Condition, ToolCall
from datasheet_rag.eval.dataset import Need
from datasheet_rag.eval.grade import Citation, FinalAnswer

#: Runs `argv` in `cwd` and returns stdout lines. Swapped out in tests.
Runner = Callable[[Sequence[str], Path, float], list[str]]

#: Wall-clock limit for one `claude -p` process.
RUN_TIMEOUT_S = 20 * 60
#: A runaway guard per run, in list-price dollars (Claude Code's own estimate).
RUN_BUDGET_USD = 3.0
#: Records' model ids start with this, so they never mix with Bedrock runs.
MODEL_PREFIX = "claude-code:"

_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Your full answer, with values, units and the conditions they apply to.",
        },
        "value": {
            "type": "number",
            "description": "When the question asks for one number: that number (the limit or "
            "figure asked for, not a range).",
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
                    "document": {
                        "type": "string",
                        "description": "The doc_id, or the PDF file name.",
                    },
                    "page": {"type": "integer", "description": "1-based PDF page."},
                },
                "required": ["document", "page"],
            },
        },
    },
    "required": ["answer", "citations"],
}

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string", "description": "One sentence."},
    },
    "required": ["correct", "reason"],
}

_BASE = (
    "You help a hardware engineer who is designing a circuit board. Answer their question "
    "about an electronic part or application note. Be precise: give values with units and "
    "the conditions they apply to. If the source does not state the answer, say so rather "
    "than guess. {sources} Your final answer is structured output: put your full answer in "
    "`answer`; when the question asks for one number, also fill `value` and `unit`; list in "
    "`citations` every document page your answer rests on."
)
_SOURCES: dict[Condition, str] = {
    "A": "You have no documents and no tools. Answer from what you know; leave citations empty.",
    "B": "The project's datasheets and application notes are the PDF files in ./pdfs. Cite a "
    "page by the PDF file name and its 1-based page.",
    "C": "The project's datasheets and application notes are indexed in datasheet-rag; use its "
    "tools. Every chunk result carries a doc_id and a 1-based PDF page; cite by doc_id.",
}


# ---------------------------------------------------------------------------
# The usage meter
# ---------------------------------------------------------------------------


class Meter:
    """The latest 5-hour and weekly utilization (0-1) any run reported.

    Shared by the worker threads. ``allows`` is the runner's gate: no new
    run once either meter is at its limit, or once a run was refused for
    rate limits.
    """

    def __init__(self, stop_at: float, weekly_ceiling: float):
        self.stop_at = stop_at
        self.weekly_ceiling = weekly_ceiling
        self.five_hour: float | None = None
        self.seven_day: float | None = None
        self.five_hour_resets: int | None = None
        self.seven_day_resets: int | None = None
        self.refused = False
        self._lock = threading.Lock()

    def update(self, info: dict[str, Any]) -> None:
        """Take one run's reading.

        Parallel processes can report out of order, so an older, lower
        reading may arrive after a newer one. Within one window usage only
        grows: keep the highest reading until the window's reset time
        changes, which means a new window started.
        """
        windows = info.get("unifiedWindows") or {}

        def merge(
            old: float | None, old_reset: int | None, w: dict[str, Any]
        ) -> tuple[float | None, int | None]:
            if "utilization" not in w:
                return old, old_reset
            new, reset = float(w["utilization"]), w.get("resetsAt")
            if old is None or reset != old_reset:
                return new, reset
            return max(old, new), reset

        with self._lock:
            self.five_hour, self.five_hour_resets = merge(
                self.five_hour, self.five_hour_resets, windows.get("five_hour") or {}
            )
            self.seven_day, self.seven_day_resets = merge(
                self.seven_day, self.seven_day_resets, windows.get("seven_day") or {}
            )
            if info.get("status") not in (None, "allowed", "allowed_warning"):
                self.refused = True

    def stop_reason(self) -> str | None:
        """Why no new run may start, or None."""
        with self._lock:
            if self.refused:
                return "a run was refused for usage limits"
            if self.five_hour is not None and self.five_hour >= self.stop_at:
                return f"5-hour meter at {self.five_hour:.0%} (stop at {self.stop_at:.0%})"
            if self.seven_day is not None and self.seven_day >= self.weekly_ceiling:
                return f"weekly meter at {self.seven_day:.0%} (ceiling {self.weekly_ceiling:.0%})"
            return None

    def allows(self) -> bool:
        return self.stop_reason() is None

    def describe(self) -> str:
        def at(value: float | None, reset: int | None) -> str:
            if value is None:
                return "unknown"
            when = time.strftime("%a %H:%M", time.localtime(reset)) if reset else "?"
            return f"{value:.0%} (resets {when})"

        with self._lock:
            return (
                f"5-hour {at(self.five_hour, self.five_hour_resets)}, "
                f"weekly {at(self.seven_day, self.seven_day_resets)}"
            )


# ---------------------------------------------------------------------------
# One `claude -p` run
# ---------------------------------------------------------------------------


class ClaudeResult(BaseModel):
    """What one `claude -p --output-format stream-json` run reported."""

    model: str | None = None
    version: str | None = None
    mcp_servers: dict[str, str] = Field(default_factory=dict)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    structured: dict[str, Any] | None = None
    text: str = ""
    subtype: str | None = None
    is_error: bool = False
    turns: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    duration_s: float = 0.0
    rate_limits: list[dict[str, Any]] = Field(default_factory=list)


def parse_stream(lines: Iterable[str]) -> ClaudeResult:
    """Read the events a stream-json run printed. Non-JSON lines are skipped."""
    res = ClaudeResult()
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = e.get("type")
        if kind == "system" and e.get("subtype") == "init":
            res.model = e.get("model")
            res.version = e.get("claude_code_version")
            res.mcp_servers = {s["name"]: s.get("status", "") for s in e.get("mcp_servers", [])}
        elif kind == "assistant":
            for b in e.get("message", {}).get("content", []):
                if b.get("type") == "tool_use" and b.get("name") != "StructuredOutput":
                    res.tool_calls.append(ToolCall(name=b["name"], input=b.get("input") or {}))
        elif kind == "rate_limit_event":
            res.rate_limits.append(e.get("rate_limit_info") or {})
        elif kind == "result":
            u = e.get("usage") or {}
            res.structured = e.get("structured_output")
            res.text = str(e.get("result") or "")
            res.subtype = e.get("subtype")
            res.is_error = bool(e.get("is_error"))
            res.turns = int(e.get("num_turns") or 0)
            res.input_tokens = int(u.get("input_tokens") or 0)
            res.cache_read_tokens = int(u.get("cache_read_input_tokens") or 0)
            res.cache_write_tokens = int(u.get("cache_creation_input_tokens") or 0)
            res.output_tokens = int(u.get("output_tokens") or 0)
            res.cost_usd = e.get("total_cost_usd")
            res.duration_s = float(e.get("duration_ms") or 0) / 1000.0
    return res


def subprocess_runner(argv: Sequence[str], cwd: Path, timeout: float) -> list[str]:
    out = subprocess.run(
        list(argv), cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    lines = out.stdout.splitlines()
    if not any('"type":"result"' in ln or '"type": "result"' in ln for ln in lines):
        raise RuntimeError(
            f"claude exited {out.returncode} without a result: {(out.stderr or out.stdout)[-500:]}"
        )
    return lines


def _common(model: str) -> list[str]:
    return [
        "claude",
        "-p",
        "--model",
        model,
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--permission-prompts",
        "none",
        "--no-session-persistence",
        "--output-format",
        "stream-json",
        "--verbose",
        "--max-budget-usd",
        str(RUN_BUDGET_USD),
    ]


def agent_command(
    condition: Condition, model: str, question: str, workspace: Workspace
) -> list[str]:
    argv = _common(model)
    if condition == "A":
        argv += ["--tools", ""]
    elif condition == "B":
        argv += ["--tools", "Read,Grep,Glob", "--allowedTools", "Read Grep Glob"]
    else:
        argv += [
            "--tools",
            "",
            "--mcp-config",
            str(workspace.mcp_config),
            "--allowedTools",
            "mcp__datasheet-rag",
        ]
    argv += [
        "--append-system-prompt",
        _BASE.format(sources=_SOURCES[condition]),
        "--json-schema",
        json.dumps(_ANSWER_SCHEMA),
        question,
    ]
    return argv


_FILE_ID = re.compile(r"__([0-9a-f]{8})\.pdf$")


def to_final(structured: dict[str, Any]) -> FinalAnswer:
    """The structured output as a FinalAnswer. A B citation names a PDF file
    ("LAN8720A__6222576e.pdf"); its id prefix is what the grader matches."""
    cites: list[Citation] = []
    for c in structured.get("citations") or []:
        doc = str(c.get("document", "")).strip()
        m = _FILE_ID.search(Path(doc).name)
        page = c.get("page")
        cites.append(
            Citation(doc_id=m.group(1) if m else doc, page=page if isinstance(page, int) else None)
        )
    value = structured.get("value")
    return FinalAnswer(
        answer=str(structured.get("answer", "")),
        value=float(value) if isinstance(value, int | float) else None,
        unit=structured.get("unit"),
        not_stated=bool(structured.get("not_stated", False)),
        citations=cites,
    )


def _take(res: ClaudeResult, meter: Meter | None) -> None:
    if meter is not None:
        for info in res.rate_limits:
            meter.update(info)
    if res.is_error and res.structured is None and not (res.subtype or "").startswith("error_max"):
        raise RuntimeError(f"claude run failed ({res.subtype}): {res.text[:300]}")


def run_condition(
    need: Need,
    condition: Condition,
    run: AgentRun,
    *,
    model: str,
    workspace: Workspace,
    meter: Meter | None = None,
    runner: Runner = subprocess_runner,
) -> None:
    """One agent run: fill ``run`` from a `claude -p` process.

    A run that stops without an answer (Claude Code's own budget or turn
    limit) keeps final None and says why in ``run.stop``; the grader scores
    it wrong. A failed process raises, so the record is retried later.
    """
    lines = runner(
        agent_command(condition, model, need.question, workspace), workspace.root, RUN_TIMEOUT_S
    )
    res = parse_stream(lines)
    _take(res, meter)
    if condition == "C" and res.mcp_servers.get("datasheet-rag") != "connected":
        raise RuntimeError(f"datasheet-rag MCP server not connected: {res.mcp_servers}")
    run.runner = "claude-code"
    run.runner_version = res.version
    run.turns = res.turns
    run.tool_calls = res.tool_calls
    run.input_tokens = res.input_tokens
    run.cache_read_tokens = res.cache_read_tokens
    run.cache_write_tokens = res.cache_write_tokens
    run.output_tokens = res.output_tokens
    run.wall_s = res.duration_s
    run.reported_cost_usd = res.cost_usd
    if res.structured is not None:
        run.final = to_final(res.structured)
        run.stop = "submitted"
    else:
        run.stop = res.subtype or "no structured output"


# ---------------------------------------------------------------------------
# The judge, as a Converse stand-in
# ---------------------------------------------------------------------------


class ClaudeJudge:
    """Answers :func:`datasheet_rag.eval.grade.judge`'s Converse call with a
    `claude -p` run, so grading spends the same subscription as the agents.

    Replaces Claude Code's system prompt with the judge's own and gives it no
    tools. Claude Code has no temperature setting, so a regrade may differ.
    """

    def __init__(
        self,
        model: str,
        *,
        workdir: Path,
        meter: Meter | None = None,
        runner: Runner = subprocess_runner,
    ):
        self.model = model
        self.workdir = workdir
        self.meter = meter
        self.runner = runner

    @property
    def converse(self) -> Callable[..., Any]:
        return self._converse

    def _converse(self, **kw: Any) -> dict[str, Any]:
        system = "\n\n".join(b["text"] for b in kw.get("system", []) if "text" in b)
        prompt = "\n\n".join(
            b["text"] for m in kw.get("messages", []) for b in m["content"] if "text" in b
        )
        argv = [
            *_common(self.model),
            "--tools",
            "",
            "--system-prompt",
            system,
            "--json-schema",
            json.dumps(_VERDICT_SCHEMA),
            prompt,
        ]
        res = parse_stream(self.runner(argv, self.workdir, RUN_TIMEOUT_S))
        _take(res, self.meter)
        if res.structured is None:
            raise RuntimeError(f"judge gave no verdict ({res.subtype}): {res.text[:300]}")
        return {
            "output": {"message": {"content": [{"toolUse": {"input": res.structured}}]}},
            "usage": {
                "inputTokens": res.input_tokens + res.cache_read_tokens + res.cache_write_tokens,
                "outputTokens": res.output_tokens,
            },
            "costUsd": res.cost_usd,
        }


def probe(
    model: str, *, workdir: Path, meter: Meter | None = None, runner: Runner = subprocess_runner
) -> str:
    """The full model name an alias ("sonnet") resolves to now, and a first
    meter reading. One tiny run: aliases move to newer models over time, so
    records name the model that actually answered."""
    argv = [*_common(model), "--tools", "", "--system-prompt", "Reply with: ok", "ok"]
    res = parse_stream(runner(argv, workdir, RUN_TIMEOUT_S))
    _take(res, meter)
    if not res.model:
        raise RuntimeError(f"could not resolve model {model!r}: {res.text[:300]}")
    return res.model


# ---------------------------------------------------------------------------
# The workspace
# ---------------------------------------------------------------------------


class Workspace:
    """The directory every run starts in.

    ``pdfs/`` holds real copies of the project's PDFs (Claude Code's Glob and
    Grep do not follow symlinks), named ``<part>__<doc_id[:8]>.pdf`` so a B
    citation maps back to its document. ``mcp-c.json`` starts the user's own
    datasheet-rag MCP command, pinned to the project.
    """

    def __init__(self, root: Path):
        self.root = root

    @property
    def mcp_config(self) -> Path:
        return self.root / "mcp-c.json"

    @property
    def pdf_dir(self) -> Path:
        return self.root / "pdfs"

    def prepare(
        self,
        documents: Iterable[dict[str, Any]],
        load_pdf: Callable[[str], bytes],
        project: str,
        *,
        mcp_command: str = "rag-mcp",
    ) -> int:
        """Write what is missing; returns how many PDFs were copied."""
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for d in documents:
            label = re.sub(r"[^A-Za-z0-9._-]+", "_", d.get("mpn") or d.get("title") or "document")
            path = self.pdf_dir / f"{label.strip('_')[:60]}__{d['doc_id'][:8]}.pdf"
            if not path.exists():
                path.write_bytes(load_pdf(d["doc_id"]))
                copied += 1
        command = shutil.which(mcp_command) or mcp_command
        self.mcp_config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "datasheet-rag": {
                            "command": command,
                            "args": [],
                            "env": {"RAG_DEFAULT_PROJECT_ID": project},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return copied
