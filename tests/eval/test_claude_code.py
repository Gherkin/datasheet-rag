"""The Claude Code runner for the answer eval (GH #93). No `claude` process: runs are faked
with stream-json lines shaped like the ones Claude Code 2.1.286 printed in the probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from datasheet_rag.eval.agent import AgentRun
from datasheet_rag.eval.answers import AnswerRecord, load_records, run_answers
from datasheet_rag.eval.claude_code import (
    ClaudeJudge,
    Meter,
    Workspace,
    agent_command,
    parse_stream,
    probe,
    run_condition,
    to_final,
)
from datasheet_rag.eval.dataset import Need
from datasheet_rag.eval.grade import grade

DOC = "6222576e55573e3d1753faa2708931880b1457fe643c8578ed93782b6a17b3da"
NEED = Need(
    need_id="N1",
    question="Does the LAN8720A need an external reset on nRST?",
    category="conceptual",
    answer="Yes: a hardware reset is required following power-up.",
    grading={"type": "text", "must_mention": ["required"], "must_not": []},
    doc_id=DOC,
    pages=[33, 58],
)


def _stream(
    *,
    structured: dict[str, Any] | None = None,
    mcp: str | None = "connected",
    five: float = 0.20,
    week: float = 0.50,
    status: str = "allowed",
    is_error: bool = False,
    subtype: str = "success",
    tools: tuple[str, ...] = ("mcp__datasheet-rag__search",),
    model: str = "claude-sonnet-5-5",
) -> list[str]:
    init = {
        "type": "system",
        "subtype": "init",
        "model": model,
        "claude_code_version": "2.1.286",
        "mcp_servers": [{"name": "datasheet-rag", "status": mcp}] if mcp else [],
    }
    uses = [{"type": "tool_use", "name": t, "input": {"query": "nRST"}} for t in tools]
    uses.append({"type": "tool_use", "name": "StructuredOutput", "input": structured or {}})
    rate = {
        "type": "rate_limit_event",
        "rate_limit_info": {
            "status": status,
            "unifiedWindows": {
                "five_hour": {"utilization": five, "resetsAt": 1790826000},
                "seven_day": {"utilization": week, "resetsAt": 1790992800},
            },
        },
    }
    result = {
        "type": "result",
        "subtype": subtype,
        "is_error": is_error,
        "num_turns": 4,
        "duration_ms": 11000,
        "total_cost_usd": 0.096,
        "usage": {
            "input_tokens": 6,
            "cache_creation_input_tokens": 19985,
            "cache_read_input_tokens": 27803,
            "output_tokens": 1063,
        },
        "structured_output": structured,
        "result": "done",
    }
    msg = {"type": "assistant", "message": {"content": uses}}
    return ["not json", *(json.dumps(e) for e in (init, msg, rate, result))]


ANSWER = {
    "answer": "Yes, a hardware reset is required following power-up.",
    "citations": [{"document": DOC, "page": 33}],
}


class FakeRunner:
    def __init__(self, *outputs: list[str]):
        self.outputs = list(outputs)
        self.calls: list[list[str]] = []

    def __call__(self, argv: Any, cwd: Path, timeout: float) -> list[str]:
        self.calls.append(list(argv))
        return self.outputs.pop(0)


# ---- parsing -------------------------------------------------------------------


def test_parse_stream_reads_every_event() -> None:
    res = parse_stream(_stream(structured=ANSWER))
    assert (res.model, res.version) == ("claude-sonnet-5-5", "2.1.286")
    assert res.mcp_servers == {"datasheet-rag": "connected"}
    assert [t.name for t in res.tool_calls] == ["mcp__datasheet-rag__search"]
    assert res.structured == ANSWER
    assert (res.cache_write_tokens, res.cache_read_tokens, res.output_tokens) == (
        19985,
        27803,
        1063,
    )
    assert res.cost_usd == 0.096 and res.turns == 4 and res.duration_s == 11.0
    assert res.rate_limits[0]["unifiedWindows"]["five_hour"]["utilization"] == 0.20


# ---- commands ------------------------------------------------------------------


def test_each_condition_gets_only_its_tools(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    for cond in "ABC":
        argv = agent_command(cond, "claude-sonnet-5-5", "q?", ws)  # type: ignore[arg-type]
        # Isolation from the user's own setup, every time.
        assert argv[argv.index("--setting-sources") + 1] == ""
        assert "--strict-mcp-config" in argv and "--json-schema" in argv
        assert argv[-1] == "q?"
        tools = argv[argv.index("--tools") + 1]
        if cond == "A":
            assert tools == "" and "--mcp-config" not in argv
        elif cond == "B":
            assert tools == "Read,Grep,Glob" and "--mcp-config" not in argv
        else:
            assert tools == "" and argv[argv.index("--mcp-config") + 1] == str(ws.mcp_config)
            assert argv[argv.index("--allowedTools") + 1] == "mcp__datasheet-rag"


def test_b_citations_map_file_names_to_doc_ids() -> None:
    final = to_final(
        {
            "answer": "x",
            "value": 5,
            "citations": [
                {"document": "LAN8720A__6222576e.pdf", "page": 33},
                {"document": "./pdfs/LAN8720A__6222576e.pdf", "page": 58},
                {"document": DOC, "page": 59},
            ],
        }
    )
    assert [(c.doc_id, c.page) for c in final.citations] == [
        ("6222576e", 33),
        ("6222576e", 58),
        (DOC, 59),
    ]
    assert final.value == 5.0


# ---- one run -------------------------------------------------------------------


def test_run_condition_fills_the_run_and_the_meter(tmp_path: Path) -> None:
    meter = Meter(stop_at=0.85, weekly_ceiling=0.80)
    run = AgentRun()
    run_condition(
        NEED, "C", run, model="m", workspace=Workspace(tmp_path), meter=meter,
        runner=FakeRunner(_stream(structured=ANSWER, five=0.31)),
    )  # fmt: skip
    assert run.runner == "claude-code" and run.runner_version == "2.1.286"
    assert run.final is not None and run.final.citations[0].page == 33
    assert run.reported_cost_usd == 0.096 and run.turns == 4
    assert meter.five_hour == 0.31


def test_c_without_datasheet_rag_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not connected"):
        run_condition(
            NEED, "C", AgentRun(), model="m", workspace=Workspace(tmp_path),
            runner=FakeRunner(_stream(structured=ANSWER, mcp="failed")),
        )  # fmt: skip


def test_a_failed_process_raises_but_a_budget_stop_is_an_unanswered_run(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    with pytest.raises(RuntimeError, match="failed"):
        run_condition(
            NEED, "B", AgentRun(), model="m", workspace=ws,
            runner=FakeRunner(_stream(is_error=True, subtype="error_during_execution")),
        )  # fmt: skip
    run = AgentRun()
    run_condition(
        NEED, "B", run, model="m", workspace=ws,
        runner=FakeRunner(_stream(is_error=True, subtype="error_max_budget_usd")),
    )  # fmt: skip
    assert run.final is None and run.stop == "error_max_budget_usd"


# ---- the meter -----------------------------------------------------------------


def _info(five: float, week: float, status: str = "allowed") -> dict[str, Any]:
    return {
        "status": status,
        "unifiedWindows": {
            "five_hour": {"utilization": five, "resetsAt": 1},
            "seven_day": {"utilization": week, "resetsAt": 2},
        },
    }


def test_meter_stops_at_either_limit_or_a_refusal() -> None:
    m = Meter(stop_at=0.85, weekly_ceiling=0.80)
    assert m.allows()  # no reading yet
    m.update(_info(0.50, 0.40))
    assert m.allows()
    m.update(_info(0.85, 0.40))
    assert not m.allows() and "5-hour" in (m.stop_reason() or "")
    w = Meter(stop_at=0.85, weekly_ceiling=0.80)
    w.update(_info(0.10, 0.80))
    assert "weekly" in (w.stop_reason() or "")
    r = Meter(stop_at=0.85, weekly_ceiling=0.80)
    r.update(_info(0.10, 0.10, status="rejected"))
    assert "refused" in (r.stop_reason() or "")


def test_meter_keeps_the_highest_reading_until_the_window_resets() -> None:
    m = Meter(stop_at=0.85, weekly_ceiling=0.80)
    m.update(_info(0.50, 0.40))
    # A process that started earlier reports an older, lower reading late.
    m.update(_info(0.30, 0.35))
    assert (m.five_hour, m.seven_day) == (0.50, 0.40)
    # A new 5-hour window (another reset time) starts from its own reading.
    later = _info(0.05, 0.41)
    later["unifiedWindows"]["five_hour"]["resetsAt"] = 99
    m.update(later)
    assert (m.five_hour, m.seven_day) == (0.05, 0.41)


# ---- judge and probe -----------------------------------------------------------


def test_claude_judge_answers_the_graders_converse_call(tmp_path: Path) -> None:
    verdict = {"correct": True, "reason": "states the requirement"}
    runner = FakeRunner(_stream(structured=verdict, tools=(), model="claude-opus-5-5"))
    judge = ClaudeJudge("claude-opus-5-5", workdir=tmp_path, runner=runner)
    final = to_final(ANSWER)
    g = grade(NEED, final, client=judge, judge_model="claude-code:claude-opus-5-5")
    assert g.correct and g.grounded
    assert g.judge_cost_usd == 0.096
    argv = runner.calls[0]
    # The judge gets its own system prompt and no tools.
    assert "--system-prompt" in argv and argv[argv.index("--tools") + 1] == ""
    assert "Reference answer" in argv[-1]


def test_probe_resolves_the_alias(tmp_path: Path) -> None:
    meter = Meter(0.85, 0.80)
    got = probe("sonnet", workdir=tmp_path, meter=meter, runner=FakeRunner(_stream(five=0.42)))
    assert got == "claude-sonnet-5-5" and meter.five_hour == 0.42


# ---- workspace -----------------------------------------------------------------


def test_workspace_copies_pdfs_once_and_pins_the_project(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    docs = [{"doc_id": DOC, "mpn": "LAN8720A", "title": None}]
    loads: list[str] = []

    def load(doc_id: str) -> bytes:
        loads.append(doc_id)
        return b"%PDF-1.4"

    assert ws.prepare(docs, load, "netdaq") == 1
    assert (ws.pdf_dir / "LAN8720A__6222576e.pdf").read_bytes() == b"%PDF-1.4"
    assert not (ws.pdf_dir / "LAN8720A__6222576e.pdf").is_symlink()
    assert ws.prepare(docs, load, "netdaq") == 0 and loads == [DOC]
    cfg = json.loads(ws.mcp_config.read_text())["mcpServers"]["datasheet-rag"]
    assert cfg["env"] == {"RAG_DEFAULT_PROJECT_ID": "netdaq"}


# ---- with the runner -----------------------------------------------------------


def test_gate_stops_new_runs_and_reported_costs_are_used(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    meter = Meter(stop_at=0.30, weekly_ceiling=0.80)
    agent_runs = FakeRunner(_stream(structured=ANSWER, five=0.35))
    judge_runs = FakeRunner(_stream(structured={"correct": True, "reason": "ok"}, tools=()))
    needs = [NEED, NEED.model_copy(update={"need_id": "N2"})]
    new = run_answers(
        needs,
        ("C",),
        model_id="claude-code:claude-sonnet-5-5",
        judge_model="claude-code:claude-opus-5-5",
        client=ClaudeJudge("claude-opus-5-5", workdir=tmp_path, meter=meter, runner=judge_runs),
        agent=lambda need, cond, run: run_condition(
            need,
            cond,
            run,
            model="m",
            workspace=Workspace(tmp_path),
            meter=meter,
            runner=agent_runs,
        ),  # fmt: skip
        gate=meter.allows,
        out_path=out,
        workers=1,
    )
    # The first run pushed the meter to 35%, past the 30% stop: N2 never started.
    assert [r.need_id for r in new] == ["N1"]
    [rec] = load_records(out)
    assert isinstance(rec, AnswerRecord) and rec.grade is not None and rec.grade.grounded
    assert rec.cost_usd() == pytest.approx(0.096 * 2)
