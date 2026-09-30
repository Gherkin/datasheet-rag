"""The answer-level eval: grounded answer rate by condition (GH #93).

Runs every need under each condition with the same model, grades each final
answer (:mod:`datasheet_rag.eval.grade`), and appends one JSONL record per
(need, condition, model) as it finishes. A rerun skips the records already
written, so an interrupted run resumes instead of paying again.

The headline is the **grounded answer rate**: the share of needs whose
answer is correct and cites a gold page. The value of datasheet-rag is
C − A; C − B compares it with an agent that reads the PDFs itself, on
accuracy and on tokens and wall time. Each difference is paired by need and
carries a bootstrap interval (:mod:`datasheet_rag.eval.stats`).
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from datasheet_rag.costs import token_cost_usd
from datasheet_rag.eval.agent import (
    CONDITIONS,
    AgentRun,
    Condition,
    ToolSet,
    run_agent,
)
from datasheet_rag.eval.dataset import Need
from datasheet_rag.eval.grade import Converse, Grade
from datasheet_rag.eval.grade import grade as grade_answer
from datasheet_rag.eval.stats import Interval, PairedDiff, bootstrap_ci, paired_diff


class LockedEmbedder:
    """One embedding model shared by the worker threads, one call at a time.

    Loading a model per thread would hold several copies in (GPU) memory,
    and the eval measures answers, not embedding throughput.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def locked(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)

        return locked


class AnswerRecord(BaseModel):
    """One agent run on one need, graded."""

    need_id: str
    category: str
    condition: Condition
    model: str
    judge_model: str = ""
    # Kept on a failed run too: the turns before the failure were billed.
    run: AgentRun | None = None
    grade: Grade | None = None
    # Set when the run itself failed (an API error, not a wrong answer).
    # Such a record is retried on the next run.
    error: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.need_id, self.condition, self.model)

    def cost_usd(self) -> float | None:
        """Agent plus judge spend at list price; None for an unpriced model."""
        agent = judge = 0.0
        if self.run is not None:
            a = token_cost_usd(
                self.model,
                input_tokens=self.run.input_tokens,
                output_tokens=self.run.output_tokens,
                cache_read_tokens=self.run.cache_read_tokens,
                cache_write_tokens=self.run.cache_write_tokens,
            )
            if a is None:
                return None
            agent = a
        if self.grade is not None and self.grade.judge_input_tokens:
            j = token_cost_usd(
                self.judge_model,
                input_tokens=self.grade.judge_input_tokens,
                output_tokens=self.grade.judge_output_tokens,
            )
            if j is None:
                return None
            judge = j
        return agent + judge


def load_records(path: Path) -> list[AnswerRecord]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [AnswerRecord.model_validate_json(line) for line in fh if line.strip()]


def run_answers(
    needs: Sequence[Need],
    conditions: Sequence[Condition],
    *,
    model_id: str,
    judge_model: str,
    client: Converse,
    toolset_for: Callable[[Condition], ToolSet],
    out_path: Path,
    workers: int = 4,
    progress: Callable[[AnswerRecord], None] | None = None,
    max_usd: float | None = None,
) -> list[AnswerRecord]:
    """Run and grade every (need, condition) not already in ``out_path``.

    ``toolset_for`` is called in the worker thread, so a tool set that holds
    a SQLite connection can be built per thread.

    ``max_usd`` caps this call's spend: once the finished runs reach it, no
    new run starts (the ones in flight finish). Work is ordered need by need
    with every condition together, so a stop leaves whole pairs behind.
    """
    done = {r.key for r in load_records(out_path) if r.error is None}
    todo = [(n, c) for n in needs for c in conditions if (n.need_id, c, model_id) not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    local = threading.local()
    spent = 0.0

    def tools(cond: Condition) -> ToolSet:
        cache: dict[Condition, ToolSet] = getattr(local, "tools", None) or {}
        local.tools = cache
        if cond not in cache:
            cache[cond] = toolset_for(cond)
        return cache[cond]

    def one(need: Need, cond: Condition) -> AnswerRecord | None:
        nonlocal spent
        with write_lock:
            if max_usd is not None and spent >= max_usd:
                return None
        rec = AnswerRecord(
            need_id=need.need_id,
            category=need.category,
            condition=cond,
            model=model_id,
            judge_model=judge_model,
            run=AgentRun(),
        )
        try:
            assert rec.run is not None
            run_agent(client, model_id, need.question, tools(cond), run=rec.run)
            rec.grade = grade_answer(need, rec.run.final, client=client, judge_model=judge_model)
        except Exception as e:  # noqa: BLE001 - recorded and retried next run
            rec.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
        with write_lock:
            # An unpriced model cannot be capped; the CLI refuses that pairing.
            spent += rec.cost_usd() or 0.0
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(rec.model_dump_json() + "\n")
        return rec

    new: list[AnswerRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, n, c) for n, c in todo]
        for f in as_completed(futures):
            rec = f.result()
            if rec is None:
                continue
            new.append(rec)
            if progress is not None:
                progress(rec)
    return new


def regrade_records(
    path: Path,
    needs: Sequence[Need],
    *,
    client: Converse,
    judge_model: str,
    workers: int = 4,
) -> tuple[int, list[str]]:
    """Grade the saved answers in ``path`` again with the current grader.

    The agents are not rerun: a grading fix costs judge calls, not another
    full run. Only records of ``needs`` that finished are touched; the file
    is rewritten in place (through a temporary file, so a crash leaves the
    old one). Returns how many were regraded and the keys whose judge call
    failed, which keep their old grade. A record's cost then counts the new
    judge call, not the one it replaced.
    """
    by_id = {n.need_id: n for n in needs}
    records = load_records(path)
    targets = [r for r in records if r.need_id in by_id and r.error is None and r.run is not None]
    failed: list[str] = []
    lock = threading.Lock()

    def one(rec: AnswerRecord) -> None:
        assert rec.run is not None
        try:
            new = grade_answer(
                by_id[rec.need_id], rec.run.final, client=client, judge_model=judge_model
            )
        except Exception as e:  # noqa: BLE001 - keep the old grade, report it
            with lock:
                failed.append(f"{rec.need_id}/{rec.condition}: {type(e).__name__}: {e}")
            return
        rec.grade = new
        rec.judge_model = judge_model

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, targets))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
    tmp.replace(path)
    return len(targets) - len(failed), failed


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class Rate(BaseModel):
    value: float
    ci: Interval | None


class ConditionSummary(BaseModel):
    condition: Condition
    n: int
    grounded: Rate
    correct: Rate
    grounded_by_category: dict[str, Rate] = Field(default_factory=dict)
    # Every input token the agent read, cached or not: the measure of how
    # much context a condition needs, whatever the cache saved on the bill.
    mean_context_tokens: float
    mean_output_tokens: float
    # Share of context tokens read from the prompt cache.
    cache_read_share: float
    mean_tool_calls: float
    mean_wall_s: float
    # Agent + judge, at list price. None when a model is not in the table.
    mean_cost_usd: float | None
    # How often each grader decided, and how many runs never submitted.
    graders: dict[str, int] = Field(default_factory=dict)
    not_submitted: int = 0


class Comparison(BaseModel):
    """``later`` minus ``earlier`` over the needs both ran."""

    earlier: Condition
    later: Condition
    n: int
    grounded: PairedDiff
    correct: PairedDiff
    context_tokens: float
    wall_s: float
    cost_usd: float | None


class AnswerReport(BaseModel):
    model: str
    conditions: list[ConditionSummary]
    comparisons: list[Comparison]
    # Needs left out because a condition has no graded record for them.
    incomplete: list[str] = Field(default_factory=list)
    # "need_id/condition" runs that errored (API failure) and were never
    # graded. A rerun retries them.
    failed: list[str] = Field(default_factory=list)
    # Everything these records cost, failed and superseded runs included.
    total_cost_usd: float | None = None


def _rate(values: list[float], clusters: list[str]) -> Rate:
    return Rate(
        value=sum(values) / len(values) if values else 0.0, ci=bootstrap_ci(values, clusters)
    )


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _mean_cost(recs: Sequence[AnswerRecord]) -> float | None:
    costs = [r.cost_usd() for r in recs]
    if any(c is None for c in costs):
        return None
    return _mean(c for c in costs if c is not None)


def build_report(
    records: Sequence[AnswerRecord],
    conditions: Sequence[Condition],
    model_id: str,
) -> AnswerReport:
    """Aggregate graded records. Only needs every condition graded count,
    so each condition is measured on the same questions and pairs cleanly."""
    graded: dict[Condition, dict[str, AnswerRecord]] = {c: {} for c in conditions}
    errored: set[tuple[str, Condition]] = set()
    for r in records:
        if r.model != model_id or r.condition not in graded:
            continue
        if r.error is not None or r.grade is None or r.run is None:
            errored.add((r.need_id, r.condition))
            continue
        graded[r.condition][r.need_id] = r  # a later record wins
    # A run that failed and then succeeded on a rerun is not a failure.
    failed = sorted(f"{n}/{c}" for n, c in errored if n not in graded[c])

    all_ids = set().union(*(set(g) for g in graded.values())) if graded else set()
    common = sorted(i for i in all_ids if all(i in g for g in graded.values()))
    incomplete = sorted(all_ids - set(common))

    summaries: list[ConditionSummary] = []
    for c in conditions:
        recs = [graded[c][i] for i in common]
        by_cat: dict[str, list[AnswerRecord]] = {}
        for r in recs:
            by_cat.setdefault(r.category, []).append(r)
        graders: dict[str, int] = {}
        for r in recs:
            assert r.grade is not None
            graders[r.grade.grader] = graders.get(r.grade.grader, 0) + 1

        def rate(rs: list[AnswerRecord], field: str) -> Rate:
            return _rate([float(getattr(r.grade, field)) for r in rs], [r.need_id for r in rs])

        summaries.append(
            ConditionSummary(
                condition=c,
                n=len(recs),
                grounded=rate(recs, "grounded"),
                correct=rate(recs, "correct"),
                grounded_by_category={
                    cat: rate(rs, "grounded") for cat, rs in sorted(by_cat.items())
                },
                mean_context_tokens=_mean(r.run.context_tokens for r in recs if r.run),
                mean_output_tokens=_mean(r.run.output_tokens for r in recs if r.run),
                cache_read_share=sum(r.run.cache_read_tokens for r in recs if r.run)
                / max(1, sum(r.run.context_tokens for r in recs if r.run)),
                mean_tool_calls=_mean(len(r.run.tool_calls) for r in recs if r.run),
                mean_wall_s=_mean(r.run.wall_s for r in recs if r.run),
                mean_cost_usd=_mean_cost(recs),
                graders=graders,
                not_submitted=sum(1 for r in recs if r.run and r.run.final is None),
            )
        )

    comparisons: list[Comparison] = []
    order = [c for c in CONDITIONS if c in conditions]
    for later_i, later in enumerate(order):
        for earlier in order[:later_i]:
            a = [graded[earlier][i] for i in common]
            b = [graded[later][i] for i in common]

            def diff(field: str) -> PairedDiff:
                return paired_diff(
                    [float(getattr(r.grade, field)) for r in a],
                    [float(getattr(r.grade, field)) for r in b],
                    common,
                )

            comparisons.append(
                Comparison(
                    earlier=earlier,
                    later=later,
                    n=len(common),
                    grounded=diff("grounded"),
                    correct=diff("correct"),
                    context_tokens=_mean(r.run.context_tokens for r in b if r.run)
                    - _mean(r.run.context_tokens for r in a if r.run),
                    wall_s=_mean(r.run.wall_s for r in b if r.run)
                    - _mean(r.run.wall_s for r in a if r.run),
                    cost_usd=_cost_diff(a, b),
                )
            )
    spent = [r.cost_usd() for r in records if r.model == model_id]
    return AnswerReport(
        model=model_id,
        conditions=summaries,
        comparisons=comparisons,
        incomplete=incomplete,
        failed=failed,
        total_cost_usd=None if any(c is None for c in spent) else sum(c or 0.0 for c in spent),
    )


def _cost_diff(a: Sequence[AnswerRecord], b: Sequence[AnswerRecord]) -> float | None:
    ca, cb = _mean_cost(a), _mean_cost(b)
    return None if ca is None or cb is None else cb - ca
