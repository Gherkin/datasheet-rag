"""Running and reporting the grounded-answer eval (GH #93). No AWS."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from datasheet_rag.eval.agent import NoTools
from datasheet_rag.eval.answers import (
    AnswerRecord,
    LockedEmbedder,
    build_report,
    compare_runs,
    load_records,
    regrade_records,
    run_answers,
)
from datasheet_rag.eval.dataset import Need, load_need_ids

DOC = "d44efe998b6632d4ed49236a1eed2792fc74fc047e2f3bec3fe09399b16f2d96"
M = "global.anthropic.claude-sonnet-4-6"
G = {"type": "numeric", "value": 5.25, "unit": "V", "bound": "max", "tolerance": 0}


def _needs(n: int) -> list[Need]:
    return [
        Need(
            need_id=f"N{i}",
            question=f"question {i}",
            category="table_spec",
            answer="5.25 V",
            grading=G,
            doc_id=DOC,
            pages=[7],
        )
        for i in range(n)
    ]


class Guided(NoTools):
    """No tools, but a guidance line the fake model can tell apart."""

    def __init__(self, cond: str):
        self.cond = cond

    def guidance(self) -> str:
        return f"CONDITION {self.cond}"


class FakeModel:
    """A (thread-safe) model: A guesses wrong, B and C answer right and cite
    the gold page, and as judge it accepts every answer it is shown (20 in,
    2 out). ``fail_once`` makes the first call for that question raise."""

    def __init__(self, fail_once: str | None = None):
        self.calls = 0
        self.judge_calls = 0
        self.fail_once = fail_once
        self._lock = threading.Lock()

    def converse(self, **kw: Any) -> dict[str, Any]:
        with self._lock:
            self.calls += 1
            question = kw["messages"][0]["content"][0]["text"]
            if question == self.fail_once:
                self.fail_once = None
                raise RuntimeError("ThrottlingException")
        tools = [t.get("toolSpec", {}).get("name") for t in kw["toolConfig"]["tools"]]
        if "verdict" in tools:
            with self._lock:
                self.judge_calls += 1
            verdict = {"toolUse": {"input": {"correct": True, "reason": "ok"}}}
            return {
                "output": {"message": {"content": [verdict]}},
                "usage": {"inputTokens": 20, "outputTokens": 2},
            }
        cond = kw["system"][0]["text"].split("CONDITION ")[1][0]
        if cond == "A":
            args: dict[str, Any] = {"answer": "6 V", "value": 6.0, "unit": "V"}
        else:
            args = {
                "answer": "5.25 V",
                "value": 5.25,
                "unit": "V",
                "citations": [{"doc_id": DOC, "page": 7}],
            }
        use = {"toolUse": {"toolUseId": "u", "name": "submit_answer", "input": args}}
        return {
            "output": {"message": {"role": "assistant", "content": [use]}},
            "usage": {"inputTokens": 100 if cond == "C" else 10, "outputTokens": 1},
        }


def _run(model: FakeModel, out: Path, needs: list[Need], conds: tuple[str, ...] = ("A", "B", "C")):
    return run_answers(
        needs,
        conds,  # type: ignore[arg-type]
        model_id=M,
        judge_model=M,
        client=model,
        toolset_for=Guided,
        out_path=out,
        workers=3,
    )


def test_runs_every_pair_and_resumes(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    model = FakeModel()
    new = _run(model, out, _needs(4))
    # 12 agent runs; B's and C's right values then go to the judge (A's
    # wrong one fails in code).
    assert len(new) == 12
    assert (model.calls, model.judge_calls) == (20, 8)
    assert len(load_records(out)) == 12
    # A rerun finds everything done and calls nothing.
    assert _run(model, out, _needs(4)) == []
    assert model.calls == 20


def test_failed_run_is_recorded_then_retried(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    model = FakeModel(fail_once="question 1")
    _run(model, out, _needs(2), conds=("A",))
    report = build_report(load_records(out), ["A"], M)
    assert report.failed == ["N1/A"]
    assert report.conditions[0].n == 1

    retried = _run(model, out, _needs(2), conds=("A",))
    assert [r.need_id for r in retried] == ["N1"]
    report = build_report(load_records(out), ["A"], M)
    assert report.failed == []
    assert report.conditions[0].n == 2


def test_report_rates_and_paired_differences(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    _run(FakeModel(), out, _needs(12))
    report = build_report(load_records(out), ["A", "B", "C"], M)
    by = {s.condition: s for s in report.conditions}
    assert by["A"].grounded.value == 0.0
    assert by["B"].grounded.value == by["C"].grounded.value == 1.0
    assert by["C"].mean_context_tokens == 100.0
    assert by["C"].graders == {"numeric+judge": 12}
    assert by["A"].graders == {"numeric": 12}

    cmp = {(c.later, c.earlier): c for c in report.comparisons}
    assert set(cmp) == {("B", "A"), ("C", "A"), ("C", "B")}
    assert cmp[("C", "A")].grounded.verdict == "better"
    assert cmp[("C", "A")].grounded.diff == 1.0
    assert cmp[("C", "B")].grounded.verdict == "no change"
    assert cmp[("C", "B")].context_tokens == 90.0

    # Priced from costs.CLAUDE_TOKEN_PRICES: $3 in and $15 out per million.
    judge = (20 * 3 + 2 * 15) / 1e6
    a, b = (10 * 3 + 1 * 15) / 1e6, (10 * 3 + 1 * 15) / 1e6 + judge
    c = (100 * 3 + 1 * 15) / 1e6 + judge
    assert by["C"].mean_cost_usd == pytest.approx(c)
    assert cmp[("C", "B")].cost_usd == pytest.approx(c - b)
    assert report.total_cost_usd == pytest.approx(12 * (a + b + c))


def test_budget_stops_new_runs(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    model = FakeModel()
    # N0 costs A $0.000045, B $0.000135, C $0.000405. After B the spend
    # ($0.00018) is under the cap, so C starts; after C it is over, and with
    # one worker nothing after it starts.
    new = run_answers(
        _needs(5),
        ("A", "B", "C"),
        model_id=M,
        judge_model=M,
        client=model,
        toolset_for=Guided,
        out_path=out,
        workers=1,
        max_usd=0.0003,
    )
    assert [(r.need_id, r.condition) for r in new] == [("N0", "A"), ("N0", "B"), ("N0", "C")]
    assert (model.calls, model.judge_calls) == (5, 2)


def test_regrade_rejudges_saved_answers_without_rerunning(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    _run(FakeModel(), out, _needs(2))

    class Strict(FakeModel):
        """A judge that now rejects everything."""

        def converse(self, **kw: Any) -> dict[str, Any]:
            resp = super().converse(**kw)
            for block in resp["output"]["message"]["content"]:
                if "input" in block.get("toolUse", {}) and "correct" in block["toolUse"]["input"]:
                    block["toolUse"]["input"]["correct"] = False
            return resp

    strict = Strict()
    n, failed = regrade_records(out, _needs(2), client=strict, judge_model=M)
    assert (n, failed) == (6, [])
    # Only judge calls: no agent was run again.
    assert strict.calls == strict.judge_calls == 4
    report = build_report(load_records(out), ["A", "B", "C"], M)
    assert all(s.correct.value == 0.0 for s in report.conditions)
    assert len(load_records(out)) == 6


def test_report_pairs_only_needs_every_condition_graded(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    _run(FakeModel(), out, _needs(3), conds=("A", "C"))
    _run(FakeModel(), out, _needs(2), conds=("B",))
    report = build_report(load_records(out), ["A", "B", "C"], M)
    assert report.incomplete == ["N2"]
    assert all(s.n == 2 for s in report.conditions)


def test_report_ignores_other_models(tmp_path: Path) -> None:
    rec = AnswerRecord(need_id="N0", category="table_spec", condition="A", model="other")
    report = build_report([rec], ["A"], M)
    assert report.conditions[0].n == 0


class WorseC(FakeModel):
    """Condition C now answers even-numbered questions wrong."""

    def converse(self, **kw: Any) -> dict[str, Any]:
        resp = super().converse(**kw)
        q = kw["messages"][0]["content"][0]["text"]
        if "CONDITION C" in kw["system"][0]["text"] and int(q.split()[-1]) % 2 == 0:
            for b in resp["output"]["message"]["content"]:
                b["toolUse"]["input"].update(answer="6 V", value=6.0)
        return resp


def test_compare_runs_pairs_two_runs_of_c(tmp_path: Path) -> None:
    base, variant = tmp_path / "base.jsonl", tmp_path / "variant.jsonl"
    _run(FakeModel(), base, _needs(20), conds=("C",))
    _run(WorseC(), variant, _needs(19), conds=("C",))
    cmp = compare_runs(load_records(base), load_records(variant), "C")
    assert cmp.n == 19
    assert cmp.base.grounded.value == 1.0
    assert cmp.variant.grounded.value == pytest.approx(9 / 19)
    assert cmp.grounded.verdict == "worse"
    assert [f.need_id for f in cmp.flipped] == sorted(f"N{i}" for i in range(0, 19, 2))
    assert all(f.base_grounded and not f.variant_grounded for f in cmp.flipped)
    assert cmp.only_base == ["N19"] and cmp.only_variant == []


def test_compare_runs_refuses_a_file_with_two_models(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    _run(FakeModel(), out, _needs(2), conds=("C",))
    other = [r.model_copy(update={"model": "other"}) for r in load_records(out)]
    with pytest.raises(ValueError, match="several models"):
        compare_runs([*load_records(out), *other][:3], load_records(out), "C")


def test_compare_runs_needs_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no need was graded"):
        compare_runs([], [], "C")


def test_need_list_skips_comments_and_blanks(tmp_path: Path) -> None:
    p = tmp_path / "ids.txt"
    p.write_text("# header\nN1\n\n  N2  # why\n#N3\n", encoding="utf-8")
    assert load_need_ids(p) == ["N1", "N2"]


def test_committed_ab_subset_names_real_needs() -> None:
    from datasheet_rag.eval.dataset import load_needs

    ids = load_need_ids("eval/needs-ab.txt")
    known = {n.need_id for n in load_needs("eval/needs-mined.jsonl")}
    assert len(ids) == len(set(ids)) > 30
    assert set(ids) <= known


def test_locked_embedder_passes_calls_through() -> None:
    class Inner:
        dim = 8

        def embed_one(self, text: str) -> list[float]:
            return [float(len(text))]

    e = LockedEmbedder(Inner())
    assert e.embed_one("abc") == [3.0]
    assert e.dim == 8
