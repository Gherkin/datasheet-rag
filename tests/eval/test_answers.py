"""Running and reporting the grounded-answer eval (GH #93). No AWS."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from datasheet_rag.eval.agent import NoTools
from datasheet_rag.eval.answers import (
    AnswerRecord,
    LockedEmbedder,
    build_report,
    load_records,
    run_answers,
)
from datasheet_rag.eval.dataset import Need

DOC = "d44efe998b6632d4ed49236a1eed2792fc74fc047e2f3bec3fe09399b16f2d96"
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
    the gold page. ``fail_once`` makes the first call for that question raise."""

    def __init__(self, fail_once: str | None = None):
        self.calls = 0
        self.fail_once = fail_once
        self._lock = threading.Lock()

    def converse(self, **kw: Any) -> dict[str, Any]:
        with self._lock:
            self.calls += 1
            question = kw["messages"][0]["content"][0]["text"]
            if question == self.fail_once:
                self.fail_once = None
                raise RuntimeError("ThrottlingException")
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
        model_id="m",
        judge_model="m",
        client=model,
        toolset_for=Guided,
        out_path=out,
        workers=3,
    )


def test_runs_every_pair_and_resumes(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    model = FakeModel()
    new = _run(model, out, _needs(4))
    assert len(new) == 12 and model.calls == 12
    assert len(load_records(out)) == 12
    # A rerun finds everything done and calls nothing.
    assert _run(model, out, _needs(4)) == []
    assert model.calls == 12


def test_failed_run_is_recorded_then_retried(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    model = FakeModel(fail_once="question 1")
    _run(model, out, _needs(2), conds=("A",))
    report = build_report(load_records(out), ["A"], "m")
    assert report.failed == ["N1/A"]
    assert report.conditions[0].n == 1

    retried = _run(model, out, _needs(2), conds=("A",))
    assert [r.need_id for r in retried] == ["N1"]
    report = build_report(load_records(out), ["A"], "m")
    assert report.failed == []
    assert report.conditions[0].n == 2


def test_report_rates_and_paired_differences(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    _run(FakeModel(), out, _needs(12))
    report = build_report(load_records(out), ["A", "B", "C"], "m")
    by = {s.condition: s for s in report.conditions}
    assert by["A"].grounded.value == 0.0
    assert by["B"].grounded.value == by["C"].grounded.value == 1.0
    assert by["C"].mean_input_tokens == 100.0
    assert by["C"].graders == {"numeric": 12}

    cmp = {(c.later, c.earlier): c for c in report.comparisons}
    assert set(cmp) == {("B", "A"), ("C", "A"), ("C", "B")}
    assert cmp[("C", "A")].grounded.verdict == "better"
    assert cmp[("C", "A")].grounded.diff == 1.0
    assert cmp[("C", "B")].grounded.verdict == "no change"
    assert cmp[("C", "B")].input_tokens == 90.0


def test_report_pairs_only_needs_every_condition_graded(tmp_path: Path) -> None:
    out = tmp_path / "answers.jsonl"
    _run(FakeModel(), out, _needs(3), conds=("A", "C"))
    _run(FakeModel(), out, _needs(2), conds=("B",))
    report = build_report(load_records(out), ["A", "B", "C"], "m")
    assert report.incomplete == ["N2"]
    assert all(s.n == 2 for s in report.conditions)


def test_report_ignores_other_models(tmp_path: Path) -> None:
    rec = AnswerRecord(need_id="N0", category="table_spec", condition="A", model="other")
    report = build_report([rec], ["A"], "m")
    assert report.conditions[0].n == 0


def test_locked_embedder_passes_calls_through() -> None:
    class Inner:
        dim = 8

        def embed_one(self, text: str) -> list[float]:
            return [float(len(text))]

    e = LockedEmbedder(Inner())
    assert e.embed_one("abc") == [3.0]
    assert e.dim == 8
