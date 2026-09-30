"""Answer grading for the grounded-answer eval (GH #93). No AWS: the judge is faked."""

from __future__ import annotations

from typing import Any

import pytest

from datasheet_rag.eval.dataset import Need
from datasheet_rag.eval.grade import (
    Citation,
    FinalAnswer,
    cites_gold,
    grade,
    grade_numeric,
    grade_set,
    parse_unit,
)

DOC = "d44efe998b6632d4ed49236a1eed2792fc74fc047e2f3bec3fe09399b16f2d96"


def _need(grading: dict[str, Any], **kw: Any) -> Need:
    base: dict[str, Any] = dict(
        need_id="N1",
        question="What is the absolute maximum VBUS voltage?",
        category="table_spec",
        answer="-0.3 V to +5.25 V",
        grading=grading,
        doc_id=DOC,
        pages=[2111],
    )
    base.update(kw)
    return Need(**base)


class FakeJudge:
    """Records the prompt and returns a fixed verdict."""

    def __init__(self, correct: bool = True):
        self.correct = correct
        self.prompts: list[str] = []

    def converse(self, **kw: Any) -> dict[str, Any]:
        self.prompts.append(kw["messages"][0]["content"][0]["text"])
        return {
            "output": {
                "message": {
                    "content": [{"toolUse": {"input": {"correct": self.correct, "reason": "fake"}}}]
                }
            }
        }


# ---- units ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "factor", "dim"),
    [
        ("V", 1.0, "V"),
        ("mV", 1e-3, "V"),
        ("kOhm", 1e3, "ohm"),
        ("kΩ", 1e3, "ohm"),
        ("Ω", 1.0, "ohm"),
        ("ohms", 1.0, "ohm"),
        ("MΩ", 1e6, "ohm"),
        ("mΩ", 1e-3, "ohm"),
        ("µA", 1e-6, "A"),
        ("uF", 1e-6, "F"),
        ("nF", 1e-9, "F"),
        ("uH", 1e-6, "H"),
        ("kHz", 1e3, "Hz"),
        ("ppm", 1.0, "ppm"),
        ("% of VREF1", 1.0, "%"),
        ("V/µs", 1e6, "V/s"),
    ],
)
def test_parse_unit(unit: str, factor: float, dim: str) -> None:
    got = parse_unit(unit)
    assert got is not None
    assert got[0] == pytest.approx(factor)
    assert got[1] == dim


def test_parse_unit_unknown() -> None:
    assert parse_unit("furlongs") is None
    assert parse_unit("pin") is None


# ---- numeric ----------------------------------------------------------------


G = {"type": "numeric", "value": 5.25, "unit": "V", "bound": "max", "tolerance": 0}


def test_numeric_exact_and_prefixed() -> None:
    assert grade_numeric(G, 5.25, "V") is True
    assert grade_numeric(G, 5250, "mV") is True
    assert grade_numeric(G, 6.0, "V") is False


def test_numeric_cross_prefix_capacitance() -> None:
    g = {"type": "numeric", "value": 0.1, "unit": "uF", "bound": "typ", "tolerance": 0}
    assert grade_numeric(g, 100, "nF") is True


def test_numeric_tolerance() -> None:
    g = {**G, "tolerance": 0.1}
    assert grade_numeric(g, 5.3, "V") is True
    assert grade_numeric(g, 5.4, "V") is False


def test_numeric_undecidable_goes_to_judge() -> None:
    assert grade_numeric(G, None, "V") is None
    assert grade_numeric(G, 5.25, None) is None
    # Wrong dimension: the grader cannot compare, it does not mark it wrong.
    assert grade_numeric(G, 5.25, "A") is None


# ---- sets -------------------------------------------------------------------


def test_set_plain_items() -> None:
    items = ["133", "69.8", "45.3", "30.9"]
    assert grade_set(items, "Class 1: 133 Ω, class 2: 69.8 Ω, class 3: 45.3 Ω, class 4: 30.9 Ω")
    assert not grade_set(items, "Class 1: 133 Ω, class 2: 69.8 Ω")


def test_set_does_not_match_inside_a_longer_number() -> None:
    assert not grade_set(["8.7"], "the current is 18.75 A")
    assert grade_set(["8.7"], "the current is 8.7 A")


def test_set_pin_map_needs_both_sides_in_one_clause() -> None:
    items = ["1=anode", "2=NC", "3=cathode"]
    assert grade_set(items, "Pin 1: anode; pin 2: not connected; pin 3: cathode.")
    assert grade_set(items, "| 1 | Anode |\n| 2 | NC |\n| 3 | Cathode |")
    # Swapped anode and cathode: every word is there, the mapping is wrong.
    assert not grade_set(items, "Pin 1: cathode; pin 2: NC; pin 3: anode.")


def test_set_part_numbers() -> None:
    items = ["TPS62A01APDDCR", "TPS62A01ADRLR"]
    assert grade_set(items, "Use TPS62A01APDDCR (SOT-23) or TPS62A01ADRLR (SOT-563).")
    assert not grade_set(items, "Use TPS62A01ADDCR.")


# ---- grounding --------------------------------------------------------------


def test_cites_gold_needs_right_doc_and_page() -> None:
    need = _need(G)
    assert cites_gold(need, [Citation(doc_id=DOC, page=2111)])
    assert cites_gold(need, [Citation(doc_id=DOC[:10], page=2111)])
    assert not cites_gold(need, [Citation(doc_id=DOC, page=2110)])
    assert not cites_gold(need, [Citation(doc_id="ffff" + DOC[4:], page=2111)])
    # Too short a prefix to name one document.
    assert not cites_gold(need, [Citation(doc_id=DOC[:4], page=2111)])


# ---- whole grade ------------------------------------------------------------


def test_grade_numeric_correct_but_uncited_is_not_grounded() -> None:
    judge = FakeJudge()
    g = grade(
        _need(G), FinalAnswer(answer="5.25 V", value=5.25, unit="V"), client=judge, judge_model="m"
    )
    assert g.correct and not g.grounded
    assert g.grader == "numeric"
    assert judge.prompts == []


def test_grade_numeric_grounded() -> None:
    final = FinalAnswer(
        answer="5.25 V", value=5.25, unit="V", citations=[Citation(doc_id=DOC, page=2111)]
    )
    g = grade(_need(G), final, client=FakeJudge(), judge_model="m")
    assert g.correct and g.grounded


def test_grade_numeric_without_value_asks_the_judge() -> None:
    judge = FakeJudge(correct=True)
    g = grade(_need(G), FinalAnswer(answer="about 5.25 V"), client=judge, judge_model="m")
    assert g.grader == "numeric->judge"
    assert g.correct
    assert "max 5.25 V" in judge.prompts[0]


def test_grade_text_sends_rubric_and_notes_to_judge() -> None:
    need = _need(
        {"type": "text", "must_mention": ["81", "analog"], "must_not": ["digital"]},
        notes="Pin 84 is a different signal.",
    )
    judge = FakeJudge(correct=False)
    g = grade(need, FinalAnswer(answer="pin 84"), client=judge, judge_model="m")
    assert g.grader == "judge" and not g.correct
    prompt = judge.prompts[0]
    assert "- 81" in prompt and "- digital" in prompt and "Pin 84 is a different signal" in prompt


def test_unanswerable_correct_abstention_is_grounded() -> None:
    need = _need(
        {"type": "text", "must_mention": ["not specified"], "must_not": []},
        answerable=False,
        doc_id=None,
        pages=[],
    )
    g = grade(
        need,
        FinalAnswer(answer="Not stated.", not_stated=True),
        client=FakeJudge(),
        judge_model="m",
    )
    assert g.correct and g.grounded


def test_no_submission_is_wrong() -> None:
    g = grade(_need(G), None, client=FakeJudge(), judge_model="m")
    assert not g.correct and not g.grounded
