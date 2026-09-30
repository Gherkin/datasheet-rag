"""Grade an agent's final answer against a need's answer key (GH #93).

Numeric and set keys are graded in code: a value within tolerance of the key
in a compatible unit, or every listed item present. Only free-text keys go
to an LLM judge, which reads the reference answer, the facts it must state
and the claims that make it wrong.

A numeric answer the code cannot read (no value, or a unit it cannot
convert) also goes to the judge rather than scoring zero, so a formatting
slip is not counted as a wrong fact. The record says which grader decided.

Grounded means correct *and* citing a gold page of the key's document: the
answer can be checked against its source. An unanswerable need has no page
to cite, so a correct "not stated" counts as grounded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from datasheet_rag.eval.dataset import Need
from datasheet_rag.eval.gold import normalize

Grader = Literal["numeric", "set", "judge", "numeric->judge"]


class Citation(BaseModel):
    doc_id: str = ""
    page: int | None = None


class FinalAnswer(BaseModel):
    """What the agent hands in through the ``submit_answer`` tool."""

    answer: str = ""
    value: float | None = None
    unit: str | None = None
    not_stated: bool = False
    citations: list[Citation] = Field(default_factory=list)


class Grade(BaseModel):
    correct: bool
    grounded: bool
    grader: Grader
    reason: str = ""


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

_PREFIX = {
    "": 1.0,
    "k": 1e3,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
}
# Longest first so "ohm" is not read as a prefixed "h".
_BASES = ("ohm", "ppm", "Hz", "dB", "%", "V", "A", "F", "H", "s", "W")


def _clean_unit(unit: str) -> str:
    u = unit.strip()
    for a, b in (("µ", "u"), ("μ", "u"), ("Ω", "ohm"), ("Ohm", "ohm"), ("ohms", "ohm")):
        u = u.replace(a, b)
    # "% of VREF1", "V rms": the qualifier names the reference, not the unit.
    u = u.split(" ", 1)[0] if " " in u else u
    return "ohm" if u.lower() in ("ohm", "ohms", "r") else u


def _simple_unit(u: str) -> tuple[float, str] | None:
    for base in _BASES:
        if base == "ohm" and u.lower().endswith("ohm"):
            prefix = u[: -len("ohm")]
        elif u.endswith(base):
            prefix = u[: -len(base)]
        else:
            continue
        if prefix in _PREFIX:
            return _PREFIX[prefix], base
    return None


def parse_unit(unit: str) -> tuple[float, str] | None:
    """``(factor to the base unit, dimension)``, e.g. ``"kOhm" -> (1e3, "ohm")``
    and ``"V/us" -> (1e6, "V/s")``. None when the unit is not understood."""
    u = _clean_unit(unit)
    if "/" in u:
        num, den = u.split("/", 1)
        a, b = _simple_unit(num), _simple_unit(den)
        if a is None or b is None:
            return None
        return a[0] / b[0], f"{a[1]}/{b[1]}"
    return _simple_unit(u)


def grade_numeric(grading: dict[str, Any], value: float | None, unit: str | None) -> bool | None:
    """Whether ``value unit`` matches the key. None when it cannot tell."""
    if value is None or not unit:
        return None
    want = parse_unit(str(grading["unit"]))
    got = parse_unit(unit)
    if want is None or got is None or want[1] != got[1]:
        return None
    target = float(grading["value"]) * want[0]
    tol = float(grading.get("tolerance") or 0.0) * want[0]
    # A little slack for float rounding (0.1 uF typed as 100 nF).
    return abs(value * got[0] - target) <= tol + 1e-6 * abs(target)


# ---------------------------------------------------------------------------
# Sets
# ---------------------------------------------------------------------------

_ALIASES = {"nc": ("nc", "no connect", "not connected", "no connection", "n/c")}


def _has_token(token: str, text: str) -> bool:
    options = _ALIASES.get(token, (token,))
    return any(
        re.search(rf"(?<![a-z0-9.]){re.escape(t)}(?![a-z0-9])", text) is not None for t in options
    )


def grade_set(items: Sequence[str], answer: str) -> bool:
    """Every item appears in the answer. A pin map item ``"a=b"`` passes when
    ``a`` and ``b`` appear in one clause ("pin 1: anode", "EN (pin 1)")."""
    text = normalize(answer)
    # Split before normalizing, which folds newlines into spaces. A table row
    # is one line, so it stays one clause.
    clauses = [normalize(c) for c in re.split(r"[;,\n]|\.\s", answer) if c.strip()]
    for item in items:
        norm = normalize(item)
        if "=" in norm:
            a, b = (s.strip() for s in norm.split("=", 1))
            if not any(_has_token(a, c) and _has_token(b, c) for c in clauses):
                return False
        elif not _has_token(norm, text):
            return False
    return True


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class Converse(Protocol):
    """The slice of a boto3 ``bedrock-runtime`` client the eval uses.

    A property returning a callable, because the boto3 stubs type
    ``converse``'s keywords with a TypedDict a plain method would not match.
    """

    @property
    def converse(self) -> Callable[..., Any]: ...


_JUDGE_SYSTEM = (
    "You grade answers to questions about electronic parts against an answer "
    "key written from the datasheet. Be strict about facts and lenient about "
    "wording: equivalent units, phrasing and order are fine. An answer is "
    "correct only when it states every key fact, makes none of the "
    "disqualifying claims, and gives no value for the asked quantity that "
    "conflicts with the key. Extra correct detail is fine. When the key says "
    "the documents do not state the answer, the answer is correct only if it "
    "says so instead of supplying a value. Call the verdict tool."
)

_VERDICT_TOOL = {
    "toolSpec": {
        "name": "verdict",
        "description": "Record whether the answer is correct.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "correct": {"type": "boolean"},
                    "reason": {"type": "string", "description": "One sentence."},
                },
                "required": ["correct", "reason"],
            }
        },
    }
}


def judge(
    client: Converse,
    model_id: str,
    need: Need,
    answer: str,
    *,
    key_facts: Sequence[str],
    must_not: Sequence[str] = (),
) -> tuple[bool, str]:
    """Ask the judge model whether ``answer`` matches the key."""
    prompt = "\n\n".join(
        [
            f"Question:\n{need.question}",
            f"Reference answer:\n{need.answer}",
            "Key facts the answer must state:\n" + "\n".join(f"- {f}" for f in key_facts),
            "Disqualifying claims:\n" + ("\n".join(f"- {m}" for m in must_not) or "(none)"),
            f"Grader notes:\n{need.notes or '(none)'}",
            f"Answer to grade:\n{answer or '(empty)'}",
        ]
    )
    resp = client.converse(
        modelId=model_id,
        system=[{"text": _JUDGE_SYSTEM}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        toolConfig={"tools": [_VERDICT_TOOL], "toolChoice": {"tool": {"name": "verdict"}}},
        inferenceConfig={"maxTokens": 1024},
    )
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            data = block["toolUse"]["input"]
            return bool(data.get("correct")), str(data.get("reason", ""))
    raise RuntimeError(f"judge returned no verdict: {json.dumps(resp['output'])[:300]}")


# ---------------------------------------------------------------------------
# Whole grade
# ---------------------------------------------------------------------------


def cites_gold(need: Need, citations: Sequence[Citation]) -> bool:
    """A citation names the key's document and one of its gold pages. A doc
    id may be shortened to a prefix of at least 8 characters."""
    if need.doc_id is None:
        return False
    for c in citations:
        cid = c.doc_id.strip().lower()
        if len(cid) >= 8 and need.doc_id.startswith(cid) and c.page in need.pages:
            return True
    return False


def grade(
    need: Need,
    final: FinalAnswer | None,
    *,
    client: Converse,
    judge_model: str,
) -> Grade:
    """Grade one final answer. No answer (the agent never submitted) is wrong."""
    if final is None:
        return Grade(correct=False, grounded=False, grader="judge", reason="no answer submitted")
    g = need.grading
    kind = g.get("type")
    if kind == "numeric" and need.answerable:
        verdict = grade_numeric(g, final.value, final.unit)
        if verdict is not None:
            grader: Grader = "numeric"
            correct, reason = (
                verdict,
                f"value {final.value} {final.unit} vs key {g['value']} {g['unit']}",
            )
        else:
            grader = "numeric->judge"
            correct, reason = judge(
                client,
                judge_model,
                need,
                final.answer,
                key_facts=[f"{g['bound']} {g['value']} {g['unit']}"],
            )
    elif kind == "set" and need.answerable:
        grader = "set"
        correct = grade_set([str(i) for i in g["items"]], final.answer)
        reason = "all items present" if correct else f"missing some of {g['items']}"
    else:
        grader = "judge"
        correct, reason = judge(
            client,
            judge_model,
            need,
            final.answer,
            key_facts=[str(f) for f in g.get("must_mention", [])] or [need.answer],
            must_not=[str(m) for m in g.get("must_not", [])],
        )
    grounded = correct and (not need.answerable or cites_gold(need, final.citations))
    return Grade(correct=correct, grounded=grounded, grader=grader, reason=reason)
