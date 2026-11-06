"""The identity rule: when a new PDF is a revision of a stored document (GH #124).

By default nothing links automatically: a revision joins a document only
when the user says so (``rag ingest --new-revision-of D12``, ``rag revisions
add``). A store may also hold one *identity rule*, an expression over
registered fields::

    intel-id OR ti-id OR (acme-pn AND doc_type)

``AND`` binds tighter than ``OR``; parentheses group. When a new PDF is
ingested without ``--new-revision-of``, the rule is checked against every
document using the field values the ingest sets. A field matches only when
both sides have it and the values are equal; a missing field never matches.
For a list field the values must be equal as sets, not just overlap: shared
part numbers are how unrelated documents in a family overlap.

One matching document: the PDF joins it as its newest revision. None: it
becomes a new document. Two or more: the ingest stops and names them, and
nothing is merged.

The rule acts at ingest only. Setting or changing it does not relink stored
documents.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from datasheet_rag.store.documents import get_setting, set_setting
from datasheet_rag.store.fields import (
    FieldValue,
    MetadataFieldError,
    field_scopes,
    get_document_values,
)

_SETTING_KEY = "identity_rule"
_TOKEN_RE = re.compile(r"\s*(\(|\)|[A-Za-z0-9][A-Za-z0-9_-]*)")


class IdentityRuleError(ValueError):
    """A rule that does not parse or names fields it cannot use."""


@dataclass(frozen=True)
class _Node:
    op: str  # "field" | "and" | "or"
    name: str = ""
    children: tuple[_Node, ...] = ()

    def text(self, top: bool = True) -> str:
        if self.op == "field":
            return self.name
        joined = f" {self.op.upper()} ".join(c.text(False) for c in self.children)
        # An AND inside an OR needs no parentheses; an OR inside an AND does.
        return joined if top or self.op == "and" else f"({joined})"


class IdentityRule:
    """A parsed identity rule."""

    def __init__(self, source: str):
        self._root = _parse(source)

    def __str__(self) -> str:
        return self._root.text()

    def __repr__(self) -> str:
        return f"IdentityRule({str(self)!r})"

    def fields(self) -> list[str]:
        """Field names the rule uses, in first-use order."""
        out: list[str] = []

        def walk(n: _Node) -> None:
            if n.op == "field":
                if n.name not in out:
                    out.append(n.name)
            for c in n.children:
                walk(c)

        walk(self._root)
        return out

    def matches(self, a: Mapping[str, FieldValue], b: Mapping[str, FieldValue]) -> bool:
        """Whether two sets of field values name the same document."""

        def ev(n: _Node) -> bool:
            if n.op == "field":
                return _equal(a, b, n.name)
            if n.op == "and":
                return all(ev(c) for c in n.children)
            return any(ev(c) for c in n.children)

        return ev(self._root)

    def matched_fields(self, a: Mapping[str, FieldValue], b: Mapping[str, FieldValue]) -> list[str]:
        """The fields that are equal on both sides (for the ingest message)."""
        return [f for f in self.fields() if _equal(a, b, f)]


def _norm(value: FieldValue | None) -> frozenset[str]:
    if value is None:
        return frozenset()
    vals = [value] if isinstance(value, str) else value
    return frozenset(v.strip().casefold() for v in vals if v.strip())


def _equal(a: Mapping[str, FieldValue], b: Mapping[str, FieldValue], name: str) -> bool:
    """Both sides hold a value for name and the values are equal."""
    left, right = _norm(a.get(name)), _norm(b.get(name))
    return bool(left) and left == right


def _parse(source: str) -> _Node:
    tokens: list[str] = []
    pos = 0
    text = source.strip()
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise IdentityRuleError(f"cannot read the rule at {text[pos:]!r}")
        tokens.append(m.group(1))
        pos = m.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    if not tokens:
        raise IdentityRuleError("the rule is empty")
    i = 0

    def peek() -> str | None:
        return tokens[i] if i < len(tokens) else None

    def take() -> str:
        nonlocal i
        tok = tokens[i]
        i += 1
        return tok

    def expr() -> _Node:
        parts = [term()]
        while (t := peek()) is not None and t.upper() == "OR":
            take()
            parts.append(term())
        return parts[0] if len(parts) == 1 else _Node("or", children=tuple(parts))

    def term() -> _Node:
        parts = [factor()]
        while (t := peek()) is not None and t.upper() == "AND":
            take()
            parts.append(factor())
        return parts[0] if len(parts) == 1 else _Node("and", children=tuple(parts))

    def factor() -> _Node:
        t = peek()
        if t is None:
            raise IdentityRuleError("the rule ends where a field name was expected")
        if t == "(":
            take()
            node = expr()
            if peek() != ")":
                raise IdentityRuleError("a '(' is not closed")
            take()
            return node
        if t == ")" or t.upper() in ("AND", "OR"):
            raise IdentityRuleError(f"expected a field name, found {t!r}")
        take()
        if t != t.lower():
            raise IdentityRuleError(f"field names are lower case: {t!r}")
        return _Node("field", name=t)

    root = expr()
    if i != len(tokens):
        raise IdentityRuleError(f"unexpected {tokens[i]!r} in the rule")
    return root


def check_rule_fields(conn: sqlite3.Connection, rule: IdentityRule) -> None:
    """Raise :class:`IdentityRuleError` unless every field is registered and
    document-scoped (a project-scoped value differs by board, so it cannot
    say which datasheet a PDF is)."""
    scopes = field_scopes(conn)
    unknown = [f for f in rule.fields() if f not in scopes]
    if unknown:
        raise IdentityRuleError(
            f"unregistered field(s) in the rule: {', '.join(unknown)} "
            "(register with `rag fields add <name>`)"
        )
    project = [f for f in rule.fields() if scopes[f] == "project"]
    if project:
        raise IdentityRuleError(
            f"field(s) {', '.join(project)} hold one value per project and cannot "
            "identify a document"
        )


def get_identity_rule(conn: sqlite3.Connection) -> IdentityRule | None:
    raw = get_setting(conn, _SETTING_KEY)
    return IdentityRule(raw) if raw else None


def set_identity_rule(conn: sqlite3.Connection, source: str | None) -> IdentityRule | None:
    """Store (or with ``None``/blank clear) the identity rule. Commits."""
    if source is None or not source.strip():
        set_setting(conn, _SETTING_KEY, None)
        conn.commit()
        return None
    rule = IdentityRule(source)
    check_rule_fields(conn, rule)
    set_setting(conn, _SETTING_KEY, str(rule))
    conn.commit()
    return rule


@dataclass(frozen=True)
class RuleMatch:
    document_id: int
    fields: list[str]


def match_documents(
    conn: sqlite3.Connection,
    rule: IdentityRule,
    values: Mapping[str, FieldValue],
    *,
    exclude: int | None = None,
) -> list[RuleMatch]:
    """Every document whose values match ``values`` under ``rule``.

    ``values`` are the new revision's document-scoped field values (the
    ingest patch). Documents' own values are compared, not revision
    overrides. ``exclude`` skips one document (the PDF's own, on re-ingest).
    """
    wanted = rule.fields()
    if not any(f in values for f in wanted):
        return []
    out: list[RuleMatch] = []
    for document_id, doc_values in sorted(get_document_values(conn, None, fields=wanted).items()):
        if document_id == exclude:
            continue
        if rule.matches(values, doc_values):
            out.append(RuleMatch(document_id, rule.matched_fields(values, doc_values)))
    return out


__all__ = [
    "IdentityRule",
    "IdentityRuleError",
    "MetadataFieldError",
    "RuleMatch",
    "check_rule_fields",
    "get_identity_rule",
    "match_documents",
    "set_identity_rule",
]
