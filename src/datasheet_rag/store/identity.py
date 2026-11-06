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

At ingest the rule only places the new PDF. Setting or changing it does not
relink stored documents on its own; :func:`apply_rule` (``rag revisions
apply-rule``) does that on request (GH #130).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from datasheet_rag.store.documents import (
    RevisionError,
    format_document_id,
    get_setting,
    ingest_times,
    merge_group,
    revisions_of,
    set_setting,
)
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

    def conjunctions(self) -> list[tuple[str, ...]]:
        """The rule as an OR of ANDs: two value sets match exactly when, for
        some conjunction, every field in it is equal on both sides."""

        def dnf(n: _Node) -> list[tuple[str, ...]]:
            if n.op == "field":
                return [(n.name,)]
            if n.op == "or":
                return [c for child in n.children for c in dnf(child)]
            out: list[tuple[str, ...]] = [()]
            for child in n.children:
                out = [
                    tuple(dict.fromkeys((*left, *right))) for left in out for right in dnf(child)
                ]
            return out

        return list(dict.fromkeys(dnf(self._root)))


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


#: What :func:`apply_rule` did with one group of matching documents.
#:
#: * ``merged``   – linked into one document;
#: * ``chain``    – not every pair in the group matches (A~B, B~C, A≁C); skipped;
#: * ``several``  – more than one document already has several revisions; skipped;
#: * ``conflict`` – the documents hold different values for a field; skipped.
GroupStatus = Literal["merged", "chain", "several", "conflict"]


@dataclass(frozen=True)
class RuleGroup:
    status: GroupStatus
    #: The documents in the group, ascending.
    document_ids: tuple[int, ...]
    #: The document the group was (or would be) merged into.
    target: int | None = None
    #: The merged document's rev_ids, newest first.
    order: tuple[str, ...] = ()
    #: True when ``target`` already had several revisions and their order
    #: was kept, with the other PDFs put on top.
    kept_order: bool = False
    detail: str = ""


def _groups(
    rule: IdentityRule, values: Mapping[int, Mapping[str, FieldValue]]
) -> tuple[list[list[int]], set[tuple[int, int]]]:
    """The connected groups of matching documents (two or more each), and
    every matching pair. Documents are bucketed per conjunction of the rule,
    so no pair is compared that shares no value."""
    edges: set[tuple[int, int]] = set()
    for conj in rule.conjunctions():
        buckets: dict[tuple[frozenset[str], ...], list[int]] = {}
        for document_id, vals in values.items():
            key = tuple(_norm(vals.get(f)) for f in conj)
            if all(key):
                buckets.setdefault(key, []).append(document_id)
        for members in buckets.values():
            members.sort()
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    edges.add((a, b))

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    return sorted(sorted(g) for g in groups.values()), edges


def apply_rule(
    conn: sqlite3.Connection, rule: IdentityRule, *, dry_run: bool = False
) -> list[RuleGroup]:
    """Link stored documents that ``rule`` says are the same (GH #130).

    Match semantics are those of ingest: documents' own values, a missing
    field never matches, list fields must be equal as sets. A group is
    merged only when every pair in it matches and at most one of its
    documents already has several revisions. That document is kept, with its
    order; otherwise the lowest-numbered document is kept. The other PDFs go
    on top, oldest ingest first. A group whose documents hold different values
    for a field is skipped.

    With ``dry_run`` the same work is done and then rolled back. Commits
    otherwise.
    """
    values = get_document_values(conn, None, fields=rule.fields())
    groups, edges = _groups(rule, values)
    out: list[RuleGroup] = []
    if not groups:
        return out
    conn.execute("SAVEPOINT apply_rule")
    try:
        for group in groups:
            out.append(_apply_group(conn, group, edges))
    except BaseException:
        conn.execute("ROLLBACK TO apply_rule")
        conn.execute("RELEASE apply_rule")
        raise
    if dry_run:
        conn.execute("ROLLBACK TO apply_rule")
    conn.execute("RELEASE apply_rule")
    if not dry_run and conn.in_transaction:
        conn.commit()
    return out


def _apply_group(
    conn: sqlite3.Connection, group: list[int], edges: set[tuple[int, int]]
) -> RuleGroup:
    ids = tuple(group)
    missing = [(a, b) for i, a in enumerate(group) for b in group[i + 1 :] if (a, b) not in edges]
    if missing:
        a, b = missing[0]
        more = f" (and {len(missing) - 1} more pair(s))" if len(missing) > 1 else ""
        return RuleGroup(
            "chain",
            ids,
            detail=f"{format_document_id(a)} and {format_document_id(b)} do not match "
            f"each other{more}",
        )
    revisions = {d: revisions_of(conn, d) for d in group}
    multi = [d for d in group if len(revisions[d]) > 1]
    if len(multi) > 1:
        return RuleGroup(
            "several",
            ids,
            detail=f"{', '.join(format_document_id(d) for d in multi)} each have several revisions",
        )
    target = multi[0] if multi else group[0]
    base = revisions[target] if multi else []
    newcomers = [r for d in group if not (multi and d == target) for r in revisions[d]]
    times = ingest_times(conn, newcomers)
    newcomers.sort(key=lambda r: (times.get(r, ""), r))
    order = (*reversed(newcomers), *base)
    conn.execute("SAVEPOINT apply_group")
    try:
        merge_group(conn, target, [d for d in group if d != target], order)
    except RevisionError as e:
        conn.execute("ROLLBACK TO apply_group")
        conn.execute("RELEASE apply_group")
        return RuleGroup("conflict", ids, target=target, detail=str(e))
    conn.execute("RELEASE apply_group")
    return RuleGroup("merged", ids, target=target, order=order, kept_order=bool(multi))


__all__ = [
    "GroupStatus",
    "IdentityRule",
    "IdentityRuleError",
    "MetadataFieldError",
    "RuleGroup",
    "RuleMatch",
    "apply_rule",
    "check_rule_fields",
    "get_identity_rule",
    "match_documents",
    "set_identity_rule",
]
