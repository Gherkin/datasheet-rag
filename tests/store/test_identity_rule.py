"""The identity rule (GH #124): parsing, matching, and where it is refused."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from datasheet_rag.store.documents import document_of
from datasheet_rag.store.fields import MetadataFieldError, register_field, remove_field
from datasheet_rag.store.identity import (
    IdentityRule,
    IdentityRuleError,
    get_identity_rule,
    match_documents,
    set_identity_rule,
)
from datasheet_rag.store.metadata import set_metadata
from datasheet_rag.store.schema import connect

EMBED_DIM = 4


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    c = connect(":memory:", embedding_dim=EMBED_DIM)
    try:
        for name in ("intel-id", "ti-id", "acme-pn"):
            register_field(c, name, shape="list" if name == "acme-pn" else "single")
        yield c
    finally:
        c.close()


@pytest.mark.parametrize(
    ("source", "canonical"),
    [
        ("intel-id", "intel-id"),
        ("intel-id or ti-id", "intel-id OR ti-id"),
        (
            "intel-id OR ti-id OR (acme-pn AND doc_type)",
            "intel-id OR ti-id OR acme-pn AND doc_type",
        ),
        ("(intel-id OR ti-id) AND doc_type", "(intel-id OR ti-id) AND doc_type"),
        ("a AND b OR c", "a AND b OR c"),
    ],
)
def test_rules_parse_with_and_binding_tighter(source: str, canonical: str) -> None:
    assert str(IdentityRule(source)) == canonical


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("", "empty"),
        ("intel-id OR", "field name was expected"),
        ("(intel-id", "not closed"),
        ("intel-id ti-id", "unexpected"),
        ("AND intel-id", "expected a field name"),
        ("Intel-ID", "lower case"),
        ("intel-id = 3", "cannot read"),
    ],
)
def test_bad_rules_are_refused(source: str, message: str) -> None:
    with pytest.raises(IdentityRuleError, match=message):
        IdentityRule(source)


def test_matching_needs_the_field_on_both_sides() -> None:
    rule = IdentityRule("intel-id OR (acme-pn AND doc_type)")
    assert rule.matches({"intel-id": "123"}, {"intel-id": "123"})
    assert not rule.matches({"intel-id": "123"}, {})
    assert not rule.matches({}, {})  # a missing field never matches
    assert not rule.matches({"intel-id": "123"}, {"intel-id": "124"})
    # AND needs both; a value compares case-insensitively.
    both = {"acme-pn": ["P1"], "doc_type": "datasheet"}
    assert rule.matches(both, {"acme-pn": ["p1"], "doc_type": "Datasheet"})
    assert not rule.matches(both, {"acme-pn": ["P1"], "doc_type": "errata"})


def test_list_fields_must_be_equal_not_overlap() -> None:
    rule = IdentityRule("acme-pn")
    assert rule.matches({"acme-pn": ["P1", "P2"]}, {"acme-pn": ["P2", "P1"]})
    assert not rule.matches({"acme-pn": ["P1", "P2"]}, {"acme-pn": ["P1"]})


def test_rule_fields_must_be_registered_and_document_scoped(conn: sqlite3.Connection) -> None:
    with pytest.raises(IdentityRuleError, match="unregistered"):
        set_identity_rule(conn, "nope-id")
    with pytest.raises(IdentityRuleError, match="one value per project"):
        set_identity_rule(conn, "subsystem")
    assert get_identity_rule(conn) is None


def test_the_rule_is_stored_and_cleared(conn: sqlite3.Connection) -> None:
    set_identity_rule(conn, "intel-id or ti-id")
    rule = get_identity_rule(conn)
    assert rule is not None and str(rule) == "intel-id OR ti-id"
    set_identity_rule(conn, None)
    assert get_identity_rule(conn) is None


def test_a_field_the_rule_uses_cannot_be_removed(conn: sqlite3.Connection) -> None:
    set_identity_rule(conn, "intel-id")
    with pytest.raises(MetadataFieldError, match="identity rule"):
        remove_field(conn, "intel-id")
    set_identity_rule(conn, None)
    remove_field(conn, "intel-id")


def test_match_documents_compares_document_values(conn: sqlite3.Connection) -> None:
    set_metadata(conn, "a", fields={"intel-id": "123"})
    set_metadata(conn, "b", fields={"ti-id": "SLOS1"})
    set_metadata(conn, "c", fields={"intel-id": "999"})
    rule = IdentityRule("intel-id OR ti-id")

    hits = match_documents(conn, rule, {"intel-id": "123", "ti-id": "SLOS1"})
    assert [h.document_id for h in hits] == sorted([document_of(conn, "a"), document_of(conn, "b")])
    hits = match_documents(conn, rule, {"intel-id": "123"})
    assert [(h.document_id, h.fields) for h in hits] == [(document_of(conn, "a"), ["intel-id"])]
    assert match_documents(conn, rule, {"acme-pn": ["X"]}) == []
