"""The identity rule (GH #124): parsing, matching, and where it is refused."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata
from datasheet_rag.store.documents import add_revision, document_of, revisions_of
from datasheet_rag.store.fields import (
    MetadataFieldError,
    get_document_values,
    register_field,
    remove_field,
)
from datasheet_rag.store.identity import (
    IdentityRule,
    IdentityRuleError,
    apply_rule,
    get_identity_rule,
    match_documents,
    set_identity_rule,
)
from datasheet_rag.store.metadata import set_metadata
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

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


def test_conjunctions_are_the_rule_as_an_or_of_ands() -> None:
    rule = IdentityRule("a OR b AND (c OR d)")
    assert rule.conjunctions() == [("a",), ("b", "c"), ("b", "d")]
    assert IdentityRule("(a OR b) AND a").conjunctions() == [("a",), ("b", "a")]


# ---- apply_rule (GH #130) ----------------------------------------------------


def _ingest(conn: sqlite3.Connection, rev_id: str, at: str, **fields: str | list[str]) -> int:
    """A stored PDF ingested at ``at``, in a document of its own."""
    insert_chunks(
        conn,
        [
            Chunk(
                id=f"{rev_id}:L2:0",
                rev_id=rev_id,
                level=ChunkLevel.MICRO,
                text="x",
                context_text="x",
                token_count=1,
                metadata=ChunkMetadata(rev_id=rev_id, page_numbers=[1]),
            )
        ],
    )
    conn.execute("UPDATE chunks SET created_at = ? WHERE rev_id = ?", (at, rev_id))
    conn.commit()
    if fields:
        set_metadata(conn, rev_id, fields=fields)
    document_id = document_of(conn, rev_id)
    assert document_id is not None
    return document_id


def test_apply_rule_links_matches_by_ingest_time(conn: sqlite3.Connection) -> None:
    d_new = _ingest(conn, "new", "2026-03-01", **{"intel-id": "123"})
    d_old = _ingest(conn, "old", "2026-01-01", **{"intel-id": "123"})
    d_mid = _ingest(conn, "mid", "2026-02-01", **{"ti-id": "S1", "intel-id": "123"})
    d_other = _ingest(conn, "other", "2026-01-01", **{"intel-id": "999"})
    rule = IdentityRule("intel-id OR ti-id")

    planned = apply_rule(conn, rule, dry_run=True)
    assert [(g.status, g.document_ids, g.target) for g in planned] == [
        ("merged", (d_new, d_old, d_mid), d_new)
    ]
    assert planned[0].order == ("new", "mid", "old") and not planned[0].kept_order
    assert document_of(conn, "old") == d_old  # the dry run changed nothing

    done = apply_rule(conn, rule)
    assert done == planned
    assert revisions_of(conn, d_new) == ["new", "mid", "old"]
    assert document_of(conn, "other") == d_other
    assert get_document_values(conn, [d_new])[d_new] == {"intel-id": "123", "ti-id": "S1"}
    assert apply_rule(conn, rule) == []  # nothing left to link


def test_apply_rule_keeps_the_order_set_by_hand(conn: sqlite3.Connection) -> None:
    d1 = _ingest(conn, "rev-b", "2026-05-01", **{"intel-id": "1"})
    _ingest(conn, "rev-a", "2026-06-01")
    add_revision(conn, d1, "rev-a")
    add_revision(conn, d1, "rev-b")  # by hand: rev-b newer, though ingested first
    _ingest(conn, "rev-c", "2026-07-01", **{"intel-id": "1"})
    _ingest(conn, "rev-0", "2026-04-01", **{"intel-id": "1"})

    (group,) = apply_rule(conn, IdentityRule("intel-id"))
    assert group.status == "merged" and group.target == d1 and group.kept_order
    assert revisions_of(conn, d1) == ["rev-c", "rev-0", "rev-b", "rev-a"]


def test_apply_rule_skips_a_chain(conn: sqlite3.Connection) -> None:
    a = _ingest(conn, "a", "2026-01-01", **{"intel-id": "1"})
    b = _ingest(conn, "b", "2026-01-02", **{"intel-id": "1", "ti-id": "T"})
    c = _ingest(conn, "c", "2026-01-03", **{"ti-id": "T"})
    (group,) = apply_rule(conn, IdentityRule("intel-id OR ti-id"))
    assert group.status == "chain" and group.document_ids == (a, b, c)
    assert f"D{a} and D{c} do not match" in group.detail
    assert [document_of(conn, r) for r in ("a", "b", "c")] == [a, b, c]


def test_apply_rule_skips_two_documents_with_several_revisions(conn: sqlite3.Connection) -> None:
    d1 = _ingest(conn, "a1", "2026-01-01", **{"intel-id": "1"})
    _ingest(conn, "a2", "2026-01-02")
    add_revision(conn, d1, "a2")
    d2 = _ingest(conn, "b1", "2026-01-03", **{"intel-id": "1"})
    _ingest(conn, "b2", "2026-01-04")
    add_revision(conn, d2, "b2")
    (group,) = apply_rule(conn, IdentityRule("intel-id"))
    assert group.status == "several" and group.document_ids == (d1, d2)
    assert revisions_of(conn, d1) == ["a2", "a1"] and revisions_of(conn, d2) == ["b2", "b1"]


def test_apply_rule_skips_a_conflict_and_links_the_rest(conn: sqlite3.Connection) -> None:
    a = _ingest(conn, "a", "2026-01-01", **{"intel-id": "1", "manufacturer": "TI"})
    b = _ingest(conn, "b", "2026-01-02", **{"intel-id": "1", "manufacturer": "ST"})
    c = _ingest(conn, "c", "2026-01-03", **{"intel-id": "2"})
    d = _ingest(conn, "d", "2026-01-04", **{"intel-id": "2"})
    first, second = apply_rule(conn, IdentityRule("intel-id"))
    assert first.status == "conflict" and first.document_ids == (a, b)
    assert "manufacturer" in first.detail
    assert document_of(conn, "b") == b
    assert get_document_values(conn, [a, b]) == {
        a: {"intel-id": "1", "manufacturer": "TI"},
        b: {"intel-id": "1", "manufacturer": "ST"},
    }
    assert second.status == "merged" and revisions_of(conn, c) == ["d", "c"]
    assert document_of(conn, "d") == c and d != c


def test_apply_rule_needs_list_fields_equal(conn: sqlite3.Connection) -> None:
    _ingest(conn, "a", "2026-01-01", **{"acme-pn": ["P1", "P2"]})
    _ingest(conn, "b", "2026-01-02", **{"acme-pn": ["P1"]})
    _ingest(conn, "c", "2026-01-03", **{"acme-pn": ["p2", "p1"]})
    (group,) = apply_rule(conn, IdentityRule("acme-pn"))
    assert group.status == "merged"
    assert group.order == ("c", "a")
