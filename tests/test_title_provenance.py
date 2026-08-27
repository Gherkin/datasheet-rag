"""Title provenance: a re-ingest must not demote a human- or LLM-chosen title.

Covers every (stored source, incoming source) pair, not just the blank
case — an auto-detected title that is merely *wrong* ("Contents") used to
win on re-ingest simply because it was non-empty. See issue #32.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.store.metadata import (
    get_title_source,
    set_metadata,
    title_source_of,
)
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks, set_doc_title

DOC = "c" * 64


def _chunk(title: str) -> Chunk:
    md = ChunkMetadata(
        doc_id=DOC,
        doc_title=title,
        chapter_title="",
        section_title="",
        page_numbers=[1],
        layout_type=LayoutType.TEXT,
        context_string="",
    )
    return Chunk(
        id=f"{DOC}:L1:0",
        doc_id=DOC,
        level=ChunkLevel.MICRO,
        text="t",
        context_text="t",
        token_count=1,
        metadata=md,
    )


@pytest.fixture
def conn(tmp_path: Path):
    return connect(tmp_path / "s.sqlite", embedding_dim=get_settings().embedding_dimensions)


def _title(conn) -> str:
    return conn.execute("SELECT doc_title FROM chunks WHERE doc_id = ?", (DOC,)).fetchone()[0]


def _ingest(conn, title: str, **kw) -> None:
    """Simulate an ingest: chunks arrive carrying a parser-derived title."""
    insert_chunks(conn, [_chunk(title)], project_id="p", **kw)


# -- the case #32 is about: a wrong-but-present auto title ------------------


@pytest.mark.parametrize("stored_source", ["manual", "inferred"])
def test_reingest_does_not_clobber_better_sourced_title(conn, stored_source: str) -> None:
    _ingest(conn, "Contents")
    set_doc_title(conn, DOC, "STM32H743 Reference Manual", source=stored_source)

    _ingest(conn, "Contents")  # re-ingest re-derives the same bad title

    assert _title(conn) == "STM32H743 Reference Manual"
    assert get_title_source(conn, DOC) == stored_source


def test_reingest_replaces_an_auto_title(conn) -> None:
    """Same-rank writes still land, so a genuinely better parse is taken."""
    _ingest(conn, "Old Auto Title")
    _ingest(conn, "New Auto Title")
    assert _title(conn) == "New Auto Title"


def test_reingest_with_force_overrides_provenance(conn) -> None:
    _ingest(conn, "Contents")
    set_doc_title(conn, DOC, "Hand Written", source="manual")

    _ingest(conn, "Parser Title", force_title=True)

    assert _title(conn) == "Parser Title"


# -- write-path precedence, source pair by source pair ----------------------


@pytest.mark.parametrize(
    ("stored", "incoming", "expected"),
    [
        ("auto", "auto", "New"),
        ("auto", "inferred", "New"),
        ("auto", "manual", "New"),
        ("inferred", "auto", "Stored"),
        ("inferred", "inferred", "New"),
        ("inferred", "manual", "New"),
        ("manual", "auto", "Stored"),
        ("manual", "inferred", "Stored"),
        ("manual", "manual", "New"),
    ],
)
def test_set_doc_title_precedence(conn, stored: str, incoming: str, expected: str) -> None:
    _ingest(conn, "Stored")
    set_doc_title(conn, DOC, "Stored", source=stored)

    updated = set_doc_title(conn, DOC, "New", source=incoming)

    assert _title(conn) == expected
    # A refused write reports zero rows touched rather than raising.
    assert (updated > 0) is (expected == "New")


def test_refused_write_leaves_provenance_alone(conn) -> None:
    _ingest(conn, "Stored")
    set_doc_title(conn, DOC, "Stored", source="manual")

    set_doc_title(conn, DOC, "New", source="auto")

    assert get_title_source(conn, DOC) == "manual"


def test_force_beats_precedence(conn) -> None:
    _ingest(conn, "Stored")
    set_doc_title(conn, DOC, "Stored", source="manual")

    set_doc_title(conn, DOC, "New", source="auto", force=True)

    assert _title(conn) == "New"
    assert get_title_source(conn, DOC) == "auto"


# -- clearing a title is a deliberate act, and sticks -----------------------


def test_manually_cleared_title_is_not_refilled_by_reingest(conn) -> None:
    _ingest(conn, "Contents")
    set_doc_title(conn, DOC, "", source="manual")

    _ingest(conn, "Contents")

    assert _title(conn) == ""


# -- the b3a7388 case still holds ------------------------------------------


def test_reembed_keeps_inferred_title(conn) -> None:
    """A cached chunk graph carrying a blank must not wipe a stored title."""
    _ingest(conn, "")
    set_doc_title(conn, DOC, "Inferred Title", source="inferred")

    _ingest(conn, "")

    assert _title(conn) == "Inferred Title"


# -- legacy stores ----------------------------------------------------------


def test_legacy_title_inferred_boolean_reads_as_inferred(conn) -> None:
    _ingest(conn, "Contents")
    set_metadata(conn, DOC, attributes={"title_inferred": True})

    assert get_title_source(conn, DOC) == "inferred"

    _ingest(conn, "Contents")  # must not demote a pre-migration inferred title
    assert _title(conn) == "Contents"


def test_legacy_boolean_is_retired_on_next_write(conn) -> None:
    _ingest(conn, "Contents")
    set_metadata(conn, DOC, attributes={"title_inferred": True})

    set_doc_title(conn, DOC, "Hand Written", source="manual")

    from datasheet_rag.store.metadata import get_metadata

    attrs = get_metadata(conn, DOC).attributes
    assert attrs["title_source"] == "manual"
    assert "title_inferred" not in attrs


def test_title_source_of_defaults_to_auto() -> None:
    assert title_source_of({}) == "auto"
    assert title_source_of({"title_source": "nonsense"}) == "auto"


def test_provenance_does_not_leave_a_null_legacy_key(conn) -> None:
    """Retiring `title_inferred` must drop the key, not store a null.

    `set_metadata` documents "set an attribute to None to drop it", but
    only honoured that when the sidecar row already existed — the first
    write for a document took the create branch and stored the null.
    """
    _ingest(conn, "Contents")
    set_doc_title(conn, DOC, "Hand Written", source="manual")

    from datasheet_rag.store.metadata import get_metadata

    assert get_metadata(conn, DOC).attributes == {"title_source": "manual"}
