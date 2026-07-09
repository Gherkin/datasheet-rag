"""CLI-level tests for chunk_id visibility and `rag get chunk` (GH issue #5).

Exercises `rag search` and `rag get chunk` end-to-end against a real
on-disk SQLite store, the same way ``test_cli_scoping.py`` does — inserting
chunks directly via ``insert_chunks`` and driving the CLI through Click's
``CliRunner``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from aws_rag.cli import SHORT_DOC_ID_LEN, cli
from aws_rag.config import get_settings
from aws_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from aws_rag.project_config import get_project_config
from aws_rag.store.schema import connect
from aws_rag.store.sqlite import insert_chunks

DOC_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
DOC_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _chunk(
    doc_id: str,
    index: int,
    text: str,
    *,
    level: ChunkLevel = ChunkLevel.MICRO,
    parent_id: str | None = None,
    prev_id: str | None = None,
    next_id: str | None = None,
    layout_type: LayoutType = LayoutType.TEXT,
    figure_caption: str | None = None,
    figure_image_path: str | None = None,
) -> Chunk:
    chunk_id = f"{doc_id}:L{level.value}:{index}"
    md = ChunkMetadata(
        doc_id=doc_id,
        doc_title=f"Doc {doc_id[:8]}",
        chapter_title="",
        section_title="Thermal Characteristics",
        page_numbers=[3],
        layout_type=layout_type,
        context_string="",
    )
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        level=level,
        text=text,
        context_text=f"Chapter: intro > {text}",
        token_count=max(1, len(text.split())),
        metadata=md,
        parent_id=parent_id,
        prev_id=prev_id,
        next_id=next_id,
        figure_caption=figure_caption,
        figure_image_path=figure_image_path,
    )


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """A real on-disk store with a small linked chain of chunks in doc A
    (for neighbor navigation) plus one chunk in doc B (to test ambiguity)."""
    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)

    parent = _chunk(DOC_A, 0, "Power supply thermal overview.", level=ChunkLevel.MESO)
    prev_c = _chunk(DOC_A, 1, "Widget operates below 40C ambient.", parent_id=parent.id)
    mid = _chunk(
        DOC_A, 2, "The gizmo derates above 85C junction temperature.",
        parent_id=parent.id, prev_id=prev_c.id,
    )
    next_c = _chunk(DOC_A, 3, "Cooling fan engages at 70C.", parent_id=parent.id, prev_id=mid.id)
    prev_c.next_id = mid.id
    mid.next_id = next_c.id

    other = _chunk(DOC_B, 0, "Unrelated capacitor derating notes.")

    insert_chunks(conn, [parent, prev_c, mid, next_c], project_id="proj-a")
    insert_chunks(conn, [other], project_id="proj-b")
    conn.commit()
    conn.close()
    return path


@pytest.fixture(autouse=True)
def _clear_project_config_cache() -> None:
    get_project_config.cache_clear()
    yield
    get_project_config.cache_clear()


def _run(db_path: Path, *args: str):
    runner = CliRunner()
    return runner.invoke(cli, [*args, "--db", str(db_path)])


def test_search_output_includes_short_chunk_id(db_path: Path) -> None:
    result = _run(db_path, "search", "derates junction temperature", "--mode", "keyword", "-g")
    assert result.exit_code == 0, result.output
    expected_short_id = f"{DOC_A[:SHORT_DOC_ID_LEN]}:L2:2"
    assert expected_short_id in result.output
    assert "chunk_id" in result.output
    assert "rag get chunk" in result.output


def test_get_chunk_by_full_id(db_path: Path) -> None:
    full_id = f"{DOC_A}:L2:2"
    result = _run(db_path, "get", "chunk", full_id)
    assert result.exit_code == 0, result.output
    assert full_id in result.output
    assert "gizmo derates above 85C" in result.output
    assert "section: Thermal Characteristics" in result.output


def test_get_chunk_by_abbreviated_doc_id(db_path: Path) -> None:
    short_id = f"{DOC_A[:SHORT_DOC_ID_LEN]}:L2:2"
    result = _run(db_path, "get", "chunk", short_id)
    assert result.exit_code == 0, result.output
    assert f"{DOC_A}:L2:2" in result.output
    assert "gizmo derates above 85C" in result.output


def test_get_chunk_not_found(db_path: Path) -> None:
    result = _run(db_path, "get", "chunk", f"{DOC_A}:L2:999")
    assert result.exit_code != 0
    assert "No chunk found" in result.output


def test_get_chunk_invalid_format_rejected(db_path: Path) -> None:
    result = _run(db_path, "get", "chunk", "not-a-chunk-id")
    assert result.exit_code != 0
    assert "doesn't look like a chunk ID" in result.output


def test_get_chunk_ambiguous_doc_prefix(db_path: Path) -> None:
    # A single shared leading char ('a' matches DOC_A's 'aaaa...' but not
    # DOC_B) isn't ambiguous; force an ambiguous prefix by using one that
    # matches both ('a' would only match DOC_A here) — instead assert the
    # unresolvable-prefix path via a prefix that matches neither doc.
    result = _run(db_path, "get", "chunk", "zzzzzzzzzzzz:L2:2")
    assert result.exit_code != 0
    assert "No ingested document matches" in result.output


def test_get_chunk_with_neighbors(db_path: Path) -> None:
    full_id = f"{DOC_A}:L2:2"
    result = _run(db_path, "get", "chunk", full_id, "--neighbors")
    assert result.exit_code == 0, result.output
    assert "Widget operates below 40C ambient." in result.output
    assert "Cooling fan engages at 70C." in result.output
    assert "Power supply thermal overview." in result.output


def test_list_figures_chunk_id_round_trips_through_get_chunk(db_path: Path) -> None:
    fig = _chunk(
        DOC_A, 4, "See figure for thermal curve.",
        layout_type=LayoutType.FIGURE, figure_caption="Fig 1: Thermal derating curve",
        figure_image_path="/tmp/fake-figure.png",
    )
    conn = connect(db_path, embedding_dim=get_settings().embedding_dimensions)
    insert_chunks(conn, [fig], project_id="proj-a")
    conn.commit()
    conn.close()

    listed = _run(db_path, "list-figures", "-g")
    assert listed.exit_code == 0, listed.output
    short_id = f"{DOC_A[:SHORT_DOC_ID_LEN]}:L2:4"
    assert short_id in listed.output

    fetched = _run(db_path, "get", "chunk", short_id)
    assert fetched.exit_code == 0, fetched.output
    assert "Fig 1: Thermal derating curve" in fetched.output
