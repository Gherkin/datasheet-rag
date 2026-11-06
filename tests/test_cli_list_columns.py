"""CLI-level tests for `rag list --columns` (GH #38, "rag list should take a
display format").

The two built-in views show a fixed set of columns. `--columns` names the
columns to print: built-in columns and registered fields (GH #120), each
field a column of its own.

Driven end-to-end through Click's ``CliRunner`` against a real on-disk store,
mirroring ``test_cli_metadata.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from datasheet_rag.cli import cli
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.project_config import get_project_config
from datasheet_rag.store.metadata import set_needs_reembed
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

DOC_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
DOC_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _chunk(rev_id: str) -> Chunk:
    md = ChunkMetadata(
        rev_id=rev_id,
        doc_title=f"Doc {rev_id[:8]}",
        chapter_title="",
        section_title="",
        page_numbers=[1],
        layout_type=LayoutType.TEXT,
        context_string="",
    )
    return Chunk(
        id=f"{rev_id}:L1:0",
        rev_id=rev_id,
        level=ChunkLevel.MICRO,
        text="chunk 0",
        context_text="chunk 0",
        token_count=2,
        metadata=md,
    )


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Two documents with sidecar rows: A carries two user fields, B one."""
    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)
    insert_chunks(conn, [_chunk(DOC_A)])
    insert_chunks(conn, [_chunk(DOC_B)])
    conn.commit()
    conn.close()

    _meta(
        path,
        DOC_A,
        "--add-project",
        "proj-a",
        "--add-project",
        "proj-b",
        "--mpn",
        "STM32H743VIT6",
        "--tag",
        "mcu",
    )
    for name in ("drawing", "reviewed_by"):
        CliRunner().invoke(cli, ["fields", "add", name, "--db", str(path)])
    _meta(path, DOC_A, "--field", "drawing=B", "--field", "reviewed_by=hector")
    _meta(path, DOC_B, "--mpn", "TLV6722", "--field", "drawing=C")
    return path


@pytest.fixture(autouse=True)
def _clear_project_config_cache() -> None:
    get_project_config.cache_clear()
    yield
    get_project_config.cache_clear()


def _meta(db_path: Path, *args: str):
    runner = CliRunner()
    return runner.invoke(cli, ["metadata", *args, "--db", str(db_path)])


def _list(db_path: Path, *args: str):
    """`rag list [options]`, widened past the default 80 columns.

    At 80, Rich squeezes rev_id until it wraps mid-hash, which would have
    these tests asserting on the wrapping rather than on the columns.
    """
    runner = CliRunner()
    return runner.invoke(
        cli, ["list", "--global", *args, "--db", str(db_path)], env={"COLUMNS": "200"}
    )


def _header(output: str) -> str:
    """The table's header row — the line holding the column labels."""
    return next(line for line in output.splitlines() if "rev_id" in line)


def _row(output: str, rev_id: str) -> str:
    return next(line for line in output.splitlines() if rev_id[:10] in line)


def _mark_stale(db_path: Path, rev_id: str) -> None:
    conn = connect(db_path, embedding_dim=get_settings().embedding_dimensions)
    try:
        set_needs_reembed(conn, rev_id, True)
    finally:
        conn.close()


def test_columns_shows_user_fields(db_path: Path) -> None:
    result = _list(db_path, "--columns", "mpn,drawing,reviewed_by")
    assert result.exit_code == 0, result.output
    assert "drawing" in _header(result.output)
    assert "B" in _row(result.output, DOC_A)
    assert "hector" in _row(result.output, DOC_A)
    # B has drawing but no reviewer — the missing cell reads as an em-dash.
    assert "C" in _row(result.output, DOC_B)
    assert "hector" not in _row(result.output, DOC_B)


def test_columns_repeatable_and_comma_separated_agree(db_path: Path) -> None:
    comma = _list(db_path, "--columns", "mpn,drawing")
    repeated = _list(db_path, "-c", "mpn", "-c", "drawing")
    assert comma.exit_code == 0, comma.output
    assert repeated.exit_code == 0, repeated.output
    assert comma.output == repeated.output


def test_columns_replaces_the_built_in_views(db_path: Path) -> None:
    result = _list(db_path, "--columns", "drawing")
    assert result.exit_code == 0, result.output
    header = _header(result.output)
    assert "drawing" in header
    for dropped in ("title", "chunks", "pages", "ingested"):
        assert dropped not in header


def test_columns_wins_over_the_filter_implied_wide_view(db_path: Path) -> None:
    """A filter implies --wide, but an explicit column list still decides."""
    result = _list(db_path, "--tag", "mcu", "--columns", "drawing")
    assert result.exit_code == 0, result.output
    assert _header(result.output).split("┃") == _header(
        _list(db_path, "--columns", "drawing").output
    ).split("┃")
    assert DOC_A[:10] in result.output
    assert DOC_B[:10] not in result.output


def test_rev_id_is_always_shown(db_path: Path) -> None:
    result = _list(db_path, "--columns", "drawing")
    assert result.exit_code == 0, result.output
    assert DOC_A[:10] in result.output


def test_built_in_columns_are_selectable(db_path: Path) -> None:
    result = _list(db_path, "--columns", "chunks,tags,mpn")
    assert result.exit_code == 0, result.output
    assert "mcu" in _row(result.output, DOC_A)
    assert "STM32H743VIT6" in _row(result.output, DOC_A)


def test_projects_column_lists_every_project(db_path: Path) -> None:
    for spelling in ("projects", "project", "project_id"):
        result = _list(db_path, "--columns", spelling)
        assert result.exit_code == 0, result.output
        assert "proj-a, proj-b" in _row(result.output, DOC_A)


def test_unknown_column_is_rejected_instead_of_printing_blanks(db_path: Path) -> None:
    result = _list(db_path, "--columns", "mpn,revsion")
    assert result.exit_code != 0
    assert "unknown column(s) revsion" in result.output


def test_stale_marker_still_shown_with_custom_columns(db_path: Path) -> None:
    _mark_stale(db_path, DOC_A)
    result = _list(db_path, "--columns", "mpn")
    assert result.exit_code == 0, result.output
    assert "stale" in _header(result.output)
    assert "re-embed" in _row(result.output, DOC_A)


def test_stale_column_not_duplicated_when_asked_for(db_path: Path) -> None:
    _mark_stale(db_path, DOC_A)
    result = _list(db_path, "--columns", "stale,mpn")
    assert result.exit_code == 0, result.output
    assert _header(result.output).count("stale") == 1


def test_columns_with_wide_is_rejected(db_path: Path) -> None:
    result = _list(db_path, "--columns", "mpn", "--wide")
    assert result.exit_code != 0
    assert "drop --wide" in result.output


def test_columns_with_s3_is_rejected(db_path: Path) -> None:
    result = _list(db_path, "--columns", "mpn", "--s3")
    assert result.exit_code != 0
    assert "--s3" in result.output


def test_empty_columns_value_is_rejected(db_path: Path) -> None:
    result = _list(db_path, "--columns", "")
    assert result.exit_code != 0
    assert "at least one column" in result.output


def test_help_documents_field_columns() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--help"])
    assert result.exit_code == 0, result.output
    assert "--columns" in result.output
    assert "registered field" in result.output
    assert "attr" not in result.output
