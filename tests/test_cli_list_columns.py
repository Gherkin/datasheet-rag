"""CLI-level tests for `rag list --columns` (GH #38, "rag list should take a
display format").

The two built-in views show a fixed set of columns, so a document's free-form
`attributes` — the keys `rag metadata --attr` writes — had no way of reaching
the terminal. `--columns` names the columns to print, falling back to an
attribute lookup for any name that isn't a built-in column.

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
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

DOC_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
DOC_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _chunk(doc_id: str) -> Chunk:
    md = ChunkMetadata(
        doc_id=doc_id,
        doc_title=f"Doc {doc_id[:8]}",
        chapter_title="",
        section_title="",
        page_numbers=[1],
        layout_type=LayoutType.TEXT,
        context_string="",
    )
    return Chunk(
        id=f"{doc_id}:L1:0",
        doc_id=doc_id,
        level=ChunkLevel.MICRO,
        text="chunk 0",
        context_text="chunk 0",
        token_count=2,
        metadata=md,
    )


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Two documents with sidecar rows: A carries attributes, B carries one."""
    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)
    insert_chunks(conn, [_chunk(DOC_A)], project_id="proj-a")
    insert_chunks(conn, [_chunk(DOC_B)], project_id="proj-a")
    conn.commit()
    conn.close()

    _meta(path, DOC_A, "--project-id", "proj-a", "--mpn", "STM32H743VIT6", "--tag", "mcu")
    _meta(path, DOC_A, "--attr", "revision=B", "--attr", "reviewed_by=hector")
    _meta(path, DOC_B, "--mpn", "TLV6722", "--attr", "revision=C")
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

    At 80, Rich squeezes doc_id until it wraps mid-hash, which would have
    these tests asserting on the wrapping rather than on the columns.
    """
    runner = CliRunner()
    return runner.invoke(
        cli, ["list", "--global", *args, "--db", str(db_path)], env={"COLUMNS": "200"}
    )


def _header(output: str) -> str:
    """The table's header row — the line holding the column labels."""
    return next(line for line in output.splitlines() if "doc_id" in line)


def _row(output: str, doc_id: str) -> str:
    return next(line for line in output.splitlines() if doc_id[:10] in line)


def test_columns_shows_arbitrary_attribute(db_path: Path) -> None:
    result = _list(db_path, "--columns", "mpn,revision,reviewed_by")
    assert result.exit_code == 0, result.output
    assert "revision" in _header(result.output)
    assert "B" in _row(result.output, DOC_A)
    assert "hector" in _row(result.output, DOC_A)
    # B has revision but no reviewer — the missing cell reads as an em-dash.
    assert "C" in _row(result.output, DOC_B)
    assert "hector" not in _row(result.output, DOC_B)


def test_columns_repeatable_and_comma_separated_agree(db_path: Path) -> None:
    comma = _list(db_path, "--columns", "mpn,revision")
    repeated = _list(db_path, "-c", "mpn", "-c", "revision")
    assert comma.exit_code == 0, comma.output
    assert repeated.exit_code == 0, repeated.output
    assert comma.output == repeated.output


def test_columns_replaces_the_built_in_views(db_path: Path) -> None:
    result = _list(db_path, "--columns", "revision")
    assert result.exit_code == 0, result.output
    header = _header(result.output)
    assert "revision" in header
    for dropped in ("title", "chunks", "pages", "ingested"):
        assert dropped not in header


def test_columns_wins_over_the_filter_implied_wide_view(db_path: Path) -> None:
    """A filter implies --wide, but an explicit column list still decides."""
    result = _list(db_path, "--tag", "mcu", "--columns", "revision")
    assert result.exit_code == 0, result.output
    assert _header(result.output).split("┃") == _header(
        _list(db_path, "--columns", "revision").output
    ).split("┃")
    assert DOC_A[:10] in result.output
    assert DOC_B[:10] not in result.output


def test_doc_id_is_always_shown(db_path: Path) -> None:
    result = _list(db_path, "--columns", "revision")
    assert result.exit_code == 0, result.output
    assert DOC_A[:10] in result.output


def test_built_in_columns_are_selectable(db_path: Path) -> None:
    result = _list(db_path, "--columns", "chunks,tags,mpn")
    assert result.exit_code == 0, result.output
    assert "mcu" in _row(result.output, DOC_A)
    assert "STM32H743VIT6" in _row(result.output, DOC_A)


def test_model_field_names_are_accepted_as_aliases(db_path: Path) -> None:
    """`project_id` is what `rag metadata` prints; it must not read as an attr."""
    result = _list(db_path, "--columns", "project_id")
    assert result.exit_code == 0, result.output
    assert "proj-a" in _row(result.output, DOC_A)


def test_attr_prefix_forces_the_attribute_reading(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--attr", "tags=from-attributes")
    result = _list(db_path, "--columns", "tags,attr:tags")
    assert result.exit_code == 0, result.output
    row = _row(result.output, DOC_A)
    # The built-in column shows the tag list, the prefixed one the attribute.
    assert "mcu" in row
    assert "from-attributes" in row


def test_unknown_attribute_column_warns_instead_of_printing_blanks(db_path: Path) -> None:
    result = _list(db_path, "--columns", "revsion")
    assert result.exit_code == 0, result.output
    assert "No listed document has the attribute" in result.output
    assert "revsion" in result.output


def test_known_attribute_column_does_not_warn(db_path: Path) -> None:
    result = _list(db_path, "--columns", "revision")
    assert result.exit_code == 0, result.output
    assert "No listed document has the attribute" not in result.output


def test_stale_marker_still_shown_with_custom_columns(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--attr", "needs_reembed=1")
    result = _list(db_path, "--columns", "mpn")
    assert result.exit_code == 0, result.output
    assert "stale" in _header(result.output)
    assert "re-embed" in _row(result.output, DOC_A)


def test_stale_column_not_duplicated_when_asked_for(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--attr", "needs_reembed=1")
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


def test_help_documents_attribute_columns() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--help"])
    assert result.exit_code == 0, result.output
    assert "--columns" in result.output
    assert "attribute" in result.output
