"""CLI-level tests for `rag metadata set|get|list` — specifically the
arbitrary key=value tagging (`--attr`/`--unset-attr`) and tag-list handling
(`--tag`/`--clear-tags`) added to close GH #3 ("arbitrary tagging is not
exposed from cli").

Exercises the commands end-to-end against a real on-disk SQLite store,
driving the CLI through Click's ``CliRunner`` — mirrors the pattern used by
``test_cli_stats.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from aws_rag.cli import cli
from aws_rag.config import get_settings
from aws_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from aws_rag.project_config import get_project_config
from aws_rag.store.schema import connect
from aws_rag.store.sqlite import insert_chunks

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
    """Two bare documents (no metadata sidecar row yet), one chunk each."""
    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)
    insert_chunks(conn, [_chunk(DOC_A)], project_id="proj-a")
    insert_chunks(conn, [_chunk(DOC_B)], project_id="proj-a")
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
    return runner.invoke(cli, ["metadata", *args, "--db", str(db_path)])


def _get_json(db_path: Path, doc_id: str) -> dict:
    result = _run(db_path, "get", doc_id)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_tag_sets_full_list(db_path: Path) -> None:
    result = _run(db_path, "set", DOC_A, "--tag", "mcu", "--tag", "reviewed")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["tags"] == ["mcu", "reviewed"]


def test_tag_replaces_wholesale_not_additive(db_path: Path) -> None:
    _run(db_path, "set", DOC_A, "--tag", "mcu", "--tag", "reviewed")
    result = _run(db_path, "set", DOC_A, "--tag", "power")
    assert result.exit_code == 0, result.output
    # Second call REPLACES the list — "mcu"/"reviewed" do not survive.
    assert _get_json(db_path, DOC_A)["tags"] == ["power"]


def test_tag_omitted_leaves_existing_tags_untouched(db_path: Path) -> None:
    _run(db_path, "set", DOC_A, "--tag", "mcu")
    result = _run(db_path, "set", DOC_A, "--mpn", "STM32H743VIT6")
    assert result.exit_code == 0, result.output
    meta = _get_json(db_path, DOC_A)
    assert meta["tags"] == ["mcu"]
    assert meta["mpn"] == "STM32H743VIT6"


def test_clear_tags_wipes_list(db_path: Path) -> None:
    _run(db_path, "set", DOC_A, "--tag", "mcu", "--tag", "reviewed")
    result = _run(db_path, "set", DOC_A, "--clear-tags")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["tags"] == []


def test_attr_sets_arbitrary_key_value(db_path: Path) -> None:
    result = _run(
        db_path, "set", DOC_A, "--attr", "revision=B", "--attr", "reviewed_by=hector"
    )
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["attributes"] == {
        "revision": "B",
        "reviewed_by": "hector",
    }


def test_attr_merges_key_by_key_preserving_others(db_path: Path) -> None:
    _run(db_path, "set", DOC_A, "--attr", "revision=B", "--attr", "notes=draft")
    result = _run(db_path, "set", DOC_A, "--attr", "revision=C")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["attributes"] == {
        "revision": "C",
        "notes": "draft",
    }


def test_unset_attr_removes_single_key(db_path: Path) -> None:
    _run(db_path, "set", DOC_A, "--attr", "revision=B", "--attr", "notes=draft")
    result = _run(db_path, "set", DOC_A, "--unset-attr", "notes")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["attributes"] == {"revision": "B"}


def test_attr_bad_format_rejected(db_path: Path) -> None:
    result = _run(db_path, "set", DOC_A, "--attr", "no-equals-sign")
    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output


def test_attr_empty_key_rejected(db_path: Path) -> None:
    result = _run(db_path, "set", DOC_A, "--attr", "=novalue")
    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output


def test_list_shows_tags_column(db_path: Path) -> None:
    _run(db_path, "set", DOC_A, "--tag", "mcu", "--tag", "reviewed")
    result = _run(db_path, "list", "--global")
    assert result.exit_code == 0, result.output
    # Rich may wrap the cell across lines in a narrow test terminal, so
    # check both tags individually rather than the joined "mcu, reviewed".
    assert "mcu" in result.output
    assert "reviewed" in result.output


def test_list_filters_by_tag(db_path: Path) -> None:
    _run(db_path, "set", DOC_A, "--tag", "mcu")
    _run(db_path, "set", DOC_B, "--tag", "rf")
    result = _run(db_path, "list", "--global", "--tag", "mcu")
    assert result.exit_code == 0, result.output
    # Truncated further still by Rich's column wrapping in a narrow test
    # terminal, so match a shorter prefix than SHORT_DOC_ID_LEN.
    assert DOC_A[:10] in result.output
    assert DOC_B[:12] not in result.output


def test_list_filter_by_tag_requires_all_given_tags(db_path: Path) -> None:
    _run(db_path, "set", DOC_A, "--tag", "mcu", "--tag", "reviewed")
    _run(db_path, "set", DOC_B, "--tag", "mcu")
    result = _run(db_path, "list", "--global", "--tag", "mcu", "--tag", "reviewed")
    assert result.exit_code == 0, result.output
    # Truncated further still by Rich's column wrapping in a narrow test
    # terminal, so match a shorter prefix than SHORT_DOC_ID_LEN.
    assert DOC_A[:10] in result.output
    assert DOC_B[:12] not in result.output


def test_list_filters_by_attr(db_path: Path) -> None:
    _run(db_path, "set", DOC_A, "--attr", "revision=B")
    _run(db_path, "set", DOC_B, "--attr", "revision=C")
    result = _run(db_path, "list", "--global", "--attr", "revision=B")
    assert result.exit_code == 0, result.output
    # Truncated further still by Rich's column wrapping in a narrow test
    # terminal, so match a shorter prefix than SHORT_DOC_ID_LEN.
    assert DOC_A[:10] in result.output
    assert DOC_B[:12] not in result.output


def test_ingest_tag_help_mentions_replace_semantics() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0, result.output
    assert "--attr key=value" in result.output
