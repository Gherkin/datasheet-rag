"""CLI-level tests for project scoping (.rag.toml discovery, --global, --project-id).

Exercises ``rag list`` end-to-end against a real on-disk SQLite store —
the simplest of the scoped query commands (no embeddings/vector search
involved) — to verify the precedence chain implemented by
``resolve_cli_project_id`` actually reaches the command output.
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


def _chunk(chunk_id: str, doc_id: str) -> Chunk:
    md = ChunkMetadata(
        doc_id=doc_id,
        doc_title=f"Doc {doc_id}",
        chapter_title="",
        section_title="",
        page_numbers=[1],
        layout_type=LayoutType.TEXT,
        context_string="",
    )
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        level=ChunkLevel.MACRO,
        text="some text",
        context_text="some text",
        token_count=2,
        metadata=md,
    )


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """A real on-disk store with two docs tagged to different projects."""
    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)
    insert_chunks(conn, [_chunk("doc-a:0", "doc-a")], project_id="proj-a")
    insert_chunks(conn, [_chunk("doc-b:0", "doc-b")], project_id="proj-b")
    conn.commit()
    conn.close()
    return path


@pytest.fixture(autouse=True)
def _clear_project_config_cache() -> None:
    """``get_project_config`` is lru_cache'd on cwd — reset between tests."""
    get_project_config.cache_clear()
    yield
    get_project_config.cache_clear()


def _run_list(db_path: Path, *extra_args: str) -> str:
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--db", str(db_path), *extra_args])
    assert result.exit_code == 0, result.output
    return result.output


def test_rag_toml_scopes_list_by_default(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".rag.toml").write_text('project_id = "proj-a"\n')
    monkeypatch.chdir(project_dir)
    get_project_config.cache_clear()

    output = _run_list(db_path)

    assert "doc-a" in output
    assert "doc-b" not in output


def test_global_flag_overrides_rag_toml_scoping(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".rag.toml").write_text('project_id = "proj-a"\n')
    monkeypatch.chdir(project_dir)
    get_project_config.cache_clear()

    output = _run_list(db_path, "--global")

    assert "doc-a" in output
    assert "doc-b" in output


def test_explicit_project_id_overrides_global_and_rag_toml(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".rag.toml").write_text('project_id = "proj-a"\n')
    monkeypatch.chdir(project_dir)
    get_project_config.cache_clear()

    output = _run_list(db_path, "--global", "--project-id", "proj-b")

    assert "doc-b" in output
    assert "doc-a" not in output


def test_no_rag_toml_lists_everything_by_default(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.chdir(empty_dir)
    get_project_config.cache_clear()

    output = _run_list(db_path)

    assert "doc-a" in output
    assert "doc-b" in output
