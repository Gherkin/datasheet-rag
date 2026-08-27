"""CLI-level tests for `rag get fig` (the CLI equivalent of the MCP
``get_figure`` tool — GH follow-up to issue #5's chunk-id fix).

Exercises the command end-to-end against a real on-disk SQLite store with a
real PNG file on disk for the figure's ``figure_image_path``, driving the
CLI through Click's ``CliRunner`` the same way ``test_cli_get_chunk.py`` does.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from click.testing import CliRunner

from datasheet_rag.cli import SHORT_DOC_ID_LEN, cli
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.project_config import get_project_config
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

DOC_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

# A real, tiny, valid 1x1 transparent PNG so image_bytes() round-trips cleanly.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _chunk(
    doc_id: str,
    index: int,
    text: str,
    *,
    layout_type: LayoutType = LayoutType.TEXT,
    figure_caption: str | None = None,
    figure_description: str | None = None,
    figure_image_path: str | None = None,
    figure_s3_key: str | None = None,
) -> Chunk:
    chunk_id = f"{doc_id}:L2:{index}"
    md = ChunkMetadata(
        doc_id=doc_id,
        doc_title="Doc A",
        chapter_title="",
        section_title="Thermal Characteristics",
        page_numbers=[7],
        layout_type=layout_type,
        context_string="",
    )
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        level=ChunkLevel.MICRO,
        text=text,
        context_text=text,
        token_count=max(1, len(text.split())),
        metadata=md,
        figure_caption=figure_caption,
        figure_description=figure_description,
        figure_image_path=figure_image_path,
        figure_s3_key=figure_s3_key,
    )


@pytest.fixture()
def figure_path(tmp_path: Path) -> Path:
    p = tmp_path / "figures" / "fig.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_PNG_BYTES)
    return p


@pytest.fixture()
def db_path(tmp_path: Path, figure_path: Path) -> Path:
    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)

    fig = _chunk(
        DOC_A,
        0,
        "See figure for thermal derating curve.",
        layout_type=LayoutType.FIGURE,
        figure_caption="Fig 1: Thermal derating curve",
        figure_description="A line chart showing power derating vs ambient temperature.",
        figure_image_path=str(figure_path),
    )
    text_chunk = _chunk(DOC_A, 1, "Plain paragraph, not a figure.")
    orphan_fig = _chunk(
        DOC_A,
        2,
        "Figure chunk with no usable image source.",
        layout_type=LayoutType.FIGURE,
    )

    insert_chunks(conn, [fig, text_chunk, orphan_fig], project_id="proj-a")
    conn.commit()
    conn.close()
    return path


@pytest.fixture(autouse=True)
def _clear_project_config_cache() -> None:
    get_project_config.cache_clear()
    yield
    get_project_config.cache_clear()


def test_get_figure_saves_exact_bytes_by_full_id(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_id = f"{DOC_A}:L2:0"
    workdir = tmp_path / "work1"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    result = CliRunner().invoke(cli, ["get", "fig", full_id, "--db", str(db_path)])
    assert result.exit_code == 0, result.output

    expected_name = f"{DOC_A[:SHORT_DOC_ID_LEN]}_L2_0.png"
    saved = workdir / expected_name
    assert saved.exists()
    assert saved.read_bytes() == _PNG_BYTES
    assert expected_name in result.output
    assert "Fig 1: Thermal derating curve" in result.output
    assert "A line chart showing power derating" in result.output
    assert "page=7" in result.output
    assert "section=Thermal Characteristics" in result.output


def test_get_figure_by_abbreviated_doc_id(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    short_id = f"{DOC_A[:SHORT_DOC_ID_LEN]}:L2:0"
    workdir = tmp_path / "work2"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    result = CliRunner().invoke(cli, ["get", "fig", short_id, "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    saved = workdir / f"{DOC_A[:SHORT_DOC_ID_LEN]}_L2_0.png"
    assert saved.exists()
    assert saved.read_bytes() == _PNG_BYTES


def test_get_figure_explicit_output_path(tmp_path: Path, db_path: Path) -> None:
    full_id = f"{DOC_A}:L2:0"
    dest = tmp_path / "custom-name.png"
    result = CliRunner().invoke(cli, ["get", "fig", full_id, "--db", str(db_path), "-o", str(dest)])
    assert result.exit_code == 0, result.output
    assert dest.exists()
    assert dest.read_bytes() == _PNG_BYTES


def test_get_figure_output_directory(tmp_path: Path, db_path: Path) -> None:
    full_id = f"{DOC_A}:L2:0"
    out_dir = tmp_path / "figs_out"
    out_dir.mkdir()
    result = CliRunner().invoke(
        cli, ["get", "fig", full_id, "--db", str(db_path), "-o", str(out_dir) + "/"]
    )
    assert result.exit_code == 0, result.output
    expected = out_dir / f"{DOC_A[:SHORT_DOC_ID_LEN]}_L2_0.png"
    assert expected.exists()
    assert expected.read_bytes() == _PNG_BYTES


def test_get_figure_rejects_non_figure_chunk(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    full_id = f"{DOC_A}:L2:1"
    result = CliRunner().invoke(cli, ["get", "fig", full_id, "--db", str(db_path)])
    assert result.exit_code != 0
    assert "not a figure" in result.output


def test_get_figure_missing_image_source(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    full_id = f"{DOC_A}:L2:2"
    result = CliRunner().invoke(cli, ["get", "fig", full_id, "--db", str(db_path)])
    assert result.exit_code != 0
    assert "no usable figure source" in result.output


def test_get_figure_unknown_chunk_id(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["get", "fig", f"{DOC_A}:L2:999", "--db", str(db_path)])
    assert result.exit_code != 0
    assert "unknown chunk_id" in result.output


def test_get_figure_invalid_chunk_id_format(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["get", "fig", "not-a-chunk-id", "--db", str(db_path)])
    assert result.exit_code != 0
    assert "doesn't look like a chunk ID" in result.output
