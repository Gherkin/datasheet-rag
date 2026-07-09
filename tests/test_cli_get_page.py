"""CLI-level tests for `rag get page` (the CLI equivalent of the MCP
``show_page`` tool — renders a single PDF page to a static PNG on disk).

Exercises the command end-to-end: a real multi-page PDF (built with
PyMuPDF) is written to the local PDF store, a real doc_id/chunk row is
inserted into a SQLite store so doc_id resolution has something to match,
and the CLI is driven through Click's ``CliRunner``. The rendered PNG is
compared byte-for-byte against calling ``pdf2image.convert_from_bytes``
directly on the same PDF/page/dpi, so the test doesn't hardcode fragile
pixel dimensions.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from click.testing import CliRunner
from pdf2image import convert_from_bytes

from aws_rag.cli import SHORT_DOC_ID_LEN, cli
from aws_rag.config import get_settings
from aws_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from aws_rag.project_config import get_project_config
from aws_rag.store.schema import connect
from aws_rag.store.sqlite import insert_chunks

DOC_ID = "a" * 64          # has chunks + a real PDF on disk
DOC_NO_PDF = "b" * 64      # has chunks but no PDF file on disk
PAGE_COUNT = 3


def _make_pdf(path: Path, n_pages: int = PAGE_COUNT) -> None:
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((50, 100), f"PAGE {i + 1}", fontsize=24)
    doc.save(str(path))
    doc.close()


def _chunk(doc_id: str) -> Chunk:
    md = ChunkMetadata(
        doc_id=doc_id, doc_title="Doc", section_title="", chapter_title="",
        page_numbers=[1], layout_type=LayoutType.TEXT, context_string="",
    )
    return Chunk(
        id=f"{doc_id}:L2:0", doc_id=doc_id, level=ChunkLevel.MICRO,
        text="t", context_text="t", token_count=1, metadata=md,
    )


@pytest.fixture()
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    settings = get_settings()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    monkeypatch.setattr(settings, "pdf_dir", pdf_dir)
    monkeypatch.setattr(settings, "s3_bucket", None)

    _make_pdf(pdf_dir / f"{DOC_ID}.pdf")
    # DOC_NO_PDF deliberately has no file written under pdf_dir.

    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)
    insert_chunks(conn, [_chunk(DOC_ID)], project_id="proj-a")
    insert_chunks(conn, [_chunk(DOC_NO_PDF)], project_id="proj-a")
    conn.commit()
    conn.close()
    return path


@pytest.fixture(autouse=True)
def _clear_project_config_cache() -> None:
    get_project_config.cache_clear()
    yield
    get_project_config.cache_clear()


def _expected_png_bytes(pdf_path: Path, page: int, dpi: int = 150) -> bytes:
    import io

    images = convert_from_bytes(pdf_path.read_bytes(), first_page=page, last_page=page, dpi=dpi)
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return buf.getvalue()


def test_get_page_positional_arg(tmp_path, db_path, monkeypatch) -> None:
    workdir = tmp_path / "work1"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    result = CliRunner().invoke(cli, ["get", "page", DOC_ID, "2", "--db", str(db_path)])
    assert result.exit_code == 0, result.output

    expected_name = f"{DOC_ID[:SHORT_DOC_ID_LEN]}_p2.png"
    saved = workdir / expected_name
    assert saved.exists()
    assert saved.read_bytes() == _expected_png_bytes(
        get_settings().pdf_dir / f"{DOC_ID}.pdf", 2
    )
    assert expected_name in result.output


def test_get_page_via_option_flag(tmp_path, db_path, monkeypatch) -> None:
    workdir = tmp_path / "work2"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    result = CliRunner().invoke(cli, ["get", "page", DOC_ID, "--page", "3", "--db", str(db_path)])
    assert result.exit_code == 0, result.output

    saved = workdir / f"{DOC_ID[:SHORT_DOC_ID_LEN]}_p3.png"
    assert saved.exists()
    assert saved.read_bytes() == _expected_png_bytes(
        get_settings().pdf_dir / f"{DOC_ID}.pdf", 3
    )


def test_get_page_positional_and_option_disagree_are_distinct_renders(
    tmp_path, db_path, monkeypatch
) -> None:
    # Sanity check that page 1 and page 2 really do render different bytes
    # (i.e. our byte-equality assertions above are actually discriminating).
    workdir = tmp_path / "work3"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    r1 = CliRunner().invoke(cli, ["get", "page", DOC_ID, "1", "--db", str(db_path)])
    r2 = CliRunner().invoke(cli, ["get", "page", DOC_ID, "2", "--db", str(db_path)])
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output

    p1 = (workdir / f"{DOC_ID[:SHORT_DOC_ID_LEN]}_p1.png").read_bytes()
    p2 = (workdir / f"{DOC_ID[:SHORT_DOC_ID_LEN]}_p2.png").read_bytes()
    assert p1 != p2


def test_get_page_both_positional_and_option_rejected(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli, ["get", "page", DOC_ID, "2", "--page", "2", "--db", str(db_path)]
    )
    assert result.exit_code != 0
    assert "not both" in result.output


def test_get_page_neither_positional_nor_option_rejected(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["get", "page", DOC_ID, "--db", str(db_path)])
    assert result.exit_code != 0
    assert "PAGE is required" in result.output


def test_get_page_explicit_output_path(tmp_path, db_path) -> None:
    dest = tmp_path / "custom-name.png"
    result = CliRunner().invoke(
        cli, ["get", "page", DOC_ID, "1", "--db", str(db_path), "-o", str(dest)]
    )
    assert result.exit_code == 0, result.output
    assert dest.exists()
    assert dest.read_bytes() == _expected_png_bytes(get_settings().pdf_dir / f"{DOC_ID}.pdf", 1)


def test_get_page_output_directory(tmp_path, db_path) -> None:
    out_dir = tmp_path / "pages_out"
    out_dir.mkdir()
    result = CliRunner().invoke(
        cli, ["get", "page", DOC_ID, "1", "--db", str(db_path), "-o", str(out_dir) + "/"]
    )
    assert result.exit_code == 0, result.output
    expected = out_dir / f"{DOC_ID[:SHORT_DOC_ID_LEN]}_p1.png"
    assert expected.exists()


def test_get_page_abbreviated_doc_id(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    short_id = DOC_ID[:SHORT_DOC_ID_LEN]
    result = CliRunner().invoke(cli, ["get", "page", short_id, "1", "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / f"{short_id}_p1.png").exists()


def test_get_page_custom_dpi_changes_output(tmp_path, db_path, monkeypatch) -> None:
    workdir = tmp_path / "work_dpi"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    result = CliRunner().invoke(
        cli, ["get", "page", DOC_ID, "1", "--dpi", "72", "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output
    saved = workdir / f"{DOC_ID[:SHORT_DOC_ID_LEN]}_p1.png"
    assert saved.read_bytes() == _expected_png_bytes(
        get_settings().pdf_dir / f"{DOC_ID}.pdf", 1, dpi=72
    )
    # Different dpi than the default (150) must produce a different-sized file.
    default_bytes = _expected_png_bytes(get_settings().pdf_dir / f"{DOC_ID}.pdf", 1, dpi=150)
    assert saved.read_bytes() != default_bytes


def test_get_page_out_of_range_errors(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli, ["get", "page", DOC_ID, str(PAGE_COUNT + 10), "--db", str(db_path)]
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_get_page_missing_pdf_file_errors(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["get", "page", DOC_NO_PDF, "1", "--db", str(db_path)])
    assert result.exit_code != 0
    assert "PDF not found" in result.output


def test_get_page_unknown_doc_id_errors(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["get", "page", "z" * 64, "1", "--db", str(db_path)])
    assert result.exit_code != 0
    assert "No ingested document matches" in result.output
