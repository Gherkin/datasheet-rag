"""CLI-level tests for `rag get doc` (GH issue #14 — merges the old
`rag download` and `rag open` commands: default behavior downloads the PDF
to disk, `--host` instead starts the loopback PDF.js viewer server, same as
the old `rag open`).

Exercises the command end-to-end against a real on-disk SQLite store and a
real (fake-content) PDF file on the local store, driving the CLI through
Click's ``CliRunner`` the same way ``test_cli_get_page.py`` does. The
``--host`` path additionally makes a real HTTP request against the loopback
server it starts, to confirm the primed PDF bytes are actually served.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest
from click.testing import CliRunner

from datasheet_rag.cli import SHORT_DOC_ID_LEN, cli
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.project_config import get_project_config
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

DOC_ID = "a" * 64          # has chunks + a real PDF on disk
DOC_NO_PDF = "b" * 64      # has chunks but no PDF file on disk
_FAKE_PDF_BYTES = b"%PDF-1.4 fake pdf content for tests\n%%EOF"


def _chunk(doc_id: str, title: str = "Widget Datasheet") -> Chunk:
    md = ChunkMetadata(
        doc_id=doc_id, doc_title=title, section_title="", chapter_title="",
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

    (pdf_dir / f"{DOC_ID}.pdf").write_bytes(_FAKE_PDF_BYTES)
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


@pytest.fixture(autouse=True)
def _clear_pdf_viewer_cache() -> None:
    # pdf_viewer._pdf_cache is a process-global dict keyed by doc_id, and
    # --host mode primes it via prime_pdf_cache(). Since DOC_ID here can
    # collide with other test modules' doc_id fixtures across the shared
    # test process, make sure our --host tests don't leak fake PDF bytes
    # into (or pick up stale bytes from) any other test.
    from datasheet_rag import pdf_viewer

    pdf_viewer._pdf_cache.clear()
    yield
    pdf_viewer._pdf_cache.clear()


# ---------------------------------------------------------------------------
# Default (download) mode
# ---------------------------------------------------------------------------


def test_get_doc_default_name(tmp_path, db_path, monkeypatch) -> None:
    workdir = tmp_path / "work1"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    result = CliRunner().invoke(cli, ["get", "doc", DOC_ID, "--db", str(db_path)])
    assert result.exit_code == 0, result.output

    saved = workdir / "Widget-Datasheet.pdf"
    assert saved.exists()
    assert saved.read_bytes() == _FAKE_PDF_BYTES
    assert "Widget-Datasheet.pdf" in result.output


def test_get_doc_explicit_output_path(tmp_path, db_path) -> None:
    dest = tmp_path / "custom-name.pdf"
    result = CliRunner().invoke(
        cli, ["get", "doc", DOC_ID, "--db", str(db_path), "-o", str(dest)]
    )
    assert result.exit_code == 0, result.output
    assert dest.exists()
    assert dest.read_bytes() == _FAKE_PDF_BYTES


def test_get_doc_output_directory(tmp_path, db_path) -> None:
    out_dir = tmp_path / "docs_out"
    out_dir.mkdir()
    result = CliRunner().invoke(
        cli, ["get", "doc", DOC_ID, "--db", str(db_path), "-o", str(out_dir) + "/"]
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "Widget-Datasheet.pdf").exists()


def test_get_doc_abbreviated_doc_id(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    short_id = DOC_ID[:SHORT_DOC_ID_LEN]
    result = CliRunner().invoke(cli, ["get", "doc", short_id, "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "Widget-Datasheet.pdf").exists()


def test_get_doc_missing_pdf_file_errors(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["get", "doc", DOC_NO_PDF, "--db", str(db_path)])
    assert result.exit_code != 0
    assert "PDF not found" in result.output


def test_get_doc_unknown_doc_id_errors(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["get", "doc", "z" * 64, "--db", str(db_path)])
    assert result.exit_code != 0
    assert "No ingested document matches" in result.output


def test_get_doc_document_alias(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["get", "document", DOC_ID, "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "Widget-Datasheet.pdf").exists()


# ---------------------------------------------------------------------------
# --host mode (old `rag open`)
# ---------------------------------------------------------------------------


def test_get_doc_host_serves_pdf_bytes(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    # The command's serve loop is `while True: time.sleep(3600)` until
    # Ctrl+C; make the first sleep raise KeyboardInterrupt so the command
    # returns immediately once the server is up, instead of hanging.
    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    opened_urls: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    result = CliRunner().invoke(
        cli, ["get", "doc", DOC_ID, "--host", "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output
    assert "PDF viewer running" in result.output
    assert f"/viewer/{DOC_ID}#page=1" in result.output
    assert "Stopped." in result.output
    assert len(opened_urls) == 1
    assert f"127.0.0.1" in opened_urls[0]
    assert f"/viewer/{DOC_ID}#page=1" in opened_urls[0]

    # Confirm the server actually started and serves the primed PDF bytes —
    # extract the port from the printed 127.0.0.1 URL and hit /pdf/<doc_id>.
    port = opened_urls[0].split(":")[2].split("/")[0]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/pdf/{DOC_ID}", timeout=5) as resp:
        assert resp.read() == _FAKE_PDF_BYTES


def test_get_doc_host_no_launch_skips_browser(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    opened_urls: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

    result = CliRunner().invoke(
        cli, ["get", "doc", DOC_ID, "--host", "--no-launch", "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output
    assert opened_urls == []
    assert "Opened the 127.0.0.1 link" not in result.output


def test_get_doc_host_custom_page_in_urls(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("time.sleep", lambda *_a, **_kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr("webbrowser.open", lambda url: None)

    result = CliRunner().invoke(
        cli, ["get", "doc", DOC_ID, "--host", "--page", "7", "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output
    assert f"/viewer/{DOC_ID}#page=7" in result.output


def test_get_doc_host_missing_pdf_errors(tmp_path, db_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli, ["get", "doc", DOC_NO_PDF, "--host", "--db", str(db_path)]
    )
    assert result.exit_code != 0
    assert "PDF not found" in result.output
