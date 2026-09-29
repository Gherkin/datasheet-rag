"""How the CLI reports errors (GH #36).

A mistake the user can fix — an unreachable server, a mistyped doc_id, a bad
option value — must come out as a one-line message and exit code 1. Anything
else is a bug: the traceback stays, under a line asking for it to be
reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

import datasheet_rag.cli as cli_mod
from datasheet_rag.backend.base import RagServerError
from datasheet_rag.backend.remote import RemoteBackend
from datasheet_rag.cli import cli
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.project_config import get_project_config
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

DOC_A = "a" * 64
_PREFACE = "An unexpected error happened"


@pytest.fixture(autouse=True)
def _clear_project_config_cache() -> None:
    get_project_config.cache_clear()
    yield
    get_project_config.cache_clear()


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    md = ChunkMetadata(
        doc_id=DOC_A,
        doc_title="Doc A",
        chapter_title="",
        section_title="",
        page_numbers=[1],
        layout_type=LayoutType.TEXT,
        context_string="",
    )
    chunk = Chunk(
        id=f"{DOC_A}:L2:0",
        doc_id=DOC_A,
        level=ChunkLevel.MICRO,
        text="t",
        context_text="t",
        token_count=1,
        metadata=md,
    )
    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)
    insert_chunks(conn, [chunk], project_id="proj-a")
    conn.commit()
    conn.close()
    return path


class _Backend:
    """Resolves any doc_id; every other call raises ``error``."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def resolve_doc_id(self, doc_id: str) -> str:
        return doc_id

    def __getattr__(self, name: str) -> Any:
        def _fail(*_a: Any, **_kw: Any) -> Any:
            raise self._error

        return _fail


def _run_with(monkeypatch: pytest.MonkeyPatch, error: Exception, *args: str):
    monkeypatch.setattr(cli_mod, "_backend_for", lambda *_a, **_kw: _Backend(error))
    return CliRunner().invoke(cli, list(args))


# --- server errors: one line, whatever the command ---------------------------

_COMMANDS = [
    ("list", "--global"),
    ("metadata", DOC_A),
    ("metadata", DOC_A, "--title", "New title"),
    ("delete", DOC_A, "--yes"),
    ("get", "doc", DOC_A),
    ("inspect", "figures", "--global"),
    ("repair", "titles"),
    ("repair", "figures"),
]


def _timeout() -> RagServerError:
    """A RagServerError as RemoteBackend._request raises it for a timeout."""
    err = RagServerError(0, "timed out")
    err.__cause__ = httpx.ReadTimeout("timed out")
    return err


@pytest.mark.parametrize("args", _COMMANDS, ids=[" ".join(a[:2]) for a in _COMMANDS])
def test_server_timeout_is_a_clean_error(monkeypatch: pytest.MonkeyPatch, args: tuple) -> None:
    result = _run_with(monkeypatch, _timeout(), *args)
    assert result.exit_code == 1, result.output
    assert "Could not get an answer from the RAG server (timed out)" in result.output
    assert "Traceback" not in result.output
    assert _PREFACE not in result.output


def test_status_0_with_an_answer_is_not_called_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The embedding-dimension checks raise status 0 for an answer the server gave.
    err = RagServerError(0, "the server's store holds 768-dimensional vectors")
    result = _run_with(monkeypatch, err, "list", "--global")
    assert result.exit_code == 1
    assert "768-dimensional" in result.output
    assert "Could not get an answer" not in result.output


def test_server_401_keeps_its_specific_message(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_with(monkeypatch, RagServerError(401, "bad token"), "list", "--global")
    assert result.exit_code == 1
    assert "RAG_SERVER_TOKEN" in result.output


# --- unknown errors: preface, then the traceback -----------------------------


def test_unexpected_error_keeps_traceback_with_preface(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = RuntimeError("something broke")
    result = _run_with(monkeypatch, boom, "list", "--global")
    assert result.exit_code == 1
    assert _PREFACE in result.stderr
    assert "report the following to the developer" in result.stderr
    # The exception propagates unchanged, so Python prints its traceback.
    assert result.exception is boom


# --- mistyped doc_id ----------------------------------------------------------


def test_search_unknown_doc_id_is_a_clean_error(db_path: Path) -> None:
    result = CliRunner().invoke(
        cli, ["search", "x", "--mode", "keyword", "--doc-id", "zzzz", "--db", str(db_path)]
    )
    assert result.exit_code == 1, result.output
    assert "No ingested document matches" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.parametrize("k", ["0", "201"])
def test_search_k_past_the_pool_is_a_clean_error(db_path: Path, k: str) -> None:
    # The backend ranks only SEARCH_POOL hits; a bigger -k would be cut silently.
    result = CliRunner().invoke(
        cli, ["search", "x", "--mode", "keyword", "-k", k, "--db", str(db_path)]
    )
    assert result.exit_code == 2, result.output
    assert "must be 1 to 200" in result.output


def test_eval_on_a_broken_keyword_index_is_a_clean_error(db_path: Path, tmp_path: Path) -> None:
    # GH #91: the eval refuses a store whose keyword index is out of sync.
    conn = connect(db_path)
    conn.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('delete-all')")
    conn.commit()
    conn.close()
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        f'{{"question": "t", "category": "identifier", "doc_id": "{DOC_A}"}}\n',
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        cli, ["eval", "run", "--mode", "keyword", "--set", str(golden), "--db", str(db_path)]
    )
    assert result.exit_code == 1, result.output
    assert "rag repair fts" in result.output
    assert _PREFACE not in result.output


@pytest.mark.parametrize("extra", [(), ("--mpn", "X1")], ids=["read", "write"])
def test_metadata_unknown_doc_id_is_a_clean_error(db_path: Path, extra: tuple[str, ...]) -> None:
    result = CliRunner().invoke(cli, ["metadata", "zzzz", *extra, "--db", str(db_path)])
    assert result.exit_code == 1, result.output
    assert "No ingested document matches" in result.output
    assert _PREFACE not in result.output


# --- bad option values --------------------------------------------------------


@pytest.mark.parametrize("pages", ["abc", "36-", "-4"])
def test_reconvert_bad_pages_is_a_usage_error(tmp_path: Path, pages: str) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF")
    result = CliRunner().invoke(cli, ["repair", "reconvert", str(pdf), "--pages", pages])
    assert result.exit_code == 2, result.output
    assert "invalid page range" in result.output


# --- a server URL that is not a RAG server ------------------------------------


def _html_backend() -> RemoteBackend:
    be = RemoteBackend("http://wrong.example")
    be._client = httpx.Client(
        base_url="http://wrong.example",
        transport=httpx.MockTransport(
            lambda req: httpx.Response(200, text="<html>login</html>", request=req)
        ),
    )
    return be


def test_non_json_response_is_a_server_error() -> None:
    with pytest.raises(RagServerError, match="did not return JSON"):
        _html_backend().get_ingested_docs()


def test_non_json_response_is_a_clean_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_mod, "_backend_for", lambda *_a, **_kw: _html_backend())
    result = CliRunner().invoke(cli, ["list", "--global"])
    assert result.exit_code == 1, result.output
    assert "did not return JSON" in result.output
    assert _PREFACE not in result.output
