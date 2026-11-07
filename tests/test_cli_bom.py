"""CLI tests for `rag bom sync` (GH #125): a BOM from a CSV file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from datasheet_rag.cli import cli
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata
from datasheet_rag.project_config import get_project_config
from datasheet_rag.store.fields import register_field
from datasheet_rag.store.metadata import set_metadata
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

DOC_A = "a" * 64
DOC_B = "b" * 64


def _chunk(rev_id: str) -> Chunk:
    return Chunk(
        id=f"{rev_id}:L2:0",
        rev_id=rev_id,
        level=ChunkLevel.MICRO,
        text="chunk 0",
        context_text="chunk 0",
        token_count=2,
        metadata=ChunkMetadata(rev_id=rev_id, doc_title=f"Doc {rev_id[:4]}", page_numbers=[1]),
    )


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """DOC_A (D1) holds acme-pn 100-1; DOC_B (D2) holds mpn TLA2518."""
    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)
    register_field(conn, "acme-pn", roles=["part"])
    for doc in (DOC_A, DOC_B):
        insert_chunks(conn, [_chunk(doc)])
    set_metadata(conn, DOC_A, fields={"acme-pn": "100-1"})
    set_metadata(conn, DOC_B, fields={"mpn": ["TLA2518"]})
    conn.close()
    return path


@pytest.fixture(autouse=True)
def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    import datasheet_rag.cli as cli_module

    monkeypatch.setattr(cli_module.console, "width", 200)
    monkeypatch.chdir(tmp_path)  # no .rag.toml above
    get_project_config.cache_clear()
    yield
    get_project_config.cache_clear()


def _rag(db_path: Path, *args: str) -> Any:
    return CliRunner().invoke(cli, [*args, "--db", str(db_path)], env={"COLUMNS": "200"})


def _csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "bom.csv"
    path.write_text(text, encoding="utf-8")
    return path


def _members(db_path: Path, project: str) -> dict[int, str]:
    conn = connect(db_path, embedding_dim=get_settings().embedding_dimensions)
    try:
        return {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT document_id, source FROM document_projects WHERE project_id = ?",
                (project,),
            )
        }
    finally:
        conn.close()


def test_sync_from_csv_reports_matches_misses_and_suggestions(
    db_path: Path, tmp_path: Path
) -> None:
    bom = _csv(
        tmp_path,
        "Reference,Internal PN,MPN\nR1,100-1,RC0402\nU8,,TLA2518IRTER\nJ1,,\n",
    )
    args = ["bom", "sync", str(bom), "--project", "netdaq", "--ref-column", "Reference"]
    args += ["--column", "Internal PN=acme-pn", "--column", "MPN=mpn"]

    dry = _rag(db_path, *args, "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert "Would add 1 document(s)" in dry.output
    assert _members(db_path, "netdaq") == {}

    out = _rag(db_path, *args)
    assert out.exit_code == 0, out.output
    assert "2 BOM line(s), 1 matched, 1 with no document" in out.output  # J1 has no part
    assert "Added 1 document(s)" in out.output and "D1 Doc aaaa" in out.output
    assert "U8: mpn=TLA2518IRTER" in out.output
    assert "suggestion: D2 Doc bbbb (mpn=TLA2518)" in out.output
    assert _members(db_path, "netdaq") == {1: "bom"}


def test_sync_needs_a_project_and_known_columns(db_path: Path, tmp_path: Path) -> None:
    bom = _csv(tmp_path, "MPN\nX\n")
    out = _rag(db_path, "bom", "sync", str(bom), "--column", "MPN=mpn")
    assert out.exit_code != 0 and "--project" in out.output

    out = _rag(db_path, "bom", "sync", str(bom), "--project", "p", "--column", "PN=mpn")
    assert out.exit_code != 0 and "no column(s) 'PN'" in out.output

    out = _rag(db_path, "bom", "sync", str(bom), "--project", "p", "--column", "MPN=manufacturer")
    assert out.exit_code != 0 and "not a part field" in out.output


def test_rag_toml_project_does_not_claim_a_bom_member(db_path: Path, tmp_path: Path) -> None:
    bom = _csv(tmp_path, "Internal PN\n100-1\n")
    out = _rag(
        db_path, "bom", "sync", str(bom), "--project", "netdaq", "--column", "Internal PN=acme-pn"
    )
    assert out.exit_code == 0, out.output
    (tmp_path / ".rag.toml").write_text('project_id = "netdaq"\n')
    get_project_config.cache_clear()

    out = _rag(db_path, "metadata", "D1", "--tag", "adc")
    assert out.exit_code == 0, out.output
    assert _members(db_path, "netdaq") == {1: "bom"}

    out = _rag(db_path, "metadata", "D1", "--add-project", "netdaq")
    assert out.exit_code == 0, out.output
    assert _members(db_path, "netdaq") == {1: "manual"}


def test_two_columns_for_one_field_are_refused(db_path: Path, tmp_path: Path) -> None:
    bom = _csv(tmp_path, "MPN,Alt MPN\nX,Y\n")
    args = ["bom", "sync", str(bom), "--project", "p", "--column", "MPN=mpn"]
    out = _rag(db_path, *args, "--column", "Alt MPN=mpn")
    assert out.exit_code == 2 and "two --column options map to field 'mpn'" in out.output
    out = _rag(db_path, *args, "--column", "MPN=acme-pn")
    assert out.exit_code == 2 and "two --column options name column 'MPN'" in out.output


def test_a_non_utf8_csv_is_a_usage_error(db_path: Path, tmp_path: Path) -> None:
    bom = tmp_path / "bom.csv"
    bom.write_bytes("MPN,Value\nTLA2518,10 µF\n".encode("cp1252"))
    out = _rag(db_path, "bom", "sync", str(bom), "--project", "p", "--column", "MPN=mpn")
    assert out.exit_code == 2 and "is not UTF-8 text" in out.output
    assert out.exception is None or isinstance(out.exception, SystemExit)
