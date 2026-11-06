"""Cache files from before the doc_id → rev_id rename (GH #124)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from datasheet_rag import cache_files
from datasheet_rag.chunking.pipeline import load_chunk_graph
from datasheet_rag.cli import cli
from datasheet_rag.config import get_settings

RID = "ab" * 32


def _old_chunks() -> dict:
    meta = {"doc_id": RID, "page_numbers": [1], "form_fields": {"doc_id": "a PDF form value"}}
    chunk = {
        "id": f"{RID}:L2:0",
        "doc_id": RID,
        "level": 2,
        "text": "t",
        "context_text": "t",
        "metadata": meta,
    }
    return {"doc_id": RID, "title_hints": None, "stats": {}, "chunks": {chunk["id"]: chunk}}


@pytest.fixture()
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    out, figs = tmp_path / "cache", tmp_path / "figures"
    (figs / RID).mkdir(parents=True)
    out.mkdir()
    (out / f"{RID}_chunks.json").write_text(json.dumps(_old_chunks(), indent=2))
    (out / f"{RID}_outline.json").write_text(json.dumps({"outline": {"doc_id": RID}}))
    (figs / RID / "manifest.json").write_text(json.dumps({"doc_id": RID, "figures": []}, indent=2))
    monkeypatch.setattr(get_settings(), "output_dir", out)
    monkeypatch.setattr(get_settings(), "figures_dir", figs)
    return out, figs


def test_a_reader_refuses_a_pre_rename_file(dirs: tuple[Path, Path]) -> None:
    out, _ = dirs
    with pytest.raises(cache_files.StaleCacheError, match="rag repair caches"):
        load_chunk_graph(out / f"{RID}_chunks.json")


def test_repair_lists_then_rewrites_and_leaves_other_keys(dirs: tuple[Path, Path]) -> None:
    out, figs = dirs
    listed = CliRunner().invoke(cli, ["repair", "caches"], catch_exceptions=False)
    assert listed.exit_code == 0, listed.output
    assert "3 cached file(s) still say doc_id" in listed.output
    assert cache_files.is_stale(json.loads((out / f"{RID}_chunks.json").read_text()))

    applied = CliRunner().invoke(cli, ["repair", "caches", "--apply"], catch_exceptions=False)
    assert applied.exit_code == 0, applied.output
    assert "Rewrote 3 cached file(s)" in applied.output

    graph = load_chunk_graph(out / f"{RID}_chunks.json")
    chunk = graph.chunks[f"{RID}:L2:0"]
    assert graph.rev_id == RID and chunk.rev_id == RID and chunk.metadata.rev_id == RID
    # A doc_id key that is data, not the id, stays as it was.
    assert chunk.metadata.form_fields == {"doc_id": "a PDF form value"}
    outline_text = (out / f"{RID}_outline.json").read_text()
    assert json.loads(outline_text) == {"outline": {"rev_id": RID}} and "\n" not in outline_text
    assert json.loads((figs / RID / "manifest.json").read_text())["rev_id"] == RID

    again = CliRunner().invoke(cli, ["repair", "caches"], catch_exceptions=False)
    assert "nothing to do" in again.output


def test_the_cli_reports_a_stale_file_without_a_traceback(dirs: tuple[Path, Path]) -> None:
    result = CliRunner().invoke(cli, ["inspect", "layout", RID])
    assert result.exit_code == 1
    assert "rag repair caches" in result.output and "Traceback" not in result.output
