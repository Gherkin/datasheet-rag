"""CLI tests for revisions (GH #124): `rag revisions`, `rag metadata` on a
document or one revision, `rag list`, `rag delete` and `rag ingest` links."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from datasheet_rag.backend.models import IngestLink, IngestResult
from datasheet_rag.cli import cli
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata
from datasheet_rag.project_config import get_project_config
from datasheet_rag.store.fields import BUILTIN_FIELDS, FieldSpec
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

DOC_A = "a" * 64
DOC_B = "b" * 64
DOC_C = "c" * 64


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
    """Three PDFs, each its own document: DOC_A is D1, DOC_B D2, DOC_C D3."""
    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)
    for doc in (DOC_A, DOC_B, DOC_C):
        insert_chunks(conn, [_chunk(doc)])
    conn.close()
    return path


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    # The CLI console is built at import; COLUMNS alone does not reach it.
    import datasheet_rag.cli as cli_module

    monkeypatch.setattr(cli_module.console, "width", 200)


@pytest.fixture(autouse=True)
def _clear_project_config_cache() -> None:
    get_project_config.cache_clear()
    yield
    get_project_config.cache_clear()


def _rag(db_path: Path, *args: str) -> Any:
    result = CliRunner().invoke(
        cli, [*args, "--db", str(db_path)], env={"COLUMNS": "200"}, catch_exceptions=False
    )
    return result


def _json(output: str) -> dict[str, Any]:
    start = output.index("{")
    end = output.rindex("}") + 1
    return json.loads(output[start:end])


def test_revisions_add_list_move_unlink(db_path: Path) -> None:
    result = _rag(db_path, "revisions", "add", "D1", DOC_B[:8])
    assert result.exit_code == 0, result.output
    assert "is the newest revision of D1" in result.output

    listed = _rag(db_path, "revisions", "D1")  # bare ref means `list`
    assert listed.exit_code == 0, listed.output
    lines = [ln for ln in listed.output.splitlines() if DOC_A[:12] in ln or DOC_B[:12] in ln]
    assert DOC_B[:12] in lines[0] and "newest" in lines[0]
    assert DOC_A[:12] in lines[1]

    moved = _rag(db_path, "revisions", "move", DOC_B[:8], "--below", DOC_A[:8])
    assert moved.exit_code == 0, moved.output
    first = next(ln for ln in moved.output.splitlines() if "newest" in ln and "│" in ln)
    assert DOC_A[:12] in first

    unlinked = _rag(db_path, "revisions", "unlink", DOC_B[:8])
    assert unlinked.exit_code == 0, unlinked.output
    assert "(was a revision of D1)" in unlinked.output


def test_revisions_add_reports_a_conflict(db_path: Path) -> None:
    _rag(db_path, "metadata", DOC_A, "--manufacturer", "TI")
    _rag(db_path, "metadata", DOC_B, "--manufacturer", "ST")
    result = _rag(db_path, "revisions", "add", "D1", DOC_B)
    assert result.exit_code != 0
    assert "manufacturer" in result.output


def test_metadata_on_a_document_reaches_every_revision(db_path: Path) -> None:
    _rag(db_path, "revisions", "add", "D1", DOC_B)
    result = _rag(db_path, "metadata", "D1", "--tag", "mcu")
    assert result.exit_code == 0, result.output
    assert "Saved metadata for D1 (all 2 revisions)" in result.output
    older = _json(_rag(db_path, "metadata", DOC_A[:8]).output)
    assert older["tags"] == ["mcu"] and older["document_id"] == "D1"
    assert older["revisions"] == [DOC_B, DOC_A]


def test_this_revision_is_marked_everywhere(db_path: Path) -> None:
    _rag(db_path, "revisions", "add", "D1", DOC_B)
    _rag(db_path, "metadata", "D1", "--doc-type", "datasheet")
    result = _rag(db_path, "metadata", DOC_A, "--this-revision", "--doc-type", "errata")
    assert result.exit_code == 0, result.output
    assert f"revision {DOC_A[:12]} only" in result.output
    assert "* set on this revision only: doc_type" in result.output

    listed = _rag(db_path, "list", "--global", "--all-revisions", "--columns", "doc_type")
    row_a = next(ln for ln in listed.output.splitlines() if DOC_A[:12] in ln)
    row_b = next(ln for ln in listed.output.splitlines() if DOC_B[:12] in ln)
    assert "errata*" in row_a and "datasheet" in row_b and "*" not in row_b
    assert "set on this revision only" in listed.output


def test_a_rev_id_write_to_a_document_with_revisions_needs_the_d12(db_path: Path) -> None:
    # One revision: the rev_id and the document are the same thing.
    assert _rag(db_path, "metadata", DOC_A, "--tag", "mcu").exit_code == 0
    _rag(db_path, "revisions", "add", "D1", DOC_B)
    for args in (("--tag", "x"), ("--mpn", "A"), ("--add-project", "p")):
        result = _rag(db_path, "metadata", DOC_A, *args)
        assert result.exit_code == 2, (args, result.output)
        assert "is one of 2 revisions of D1" in result.output
        assert "Use D1 to change all of them" in result.output
    # The label and the title are the revision's own: no question to ask.
    assert _rag(db_path, "metadata", DOC_A, "--revision", "Rev A").exit_code == 0
    assert _rag(db_path, "metadata", DOC_A, "--this-revision", "--tag", "x").exit_code == 0
    assert _rag(db_path, "metadata", "D1", "--tag", "y").exit_code == 0


def test_a_label_write_on_a_rev_id_with_revisions_skips_rag_toml(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _rag(db_path, "revisions", "add", "D1", DOC_B)
    work = tmp_path / "work"
    work.mkdir()
    (work / ".rag.toml").write_text('project_id = "board-a"\ntags = ["x"]\n')
    monkeypatch.chdir(work)
    for args in (("--revision", "Rev A"), ("--title", "Old")):
        result = _rag(db_path, "metadata", DOC_A, *args)
        assert result.exit_code == 0, (args, result.output)
    meta = _json(_rag(db_path, "metadata", "D1").output)
    assert meta.get("projects", []) == [] and meta.get("tags", []) == []


def test_mpn_alias_on_a_document_builds_on_the_documents_list(db_path: Path) -> None:
    _rag(db_path, "metadata", DOC_A, "--mpn", "C")
    _rag(db_path, "metadata", DOC_A, "--this-revision", "--mpn", "A")
    result = _rag(db_path, "metadata", "D1", "--mpn-alias", "D")
    assert result.exit_code == 0, result.output
    assert _json(_rag(db_path, "metadata", "D1").output)["fields"]["mpn"] == ["C", "D"]
    result = _rag(db_path, "metadata", DOC_A, "--mpn-alias", "E")
    assert result.exit_code == 0, result.output
    assert _json(_rag(db_path, "metadata", "D1").output)["fields"]["mpn"] == ["C", "D", "E"]


def test_metadata_refuses_per_revision_options_on_a_document(db_path: Path) -> None:
    for args in (
        ("--this-revision", "--tag", "x"),
        ("--revision", "Rev C"),
        ("--title", "New"),
    ):
        result = _rag(db_path, "metadata", "D1", *args)
        assert result.exit_code == 2, result.output
        assert "whole document" in result.output
    result = _rag(db_path, "metadata", DOC_A, "--this-revision", "--add-project", "p")
    assert result.exit_code == 2 and "projects belong" in result.output


def test_revision_label_shows_in_list(db_path: Path) -> None:
    _rag(db_path, "revisions", "add", "D1", DOC_B)
    _rag(db_path, "metadata", DOC_B, "--revision", "Rev C")
    listed = _rag(db_path, "list", "--global")
    assert listed.exit_code == 0, listed.output
    header = next(ln for ln in listed.output.splitlines() if "rev_id" in ln)
    assert "document" in header and "revision" in header and "revisions" in header
    rows = [ln for ln in listed.output.splitlines() if DOC_A[:12] in ln or DOC_B[:12] in ln]
    assert len(rows) == 1 and DOC_B[:12] in rows[0] and "Rev C" in rows[0]
    every = _rag(db_path, "list", "--global", "--all-revisions")
    assert DOC_A[:12] in every.output and DOC_B[:12] in every.output


def test_delete_a_document_deletes_every_revision(db_path: Path) -> None:
    _rag(db_path, "revisions", "add", "D1", DOC_B)
    result = _rag(db_path, "delete", "D1", "--yes")
    assert result.exit_code == 0, result.output
    assert "D1 (2 revision(s))" in result.output
    listed = _rag(db_path, "list", "--global", "--all-revisions")
    assert DOC_A[:12] not in listed.output and DOC_B[:12] not in listed.output
    assert DOC_C[:12] in listed.output


def test_delete_one_revision_says_the_others_stay(db_path: Path) -> None:
    _rag(db_path, "revisions", "add", "D1", DOC_B)
    result = _rag(db_path, "delete", DOC_B, "--dry-run")
    assert "one of 2 revisions of D1; the others are kept" in result.output


def test_rule_set_points_to_apply_rule(db_path: Path) -> None:
    assert _rag(db_path, "fields", "add", "intel-id").exit_code == 0
    result = _rag(db_path, "revisions", "rule", "set", "intel-id or doc_type")
    assert result.exit_code == 0, result.output
    assert "intel-id OR doc_type" in result.output
    assert "rag revisions apply-rule --dry-run" in result.output
    assert "intel-id OR doc_type" in _rag(db_path, "revisions", "rule", "show").output
    bad = _rag(db_path, "revisions", "rule", "set", "nope")
    assert bad.exit_code != 0 and "unregistered" in bad.output
    _rag(db_path, "revisions", "rule", "clear")
    assert "No identity rule" in _rag(db_path, "revisions", "rule", "show").output


def test_apply_rule_dry_run_then_link(db_path: Path) -> None:
    assert _rag(db_path, "fields", "add", "intel-id").exit_code == 0
    for doc in (DOC_A, DOC_B):
        _rag(db_path, "metadata", doc, "--field", "intel-id=123")
    _rag(db_path, "metadata", DOC_C, "--field", "intel-id=999")

    none = _rag(db_path, "revisions", "apply-rule")
    assert none.exit_code != 0 and "no identity rule" in none.output

    _rag(db_path, "revisions", "rule", "set", "intel-id")
    dry = _rag(db_path, "revisions", "apply-rule", "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert "Would link 1 group(s)" in dry.output
    assert "D2 → D1 (ordered by ingest time)" in dry.output
    assert "Check the order of each linked document" in dry.output
    assert "Dry run: nothing was changed" in dry.output
    assert "D2" in _json(_rag(db_path, "metadata", DOC_B).output)["document_id"]

    done = _rag(db_path, "revisions", "apply-rule")
    assert done.exit_code == 0, done.output
    assert "Linked 1 group(s)" in done.output and "Dry run" not in done.output
    assert _json(_rag(db_path, "metadata", DOC_B).output)["document_id"] == "D1"
    again = _rag(db_path, "revisions", "apply-rule")
    assert "Nothing to link" in again.output


def test_apply_rule_lists_skipped_groups(db_path: Path) -> None:
    _rag(db_path, "fields", "add", "intel-id")
    _rag(db_path, "metadata", DOC_A, "--field", "intel-id=1", "--manufacturer", "TI")
    _rag(db_path, "metadata", DOC_B, "--field", "intel-id=1", "--manufacturer", "ST")
    _rag(db_path, "revisions", "rule", "set", "intel-id")
    result = _rag(db_path, "revisions", "apply-rule")
    assert result.exit_code == 0, result.output
    assert "Linked 0 group(s). Skipped 1." in result.output
    assert "Skipped D1, D2: a field holds different values" in result.output
    assert "Check the order" not in result.output


# ---- rag ingest --------------------------------------------------------------


class _FakeBackend:
    """The remote raw-PDF ingest branch, without a server or a parse."""

    def __init__(self) -> None:
        self.plans: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.link = IngestLink(action="explicit", document_id="D7")

    def list_fields(self) -> list[FieldSpec]:
        return list(BUILTIN_FIELDS)

    def plan_ingest_link(self, **kwargs: Any) -> IngestLink:
        self.plans.append(kwargs)
        return self.link

    def ingest_pdf(self, pdf_path: Path, **kwargs: Any) -> IngestResult:
        self.calls.append(kwargs)
        return IngestResult(rev_id="d" * 64, document_id="D7", link=self.link)


@pytest.fixture()
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> _FakeBackend:
    import datasheet_rag.backend as backend_pkg

    fake = _FakeBackend()
    monkeypatch.setattr(backend_pkg, "backend_mode", lambda: "remote")
    monkeypatch.setattr(backend_pkg, "compute_mode", lambda: "server")
    monkeypatch.setattr(backend_pkg, "get_backend", lambda: fake)
    return fake


def test_ingest_passes_the_link_and_prints_it(tmp_path: Path, fake_backend: _FakeBackend) -> None:
    pdf = tmp_path / "rev_c.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    result = CliRunner().invoke(
        cli, ["ingest", str(pdf), "--new-revision-of", "D7", "--revision", "Rev C"]
    )
    assert result.exit_code == 0, result.output
    assert fake_backend.plans[0]["new_revision_of"] == "D7"
    assert fake_backend.calls[0]["new_revision_of"] == "D7"
    assert fake_backend.calls[0]["metadata"].revision == "Rev C"
    assert "linked to D7 as newest revision (--new-revision-of)" in result.output


def test_ingest_stops_before_the_parse_when_the_link_fails(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    def refuse(**kwargs: Any) -> IngestLink:
        raise ValueError("the identity rule (intel-id) matches D1 and D2")

    fake_backend.plan_ingest_link = refuse  # type: ignore[method-assign]
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    result = CliRunner().invoke(cli, ["ingest", str(pdf)])
    assert result.exit_code == 1
    assert "matches D1 and D2" in result.output
    assert fake_backend.calls == []


def test_bulk_ingest_refuses_revision_options(tmp_path: Path, fake_backend: _FakeBackend) -> None:
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4\n")
    result = CliRunner().invoke(cli, ["ingest", str(tmp_path), "--new-revision-of", "D7"])
    assert result.exit_code == 1
    assert "one file at a time" in result.output
