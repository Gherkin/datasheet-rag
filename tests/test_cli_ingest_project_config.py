"""`rag ingest` applies every ``.rag.toml`` setting, ``[fields]`` included (GH #26).

Drives the remote raw-PDF branch with a fake backend, so the test checks the
metadata the CLI would send without running the parse pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from datasheet_rag.backend.models import IngestResult
from datasheet_rag.cli import cli
from datasheet_rag.store.fields import BUILTIN_FIELDS, FieldSpec


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fields: list[FieldSpec] = list(BUILTIN_FIELDS)
        self.list_fields_calls = 0

    def list_fields(self) -> list[FieldSpec]:
        self.list_fields_calls += 1
        return self.fields

    def ingest_pdf(self, pdf_path: Path, **kwargs: Any) -> IngestResult:
        self.calls.append({"pdf_path": pdf_path, **kwargs})
        return IngestResult(doc_id="d" * 64)


@pytest.fixture()
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> _FakeBackend:
    import datasheet_rag.backend as backend_pkg

    fake = _FakeBackend()
    monkeypatch.setattr(backend_pkg, "backend_mode", lambda: "remote")
    monkeypatch.setattr(backend_pkg, "compute_mode", lambda: "server")
    monkeypatch.setattr(backend_pkg, "get_backend", lambda: fake)
    return fake


def _pdf(directory: Path, name: str = "doc.pdf") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"%PDF-1.4\n")
    return path


def _ingest(*args: str):
    return CliRunner().invoke(cli, ["ingest", *args])


def test_ingest_applies_fields_from_config(tmp_path: Path, fake_backend: _FakeBackend) -> None:
    fake_backend.fields += [FieldSpec(name="revision"), FieldSpec(name="owner")]
    (tmp_path / ".rag.toml").write_text(
        '[fields]\ndoc_type = "technical-manual"\nrevision = "A"\nowner = "hw"\n'
    )
    pdf = _pdf(tmp_path / "V700")
    (pdf.parent / ".rag.toml").write_text('[fields]\nrevision = "B"\n')

    result = _ingest(str(pdf), "--field", "owner=fw")

    assert result.exit_code == 0, result.output
    patch = fake_backend.calls[0]["metadata"]
    # Nearest file wins for revision; the CLI flag wins for owner.
    assert patch.fields == {"doc_type": "technical-manual", "revision": "B", "owner": "fw"}


def test_ingest_keeps_config_subsystem_in_the_config_project(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    (tmp_path / ".rag.toml").write_text(
        'project_id = "board-a"\n[fields]\nsubsystem = "power"\nmanufacturer = "TI"\n'
    )
    pdf = _pdf(tmp_path)

    assert _ingest(str(pdf)).exit_code == 0
    assert fake_backend.calls[0]["metadata"].fields == {"subsystem": "power", "manufacturer": "TI"}

    # Another project: board-a's subsystem is left out; document fields stay.
    result = _ingest(str(pdf), "--project-id", "board-b")
    assert result.exit_code == 0, result.output
    assert fake_backend.calls[1]["project_id"] == "board-b"
    assert fake_backend.calls[1]["metadata"].fields == {"manufacturer": "TI"}
    output = " ".join(result.output.split())  # Rich wraps the note
    assert "sets subsystem for project 'board-a'; left out" in output
    assert "this ingest goes to project 'board-b'" in output

    # A padded --project-id is still board-a: the subsystem stays.
    result = _ingest(str(pdf), "--project-id", "board-a ")
    assert result.exit_code == 0, result.output
    assert fake_backend.calls[2]["metadata"].fields == {"subsystem": "power", "manufacturer": "TI"}

    # A blank --project-id is no project: the config's board-a applies.
    result = _ingest(str(pdf), "--project-id", " ")
    assert result.exit_code == 0, result.output
    assert fake_backend.calls[3]["project_id"] == "board-a"
    assert fake_backend.calls[3]["metadata"].fields == {"subsystem": "power", "manufacturer": "TI"}


def test_ingest_without_fields_sends_none(tmp_path: Path, fake_backend: _FakeBackend) -> None:
    pdf = _pdf(tmp_path)

    result = _ingest(str(pdf))

    assert result.exit_code == 0, result.output
    assert fake_backend.calls[0]["metadata"].fields is None


def test_ingest_mpn_is_repeatable_and_not_split(tmp_path: Path, fake_backend: _FakeBackend) -> None:
    result = _ingest(str(_pdf(tmp_path)), "--mpn", "INA226", "--mpn", "A,B")

    assert result.exit_code == 0, result.output
    assert fake_backend.calls[0]["metadata"].fields == {"mpn": ["INA226", "A,B"]}


def test_ingest_rejects_bad_config_without_traceback(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    (tmp_path / ".rag.toml").write_text("[fields]\nmanufacturer = 2\n")
    pdf = _pdf(tmp_path)

    result = _ingest(str(pdf))

    assert result.exit_code == 1
    assert "must be a string" in result.output
    assert "Traceback" not in result.output
    assert fake_backend.calls == []


def test_ingest_rejects_an_old_attributes_table(tmp_path: Path, fake_backend: _FakeBackend) -> None:
    (tmp_path / ".rag.toml").write_text('[attributes]\nrevision = "B"\n')

    result = _ingest(str(_pdf(tmp_path)))

    assert result.exit_code == 1
    assert "unknown key(s) attributes" in result.output
    assert "arbitrary values go under [fields]" in result.output
    assert fake_backend.calls == []


def test_bulk_ingest_skips_only_the_directory_with_a_bad_config(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    good = _pdf(tmp_path / "good")
    bad = _pdf(tmp_path / "bad")
    (bad.parent / ".rag.toml").write_text('colour = "red"\n')

    result = _ingest(str(tmp_path))

    assert result.exit_code == 0, result.output
    assert [c["pdf_path"] for c in fake_backend.calls] == [good]
    assert "1 failed" in result.output


def test_unregistered_field_fails_before_the_pdf_is_sent(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    result = _ingest(str(_pdf(tmp_path)), "--field", "acme-idd=D-1")

    assert result.exit_code == 1
    assert "unregistered field(s) acme-idd" in result.output
    assert fake_backend.calls == []


def test_unregistered_config_field_fails_before_the_pdf_is_sent(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    (tmp_path / ".rag.toml").write_text('[fields]\nrevision = "B"\n')

    result = _ingest(str(_pdf(tmp_path)))

    assert result.exit_code == 1
    assert "unregistered field(s) revision" in result.output
    assert fake_backend.calls == []


def test_bulk_ingest_reads_the_registry_once(tmp_path: Path, fake_backend: _FakeBackend) -> None:
    fake_backend.fields.append(FieldSpec(name="revision"))
    (tmp_path / ".rag.toml").write_text('[fields]\nrevision = "B"\n')
    _pdf(tmp_path / "a")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "doc.pdf").write_bytes(b"%PDF-1.4\nb\n")

    result = _ingest(str(tmp_path))

    assert result.exit_code == 0, result.output
    assert len(fake_backend.calls) == 2
    assert fake_backend.list_fields_calls == 1


def test_ingest_without_a_patch_does_not_read_the_registry(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    result = _ingest(str(_pdf(tmp_path)))

    assert result.exit_code == 0, result.output
    assert fake_backend.list_fields_calls == 0


def test_ingest_field_bad_format_rejected(tmp_path: Path, fake_backend: _FakeBackend) -> None:
    result = _ingest(str(_pdf(tmp_path)), "--field", "no-equals-sign")

    assert result.exit_code != 0
    assert "NAME=VALUE" in result.output
