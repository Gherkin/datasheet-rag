"""`rag ingest` applies every ``.rag.toml`` field, attributes included (GH #26).

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


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

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


def test_ingest_applies_doc_type_and_attributes_from_config(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    (tmp_path / ".rag.toml").write_text(
        'doc_type = "technical-manual"\n[attributes]\nrevision = "A"\nowner = "hw"\n'
    )
    pdf = _pdf(tmp_path / "V700")
    (pdf.parent / ".rag.toml").write_text('[attributes]\nrevision = "B"\n')

    result = _ingest(str(pdf), "--attr", "owner=fw")

    assert result.exit_code == 0, result.output
    patch = fake_backend.calls[0]["metadata"]
    assert patch.doc_type == "technical-manual"
    # Nearest file wins for revision; the CLI flag wins for owner.
    assert patch.attributes == {"revision": "B", "owner": "fw"}


def test_ingest_without_attributes_sends_none(tmp_path: Path, fake_backend: _FakeBackend) -> None:
    pdf = _pdf(tmp_path)

    result = _ingest(str(pdf))

    assert result.exit_code == 0, result.output
    assert fake_backend.calls[0]["metadata"].attributes is None


def test_ingest_rejects_bad_config_without_traceback(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    (tmp_path / ".rag.toml").write_text("[attributes]\nrevision = 2\n")
    pdf = _pdf(tmp_path)

    result = _ingest(str(pdf))

    assert result.exit_code == 1
    assert "must be a string" in result.output
    assert "Traceback" not in result.output
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


def test_ingest_attr_bad_format_rejected(tmp_path: Path, fake_backend: _FakeBackend) -> None:
    result = _ingest(str(_pdf(tmp_path)), "--attr", "no-equals-sign")

    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output
