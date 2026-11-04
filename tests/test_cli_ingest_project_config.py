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
from datasheet_rag.store.keys import BUILTIN_KEYS, KeySpec


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.keys: list[KeySpec] = list(BUILTIN_KEYS)
        self.list_keys_calls = 0

    def list_keys(self) -> list[KeySpec]:
        self.list_keys_calls += 1
        return self.keys

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
    assert patch.keys == {"doc_type": "technical-manual"}
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


def test_unregistered_key_fails_before_the_pdf_is_sent(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    result = _ingest(str(_pdf(tmp_path)), "--key", "acme-idd=D-1")

    assert result.exit_code == 1
    assert "unregistered key(s) acme-idd" in result.output
    assert fake_backend.calls == []


def test_config_attribute_that_is_now_a_key_is_sent_as_the_key(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    fake_backend.keys.append(KeySpec(name="revision"))
    (tmp_path / ".rag.toml").write_text('[attributes]\nrevision = "B"\nowner = "hw"\n')

    result = _ingest(str(_pdf(tmp_path)))

    assert result.exit_code == 0, result.output
    patch = fake_backend.calls[0]["metadata"]
    assert patch.keys == {"revision": "B"}
    assert patch.attributes == {"owner": "hw"}
    assert "[attributes] revision: registered key(s) now" in result.output


def test_stale_config_mpn_attribute_is_split_on_commas(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    (tmp_path / ".rag.toml").write_text('[attributes]\nmpn = "INA226, INA226A"\n')

    result = _ingest(str(_pdf(tmp_path)))

    assert result.exit_code == 0, result.output
    assert fake_backend.calls[0]["metadata"].keys == {"mpn": ["INA226", "INA226A"]}


def test_bulk_ingest_reads_the_registry_and_warns_once(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    fake_backend.keys.append(KeySpec(name="revision"))
    (tmp_path / ".rag.toml").write_text('[attributes]\nrevision = "B"\n')
    _pdf(tmp_path / "a")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "doc.pdf").write_bytes(b"%PDF-1.4\nb\n")

    result = _ingest(str(tmp_path))

    assert result.exit_code == 0, result.output
    assert len(fake_backend.calls) == 2
    assert fake_backend.list_keys_calls == 1
    assert result.output.count("registered key(s) now") == 1


def test_ingest_without_a_patch_does_not_read_the_registry(
    tmp_path: Path, fake_backend: _FakeBackend
) -> None:
    result = _ingest(str(_pdf(tmp_path)))

    assert result.exit_code == 0, result.output
    assert fake_backend.list_keys_calls == 0


def test_ingest_attr_bad_format_rejected(tmp_path: Path, fake_backend: _FakeBackend) -> None:
    result = _ingest(str(_pdf(tmp_path)), "--attr", "no-equals-sign")

    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output
