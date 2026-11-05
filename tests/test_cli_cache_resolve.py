"""`_resolve_cached_doc_id` treats the user's prefix literally, not as a glob."""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from datasheet_rag import cli
from datasheet_rag.config import get_settings


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(get_settings(), "output_dir", tmp_path)
    monkeypatch.setattr(cli, "_resolve_store_first", lambda _arg: None)
    (tmp_path / "abc123_outline.json").write_text("{}")
    return tmp_path


def test_prefix_resolves(cache: Path) -> None:
    assert cli._resolve_cached_doc_id("abc") == "abc123"


@pytest.mark.parametrize("pattern", ["*", "?", "a?c", "[a]"])
def test_glob_characters_are_literal(cache: Path, pattern: str) -> None:
    with pytest.raises(click.ClickException, match="No cached"):
        cli._resolve_cached_doc_id(pattern)
