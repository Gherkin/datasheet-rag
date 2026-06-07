"""Tests for ``.rag.toml`` discovery and CLI project-scope resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from aws_rag.project_config import (
    ProjectConfig,
    find_project_config,
    resolve_cli_project_id,
)


# ---------------------------------------------------------------------------
# find_project_config
# ---------------------------------------------------------------------------


def test_find_project_config_in_cwd(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text(
        'project_id = "proj-a"\ngroup = "power"\ntags = ["x", "y"]\n'
    )

    cfg = find_project_config(tmp_path)

    assert cfg is not None
    assert cfg.project_id == "proj-a"
    assert cfg.group == "power"
    assert cfg.tags == ["x", "y"]
    assert cfg.path == tmp_path / ".rag.toml"


def test_find_project_config_walks_up_to_parent(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text('project_id = "proj-a"\n')
    nested = tmp_path / "sub" / "dir"
    nested.mkdir(parents=True)

    cfg = find_project_config(nested)

    assert cfg is not None
    assert cfg.project_id == "proj-a"
    assert cfg.path == tmp_path / ".rag.toml"


def test_find_project_config_returns_none_when_absent(tmp_path: Path) -> None:
    assert find_project_config(tmp_path) is None


def test_find_project_config_returns_none_for_malformed_toml(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text("this is not [valid toml")

    assert find_project_config(tmp_path) is None


# ---------------------------------------------------------------------------
# resolve_cli_project_id precedence
# ---------------------------------------------------------------------------


@pytest.fixture()
def discovered(monkeypatch: pytest.MonkeyPatch) -> ProjectConfig:
    """Pretend a ``.rag.toml`` with project_id=discovered-proj was found."""
    from aws_rag import project_config as pc

    cfg = ProjectConfig(path=Path("/fake/.rag.toml"), project_id="discovered-proj")
    monkeypatch.setattr(pc, "get_project_config", lambda: cfg)
    return cfg


def test_explicit_project_id_wins_over_everything(discovered: ProjectConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_cli_project_id("explicit-proj", is_global=False) == "explicit-proj"
    assert resolve_cli_project_id("explicit-proj", is_global=True) == "explicit-proj"


def test_global_flag_forces_unscoped_over_discovered_config(discovered: ProjectConfig) -> None:
    assert resolve_cli_project_id(None, is_global=True) is None


def test_discovered_config_used_as_implicit_scope(discovered: ProjectConfig) -> None:
    assert resolve_cli_project_id(None, is_global=False) == "discovered-proj"


def test_falls_back_to_settings_default_project_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from aws_rag import project_config as pc

    monkeypatch.setattr(pc, "get_project_config", lambda: None)

    class _FakeSettings:
        default_project_id = "env-default-proj"

    monkeypatch.setattr("aws_rag.config.get_settings", lambda: _FakeSettings())

    assert resolve_cli_project_id(None, is_global=False) == "env-default-proj"


def test_resolves_to_none_when_nothing_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from aws_rag import project_config as pc

    monkeypatch.setattr(pc, "get_project_config", lambda: None)

    class _FakeSettings:
        default_project_id = None

    monkeypatch.setattr("aws_rag.config.get_settings", lambda: _FakeSettings())

    assert resolve_cli_project_id(None, is_global=False) is None
