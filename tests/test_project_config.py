"""Tests for ``.rag.toml`` discovery and CLI project-scope resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from datasheet_rag.project_config import (
    ProjectConfig,
    ProjectConfigError,
    find_project_configs,
    get_project_config_for,
    merge_project_configs,
    resolve_cli_project_id,
)

# ---------------------------------------------------------------------------
# find_project_configs
# ---------------------------------------------------------------------------


def test_find_project_configs_in_cwd(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text(
        'project_id = "proj-a"\ntags = ["x", "y"]\n[fields]\nsubsystem = "power"\n'
    )

    configs = find_project_configs(tmp_path)

    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.project_id == "proj-a"
    assert cfg.fields == {"subsystem": "power"}
    assert cfg.tags == ["x", "y"]
    assert cfg.path == tmp_path / ".rag.toml"


@pytest.mark.parametrize(("raw", "expected"), [(" proj-a ", "proj-a"), ("  ", None)])
def test_project_id_is_stripped_and_blank_is_unset(
    tmp_path: Path, raw: str, expected: str | None
) -> None:
    """Stored project names are stripped, so the query scope must be too (GH #121)."""
    (tmp_path / ".rag.toml").write_text(f'project_id = "{raw}"\n')
    assert find_project_configs(tmp_path)[0].project_id == expected


def test_find_project_configs_walks_up_to_parent(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text('project_id = "proj-a"\n')
    nested = tmp_path / "sub" / "dir"
    nested.mkdir(parents=True)

    configs = find_project_configs(nested)

    assert len(configs) == 1
    assert configs[0].project_id == "proj-a"
    assert configs[0].path == tmp_path / ".rag.toml"


def test_find_project_configs_collects_whole_chain_nearest_first(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text(
        'project_id = "proj-a"\n[fields]\nmanufacturer = "STMicro"\n'
    )
    subsystem = tmp_path / "power"
    subsystem.mkdir()
    (subsystem / ".rag.toml").write_text('[fields]\nsubsystem = "power"\n')
    mpn_dir = subsystem / "STM32H743"
    mpn_dir.mkdir()
    (mpn_dir / ".rag.toml").write_text('[fields]\nmpn = "STM32H743VIT6"\n')

    configs = find_project_configs(mpn_dir)

    assert [cfg.path for cfg in configs] == [
        mpn_dir / ".rag.toml",
        subsystem / ".rag.toml",
        tmp_path / ".rag.toml",
    ]


def test_find_project_configs_returns_empty_when_absent(tmp_path: Path) -> None:
    assert find_project_configs(tmp_path) == []


def test_find_project_configs_rejects_malformed_toml(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text("this is not [valid toml")

    with pytest.raises(ProjectConfigError, match="not valid TOML"):
        find_project_configs(tmp_path)


def test_find_project_configs_rejects_malformed_even_with_valid_ancestors(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text('project_id = "proj-a"\n')
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / ".rag.toml").write_text("not valid toml [")

    with pytest.raises(ProjectConfigError, match=str(nested / ".rag.toml")):
        find_project_configs(nested)


# ---------------------------------------------------------------------------
# Schema: [fields], and what is rejected (GH #26)
# ---------------------------------------------------------------------------


def test_loads_fields_table_and_list_mpn(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text(
        '[fields]\nmpn = ["STM32H743VIT6", "STM32H743ZIT6"]\n'
        'doc_type = "technical-manual"\nacme-id = "D-0172"\nacme-pn = ["P1", "P2"]\n'
    )

    (cfg,) = find_project_configs(tmp_path)

    assert cfg.fields == {
        "mpn": ["STM32H743VIT6", "STM32H743ZIT6"],
        "doc_type": "technical-manual",
        "acme-id": "D-0172",
        "acme-pn": ["P1", "P2"],
    }


def test_comma_mpn_string_is_one_value(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text('[fields]\nmpn = "A,B"\n')

    (cfg,) = find_project_configs(tmp_path)

    assert cfg.fields == {"mpn": "A,B"}


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('colour = "red"\n', "unknown key.*arbitrary values go under \\[fields\\]"),
        # The old forms are unknown keys like any other.
        ('mpn = "A"\n', "unknown key\\(s\\) mpn"),
        ('manufacturer = "TI"\n', "unknown key\\(s\\) manufacturer"),
        ('doc_type = "errata"\n', "unknown key\\(s\\) doc_type"),
        ('[attributes]\nrevision = "B"\n', "unknown key\\(s\\) attributes"),
        ('[keys]\nacme-id = "D-1"\n', "unknown key\\(s\\) keys"),
        # Removed in GH #123; the error says what to write instead.
        ('group = "psu"\n', "group was removed; use \\[fields\\] subsystem"),
        ('subsystem = "power"\n', "move it under \\[fields\\]"),
        ("[fields]\nmpn = 226\n", "'mpn' must be a string or a list of strings"),
        ('[fields]\ndoc_type = ["a"]\n', "'doc_type' must be a string"),
        ('[fields]\nmanufacturer = ["a"]\n', "'manufacturer' must be a string"),
        ("[fields]\nacme-id = 3\n", "'acme-id' must be a string or a list of strings"),
        ('fields = "x"\n', "'fields' must be a table"),
        ('[fields]\n"" = "v"\n', "empty name"),
        ('tags = "mcu"\n', "'tags' must be a list of strings"),
        ("tags = [1, 2]\n", "'tags' must be a list of strings"),
    ],
)
def test_rejects_schema_violations(tmp_path: Path, body: str, message: str) -> None:
    (tmp_path / ".rag.toml").write_text(body)

    with pytest.raises(ProjectConfigError, match=message):
        find_project_configs(tmp_path)


# ---------------------------------------------------------------------------
# merge_project_configs
# ---------------------------------------------------------------------------


def test_merge_project_configs_returns_none_for_empty_chain() -> None:
    assert merge_project_configs([]) is None


def test_merge_project_configs_nearest_scalar_wins(tmp_path: Path) -> None:
    nearest = ProjectConfig(path=tmp_path / "mpn" / ".rag.toml", fields={"mpn": "STM32H743VIT6"})
    middle = ProjectConfig(path=tmp_path / "subsystem" / ".rag.toml", project_id="proj-b")
    root = ProjectConfig(
        path=tmp_path / ".rag.toml",
        project_id="proj-a",
        fields={"manufacturer": "STMicro"},
    )

    merged = merge_project_configs([nearest, middle, root])

    assert merged is not None
    assert merged.path == nearest.path
    # The nearest file that sets project_id wins over the root's.
    assert merged.project_id == "proj-b"
    assert merged.fields == {"mpn": "STM32H743VIT6", "manufacturer": "STMicro"}


def test_merge_project_configs_merges_fields_by_name_nearest_wins(tmp_path: Path) -> None:
    nearest = ProjectConfig(path=tmp_path / "mpn" / ".rag.toml", fields={"revision": "B"})
    middle = ProjectConfig(path=tmp_path / "sub" / ".rag.toml", fields={"doc_type": "errata"})
    root = ProjectConfig(
        path=tmp_path / ".rag.toml",
        fields={"doc_type": "datasheet", "mpn": ["A", "B"], "revision": "A"},
    )

    merged = merge_project_configs([nearest, middle, root])

    assert merged is not None
    # Per name, nearest wins; a list is replaced, not unioned.
    assert merged.fields == {"doc_type": "errata", "mpn": ["A", "B"], "revision": "B"}


def test_merge_project_configs_unions_tags_nearest_first_deduped(tmp_path: Path) -> None:
    nearest = ProjectConfig(path=tmp_path / "mpn" / ".rag.toml", tags=["errata", "datasheet"])
    root = ProjectConfig(path=tmp_path / ".rag.toml", tags=["datasheet", "reference-manual"])

    merged = merge_project_configs([nearest, root])

    assert merged is not None
    assert merged.tags == ["errata", "datasheet", "reference-manual"]


def test_merge_project_configs_single_config_passthrough(tmp_path: Path) -> None:
    only = ProjectConfig(path=tmp_path / ".rag.toml", project_id="proj-a", tags=["x"])

    merged = merge_project_configs([only])

    assert merged == only


# ---------------------------------------------------------------------------
# get_project_config_for — resolves from an arbitrary directory, not the cwd
# ---------------------------------------------------------------------------


def test_get_project_config_for_merges_chain_from_given_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text('project_id = "netdaq"\ntags = ["netdaq"]\n')
    subsystem = tmp_path / "datasheets" / "eth_phy"
    subsystem.mkdir(parents=True)
    (subsystem / ".rag.toml").write_text('[fields]\nsubsystem = "eth_phy"\n')
    mpn_dir = subsystem / "LAN8720"
    mpn_dir.mkdir()
    (mpn_dir / ".rag.toml").write_text('[fields]\nmpn = "LAN8720A"\nmanufacturer = "Microchip"\n')

    # cwd is the project root — get_project_config_for should still resolve
    # from the PDF's own directory, not the cwd.
    monkeypatch.chdir(tmp_path)

    cfg = get_project_config_for(mpn_dir)

    assert cfg is not None
    assert cfg.project_id == "netdaq"
    assert cfg.fields == {"mpn": "LAN8720A", "manufacturer": "Microchip", "subsystem": "eth_phy"}
    assert cfg.tags == ["netdaq"]


def test_get_project_config_for_returns_none_when_chain_empty(tmp_path: Path) -> None:
    assert get_project_config_for(tmp_path) is None


# ---------------------------------------------------------------------------
# resolve_cli_project_id precedence
# ---------------------------------------------------------------------------


@pytest.fixture()
def discovered(monkeypatch: pytest.MonkeyPatch) -> ProjectConfig:
    """Pretend a ``.rag.toml`` with project_id=discovered-proj was found."""
    from datasheet_rag import project_config as pc

    cfg = ProjectConfig(path=Path("/fake/.rag.toml"), project_id="discovered-proj")
    monkeypatch.setattr(pc, "get_project_config", lambda: cfg)
    return cfg


def test_explicit_project_id_wins_over_everything(
    discovered: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert resolve_cli_project_id("explicit-proj", is_global=False) == "explicit-proj"
    assert resolve_cli_project_id("explicit-proj", is_global=True) == "explicit-proj"


def test_global_flag_forces_unscoped_over_discovered_config(discovered: ProjectConfig) -> None:
    assert resolve_cli_project_id(None, is_global=True) is None


def test_discovered_config_used_as_implicit_scope(discovered: ProjectConfig) -> None:
    assert resolve_cli_project_id(None, is_global=False) == "discovered-proj"


def test_falls_back_to_settings_default_project_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from datasheet_rag import project_config as pc

    monkeypatch.setattr(pc, "get_project_config", lambda: None)

    class _FakeSettings:
        default_project_id = "env-default-proj"

    monkeypatch.setattr("datasheet_rag.config.get_settings", lambda: _FakeSettings())

    assert resolve_cli_project_id(None, is_global=False) == "env-default-proj"


def test_resolves_to_none_when_nothing_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from datasheet_rag import project_config as pc

    monkeypatch.setattr(pc, "get_project_config", lambda: None)

    class _FakeSettings:
        default_project_id = None

    monkeypatch.setattr("datasheet_rag.config.get_settings", lambda: _FakeSettings())

    assert resolve_cli_project_id(None, is_global=False) is None
