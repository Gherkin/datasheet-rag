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
from datasheet_rag.store.metadata import RESERVED_ATTRIBUTES

# ---------------------------------------------------------------------------
# find_project_configs
# ---------------------------------------------------------------------------


def test_find_project_configs_in_cwd(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text(
        'project_id = "proj-a"\ngroup = "power"\ntags = ["x", "y"]\n'
    )

    configs = find_project_configs(tmp_path)

    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.project_id == "proj-a"
    assert cfg.group == "power"
    assert cfg.tags == ["x", "y"]
    assert cfg.path == tmp_path / ".rag.toml"


def test_find_project_configs_walks_up_to_parent(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text('project_id = "proj-a"\n')
    nested = tmp_path / "sub" / "dir"
    nested.mkdir(parents=True)

    configs = find_project_configs(nested)

    assert len(configs) == 1
    assert configs[0].project_id == "proj-a"
    assert configs[0].path == tmp_path / ".rag.toml"


def test_find_project_configs_collects_whole_chain_nearest_first(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text('project_id = "proj-a"\nmanufacturer = "STMicro"\n')
    subsystem = tmp_path / "power"
    subsystem.mkdir()
    (subsystem / ".rag.toml").write_text('subsystem = "power"\ngroup = "psu"\n')
    mpn_dir = subsystem / "STM32H743"
    mpn_dir.mkdir()
    (mpn_dir / ".rag.toml").write_text('mpn = "STM32H743VIT6"\n')

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
# Schema: doc_type, [attributes], and what is rejected (GH #26)
# ---------------------------------------------------------------------------


def test_loads_doc_type_and_attributes(tmp_path: Path) -> None:
    (tmp_path / ".rag.toml").write_text(
        'doc_type = "technical-manual"\n[attributes]\nrevision = "B"\n'
    )

    (cfg,) = find_project_configs(tmp_path)

    assert cfg.doc_type == "technical-manual"
    assert cfg.attributes == {"revision": "B"}


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('colour = "red"\n', "unknown key"),
        ("mpn = 226\n", "'mpn' must be a string"),
        ('doc_type = ["a"]\n', "'doc_type' must be a string"),
        ('tags = "mcu"\n', "'tags' must be a list of strings"),
        ("tags = [1, 2]\n", "'tags' must be a list of strings"),
        ('attributes = "x"\n', "must be a table"),
        ("[attributes]\nrevision = 2\n", "'revision' must be a string, got int — quote it"),
        ("[attributes]\nreviewed = true\n", 'reviewed = "true"'),
        ('[attributes]\nlist = ["a"]\n', "'list' must be a string, got list$"),
        ('[attributes.nested]\nkey = "v"\n', "'nested' must be a string, got dict$"),
        ('[attributes]\n"" = "v"\n', "empty key"),
    ],
)
def test_rejects_schema_violations(tmp_path: Path, body: str, message: str) -> None:
    (tmp_path / ".rag.toml").write_text(body)

    with pytest.raises(ProjectConfigError, match=message):
        find_project_configs(tmp_path)


@pytest.mark.parametrize("key", sorted(RESERVED_ATTRIBUTES))
def test_rejects_reserved_attribute_keys(tmp_path: Path, key: str) -> None:
    (tmp_path / ".rag.toml").write_text(f'[attributes]\n{key} = "x"\n')

    with pytest.raises(ProjectConfigError, match="reserved"):
        find_project_configs(tmp_path)


def test_reserved_attributes_cover_every_key_the_pipeline_writes() -> None:
    from datasheet_rag.cli import _STALE_ATTR

    assert _STALE_ATTR in RESERVED_ATTRIBUTES
    # store.metadata.set_title_source and ingest_pipeline's title hints.
    assert {"title_source", "title_inferred"} <= RESERVED_ATTRIBUTES
    assert {"running_header", "pdf_meta_title"} <= RESERVED_ATTRIBUTES


# ---------------------------------------------------------------------------
# merge_project_configs
# ---------------------------------------------------------------------------


def test_merge_project_configs_returns_none_for_empty_chain() -> None:
    assert merge_project_configs([]) is None


def test_merge_project_configs_nearest_scalar_wins(tmp_path: Path) -> None:
    nearest = ProjectConfig(
        path=tmp_path / "mpn" / ".rag.toml", mpn="STM32H743VIT6", subsystem="mcu"
    )
    middle = ProjectConfig(
        path=tmp_path / "subsystem" / ".rag.toml", subsystem="power", group="psu"
    )
    root = ProjectConfig(
        path=tmp_path / ".rag.toml", project_id="proj-a", manufacturer="STMicro", subsystem="root"
    )

    merged = merge_project_configs([nearest, middle, root])

    assert merged is not None
    assert merged.path == nearest.path
    assert merged.project_id == "proj-a"
    assert merged.manufacturer == "STMicro"
    assert merged.group == "psu"
    assert merged.mpn == "STM32H743VIT6"
    # nearest sets subsystem explicitly, so it wins over middle/root
    assert merged.subsystem == "mcu"


def test_merge_project_configs_unions_tags_nearest_first_deduped(tmp_path: Path) -> None:
    nearest = ProjectConfig(path=tmp_path / "mpn" / ".rag.toml", tags=["errata", "datasheet"])
    root = ProjectConfig(path=tmp_path / ".rag.toml", tags=["datasheet", "reference-manual"])

    merged = merge_project_configs([nearest, root])

    assert merged is not None
    assert merged.tags == ["errata", "datasheet", "reference-manual"]


def test_merge_project_configs_merges_attributes_key_by_key_nearest_wins(tmp_path: Path) -> None:
    nearest = ProjectConfig(path=tmp_path / "mpn" / ".rag.toml", attributes={"revision": "B"})
    middle = ProjectConfig(path=tmp_path / "sub" / ".rag.toml", doc_type="errata")
    root = ProjectConfig(
        path=tmp_path / ".rag.toml",
        doc_type="datasheet",
        attributes={"revision": "A", "owner": "hw"},
    )

    merged = merge_project_configs([nearest, middle, root])

    assert merged is not None
    assert merged.attributes == {"revision": "B", "owner": "hw"}
    assert merged.doc_type == "errata"


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
    (subsystem / ".rag.toml").write_text('subsystem = "eth_phy"\ngroup = "ethernet"\n')
    mpn_dir = subsystem / "LAN8720"
    mpn_dir.mkdir()
    (mpn_dir / ".rag.toml").write_text('mpn = "LAN8720A"\nmanufacturer = "Microchip"\n')

    # cwd is the project root — get_project_config_for should still resolve
    # from the PDF's own directory, not the cwd.
    monkeypatch.chdir(tmp_path)

    cfg = get_project_config_for(mpn_dir)

    assert cfg is not None
    assert cfg.project_id == "netdaq"
    assert cfg.subsystem == "eth_phy"
    assert cfg.group == "ethernet"
    assert cfg.mpn == "LAN8720A"
    assert cfg.manufacturer == "Microchip"
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
