"""Per-directory project defaults, discovered from ``.rag.toml`` files.

Lets users drop small TOML files in a directory tree to set
``project_id``/``group``/etc. once, instead of retyping them on every
``rag`` invocation. Discovery walks up from the current directory the same
way git/pyproject-style tools find their config — but unlike those tools,
*every* ``.rag.toml`` along the way is loaded and merged, nearest-wins, so
you can build a tree of configs:

    netdaq/
      .rag.toml                      # project_id, manufacturer defaults
      datasheets/
        power-subsystem/
          .rag.toml                  # subsystem = "power", group = "psu"
          STM32H743VIT6/
            .rag.toml                # mpn = "STM32H743VIT6"
            datasheet.pdf

Running ``rag ingest`` from inside ``STM32H743VIT6/`` merges all three files:
the mpn-level config wins for fields it sets (``mpn``), the subsystem-level
config fills in ``subsystem``/``group``, and the project-level config supplies
the rest (``project_id``, ``manufacturer``). ``tags`` are unioned across every
level instead of overridden.

Example ``.rag.toml``::

    project_id = "stm32-h7-devboard"
    group = "power-subsystem"
    mpn = "STM32H743VIT6"
    manufacturer = "STMicroelectronics"
    subsystem = "mcu"
    tags = ["reference-manual"]
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CONFIG_FILENAME = ".rag.toml"


@dataclass(frozen=True)
class ProjectConfig:
    """Defaults discovered from a ``.rag.toml`` file."""

    path: Path
    project_id: str | None = None
    group: str | None = None
    mpn: str | None = None
    manufacturer: str | None = None
    subsystem: str | None = None
    tags: list[str] | None = None


def _load_config(candidate: Path) -> ProjectConfig | None:
    try:
        with open(candidate, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError):
        return None
    return ProjectConfig(
        path=candidate,
        project_id=data.get("project_id"),
        group=data.get("group"),
        mpn=data.get("mpn"),
        manufacturer=data.get("manufacturer"),
        subsystem=data.get("subsystem"),
        tags=data.get("tags"),
    )


def find_project_configs(start: Path) -> list[ProjectConfig]:
    """Walk ``start`` and its parents, collecting every ``.rag.toml`` found.

    Returns them ordered from most specific (nearest ``start``) to least
    specific (closest to the filesystem root) — the order ``merge_project_configs``
    expects. A malformed file is skipped (treated as absent) rather than
    aborting the whole walk — a broken config file shouldn't break every command.
    """
    configs = []
    for directory in (start, *start.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            cfg = _load_config(candidate)
            if cfg is not None:
                configs.append(cfg)
    return configs


def merge_project_configs(configs: Sequence[ProjectConfig]) -> ProjectConfig | None:
    """Merge a most-specific-first chain of configs into one effective config.

    Scalar fields (``project_id``, ``mpn``, ``subsystem``, …) use the nearest
    value that's set — a deeper ``.rag.toml`` overrides its ancestors, the
    same way directory-local config overrides project-wide config elsewhere.
    ``tags`` are unioned across every level instead, since tags are additive
    labels rather than a single value to override (de-duplicated, nearest first).
    The merged ``path`` is the nearest file's, since that's the config a user
    editing files in this directory would reach for first.
    """
    if not configs:
        return None

    def _first(field: str) -> str | None:
        for cfg in configs:
            value = getattr(cfg, field)
            if value:
                return value
        return None

    tags: list[str] = []
    for cfg in configs:
        for tag in cfg.tags or []:
            if tag not in tags:
                tags.append(tag)

    return ProjectConfig(
        path=configs[0].path,
        project_id=_first("project_id"),
        group=_first("group"),
        mpn=_first("mpn"),
        manufacturer=_first("manufacturer"),
        subsystem=_first("subsystem"),
        tags=tags or None,
    )


@lru_cache(maxsize=1)
def get_project_config() -> ProjectConfig | None:
    """Cached, merged ``.rag.toml`` chain discovered upward from the cwd."""
    return merge_project_configs(find_project_configs(Path.cwd()))


def get_project_config_for(start: Path) -> ProjectConfig | None:
    """Merged ``.rag.toml`` chain discovered upward from ``start``.

    Unlike ``get_project_config`` (which always resolves from the cwd), this
    resolves from an arbitrary directory — e.g. a document's own directory.
    That's what ``rag ingest`` uses, so running it from the project root still
    picks up subsystem-/mpn-level ``.rag.toml`` files placed next to the PDF
    instead of only the ones above the cwd.
    """
    return merge_project_configs(find_project_configs(start.resolve()))


def resolve_cli_project_id(explicit: str | None, *, is_global: bool) -> str | None:
    """Resolve the effective ``project_id`` scope for a CLI query command.

    Precedence (most to least specific):

    1. ``explicit`` (``--project-id <id>``) — wins outright.
    2. ``is_global`` (``--global``/``-g``) — forces unscoped (``None``),
       even when a ``.rag.toml`` would otherwise scope the command.
    3. ``project_id`` from a discovered ``.rag.toml`` — implicit default scope.
    4. ``settings.default_project_id`` (``RAG_DEFAULT_PROJECT_ID`` env) — final fallback.
    5. ``None`` — unscoped/global.
    """
    if explicit:
        return explicit
    if is_global:
        return None

    config = get_project_config()
    if config is not None and config.project_id:
        return config.project_id

    from aws_rag.config import get_settings

    return get_settings().default_project_id or None
