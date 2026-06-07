"""Per-directory project defaults, discovered from a ``.rag.toml`` file.

Lets users drop a small TOML file in a project directory to set
``project_id``/``group``/etc. once, instead of retyping them on every
``rag`` invocation. Discovery walks up from the current directory the same
way git/pyproject-style tools find their config.

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


def find_project_config(start: Path) -> ProjectConfig | None:
    """Walk ``start`` and its parents looking for ``.rag.toml``.

    Returns the first match parsed into a ``ProjectConfig``, or ``None`` if
    no file is found. Malformed TOML is treated as "not found" rather than
    raising — a broken config file shouldn't break every command.
    """
    for directory in (start, *start.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
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
    return None


@lru_cache(maxsize=1)
def get_project_config() -> ProjectConfig | None:
    """Cached lookup of the ``.rag.toml`` discovered from the cwd."""
    return find_project_config(Path.cwd())


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
