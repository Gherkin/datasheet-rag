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
level instead of overridden, and ``attributes`` are merged key by key.

Example ``.rag.toml``::

    project_id = "stm32-h7-devboard"
    group = "power-subsystem"
    mpn = "STM32H743VIT6"
    manufacturer = "STMicroelectronics"
    subsystem = "mcu"
    doc_type = "reference-manual"
    tags = ["reference-manual"]

    [attributes]
    revision = "B"

The schema is closed: an unknown key, a value of the wrong type, a reserved
attribute key or a file that is not valid TOML raises ``ProjectConfigError``
rather than being dropped, since a silently ignored default mislabels every
document ingested under it (GH #26).
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from datasheet_rag.store.metadata import RESERVED_ATTRIBUTES

CONFIG_FILENAME = ".rag.toml"

_SCALAR_KEYS = ("project_id", "group", "mpn", "manufacturer", "subsystem", "doc_type")
_KNOWN_KEYS = frozenset({*_SCALAR_KEYS, "tags", "attributes"})


class ProjectConfigError(ValueError):
    """A ``.rag.toml`` that cannot be applied as written."""


@dataclass(frozen=True)
class ProjectConfig:
    """Defaults discovered from a ``.rag.toml`` file."""

    path: Path
    project_id: str | None = None
    group: str | None = None
    mpn: str | None = None
    manufacturer: str | None = None
    subsystem: str | None = None
    doc_type: str | None = None
    tags: list[str] | None = None
    attributes: dict[str, str] | None = None


def _check_attributes(candidate: Path, raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ProjectConfigError(f"{candidate}: 'attributes' must be a table, e.g. [attributes]")
    for key, value in raw.items():
        if not key:
            raise ProjectConfigError(f"{candidate}: [attributes] has an empty key")
        if key in RESERVED_ATTRIBUTES:
            raise ProjectConfigError(
                f"{candidate}: attribute {key!r} is reserved — the pipeline sets it "
                f"itself (reserved: {', '.join(sorted(RESERVED_ATTRIBUTES))})"
            )
        if not isinstance(value, str):
            # Only a plain number or bool can simply be quoted; spell a bool
            # the TOML way, not as Python's True/False.
            hint = ""
            if isinstance(value, (bool, int, float)):
                shown = str(value).lower() if isinstance(value, bool) else value
                hint = f' — quote it, e.g. {key} = "{shown}"'
            raise ProjectConfigError(
                f"{candidate}: attribute {key!r} must be a string, got {type(value).__name__}{hint}"
            )
    return dict(raw)


def _load_config(candidate: Path) -> ProjectConfig:
    try:
        with open(candidate, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ProjectConfigError(f"{candidate}: not valid TOML: {e}") from e
    except OSError as e:
        raise ProjectConfigError(f"{candidate}: cannot be read: {e}") from e

    unknown = sorted(set(data) - _KNOWN_KEYS)
    if unknown:
        raise ProjectConfigError(
            f"{candidate}: unknown key(s) {', '.join(unknown)} (known: "
            f"{', '.join(sorted(_KNOWN_KEYS))}; arbitrary keys go under [attributes])"
        )
    for key in _SCALAR_KEYS:
        if key in data and not isinstance(data[key], str):
            raise ProjectConfigError(
                f"{candidate}: {key!r} must be a string, got {type(data[key]).__name__}"
            )
    tags = data.get("tags")
    if tags is not None and not (isinstance(tags, list) and all(isinstance(t, str) for t in tags)):
        raise ProjectConfigError(f"{candidate}: 'tags' must be a list of strings")
    attributes = data.get("attributes")

    return ProjectConfig(
        path=candidate,
        project_id=data.get("project_id"),
        group=data.get("group"),
        mpn=data.get("mpn"),
        manufacturer=data.get("manufacturer"),
        subsystem=data.get("subsystem"),
        doc_type=data.get("doc_type"),
        tags=tags,
        attributes=_check_attributes(candidate, attributes) if attributes is not None else None,
    )


def find_project_configs(start: Path) -> list[ProjectConfig]:
    """Walk ``start`` and its parents, collecting every ``.rag.toml`` found.

    Returns them ordered from most specific (nearest ``start``) to least
    specific (closest to the filesystem root) — the order ``merge_project_configs``
    expects. Raises ``ProjectConfigError`` on the first file that is malformed
    or breaks the schema: skipping it would silently drop the defaults it was
    written to apply.
    """
    configs = []
    for directory in (start, *start.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            configs.append(_load_config(candidate))
    return configs


def merge_project_configs(configs: Sequence[ProjectConfig]) -> ProjectConfig | None:
    """Merge a most-specific-first chain of configs into one effective config.

    Scalar fields (``project_id``, ``mpn``, ``subsystem``, …) use the nearest
    value that's set — a deeper ``.rag.toml`` overrides its ancestors, the
    same way directory-local config overrides project-wide config elsewhere.
    ``tags`` are unioned across every level instead, since tags are additive
    labels rather than a single value to override (de-duplicated, nearest first).
    ``attributes`` merge key by key, the nearest file winning per key.
    The merged ``path`` is the nearest file's, since that's the config a user
    editing files in this directory would reach for first.
    """
    if not configs:
        return None

    def _first(field: str) -> str | None:
        for cfg in configs:
            value: str | None = getattr(cfg, field)
            if value:
                return value
        return None

    tags: list[str] = []
    for cfg in configs:
        for tag in cfg.tags or []:
            if tag not in tags:
                tags.append(tag)

    attributes: dict[str, str] = {}
    for cfg in reversed(configs):
        attributes.update(cfg.attributes or {})

    return ProjectConfig(
        path=configs[0].path,
        project_id=_first("project_id"),
        group=_first("group"),
        mpn=_first("mpn"),
        manufacturer=_first("manufacturer"),
        subsystem=_first("subsystem"),
        doc_type=_first("doc_type"),
        tags=tags or None,
        attributes=attributes or None,
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

    from datasheet_rag.config import get_settings

    return get_settings().default_project_id or None
