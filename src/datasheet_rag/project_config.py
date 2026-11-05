"""Per-directory project defaults, discovered from ``.rag.toml`` files.

Lets users drop small TOML files in a directory tree to set
``project_id``/``tags``/fields once, instead of retyping them on every
``rag`` invocation. Discovery walks up from the current directory the same
way git/pyproject-style tools find their config — but unlike those tools,
*every* ``.rag.toml`` along the way is loaded and merged, nearest-wins, so
you can build a tree of configs:

    netdaq/
      .rag.toml                      # project_id, [fields] manufacturer
      datasheets/
        power-subsystem/
          .rag.toml                  # [fields] subsystem = "power"
          STM32H743VIT6/
            .rag.toml                # [fields] mpn = ["STM32H743VIT6"]
            datasheet.pdf

Running ``rag ingest`` from inside ``STM32H743VIT6/`` merges all three files:
the mpn-level config wins for what it sets (``mpn``), the subsystem-level
config fills in ``subsystem``, and the project-level config supplies the rest
(``project_id``, ``manufacturer``). ``tags`` are unioned across every level
instead of overridden, and ``[fields]`` are merged name by name.

Example ``.rag.toml``::

    project_id = "stm32-h7-devboard"
    tags = ["reference-manual"]

    [fields]
    mpn = ["STM32H743VIT6", "STM32H743ZIT6"]
    manufacturer = "STMicroelectronics"
    doc_type = "reference-manual"
    subsystem = "mcu"            # per project: written in project_id
    acme-id = "D-0172"           # a field registered with `rag fields add`

``[fields]`` sets any registered field, built-in (``mpn``, ``manufacturer``,
``doc_type``, ``subsystem``) or user-registered (GH #120). A project-scoped
field such as ``subsystem`` is written in the config's ``project_id``
and only there: a write to another project (``--project``,
``ingest --project-id``) or one that takes the document out of the config's
project leaves it out (GH #123).

The schema is closed: an unknown key, a value of the wrong type or a file
that is not valid TOML raises ``ProjectConfigError`` rather than being
dropped, since a silently ignored default mislabels every document ingested
under it (GH #26).
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from datasheet_rag.store.fields import BUILTIN_FIELDS, FieldValue

CONFIG_FILENAME = ".rag.toml"

_SCALAR_KEYS = ("project_id",)
_KNOWN_KEYS = frozenset({*_SCALAR_KEYS, "tags", "fields"})

#: Top-level keys removed in GH #123, with what replaces them. Named in the
#: error so an old config says how to fix it, not only that it is wrong.
_REMOVED_KEYS = {
    "group": 'group was removed; use [fields] subsystem = "…"',
    "subsystem": 'subsystem is a field now; move it under [fields]: subsystem = "…"',
}


class ProjectConfigError(ValueError):
    """A ``.rag.toml`` that cannot be applied as written."""


@dataclass(frozen=True)
class ProjectConfig:
    """Defaults discovered from a ``.rag.toml`` file."""

    path: Path
    project_id: str | None = None
    tags: list[str] | None = None
    #: Registered-field values. Whether each field is registered is checked
    #: when the values are written, not here: loading a config must not need
    #: the store.
    fields: dict[str, FieldValue] | None = None


#: Shapes of the built-in fields, known without a store. A user field's shape
#: is only known to the store, so its value is checked when it is written.
_BUILTIN_SHAPES = {f.name: f.shape for f in BUILTIN_FIELDS}


def _check_field_value(candidate: Path, name: str, value: Any) -> FieldValue:
    if isinstance(value, str):
        return value
    if _BUILTIN_SHAPES.get(name) == "single":
        raise ProjectConfigError(
            f"{candidate}: field {name!r} must be a string, got {type(value).__name__}"
        )
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ProjectConfigError(
        f"{candidate}: field {name!r} must be a string or a list of strings, "
        f"got {type(value).__name__}"
    )


def _check_fields(candidate: Path, raw: Any) -> dict[str, FieldValue]:
    if not isinstance(raw, dict):
        raise ProjectConfigError(f"{candidate}: 'fields' must be a table, e.g. [fields]")
    out: dict[str, FieldValue] = {}
    for name, value in raw.items():
        if not name:
            raise ProjectConfigError(f"{candidate}: [fields] has an empty name")
        out[name] = _check_field_value(candidate, name, value)
    return out


def _load_config(candidate: Path) -> ProjectConfig:
    try:
        with open(candidate, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ProjectConfigError(f"{candidate}: not valid TOML: {e}") from e
    except OSError as e:
        raise ProjectConfigError(f"{candidate}: cannot be read: {e}") from e

    removed = sorted(set(data) & set(_REMOVED_KEYS))
    if removed:
        raise ProjectConfigError(
            f"{candidate}: " + "; ".join(_REMOVED_KEYS[k] for k in removed) + " (GH #123)"
        )
    unknown = sorted(set(data) - _KNOWN_KEYS)
    if unknown:
        raise ProjectConfigError(
            f"{candidate}: unknown key(s) {', '.join(unknown)} (known: "
            f"{', '.join(sorted(_KNOWN_KEYS))}; arbitrary values go under [fields])"
        )
    for key in _SCALAR_KEYS:
        if key in data and not isinstance(data[key], str):
            raise ProjectConfigError(
                f"{candidate}: {key!r} must be a string, got {type(data[key]).__name__}"
            )
    tags = data.get("tags")
    if tags is not None and not (isinstance(tags, list) and all(isinstance(t, str) for t in tags)):
        raise ProjectConfigError(f"{candidate}: 'tags' must be a list of strings")
    fields = _check_fields(candidate, data["fields"]) if "fields" in data else None

    return ProjectConfig(
        path=candidate,
        # Stored project names are stripped (GH #121); a blank one is unset.
        project_id=(data.get("project_id") or "").strip() or None,
        tags=tags,
        fields=fields or None,
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

    ``project_id`` uses the nearest value that's set — a deeper ``.rag.toml``
    overrides its ancestors, the same way directory-local config overrides
    project-wide config elsewhere.
    ``tags`` are unioned across every level instead, since tags are additive
    labels rather than a single value to override (de-duplicated, nearest first).
    ``[fields]`` merge name by name, the nearest file winning per name (a list
    value is replaced, not unioned). The merged ``path`` is the nearest file's,
    since that's the config a user editing files in this directory would reach
    for first.
    """
    if not configs:
        return None

    def _first(name: str) -> str | None:
        for cfg in configs:
            value: str | None = getattr(cfg, name)
            if value:
                return value
        return None

    tags: list[str] = []
    for cfg in configs:
        for tag in cfg.tags or []:
            if tag not in tags:
                tags.append(tag)

    fields: dict[str, FieldValue] = {}
    for cfg in reversed(configs):
        fields.update(cfg.fields or {})

    return ProjectConfig(
        path=configs[0].path,
        project_id=_first("project_id"),
        tags=tags or None,
        fields=fields or None,
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
