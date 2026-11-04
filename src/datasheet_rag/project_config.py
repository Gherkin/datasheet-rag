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
level instead of overridden, and ``keys``/``attributes`` are merged key by key.

Example ``.rag.toml``::

    project_id = "stm32-h7-devboard"
    group = "power-subsystem"
    mpn = ["STM32H743VIT6", "STM32H743ZIT6"]
    manufacturer = "STMicroelectronics"
    subsystem = "mcu"
    doc_type = "reference-manual"
    tags = ["reference-manual"]

    [keys]
    acme-id = "D-0172"           # a key registered with `rag keys add`

    [attributes]
    revision = "B"

``mpn``, ``manufacturer`` and ``doc_type`` are shortcuts for the built-in
keys of the same name; ``[keys]`` sets any registered key (GH #120).

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

from datasheet_rag.store.keys import BUILTIN_KEYS
from datasheet_rag.store.metadata import RESERVED_ATTRIBUTES

CONFIG_FILENAME = ".rag.toml"

_SCALAR_KEYS = ("project_id", "group", "subsystem")
#: Top-level shortcuts for built-in metadata keys: ``mpn = "X"`` means the
#: same as ``[keys] mpn = "X"``. ``mpn`` may be a string or a list.
_KEY_SHORTCUTS = ("mpn", "manufacturer", "doc_type")
_KNOWN_KEYS = frozenset({*_SCALAR_KEYS, *_KEY_SHORTCUTS, "tags", "keys", "attributes"})

#: A ``[keys]`` value: a string, or a list of strings for a list-valued key.
KeyValue = str | list[str]


class ProjectConfigError(ValueError):
    """A ``.rag.toml`` that cannot be applied as written."""


@dataclass(frozen=True)
class ProjectConfig:
    """Defaults discovered from a ``.rag.toml`` file."""

    path: Path
    project_id: str | None = None
    group: str | None = None
    subsystem: str | None = None
    tags: list[str] | None = None
    #: Registered-key values, shortcuts (``mpn``, …) folded in. Whether each
    #: key is registered is checked when the values are written, not here:
    #: loading a config must not need the store.
    keys: dict[str, KeyValue] | None = None
    attributes: dict[str, str] | None = None

    def key(self, name: str) -> KeyValue | None:
        return (self.keys or {}).get(name)


#: Shapes of the built-in keys, known without a store. A user key's shape is
#: only known to the store, so its value is checked when it is written.
_BUILTIN_SHAPES = {k.name: k.shape for k in BUILTIN_KEYS}


def _check_key_value(candidate: Path, key: str, value: Any) -> KeyValue:
    if isinstance(value, str):
        return value
    if _BUILTIN_SHAPES.get(key) == "single":
        raise ProjectConfigError(
            f"{candidate}: {key!r} must be a string, got {type(value).__name__}"
        )
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ProjectConfigError(
        f"{candidate}: {key!r} must be a string or a list of strings, got {type(value).__name__}"
    )


def _check_keys(candidate: Path, raw: Any) -> dict[str, KeyValue]:
    if not isinstance(raw, dict):
        raise ProjectConfigError(f"{candidate}: 'keys' must be a table, e.g. [keys]")
    out: dict[str, KeyValue] = {}
    for key, value in raw.items():
        if not key:
            raise ProjectConfigError(f"{candidate}: [keys] has an empty key")
        out[key] = _check_key_value(candidate, key, value)
    return out


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

    keys = _check_keys(candidate, data["keys"]) if "keys" in data else {}
    for shortcut in _KEY_SHORTCUTS:
        if shortcut not in data:
            continue
        if shortcut in keys:
            raise ProjectConfigError(
                f"{candidate}: {shortcut!r} is set both at the top level and under [keys]"
            )
        value = _check_key_value(candidate, shortcut, data[shortcut])
        if shortcut == "mpn" and isinstance(value, str) and "," in value:
            # Same rule as `--mpn A,B`: a comma string is how a config listed
            # several part numbers before mpn was a list.
            value = [v.strip() for v in value.split(",") if v.strip()]
        keys[shortcut] = value
    attrs = _check_attributes(candidate, attributes) if attributes is not None else None
    clash = sorted(set(attrs or {}) & set(keys))
    if clash:
        raise ProjectConfigError(
            f"{candidate}: {', '.join(clash)} set both under [keys] and [attributes]"
        )

    return ProjectConfig(
        path=candidate,
        project_id=data.get("project_id"),
        group=data.get("group"),
        subsystem=data.get("subsystem"),
        tags=tags,
        keys=keys or None,
        attributes=attrs,
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
    ``keys`` and ``attributes`` merge name by name, the nearest file winning
    per name across both tables (a list value is replaced, not unioned).
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
    keys: dict[str, KeyValue] = {}
    for cfg in reversed(configs):
        # One name space across both tables: a nearer [attributes] X replaces
        # an ancestor's [keys] X, and the other way round.
        for name in cfg.attributes or {}:
            keys.pop(name, None)
        for name in cfg.keys or {}:
            attributes.pop(name, None)
        attributes.update(cfg.attributes or {})
        keys.update(cfg.keys or {})

    return ProjectConfig(
        path=configs[0].path,
        project_id=_first("project_id"),
        group=_first("group"),
        subsystem=_first("subsystem"),
        tags=tags or None,
        keys=keys or None,
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
