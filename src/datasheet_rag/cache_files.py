"""The JSON files the pipeline keeps on disk, and their rename from ``doc_id``
to ``rev_id`` (GH #124).

Three kinds of file name a revision by id:

* ``<output_dir>/<rev_id>_chunks.json`` – a chunk graph: the id at the top,
  on every chunk and in every chunk's ``metadata``.
* ``<output_dir>/<rev_id>_outline.json`` – a Docling layout: the id in ``outline``.
* ``<figures_dir>/<rev_id>/manifest.json`` – the figure crops: the id at the top.

Files written before the rename say ``doc_id``. A reader that meets one stops
with :class:`StaleCacheError` instead of guessing; ``rag repair caches``
(:func:`repair_caches`) rewrites them. Only the places listed above are
touched: a ``doc_id`` key anywhere else (a PDF form field, say) is data.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

LEGACY_KEY = "doc_id"
KEY = "rev_id"


class StaleCacheError(ValueError):
    """A cache file from before the ``rev_id`` rename."""


def _holders(data: Any) -> Iterator[dict[str, Any]]:
    """The dicts in a cache file that carry the revision id."""
    if not isinstance(data, dict):
        return
    yield data
    outline = data.get("outline")
    if isinstance(outline, dict):
        yield outline
    chunks = data.get("chunks")
    if isinstance(chunks, dict):
        for chunk in chunks.values():
            if isinstance(chunk, dict):
                yield chunk
                meta = chunk.get("metadata")
                if isinstance(meta, dict):
                    yield meta


def is_stale(data: Any) -> bool:
    return any(LEGACY_KEY in d for d in _holders(data))


def check(data: Any, path: Path) -> Any:
    """``data`` itself; raises :class:`StaleCacheError` for a pre-rename file."""
    if is_stale(data):
        raise StaleCacheError(
            f"{path} was written before doc_id was renamed rev_id. "
            "Run `rag repair caches` to update the cached files, then try again."
        )
    return data


def load(path: Path) -> Any:
    """Read a cache file, refusing one from before the rename."""
    with open(path) as f:
        return check(json.load(f), path)


def _rename(data: Any) -> bool:
    changed = False
    for d in list(_holders(data)):
        if LEGACY_KEY in d:
            value = d.pop(LEGACY_KEY)
            d.setdefault(KEY, value)
            changed = True
    return changed


def cache_files(output_dir: Path, figures_dir: Path) -> list[Path]:
    """Every cache file that names a revision by id, sorted."""
    found: list[Path] = []
    if output_dir.is_dir():
        found += output_dir.glob("*_chunks.json")
        found += output_dir.glob("*_outline.json")
    if figures_dir.is_dir():
        found += figures_dir.glob("*/manifest.json")
    return sorted(found)


def repair_caches(output_dir: Path, figures_dir: Path, *, apply: bool) -> list[Path]:
    """The cache files still keyed ``doc_id``; with ``apply``, rewritten to
    ``rev_id`` in place (each through a temporary file, so a stopped run
    leaves every file whole). Safe to run again."""
    stale: list[Path] = []
    for path in cache_files(output_dir, figures_dir):
        text = path.read_text()
        data = json.loads(text)
        if not _rename(data):
            continue
        stale.append(path)
        if not apply:
            continue
        # Keep the layout the writer used: chunk graphs and manifests are
        # indented, outlines are not.
        indent = 2 if text.startswith("{\n") else None
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=indent, default=str))
        os.replace(tmp, path)
    return stale
