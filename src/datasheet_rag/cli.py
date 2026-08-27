"""CLI for the RAG pipeline."""

from __future__ import annotations

import json
import os
import random
import re
import socket
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import click
from rich.console import Console
from rich.table import Table

from datasheet_rag.config import get_settings

if TYPE_CHECKING:
    import sqlite3

    from datasheet_rag.backend import RagBackend
    from datasheet_rag.backend.base import RagServerError, SearchMode
    from datasheet_rag.backend.models import IngestedDoc
    from datasheet_rag.chunking.layout_parser import DocumentOutline
    from datasheet_rag.costs import CostEstimate
    from datasheet_rag.eval.ablation import IndexVariant
    from datasheet_rag.eval.harness import RunReport
    from datasheet_rag.ingest_pipeline import ProgressEvent
    from datasheet_rag.models.chunk import Chunk
    from datasheet_rag.store.metadata import DocMetadata

console = Console()

# doc_ids are full SHA-256 content hashes (64 hex chars). We display and
# accept abbreviated forms — like `git log --oneline` short SHAs — and
# resolve unambiguous prefixes to the full hash via `resolve_doc_id`.
SHORT_DOC_ID_LEN = 12


def _resolve_doc_id(conn: sqlite3.Connection, doc_id: str) -> str:
    """CLI wrapper around the **store-backed** `resolve_doc_id`: turns
    ambiguity/misses into ClickException.

    Domain: the sqlite index — only resolves docs that have been *embedded*.
    For commands that operate on on-disk cache artifacts (blocks/chunks/outline)
    before a doc is embedded, use `_resolve_cached_doc_id` instead.
    """
    from datasheet_rag.store import resolve_doc_id

    try:
        return resolve_doc_id(conn, doc_id)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _cache_path(doc_id: str, suffix: str) -> Path:
    """Return the conventional cache-artifact path for *doc_id*.

    All local pipeline artifacts live under ``settings.output_dir`` keyed by
    doc_id, e.g. ``{doc_id}_blocks.json`` / ``{doc_id}_chunks.json`` /
    ``{doc_id}_outline.json``.
    """
    return get_settings().output_dir / f"{doc_id}{suffix}"


def _resolve_store_first(arg: str) -> str | None:
    """Expand a doc_id prefix via the store, or return None if it isn't there.

    The store is the authoritative namespace: a doc_id printed by `rag list`
    or `rag search` should resolve the same way everywhere. Cache-reading
    commands consult this first and only fall back to prefix-matching the
    local cache, which is what lets them still work on a PDF that was parsed
    but never ingested.

    Never raises — an unreachable server or an unknown prefix both mean "not
    in the store", and the caller falls back to the cache.
    """
    try:
        return _backend_for().resolve_doc_id(arg)
    except Exception:
        return None


def _resolve_cached_doc_id(doc_id: str, suffix: str = "_outline.json") -> str:
    """Resolve a possibly-abbreviated doc_id against the local cache dir.

    Domain: the filesystem cache (``settings.output_dir``). Globs
    ``{output_dir}/{doc_id}*{suffix}`` and returns the single full doc_id that
    matches. Unlike the store-backed `_resolve_doc_id`, this sees documents
    that have a cached artifact but have not been embedded yet — which is the
    normal state for `repair chunks` / `repair embeddings` inputs.

    Raises ClickException on zero or ambiguous matches (consistent message).
    """
    settings = get_settings()

    # Store first, so an abbreviation resolves the same way it does in
    # `rag list` / `rag search`; fall back to the cache for docs not (yet)
    # ingested.
    full = _resolve_store_first(doc_id)
    if full is not None and (settings.output_dir / f"{full}{suffix}").is_file():
        return full

    matches = sorted(settings.output_dir.glob(f"{doc_id}*{suffix}"))
    if not matches:
        raise click.ClickException(
            f"No cached {suffix.lstrip('_')} matching doc_id {doc_id!r} in "
            f"{settings.output_dir}. Run the earlier pipeline stage for this "
            "document first."
        )
    resolved = {p.name.removesuffix(suffix) for p in matches}
    if len(resolved) > 1:
        names = ", ".join(sorted(resolved))
        raise click.ClickException(f"doc_id {doc_id!r} is ambiguous — matches: {names}")
    return next(iter(resolved))


def _require_docling_outline(doc_id: str) -> str:
    """Resolve *doc_id* to a cached Docling outline, or explain why there isn't one.

    The table commands read Docling's own structure confidence signals, which
    Textract's output has no equivalent of — so they are Docling-only by
    nature, not by oversight. Detect the Textract case explicitly rather than
    letting `_resolve_cached_doc_id` report a generic missing-artifact error
    that reads like the document was never ingested.
    """
    settings = get_settings()
    if not sorted(settings.output_dir.glob(f"{doc_id}*_outline.json")):
        if sorted(settings.output_dir.glob(f"{doc_id}*_blocks.json")):
            raise click.ClickException(
                f"Document {doc_id!r} was ingested with the Textract backend, and "
                "the table commands are Docling-only — they read Docling's "
                "table-structure confidence signals, which Textract's output "
                "does not provide. Re-ingest with `rag ingest --backend docling` "
                "to use them (see GitHub issue #31)."
            )
    return _resolve_cached_doc_id(doc_id, "_outline.json")


def _doc_input(arg: str, suffix: str) -> tuple[str, Path]:
    """Resolve a smart positional that is *either* a doc_id or an explicit file path.

    If *arg* is an existing file, use it directly and infer the doc_id from its
    stem (dropping the artifact suffix, e.g. ``_blocks``). Otherwise treat *arg*
    as a doc_id prefix, resolve it against the cache (`_resolve_cached_doc_id`),
    and return the conventional ``{doc_id}{suffix}`` path.

    Returns ``(doc_id, path)``.
    """
    p = Path(arg)
    if p.is_file():
        stem_suffix = suffix.removesuffix(".json")  # e.g. "_blocks"
        doc_id = p.stem.removesuffix(stem_suffix)
        return doc_id, p
    doc_id = _resolve_cached_doc_id(arg, suffix)
    return doc_id, _cache_path(doc_id, suffix)


def _backend_for(db_path: Path | None = None) -> RagBackend:
    """Return the backend a command should use.

    ``--db <path>`` always means "this specific local sqlite file" so it
    builds a LocalBackend on that path; otherwise the configured backend
    (remote when RAG_SERVER_URL is set, else local).
    """
    if db_path is not None:
        from datasheet_rag.backend import LocalBackend

        return LocalBackend(db_path)
    from datasheet_rag.backend import get_backend

    return get_backend()


def _collect_figure_uploads(graph: Any) -> tuple[dict[str, tuple[bytes, str]], list[str]]:
    """Read every croppable figure in *graph* for upload to a remote server.

    Returns ``(uploads, missing)`` where ``missing`` lists the chunk ids whose
    recorded crop could not be read here. Paths go through
    ``resolve_figure_path`` because a stored path may be relative to
    ``figures_dir`` — treating one as a plain filesystem path resolves it
    against the CWD, silently loses the crop, and leaves the server holding a
    path it cannot serve (GH #41).
    """
    from datasheet_rag.store import resolve_figure_path

    uploads: dict[str, tuple[bytes, str]] = {}
    missing: list[str] = []
    for c in graph.chunks.values():
        if not c.figure_image_path:
            continue
        p = resolve_figure_path(c.figure_image_path)
        if p is not None and p.is_file():
            uploads[c.id] = (p.read_bytes(), p.suffix.lstrip(".") or "png")
        else:
            missing.append(c.id)
    return uploads, missing


def _warn_missing_figure_crops(missing: list[str], uploaded: int) -> None:
    """Say out loud that some figures will land in the store without an image."""
    if not missing:
        return
    console.print(
        f"  [yellow]Warning:[/] {len(missing)} figure crop(s) could not be read "
        f"on this machine and will not be uploaded ({uploaded} will be). Those "
        f"chunks are stored without an image — searchable, but `show_figure` "
        f"cannot serve them. Re-run the ingest with [cyan]--force[/] to re-crop."
    )


def _require_local_db(db_path: Path | None) -> Path:
    """Return a concrete local sqlite path for commands that need raw index
    access (the eval harness: tunable RRF weights, variant-store builds).

    In remote mode these can't run against the HTTP API, so require an
    explicit local --db pointing at a copy of the corpus.
    """
    from datasheet_rag.backend import backend_mode

    if db_path is not None:
        return db_path
    if backend_mode() == "remote":
        raise click.ClickException(
            "eval is a local-only benchmarking harness (it needs raw index "
            "access and builds variant stores) — it can't run against a "
            "remote RAG server. Pass --db pointing at a local sqlite copy of "
            "the corpus, e.g. `--db ./rag.sqlite`."
        )
    return get_settings().sqlite_db_path


def _friendly_server_error(e: RagServerError) -> click.ClickException:
    """Turn a RagServerError into an actionable CLI message (esp. auth)."""
    code = getattr(e, "status_code", None)
    if code == 401:
        return click.ClickException(
            "Server rejected the credentials (401). Check RAG_SERVER_TOKEN — "
            "it may be missing, wrong, or revoked."
        )
    if code == 403:
        return click.ClickException(
            "Insufficient scope (403): this token is read-only. Ingesting and "
            "writes need a per-client ingest key (rag-server create-key --scope ingest)."
        )
    return click.ClickException(str(e))


def _backend_resolve(be: RagBackend, doc_id: str) -> str:
    """Resolve a doc_id prefix via the backend, turning errors into ClickException."""
    from datasheet_rag.backend import RagServerError

    try:
        return be.resolve_doc_id(doc_id)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    except RagServerError as e:
        raise _friendly_server_error(e) from e


def _short_chunk_id(chunk_id: str, doc_id: str) -> str:
    """Abbreviate a chunk_id's doc_id portion the same way doc_ids are
    displayed elsewhere (`ab12cd34ef56:L2:143`), keeping the
    `:L{level}:{index}` suffix intact so it round-trips through
    `_resolve_chunk_id`.
    """
    return doc_id[:SHORT_DOC_ID_LEN] + chunk_id[len(doc_id) :]


def _resolve_chunk_id(be: RagBackend, chunk_id: str) -> str:
    """Resolve a chunk_id whose doc_id portion may be abbreviated (as
    printed by `rag search` / `rag inspect figures`) to its full form.

    Chunk IDs are `{doc_id}:L{level}:{index}`; the doc_id may be given in
    full or as an unambiguous prefix, resolved the same way `doc_id`
    arguments are elsewhere in this CLI.
    """
    if ":" not in chunk_id:
        raise click.ClickException(
            f"{chunk_id!r} doesn't look like a chunk ID — expected "
            "'<doc_id>:L<level>:<index>' (see `rag search` output)."
        )
    doc_part, suffix = chunk_id.split(":", 1)
    full_doc_id = _backend_resolve(be, doc_part)
    return f"{full_doc_id}:{suffix}"


_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def _slugify(text: str | None) -> str:
    """Turn a doc title into a filesystem-safe filename stem, or '' if unusable."""
    if not text or text == "—":
        return ""
    return _SLUG_RE.sub("-", text).strip("-")[:80]


_F = TypeVar("_F", bound=Callable[..., Any])


def _db_option(fn: _F) -> _F:
    """Shared ``--db`` — the same option on 20 commands, defined once.

    Its default lives in ``settings.sqlite_db_path`` rather than in the
    declaration, so Click cannot show it: the help text spells it out instead.
    """
    return click.option(
        "--db",
        "db_path",
        type=click.Path(path_type=Path),
        default=None,
        help="SQLite store to read/write (default: settings.sqlite_db_path, "
        "normally ~/.rag/rag.sqlite).",
    )(fn)


# show_default is inherited by every subcommand's context, so this one setting
# makes Click append `[default: ...]` to the help of every option that has a
# concrete default. Options defaulting to None (and bare flags) still print
# nothing, so those document their fallback in `help=` by hand.
class OrderedGroup(click.Group):
    """A click.Group that lists its commands in a fixed, hand-picked order.

    Click sorts alphabetically, which buries `ingest` and `search` among
    commands most people touch once a year. Ordering by how often a command
    is actually reached makes `rag --help` readable top-to-bottom.
    """

    def __init__(self, *args: Any, order: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._order = order or []

    def list_commands(self, ctx: click.Context) -> list[str]:
        known = [name for name in self._order if name in self.commands]
        # Anything not named in the order falls to the end, alphabetically, so
        # a newly added command shows up rather than silently disappearing.
        return known + sorted(set(self.commands) - set(known))


@click.group(
    cls=OrderedGroup,
    context_settings={"show_default": True},
    order=[
        "ingest",
        "metadata",
        "search",
        "list",
        "get",
        "inspect",
        "config",
        "repair",
        "delete",
        "admin",
    ],
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Datasheet RAG Pipeline — electronics datasheet ingestion."""
    # Remind the user when they're using the local sqlite file rather than a
    # shared server (printed to stderr, non-failing). Skipped for `config`,
    # which is the group that sets the server up.
    if ctx.invoked_subcommand != "config":
        from datasheet_rag.backend import emit_local_notice

        emit_local_notice()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@cli.group("config", short_help="Manage configuration.")
def config_group() -> None:
    """Create and manage this install's configuration."""


def _config_env_lines() -> list[str]:
    """Render every Settings field as a commented-out config.env line.

    Schema-driven so the template never drifts from the model: each field
    becomes ``# RAG_X=<default>   # <terse description>``. The caller fills
    in (uncomments) the handful of critical fields it prompted for.
    """
    from datasheet_rag.config import Settings

    lines: list[str] = []
    for name, field in Settings.model_fields.items():
        env_key = field.alias or f"RAG_{name.upper()}"
        default = field.default
        if default is None or (isinstance(default, str) and default == ""):
            example = ""
        elif isinstance(default, Path):
            example = str(default)
        elif isinstance(default, bool):
            example = "true" if default else "false"
        elif isinstance(default, list):
            example = ",".join(str(x) for x in default)
        else:
            example = str(default)
        desc = (field.description or "").strip().split(". ")[0].split("\n")[0]
        if len(desc) > 90:
            desc = desc[:87] + "…"
        comment = f"   # {desc}" if desc else ""
        lines.append(f"# {env_key}={example}{comment}")
    return lines


@config_group.command("init", short_help="Create ~/.rag and write a config.env.")
@click.option("--force", is_flag=True, help="Overwrite an existing config.env without prompting.")
def config_init(force: bool) -> None:
    """Create ~/.rag and a documented config.env, prompting for the essentials.

    Sets up the local store directory and writes a config file with the few
    critical options filled in (from prompts) and every other option present
    but commented out with a terse description — so it doubles as reference.
    """
    settings = get_settings()
    home = settings.rag_home
    home.mkdir(parents=True, exist_ok=True)
    for sub in ("pdfs", "figures", "cache"):
        (home / sub).mkdir(parents=True, exist_ok=True)

    config_path = home / "config.env"
    if config_path.exists() and not force:
        if not click.confirm(f"{config_path} already exists — overwrite?", default=False):
            console.print("[yellow]Left existing config.env untouched.[/]")
            return

    console.rule("[bold magenta]Configure datasheet-rag[/]")
    console.print(
        "Leave the server URL blank to use a [cyan]local[/] sqlite store. "
        "Point it at a shared RAG server to collaborate with others.\n"
    )
    server_url = click.prompt(
        "Remote RAG server URL (blank = local mode)", default="", show_default=False
    ).strip()

    chosen: dict[str, str] = {}
    if server_url:
        chosen["RAG_SERVER_URL"] = server_url
        console.print(
            "[dim]Paste the shared read token for a read-only client, or this "
            "machine's own ingest key (an ingest key also reads). Ask the "
            "server admin to mint one with `rag-server create-key`.[/]"
        )
        token = click.prompt(
            "Server token / API key (blank = none)", default="", show_default=False
        ).strip()
        if token:
            chosen["RAG_SERVER_TOKEN"] = token
        console.print(
            "[dim]Embeddings run on the server in remote mode — no local model config needed.[/]"
        )
    else:
        backend = click.prompt(
            "Embedding backend", type=click.Choice(["local", "bedrock"]), default="local"
        )
        chosen["RAG_EMBEDDING_BACKEND"] = backend
        if backend == "bedrock":
            chosen["AWS_REGION"] = click.prompt("AWS region", default=settings.aws_region)

    # Render the full template, uncommenting the chosen keys.
    rendered: list[str] = [
        "# datasheet-rag configuration — generated by `rag config init`.",
        "# Uncomment and edit any line below to override a default.",
        "",
    ]
    chosen_written: set[str] = set()
    for line in _config_env_lines():
        key = line[2:].split("=", 1)[0]
        if key in chosen:
            comment = line.split("   # ", 1)
            tail = f"   # {comment[1]}" if len(comment) > 1 else ""
            rendered.append(f"{key}={chosen[key]}{tail}")
            chosen_written.add(key)
        else:
            rendered.append(line)
    # Any chosen key not present as a field (shouldn't happen) appended verbatim.
    for key, val in chosen.items():
        if key not in chosen_written:
            rendered.append(f"{key}={val}")

    config_path.write_text("\n".join(rendered) + "\n")
    console.print(f"\n[green]Wrote[/] {config_path}")
    if server_url:
        console.print(f"  Mode: [cyan]remote[/] → {server_url}")
    else:
        console.print(f"  Mode: [cyan]local[/] → {settings.sqlite_db_path}")
    console.print("  Edit the file to tweak any other option (all are listed, commented).")


# ---------------------------------------------------------------------------
# Inspect / Repair groups
# ---------------------------------------------------------------------------


@cli.group("inspect", short_help="Examine a document without changing it.")
def inspect_group() -> None:
    """Read-only reports about a document.

    Everything here prints and nothing writes, costs money, or calls an LLM.
    Use `repair` for the commands that act on what these turn up.

    `stats` and `figures` read the store, so they only see ingested
    documents; `layout` and `tables` read the local layout cache, so they
    also work on a PDF that was parsed but whose ingest never finished.
    """


@cli.group("repair", short_help="Fix or reprocess an ingested document.")
def repair_group() -> None:
    """Redo part of the pipeline for a document already in the store.

    These re-run individual stages against the cached layout artifact rather
    than re-ingesting — `rag ingest --force` would discard the layout analysis
    too, which is by far the most expensive step on a large datasheet.

    Anything here that changes a document's embedded text (`tables`,
    `figures`, `chunks`) needs `rag repair embeddings <doc-id>` afterwards
    before the change shows up in `rag search`. `titles` is the exception —
    it writes doc_title directly and is visible immediately.
    """


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------

# The column vocabulary `rag list --columns` understands. Split by where the
# value lives: the store's own row (`IngestedDoc`) or the metadata sidecar
# (`DocMetadata`). Anything *not* named here is looked up as a key in the
# sidecar's free-form `attributes` dict — those keys are arbitrary by design,
# so they can't be enumerated, and reaching them is the point of the flag.
_DOC_COLUMNS: dict[str, Callable[[IngestedDoc], Any]] = {
    "doc_id": lambda d: d.doc_id[:SHORT_DOC_ID_LEN],
    "title": lambda d: d.doc_title,
    "chunks": lambda d: d.chunk_count,
    "pages": lambda d: d.page_count,
    "ingested": lambda d: d.ingested_at,
}

_META_COLUMNS: dict[str, Callable[[DocMetadata], Any]] = {
    "project": lambda m: m.project_id,
    "group": lambda m: m.group_name,
    "mpn": lambda m: m.mpn,
    "manufacturer": lambda m: m.manufacturer,
    "subsystem": lambda m: m.subsystem,
    "doc_type": lambda m: m.doc_type,
    "tags": lambda m: m.tags,
    "updated": lambda m: m.updated_at,
    "stale": lambda m: "[yellow]re-embed[/]" if m.attributes.get(_STALE_ATTR) else "",
}

# Field names that would otherwise fall through to an (always empty)
# attribute column. Anyone who knows the model or the JSON `rag metadata`
# prints reaches for these spellings, so map them rather than silently
# printing a column of em-dashes.
_LIST_COLUMN_ALIASES = {
    "id": "doc_id",
    "doc_title": "title",
    "chunk_count": "chunks",
    "page_count": "pages",
    "ingested_at": "ingested",
    "project_id": "project",
    "group_name": "group",
    "updated_at": "updated",
}

_LIST_RIGHT_ALIGNED = {"chunks", "pages"}

# The two built-in views. They answer different questions — ingest health vs.
# cataloguing — so --wide swaps the store-stat columns for the sidecar ones
# rather than appending them. Showing all eleven at once is unreadable on an
# 80-column terminal, where Rich squeezes every cell down to nothing.
_LIST_DEFAULT_COLUMNS = ("title", "chunks", "pages", "ingested")
_LIST_WIDE_COLUMNS = ("project", "group", "mpn", "manufacturer", "subsystem", "tags")


def _parse_list_columns(columns: tuple[str, ...]) -> list[str]:
    """Flatten repeated/comma-separated --columns into a spec list."""
    parsed = [name.strip() for spec in columns for name in spec.split(",") if name.strip()]
    if not parsed:
        raise click.BadParameter("expects at least one column name", param_hint="--columns")
    return [_LIST_COLUMN_ALIASES.get(name, name) for name in parsed]


def _list_cell(name: str, doc: IngestedDoc, meta: DocMetadata | None) -> Any:
    """Value of column *name* for one row, or None when it has none.

    An ``attr:`` prefix forces the attribute reading, which is the only way to
    reach an attribute whose key collides with a built-in column name.
    """
    if name.startswith("attr:"):
        return meta.attributes.get(name[len("attr:") :]) if meta is not None else None
    if (doc_getter := _DOC_COLUMNS.get(name)) is not None:
        return doc_getter(doc)
    if meta is None:
        return None
    if (meta_getter := _META_COLUMNS.get(name)) is not None:
        return meta_getter(meta)
    return meta.attributes.get(name)


def _fmt_list_cell(value: Any) -> str:
    """Render one cell, collapsing every flavour of "nothing" to an em-dash."""
    if value is None or value == "" or value == [] or value == {}:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


@cli.command("list", short_help="List ingested documents.")
@_db_option
@click.option(
    "--project-id",
    default=None,
    help="Restrict to a project (default: scoped by .rag.toml if present).",
)
@click.option(
    "--global",
    "-g",
    "is_global",
    is_flag=True,
    help="Show every project, ignoring any .rag.toml scoping.",
)
@click.option("--group", "group_name", default=None, help="Only show documents in this group.")
@click.option("--mpn", default=None, help="Only show documents with this part number.")
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Only show documents that have this tag (repeatable — "
    "with multiple --tag, a document must have all of them).",
)
@click.option(
    "--attr",
    "attrs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Only show documents whose attributes contain this "
    "key=value pair (repeatable, all must match).",
)
@click.option(
    "--columns",
    "-c",
    "columns",
    multiple=True,
    metavar="COL,COL,...",
    help="Choose the columns to show (comma-separated, repeatable). "
    "Built-ins: title, chunks, pages, ingested, project, group, mpn, "
    "manufacturer, subsystem, doc_type, tags, updated, stale. Any other "
    "name is read as an attribute key set by `rag metadata --attr` "
    "(prefix with `attr:` to force that reading). doc_id is always shown.",
)
@click.option(
    "--wide",
    "-w",
    is_flag=True,
    help="Show the sidecar columns (project, group, mpn, manufacturer, "
    "subsystem, tags) in place of the chunk/page/ingest columns. "
    "Implied whenever a filter is given.",
)
@click.option(
    "--s3",
    "show_s3",
    is_flag=True,
    help="List raw S3 uploads instead (debug — includes documents not yet ingested).",
)
def list_docs(
    db_path: Path | None,
    project_id: str | None,
    is_global: bool,
    group_name: str | None,
    mpn: str | None,
    tags: tuple[str, ...],
    attrs: tuple[str, ...],
    columns: tuple[str, ...],
    wide: bool,
    show_s3: bool,
) -> None:
    """List ingested documents (searchable in the store).

    Shows the store's own view — title, chunk and page counts, ingest time.
    Pass --wide, or any sidecar filter, to swap those for the metadata columns
    (project, group, mpn, manufacturer, subsystem, tags) instead; filters
    narrow the listing to documents whose sidecar row matches.

    --columns replaces both views with exactly the columns you name, and is
    the only way to see free-form attributes:

      rag list --columns mpn,revision,reviewed_by

    Names that aren't built-in columns are read as attribute keys, so that
    prints `revision` and `reviewed_by` as set by
    `rag metadata <doc> --attr revision=B`. An attribute whose name collides
    with a built-in column needs the `attr:` prefix (`--columns attr:tags`).
    """
    from datasheet_rag.project_config import resolve_cli_project_id

    project_id = resolve_cli_project_id(project_id, is_global=is_global)

    if columns and wide:
        raise click.BadParameter(
            "--columns already says which columns to show; drop --wide",
            param_hint="--columns",
        )
    if columns and show_s3:
        raise click.BadParameter(
            "--s3 lists raw uploads, which have no metadata to choose columns from",
            param_hint="--columns",
        )
    chosen = _parse_list_columns(columns) if columns else None

    if show_s3:
        from datasheet_rag.storage import list_documents

        s3_docs = list_documents()
        if not s3_docs:
            console.print("[yellow]No documents found in S3.[/]")
            return

        table = Table(title="S3 Uploads")
        table.add_column("doc_id", style="cyan")
        table.add_column("S3 Prefix")
        for s3_doc in s3_docs:
            table.add_row(s3_doc["doc_id"], s3_doc["prefix"])
        console.print(table)
        return

    attr_filters: dict[str, str] = {}
    for item in attrs:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise click.BadParameter(f"--attr expects KEY=VALUE, got {item!r}", param_hint="--attr")
        attr_filters[key] = value

    filtering = bool(group_name or mpn or tags or attr_filters)
    be = _backend_for(db_path)
    docs = be.get_ingested_docs(project_id=project_id)

    # The sidecar is a separate table from the chunk store, so join by doc_id
    # rather than assuming every ingested document has a row in it.
    # Always fetched, not just when filtering: the stale marker lives in the
    # sidecar and belongs in the default view.
    try:
        meta_by_id = {
            m.doc_id: m for m in be.list_docs(project_id=project_id, group_name=group_name, mpn=mpn)
        }
    except Exception:
        meta_by_id = {}

    if filtering:
        wanted_tags = set(tags)
        docs = [
            d
            for d in docs
            if (m := meta_by_id.get(d.doc_id)) is not None
            and wanted_tags.issubset(set(m.tags))
            and all(m.attributes.get(k) == v for k, v in attr_filters.items())
        ]

    if not docs:
        if filtering:
            console.print("[yellow]No ingested documents match those filters.[/]")
        else:
            console.print(
                "[yellow]No ingested documents found.[/] Run [cyan]rag ingest[/] first "
                "(or pass --s3 to see raw uploads)."
            )
        return

    stale_ids = {
        d.doc_id
        for d in docs
        if (m := meta_by_id.get(d.doc_id)) is not None and m.attributes.get(_STALE_ATTR)
    }

    if chosen is not None:
        shown = list(chosen)
    else:
        shown = list(_LIST_WIDE_COLUMNS if (wide or filtering) else _LIST_DEFAULT_COLUMNS)
    # doc_id is the handle every other `rag` command takes, so a listing
    # without it is a dead end; the stale marker is a health flag rather than
    # metadata, and is worth showing whatever view was asked for.
    if "doc_id" not in shown:
        shown.insert(0, "doc_id")
    if stale_ids and "stale" not in shown:
        shown.insert(shown.index("doc_id") + 1, "stale")

    table = Table(title="Ingested Documents")
    for name in shown:
        label = name[len("attr:") :] if name.startswith("attr:") else name
        table.add_column(
            label,
            style="cyan" if name == "doc_id" else None,
            justify="right" if name in _LIST_RIGHT_ALIGNED else "left",
        )

    # An attribute column that is empty on every row is usually a typo or a
    # case mismatch, not a fleet of documents that all happen to lack it.
    empty_attr_cols = {
        name for name in shown if name not in _DOC_COLUMNS and name not in _META_COLUMNS
    }
    for doc in docs:
        meta = meta_by_id.get(doc.doc_id)
        row: list[str] = []
        for name in shown:
            value = _list_cell(name, doc, meta)
            if value not in (None, "", [], {}):
                empty_attr_cols.discard(name)
            row.append(_fmt_list_cell(value))
        table.add_row(*row)

    console.print(table)
    if empty_attr_cols:
        names = ", ".join(sorted(empty_attr_cols))
        console.print(
            f"  [yellow]No listed document has the attribute(s):[/] {names}. "
            "Attribute names are case-sensitive — check one document's with "
            "[cyan]rag metadata <doc-id>[/]."
        )
    if stale_ids:
        console.print(
            f"  [yellow]{len(stale_ids)} document(s) were repaired since they were "
            "last embedded[/] — their stored text has moved on from their vectors, "
            "so `rag search` still returns the old wording. Run "
            "[cyan]rag repair embeddings <doc-id>[/] on each to catch them up."
        )


@inspect_group.command("stats", short_help="Show chunk counts by zoom level.")
@click.option(
    "--project-id",
    default=None,
    help="Restrict to a project (default: scoped by .rag.toml if present).",
)
@click.option(
    "--global",
    "-g",
    "is_global",
    is_flag=True,
    help="Show stats across every project, ignoring any .rag.toml scoping.",
)
@click.option("--doc-id", default=None, help="Restrict to a single document.")
@_db_option
def stats_cmd(
    project_id: str | None,
    is_global: bool,
    doc_id: str | None,
    db_path: Path | None,
) -> None:
    """Show chunk counts — total and by zoom level — for a scope.

    A quick sanity check on corpus size (e.g. after ingesting: does the
    chunk count look right?). Same rollup as the MCP `stats` tool.
    """
    from datasheet_rag.backend import RagServerError
    from datasheet_rag.project_config import resolve_cli_project_id

    project_id = resolve_cli_project_id(project_id, is_global=is_global)
    be = _backend_for(db_path)
    if doc_id:
        doc_id = _backend_resolve(be, doc_id)

    try:
        result = be.stats(project_id=project_id, doc_id=doc_id)
    except RagServerError as e:
        raise _friendly_server_error(e) from e

    scope_bits = []
    if result.project_id:
        scope_bits.append(f"project={result.project_id}")
    if result.doc_id:
        scope_bits.append(f"doc={result.doc_id[:SHORT_DOC_ID_LEN]}")
    scope = ", ".join(scope_bits) if scope_bits else "all projects"

    table = Table(title=f"Chunk stats — {scope}")
    table.add_column("level")
    table.add_column("count", justify="right")
    for level_name in ("MACRO", "MESO", "MICRO"):
        table.add_row(level_name, str(result.by_level.get(level_name, 0)))
    table.add_row("TOTAL", str(result.total_chunks), style="bold")
    console.print(table)

    # Store-wide, so it is reported however the counts above were scoped.
    if result.fts_missing:
        console.print(
            f"  [yellow]{result.fts_missing} chunk(s) are missing from the "
            f"keyword index[/] — `--mode keyword` cannot see them and "
            f"`--mode hybrid` is running on its vector half alone. Run "
            f"[cyan]rag repair fts[/] to rebuild it."
        )


# ---------------------------------------------------------------------------
# Get (doc / page / chunk / fig — fetch something and save/show it)
# ---------------------------------------------------------------------------


class AliasedGroup(click.Group):
    """A click.Group that resolves a fixed set of long-form aliases to their
    canonical short subcommand name (``document`` -> ``doc``, ``figure`` ->
    ``fig``) before the normal command lookup runs.
    """

    _ALIASES = {"document": "doc", "figure": "fig"}

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        return super().get_command(ctx, self._ALIASES.get(cmd_name, cmd_name))


@cli.group("get", cls=AliasedGroup, short_help="Fetch a doc, page, chunk, figure, or text.")
def get_group() -> None:
    """Fetch a document, page, chunk, figure, or its text and save/show it.

    Subcommands: doc (or document), page, chunk, fig (or figure), text.

    All of these need an ingested document except `text`, which reads the
    local layout cache and so also works for a PDF that was parsed but whose
    ingest never finished.
    """


def _local_ips() -> list[str]:
    """Best-effort discovery of this host's IPv4 addresses.

    Used to print every URL that might reach the loopback PDF server —
    handy when you're SSH'd into the machine and `127.0.0.1` in the
    terminal isn't `127.0.0.1` in your browser. Always includes
    ``127.0.0.1`` first (works when running locally / port-forwarded).
    """
    import subprocess

    ips: set[str] = set()

    # The address this host would use to reach the outside world — a UDP
    # "connect" just picks a route, no packets are actually sent.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
    except OSError:
        pass

    # Every IP the hostname resolves to.
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            ips.add(ip)
    except OSError:
        pass

    # `ip -4 -o addr show` catches interfaces the above can miss (Tailscale,
    # Docker bridges, extra NICs, …) — Linux-only, best-effort.
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if "inet" in parts:
                ips.add(parts[parts.index("inet") + 1].split("/")[0])
    except (OSError, ValueError):
        pass

    ips.discard("127.0.0.1")
    return ["127.0.0.1", *sorted(ips)]


@get_group.command("doc", short_help="Fetch a document's source PDF.")
@click.argument("doc_id", type=str)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Destination file or directory (default: ./<short_doc_id>.pdf). Ignored with --host.",
)
@click.option(
    "--host",
    is_flag=True,
    help="Serve the document instead of downloading it — starts the local "
    "PDF.js viewer and prints browser URLs (see below).",
)
@click.option("--page", default=1, type=int, help="1-based page to open to (--host only).")
@click.option(
    "--launch/--no-launch",
    default=True,
    help="With --host, open the 127.0.0.1 URL in your default browser "
    "(skip it if you're connecting from a different machine). "
    "No effect without --host.",
)
@_db_option
def get_doc_cmd(
    doc_id: str,
    output_path: Path | None,
    host: bool,
    page: int,
    launch: bool,
    db_path: Path | None,
) -> None:
    """Fetch a document's source PDF (from the server / S3, or the local scan
    fallback) — save it to disk, or serve it with --host.

    By default, saves the PDF to disk. With --host, instead starts the
    PDF.js viewer server — the same one the MCP `show_pdf` tool uses, bound
    to every interface — and prints one URL per local IP address (including
    127.0.0.1) so you can pick whichever one your browser can reach:
    localhost if you're on the machine directly, the LAN/Tailscale/SSH
    address if you're remote. The server runs in this process, so the link
    only works while this command stays alive — Ctrl+C to stop serving.
    """
    be = _backend_for(db_path)
    doc_id = _backend_resolve(be, doc_id)

    if host:
        import time
        import webbrowser

        from datasheet_rag import pdf_viewer

        try:
            # Fetch via the backend (HTTP in remote mode) and prime the
            # viewer cache so the loopback server can serve it locally.
            pdf_viewer.prime_pdf_cache(doc_id, be.get_pdf_bytes(doc_id))
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

        port = pdf_viewer.ensure_pdf_server()
        local_url = f"http://127.0.0.1:{port}/viewer/{doc_id}#page={page}"

        console.print("[green]PDF viewer running — pick whichever URL your browser can reach:[/]")
        for ip in _local_ips():
            console.print(f"  http://{ip}:{port}/viewer/{doc_id}#page={page}", soft_wrap=True)

        if launch:
            webbrowser.open(local_url)
            console.print(
                "[dim]Opened the 127.0.0.1 link in your default browser "
                "(use --no-launch to skip this if you're connecting remotely).[/]"
            )
        console.print(
            "[dim]Serving from this process — keep it running to keep the link "
            "alive. Ctrl+C to stop.[/]"
        )

        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped.[/]")
        return

    title = be.get_doc_titles().get(doc_id)

    try:
        data = be.get_pdf_bytes(doc_id)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # RagServerError, etc.
        raise click.ClickException(str(exc)) from exc

    short_id = doc_id[:SHORT_DOC_ID_LEN]
    default_name = f"{_slugify(title) or short_id}.pdf"
    if output_path is None:
        dest = Path(default_name)
    elif output_path.is_dir() or str(output_path).endswith(("/", os.sep)):
        dest = output_path / default_name
    else:
        dest = output_path

    dest.write_bytes(data)
    console.print(f"[green]Saved[/] {len(data):,} bytes → [cyan]{dest}[/]")


@get_group.command("page", short_help="Render a PDF page to a PNG.")
@click.argument("doc_id", type=str)
@click.argument("page_arg", metavar="PAGE", type=int, required=False)
@click.option(
    "--page",
    "page_opt",
    type=int,
    default=None,
    help="1-based page number (alternative to the positional PAGE argument).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to save the image. Defaults to a name derived from the "
    "doc_id and page in the current directory. If a directory, the "
    "default filename is placed inside it.",
)
@click.option("--dpi", default=150, type=int, help="Render DPI for the page image.")
@_db_option
def get_page_cmd(
    doc_id: str,
    page_arg: int | None,
    page_opt: int | None,
    output_path: Path | None,
    dpi: int,
    db_path: Path | None,
) -> None:
    """Render a single PDF page to a PNG and save it to disk.

    PAGE may be given positionally (``rag get page <doc_id> 5``) or via
    ``--page`` (``rag get page <doc_id> --page 5``) — pass exactly one. This
    is the CLI equivalent of the MCP `show_page` tool, minus the inline
    chat rendering (see `rag get doc --host` for the interactive, scrollable
    viewer the MCP `show_pdf` tool uses).
    """
    if page_arg is not None and page_opt is not None:
        raise click.UsageError("Pass PAGE either positionally or via --page, not both.")
    page = page_arg if page_arg is not None else page_opt
    if page is None:
        raise click.UsageError("PAGE is required — pass it positionally or via --page.")

    from pdf2image import convert_from_bytes

    be = _backend_for(db_path)
    doc_id = _backend_resolve(be, doc_id)

    try:
        pdf_bytes = be.get_pdf_bytes(doc_id)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        images = convert_from_bytes(pdf_bytes, first_page=page, last_page=page, dpi=dpi)
    except Exception as exc:
        raise click.ClickException(f"Failed to render page {page}: {exc}") from exc
    if not images:
        short_id = doc_id[:SHORT_DOC_ID_LEN]
        raise click.ClickException(f"Page {page} not found in document {short_id}.")

    default_name = f"{doc_id[:SHORT_DOC_ID_LEN]}_p{page}.png"
    if output_path is None:
        dest = Path(default_name)
    elif output_path.is_dir() or str(output_path).endswith(("/", os.sep)):
        dest = output_path / default_name
    else:
        dest = output_path

    images[0].save(dest, format="PNG")
    console.print(f"[green]Saved[/] page {page} → [cyan]{dest}[/]")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


@cli.command("delete", short_help="Permanently delete a document.")
@click.argument("doc_id", type=str)
@_db_option
@click.option(
    "--dry-run", is_flag=True, help="Show what would be deleted without deleting anything."
)
@click.option("-y", "--yes", "assume_yes", is_flag=True, help="Skip the confirmation prompt.")
def delete_doc_cmd(doc_id: str, db_path: Path | None, dry_run: bool, assume_yes: bool) -> None:
    """Permanently delete a document.

    Removes its chunks, vectors and metadata from the store, its local PDF
    and figure-crop files, cached pipeline artifacts, and (if S3 is
    configured) the matching S3 objects. There is no undo — the document
    must be re-ingested from the source PDF to get it back.
    """
    be = _backend_for(db_path)
    doc_id = _backend_resolve(be, doc_id)
    title = be.get_doc_titles().get(doc_id)
    chunk_count = be.count_chunks(doc_id=doc_id)
    label = f"{title!r} ({doc_id[:SHORT_DOC_ID_LEN]})" if title else doc_id[:SHORT_DOC_ID_LEN]

    if dry_run:
        console.print(f"[yellow]--dry-run:[/] would delete {chunk_count} chunk(s) for {label}")
        console.print(
            "  Also removes: local PDF, figure-crop directory, cached pipeline "
            "artifacts, and any S3 objects under "
            f"[cyan]{get_settings().s3_pdf_prefix}{doc_id}/[/] and "
            f"[cyan]figures/{doc_id}/[/] (if S3 is configured)."
        )
        return

    if not assume_yes:
        if not click.confirm(
            f"Delete {chunk_count} chunk(s) for {label}? This also removes local "
            "PDF/figure files and any S3 content, and cannot be undone.",
            default=False,
        ):
            console.print("[yellow]Aborted.[/]")
            return

    try:
        deleted = be.delete_doc(doc_id)
    except Exception as exc:  # RagServerError, etc.
        raise click.ClickException(str(exc)) from exc

    console.print(f"[green]Deleted[/] {deleted} chunk(s) for {label}.")


@get_group.command("text", short_help="Dump a document's extracted text.")
@click.argument("doc_id", type=str)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the text to this file (default: print it to stdout).",
)
def get_text_cmd(doc_id: str, output: Path | None) -> None:
    """Dump a document's extracted text in reading order.

    DOC_ID is a doc_id (full hash or unambiguous prefix), resolved against
    whichever layout artifact the ingest backend cached — Docling's
    ``{doc_id}_outline.json`` or Textract's ``{doc_id}_blocks.json``. An
    explicit path to either is also accepted in place of a doc_id.

    Section titles are emitted as headings so the dump keeps the document's
    structure rather than running together as one wall of prose.
    """
    _, outline = _load_outline(doc_id)

    parts: list[str] = []
    for section in outline.all_sections_flat:
        if section.title:
            parts.append(f"{'#' * (section.level + 1)} {section.title}")
        body = section.all_text
        if body:
            parts.append(body)
    text = "\n\n".join(parts)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        console.print(f"[green]Text saved to[/] {output}")
    else:
        console.print(text)


# ---------------------------------------------------------------------------
# Inspect layout (debug helper)
# ---------------------------------------------------------------------------


def _resolve_layout_input(arg: str) -> tuple[str, str, Path]:
    """Resolve *arg* to a cached layout artifact from either ingestion backend.

    The two backends leave different layout artifacts on disk: Textract writes
    ``{doc_id}_blocks.json`` (raw blocks) and Docling writes
    ``{doc_id}_outline.json`` (a serialized `DocumentOutline`). *arg* is either
    an explicit path to one of those, or a doc_id (full hash or unambiguous
    prefix) resolved against the cache — preferring the Docling outline when
    both exist, since Docling is the default backend.

    Returns ``(doc_id, backend, path)`` where *backend* is ``"docling"`` or
    ``"textract"``.
    """
    p = Path(arg)
    if p.is_file():
        if p.name.endswith("_outline.json"):
            return p.stem.removesuffix("_outline"), "docling", p
        if p.name.endswith("_blocks.json"):
            return p.stem.removesuffix("_blocks"), "textract", p
        raise click.ClickException(
            f"{p.name} is not a recognised layout artifact — expected a "
            "Docling '_outline.json' or Textract '_blocks.json' file."
        )

    settings = get_settings()
    artifacts = (("_outline.json", "docling"), ("_blocks.json", "textract"))

    # Store first (see _resolve_store_first), cache second.
    full = _resolve_store_first(arg)
    if full is not None:
        for suffix, backend in artifacts:
            path = settings.output_dir / f"{full}{suffix}"
            if path.is_file():
                return full, backend, path

    for suffix, backend in artifacts:
        if sorted(settings.output_dir.glob(f"{arg}*{suffix}")):
            doc_id = _resolve_cached_doc_id(arg, suffix)
            return doc_id, backend, _cache_path(doc_id, suffix)

    raise click.ClickException(
        f"No cached layout artifact matching doc_id {arg!r} in "
        f"{settings.output_dir}. Run `rag ingest` for this document first "
        "(Docling writes '_outline.json', Textract writes '_blocks.json')."
    )


def _load_outline(arg: str) -> tuple[str, DocumentOutline]:
    """Load a cached layout artifact as a `DocumentOutline`, whichever backend
    produced it.

    `DocumentOutline` is the backend-neutral representation the whole chunking
    pipeline already runs on — Docling builds one directly, and Textract blocks
    are converted by `parse_textract_blocks`. Resolving to it here is what lets
    the post-ingest commands behave identically on both backends instead of
    only working on whichever one happened to write their input file.
    """
    from datasheet_rag.chunking.layout_parser import DocumentOutline, parse_textract_blocks

    doc_id, backend, path = _resolve_layout_input(arg)
    with open(path) as f:
        data = json.load(f)

    if backend == "docling":
        # The cache wraps the outline alongside its figure regions.
        return doc_id, DocumentOutline.from_dict(data["outline"])
    return doc_id, parse_textract_blocks(data, doc_id=doc_id)


class _FlaggedTable(NamedTuple):
    """One table the header detectors judged untrustworthy, for --list-flagged."""

    page: int
    caption: str
    reason: str


# Rows of the section/layout listing to show before truncating (unless --full).
_INSPECT_LAYOUT_PREVIEW = 30


def _inspect_textract_layout(blocks: list[dict[str, Any]], full: bool = False) -> None:
    """Render the block-type summary and layout hierarchy for Textract output."""
    from datasheet_rag.textract import extract_layout_elements

    by_type = extract_layout_elements(blocks)

    table = Table(title="Textract Block Types")
    table.add_column("Block Type", style="cyan")
    table.add_column("Count", justify="right")

    for bt, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        table.add_row(bt, str(len(items)))

    console.print(table)

    # Show layout hierarchy if present
    layout_blocks = [b for b in blocks if b.get("BlockType", "").startswith("LAYOUT_")]
    if layout_blocks:
        console.print(f"\n[bold]Layout blocks ({len(layout_blocks)}):[/]")
        shown = layout_blocks if full else layout_blocks[:_INSPECT_LAYOUT_PREVIEW]
        for lb in shown:
            bt = lb["BlockType"]
            page = lb.get("Page", "?")
            top = lb.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0)
            text_preview = ""
            if "Text" in lb:
                text_preview = lb["Text"][:80]
            console.print(f"  p{page} {bt:30s} top={top:.3f}  {text_preview}")

        if len(layout_blocks) > len(shown):
            console.print(
                f"  … and {len(layout_blocks) - len(shown)} more (pass --full to show all)"
            )


def _inspect_docling_layout(outline: DocumentOutline, full: bool = False) -> None:
    """Render the element-type summary and section hierarchy for Docling output."""
    summary = outline.summary()

    table = Table(title="Docling Element Types")
    table.add_column("Element Type", style="cyan")
    table.add_column("Count", justify="right")
    for et, count in sorted(summary["elements_by_type"].items(), key=lambda x: -x[1]):
        table.add_row(et, str(count))
    console.print(table)

    console.print(
        f"\n{summary['top_level_sections']} chapters, "
        f"{summary['total_sections']} sections, "
        f"{summary['total_elements']} elements "
        f"across {summary['total_pages']} pages."
    )

    # Show the section hierarchy (depth-first, indented by level).
    flat = outline.all_sections_flat
    if flat:
        console.print(f"\n[bold]Section hierarchy ({len(flat)}):[/]")
        shown = flat if full else flat[:_INSPECT_LAYOUT_PREVIEW]
        for section in shown:
            indent = "  " * (section.level + 1)
            pages = (
                f"p{section.page_start}"
                if section.page_start == section.page_end
                else f"p{section.page_start}-{section.page_end}"
            )
            title = (section.title or "(untitled)")[:80]
            console.print(f"{indent}{title}  [dim]{pages}, {section.element_count} elements[/]")
        if len(flat) > len(shown):
            console.print(f"  … and {len(flat) - len(shown)} more (pass --full to show all)")


@inspect_group.command("layout", short_help="Summarise a document's parsed layout.")
@click.argument("doc_id", type=str)
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Show the entire section/layout listing instead of the first "
    f"{_INSPECT_LAYOUT_PREVIEW} rows.",
)
def inspect_layout(doc_id: str, full: bool) -> None:
    """Show a summary of the parsed layout structure for a document.

    Works with either ingestion backend: the cached Docling
    ``{doc_id}_outline.json`` (element types + section hierarchy) or the
    Textract ``{doc_id}_blocks.json`` (block types + layout blocks). DOC_ID is
    a doc_id (full hash or unambiguous prefix) resolved against the cache, or an
    explicit path to one of those artifacts.
    """
    resolved_id, backend, path = _resolve_layout_input(doc_id)
    with open(path) as f:
        data = json.load(f)

    console.print(f"doc_id [cyan]{resolved_id}[/] — [cyan]{backend}[/] backend")

    if backend == "docling":
        from datasheet_rag.chunking.layout_parser import DocumentOutline

        _inspect_docling_layout(DocumentOutline.from_dict(data["outline"]), full=full)
    else:
        # Accept both the plain list `save_blocks` writes and a raw Textract
        # API response dict (e.g. a file fetched independently of `rag ingest`).
        blocks = data.get("Blocks", []) if isinstance(data, dict) else data
        _inspect_textract_layout(blocks, full=full)


# ---------------------------------------------------------------------------
# Chunk (multi-scale chunking pipeline)
# ---------------------------------------------------------------------------


@repair_group.command("chunks", short_help="Re-chunk a parsed document.")
@click.argument("doc_id", type=str)
@click.option(
    "--figures-manifest",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Figure manifest JSON to fold in (default: the manifest ingest "
    "cached for this document, so re-chunking keeps its figures).",
)
@click.option("--micro-tokens", default=128, type=int, help="Max tokens per MICRO chunk.")
@click.option("--meso-tokens", default=512, type=int, help="Max tokens per MESO chunk.")
@click.option(
    "--summarizer",
    type=click.Choice(["extractive", "abstractive"]),
    default="extractive",
    help="Summarization mode for MACRO chunks.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write the chunk graph (default: the cached {doc_id}_chunks.json).",
)
def chunk_cmd(
    doc_id: str,
    figures_manifest: Path | None,
    micro_tokens: int,
    meso_tokens: int,
    summarizer: str,
    output: Path | None,
) -> None:
    """Re-run multi-scale chunking against a document's cached layout.

    DOC_ID is a doc_id (full hash or unambiguous prefix), resolved against
    whichever layout artifact the ingest backend cached — Docling's
    ``{doc_id}_outline.json`` or Textract's ``{doc_id}_blocks.json``. An
    explicit path to either is also accepted in place of a doc_id. Chunks are
    written to ``{doc_id}_chunks.json`` unless --output is given.

    This is the cheap way to retune --micro-tokens / --meso-tokens: it reuses
    the cached layout analysis, where `rag ingest --force` would redo it.
    Follow with `rag repair embeddings <doc-id>` to re-index the result.

    Produces a hierarchical chunk graph at three levels (MACRO/MESO/MICRO)
    with navigation links, context enrichment, and chapter summaries.
    """
    from datasheet_rag.chunking.pipeline import (
        run_chunking_pipeline_from_outline,
        save_chunk_graph,
    )
    from datasheet_rag.chunking.splitter import SplitterConfig

    doc_id, outline = _load_outline(doc_id)

    # Default to the manifest ingest wrote for this document — without it a
    # re-chunk would silently drop every figure from the graph.
    if figures_manifest is None:
        cached_manifest = get_settings().figures_dir / doc_id / "manifest.json"
        if cached_manifest.is_file():
            figures_manifest = cached_manifest

    figure_manifest = None
    if figures_manifest:
        with open(figures_manifest) as f:
            figure_manifest = json.load(f)

    config = SplitterConfig(
        micro_max_tokens=micro_tokens,
        meso_max_tokens=meso_tokens,
    )

    graph = run_chunking_pipeline_from_outline(
        outline,
        figure_manifest=figure_manifest,
        config=config,
        summarizer_mode=summarizer,
    )

    # Save
    settings = get_settings()
    if output is None:
        output = settings.output_dir / f"{doc_id}_chunks.json"

    save_chunk_graph(graph, output)

    # Display summary
    stats = graph.stats()
    table = Table(title="Chunk Graph Summary")
    table.add_column("Level", style="cyan")
    table.add_column("Count", justify="right")

    for level_name, count in stats["by_level"].items():
        table.add_row(level_name, str(count))
    table.add_row("TOTAL", str(stats["total_chunks"]), style="bold")

    console.print(table)

    # Show MACRO chunks (chapter summaries)
    from datasheet_rag.models.chunk import ChunkLevel

    macros = graph.by_level(ChunkLevel.MACRO)
    if macros:
        console.print("\n[bold]Chapter Summaries:[/]")
        for m in macros:
            console.print(
                f"\n[cyan]{m.metadata.chapter_title}[/] (pages {m.metadata.page_numbers})"
            )
            if m.text:
                preview = m.text[:300] + "…" if len(m.text) > 300 else m.text
                console.print(f"  {preview}")
            else:
                console.print("  [yellow](no summary generated)[/]")

    # Only nag when the graph landed where `repair embeddings` will look for
    # it — with an explicit --output this is a scratch dump, not a re-index.
    if output == settings.output_dir / f"{doc_id}_chunks.json":
        _next_steps(doc_id, rechunk=False)


# ---------------------------------------------------------------------------
# Embed (Bedrock Titan + SQLite store)
# ---------------------------------------------------------------------------


@repair_group.command("embeddings", short_help="Re-embed a document into the store.")
@click.argument("doc_id", type=str)
@_db_option
@click.option("--project-id", default=None, help="Project ID to attach to every chunk.")
@click.option("--group", "group_name", default=None, help="Group name to attach to every chunk.")
@click.option("--verbose/--quiet", default=True, help="Print per-batch progress.")
def embed(
    doc_id: str,
    db_path: Path | None,
    project_id: str | None,
    group_name: str | None,
    verbose: bool,
) -> None:
    """Embed a document's chunk graph and store it.

    DOC_ID is a doc_id (full hash or unambiguous prefix); the chunk graph is
    read from the cached ``{doc_id}_chunks.json``. An explicit chunks-JSON path
    is also accepted in place of a doc_id.

    Embedding + insert run through the backend, so this writes to the remote
    server (which embeds) when RAG_SERVER_URL is set, or the local sqlite
    store otherwise. There is no dry run — embedding happens backend-side;
    use `rag ingest --show-cost` to price a run without writing.
    """
    from datasheet_rag.backend import MetadataPatch, backend_mode
    from datasheet_rag.chunking.pipeline import load_chunk_graph
    from datasheet_rag.project_config import get_project_config

    _, chunks_json = _doc_input(doc_id, "_chunks.json")

    proj_cfg = get_project_config()
    if proj_cfg is not None:
        project_id = project_id or proj_cfg.project_id
        group_name = group_name or proj_cfg.group

    console.print(f"Loading chunk graph from [cyan]{chunks_json}[/]…")
    graph = load_chunk_graph(chunks_json)
    stats = graph.stats()
    console.print(
        f"  {stats['total_chunks']} chunks "
        f"(MACRO {stats['by_level']['MACRO']}, "
        f"MESO {stats['by_level']['MESO']}, "
        f"MICRO {stats['by_level']['MICRO']})"
    )

    be = _backend_for(db_path)

    # Remote: ship figure images so the server stores them and the embedded
    # context_text can fold in any server-side descriptions.
    figures_upload: dict[str, tuple[bytes, str]] | None = None
    if backend_mode() == "remote" and db_path is None:
        figures_upload, missing = _collect_figure_uploads(graph)
        _warn_missing_figure_crops(missing, len(figures_upload))

    console.print("Embedding & writing via the backend…")
    from datasheet_rag.backend import RagServerError

    try:
        result = be.ingest_chunk_graph(
            graph,
            figures=figures_upload,
            project_id=project_id,
            group_name=group_name,
            metadata=MetadataPatch(),
            embed=True,
            describe_figures=False,
        )
    except RagServerError as e:
        raise _friendly_server_error(e) from e
    console.print(f"[green]Inserted[/] {result.inserted} chunks.")
    _report_pruned(result.pruned, indent="")
    # Vectors now match the stored text again.
    _set_stale(result.doc_id or doc_id, False)


# ---------------------------------------------------------------------------
# Search (hybrid / vector / keyword)
# ---------------------------------------------------------------------------


@cli.command(short_help="Search the RAG store.")
@click.argument("query", type=str)
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "vector", "keyword"]),
    default="hybrid",
    help="Retrieval mode.",
)
@click.option("-k", "top_k", default=10, type=int, help="Number of results.")
@_db_option
@click.option(
    "--project-id",
    default=None,
    help="Restrict to a project (default: scoped by .rag.toml if present).",
)
@click.option(
    "--global",
    "-g",
    "is_global",
    is_flag=True,
    help="Search every project, ignoring any .rag.toml scoping.",
)
@click.option("--group", "group_name", default=None, help="Restrict to a group.")
@click.option("--doc-id", "doc_ids", multiple=True, help="Restrict to one or more doc IDs.")
@click.option(
    "--level",
    type=click.Choice(["macro", "meso", "micro"]),
    default=None,
    help="Restrict to a single zoom level.",
)
@click.option(
    "--show-context/--no-show-context",
    default=False,
    help="Show context_text (full embedding-ready blob) instead of raw text.",
)
def search(
    query: str,
    mode: SearchMode,
    top_k: int,
    db_path: Path | None,
    project_id: str | None,
    is_global: bool,
    group_name: str | None,
    doc_ids: tuple[str, ...],
    level: str | None,
    show_context: bool,
) -> None:
    """Search the RAG store (local sqlite or remote server) with hybrid /
    vector / keyword retrieval. The query is embedded by the backend."""
    from datasheet_rag.backend import RagServerError
    from datasheet_rag.models.chunk import ChunkLevel
    from datasheet_rag.project_config import resolve_cli_project_id
    from datasheet_rag.store import SearchFilters

    project_id = resolve_cli_project_id(project_id, is_global=is_global)
    be = _backend_for(db_path)

    resolved_doc_ids = [be.resolve_doc_id(d) for d in doc_ids]

    level_enum = None
    if level:
        level_enum = {
            "macro": ChunkLevel.MACRO,
            "meso": ChunkLevel.MESO,
            "micro": ChunkLevel.MICRO,
        }[level]

    filters = SearchFilters(
        doc_ids=resolved_doc_ids if resolved_doc_ids else None,
        project_id=project_id,
        group_name=group_name,
        level=level_enum,
    )

    try:
        results = be.search(query, mode=mode, k=top_k, filters=filters)
    except RagServerError as e:
        raise _friendly_server_error(e) from e

    if not results:
        console.print("[yellow]No results.[/]")
        return

    table = Table(title=f"{mode.title()} search · {len(results)} results")
    table.add_column("#", justify="right", style="dim")
    table.add_column("score", justify="right", style="cyan")
    table.add_column("level", style="magenta")
    table.add_column("chunk_id", style="cyan", no_wrap=True)
    table.add_column("section")
    table.add_column("preview")

    for i, r in enumerate(results, 1):
        body = r.chunk.context_text if show_context else r.chunk.text
        preview = body[:140].replace("\n", " ") + ("…" if len(body) > 140 else "")
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            r.chunk.level.name,
            _short_chunk_id(r.chunk.id, r.chunk.doc_id),
            (r.chunk.metadata.section_title or r.chunk.metadata.chapter_title or "")[:40],
            preview,
        )

    console.print(table)
    console.print("[dim]Fetch a full chunk with: rag get chunk <chunk_id>[/]")


# Sidecar attribute marking a document whose stored text has moved on from
# its vectors. Set by the repairs that change embedded text, cleared by
# `rag repair embeddings`, and surfaced as a column in `rag list`.
_STALE_ATTR = "needs_reembed"


def _set_stale(doc_id: str | None, stale: bool) -> None:
    """Best-effort flip of the stale marker on *doc_id*'s sidecar row.

    Deliberately swallows every failure: a repair that already succeeded must
    not be reported as failed because a bookkeeping write didn't land, and
    these commands legitimately run against documents that were parsed but
    never ingested, which have no sidecar row to write to.
    """
    if not doc_id:
        return
    from datasheet_rag.backend import MetadataPatch

    try:
        be = _backend_for()
        be.set_metadata(
            be.resolve_doc_id(doc_id),
            MetadataPatch(attributes={_STALE_ATTR: True if stale else None}),
        )
    except Exception:
        pass


def _next_steps(doc_id: str | None, *, rechunk: bool) -> None:
    """Print what still has to happen before a repair shows up in `rag search`.

    Every `repair` subcommand edits something upstream of the vectors, so none
    of them take effect on their own. Repairs that patch the cached layout
    outline need the chunk graph rebuilt first; repairs that edit chunk text
    in place only need re-embedding.
    """
    _set_stale(doc_id, True)
    target = doc_id[:SHORT_DOC_ID_LEN] if doc_id else "<doc-id>"
    console.print()
    console.rule("[bold yellow]Not searchable yet[/]")
    if rechunk:
        console.print(
            "  This patched the cached layout outline. Rebuild and re-index it:\n"
            f"    [cyan]rag repair chunks {target}[/]\n"
            f"    [cyan]rag repair embeddings {target}[/]"
        )
    else:
        console.print(
            "  Re-embed to refresh the vectors before this shows up in search:\n"
            f"    [cyan]rag repair embeddings {target}[/]"
        )


def _print_chunk_detail(chunk: Chunk, *, show_context: bool = False) -> None:
    """Render one chunk's metadata + text — the CLI counterpart of the MCP
    server's ``_shape_chunk``."""
    from datasheet_rag.models.chunk import LayoutType

    pages = chunk.metadata.page_numbers
    page = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}" if pages else "—"

    console.print(f"[bold cyan]{chunk.id}[/]")
    console.print(f"  level:   {chunk.level.name}")
    console.print(f"  page:    {page}")
    section = chunk.metadata.section_title or chunk.metadata.chapter_title or "—"
    console.print(f"  section: {section}")
    console.print(f"  parent:  {chunk.parent_id or '—'}")
    console.print(f"  prev:    {chunk.prev_id or '—'}")
    console.print(f"  next:    {chunk.next_id or '—'}")

    if chunk.metadata.layout_type == LayoutType.FIGURE:
        if chunk.figure_caption:
            console.print(f"  caption: {chunk.figure_caption}")
        if chunk.figure_description:
            console.print(f"  description: {chunk.figure_description}")
        if not (chunk.figure_image_path or chunk.figure_s3_key):
            console.print("  [yellow]no figure image available[/]")

    console.print()
    console.print(chunk.context_text if show_context and chunk.context_text else chunk.text)


@get_group.command("chunk", short_help="Fetch one chunk by ID.")
@click.argument("chunk_id", type=str)
@click.option(
    "--neighbors/--no-neighbors",
    default=False,
    help="Also print the parent/prev/next chunks (mirrors the MCP "
    "get_chunk tool's include_neighbors option).",
)
@click.option(
    "--show-context/--no-show-context",
    default=False,
    help="Print context_text (embedding-ready blob) instead of raw text.",
)
@_db_option
def get_chunk_cmd(
    chunk_id: str,
    neighbors: bool,
    show_context: bool,
    db_path: Path | None,
) -> None:
    """Fetch one chunk by ID — the CLI equivalent of the MCP `get_chunk` tool.

    CHUNK_ID accepts the full id or an abbreviated form using a doc_id
    prefix, e.g. ``ab12cd34ef56:L2:143`` as printed by `rag search` /
    `rag inspect figures`.
    """
    from datasheet_rag.backend import RagServerError

    be = _backend_for(db_path)
    full_id = _resolve_chunk_id(be, chunk_id)

    try:
        chunk = be.get_chunk(full_id)
    except RagServerError as e:
        raise _friendly_server_error(e) from e

    if chunk is None:
        raise click.ClickException(f"No chunk found with id {full_id!r}.")

    _print_chunk_detail(chunk, show_context=show_context)

    if neighbors:
        links = (("parent", chunk.parent_id), ("prev", chunk.prev_id), ("next", chunk.next_id))
        for label, nid in links:
            console.print()
            console.rule(label)
            if not nid:
                console.print("[dim]— none —[/]")
                continue
            try:
                nchunk = be.get_chunk(nid)
            except RagServerError as e:
                raise _friendly_server_error(e) from e
            if nchunk is None:
                console.print(f"[dim]{nid} (not found)[/]")
                continue
            _print_chunk_detail(nchunk, show_context=show_context)


# ---------------------------------------------------------------------------
# Figures (list / inspect figure chunks in the store)
# ---------------------------------------------------------------------------


def _warn_unusable_figures(unusable: list[Any], listed: bool) -> None:
    """Point at figure chunks that exist but can never be shown."""
    if not unusable:
        return
    docs = {c.doc_id for c in unusable}
    lead = (
        f"[yellow]{len(unusable)} figure chunk(s)[/] across {len(docs)} document(s) "
        f"have no usable image"
    )
    if not listed:
        lead += " (hidden — pass [cyan]--include-unusable[/] to see them)"
    console.print(
        f"{lead}. Search will not offer them. Try [cyan]rag repair "
        f"figure-links[/] if the crops are still on disk, otherwise re-ingest "
        f"the document without --skip-figures."
    )


@inspect_group.command("figures", short_help="List figure chunks in the store.")
@click.option("--doc-id", default=None, help="Restrict to a single document.")
@click.option(
    "--project-id",
    default=None,
    help="Restrict to a project (default: scoped by .rag.toml if present).",
)
@click.option(
    "--global",
    "-g",
    "is_global",
    is_flag=True,
    help="List figures across every project, ignoring any .rag.toml scoping.",
)
@_db_option
@click.option(
    "--missing-description-only",
    is_flag=True,
    help="Only show figure chunks whose figure_description is empty.",
)
@click.option(
    "--include-unusable", is_flag=True, help="Also list figure chunks whose image cannot be served."
)
def list_figures_cmd(
    doc_id: str | None,
    project_id: str | None,
    is_global: bool,
    db_path: Path | None,
    missing_description_only: bool,
    include_unusable: bool,
) -> None:
    """List figure chunks in the store.

    Shows the ones with a usable image by default. Pass --include-unusable to
    see figure chunks whose image is missing — search will not offer those,
    and `rag repair figure-links` may be able to reattach their crops.
    """
    from datasheet_rag.project_config import resolve_cli_project_id

    project_id = resolve_cli_project_id(project_id, is_global=is_global)

    be = _backend_for(db_path)
    if doc_id:
        doc_id = _backend_resolve(be, doc_id)
    figs = be.list_figure_chunks(doc_id=doc_id, project_id=project_id, only_with_image=False)
    unusable = [c for c in figs if not c.figure_available]
    if not include_unusable:
        figs = [c for c in figs if c.figure_available]
    if missing_description_only:
        figs = [c for c in figs if not c.figure_description]

    if not figs:
        console.print("[yellow]No figure chunks match.[/]")
        _warn_unusable_figures(unusable, include_unusable)
        return

    table = Table(title=f"Figure chunks ({len(figs)})")
    table.add_column("chunk_id", style="cyan", no_wrap=True)
    table.add_column("page")
    table.add_column("section")
    table.add_column("caption")
    table.add_column("desc?", justify="center")
    table.add_column("source", style="dim")

    for c in figs:
        pages = c.metadata.page_numbers
        page = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}" if pages else ""
        if c.figure_available:
            src = "local" if c.figure_image_path else "s3"
        else:
            src = "[red]missing[/]" if c.figure_image_path else "[red]none[/]"
        table.add_row(
            _short_chunk_id(c.id, c.doc_id),
            page,
            (c.metadata.section_title or "")[:30],
            (c.figure_caption or "")[:40],
            "[green]Y[/]" if c.figure_description else "[red]N[/]",
            src,
        )
    console.print(table)
    _warn_unusable_figures(unusable, include_unusable)


@get_group.command("fig", short_help="Fetch a figure chunk's image.")
@click.argument("chunk_id", type=str)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to save the image. Defaults to a name derived from the "
    "chunk_id in the current directory. If a directory, the default "
    "filename is placed inside it.",
)
@_db_option
def get_figure_cmd(chunk_id: str, output_path: Path | None, db_path: Path | None) -> None:
    """Fetch a figure chunk's image and save it to disk.

    CHUNK_ID accepts the full id or an abbreviated form using a doc_id
    prefix, e.g. ``ab12cd34ef56:L2:143`` as printed by `rag inspect figures` /
    `rag search`. This is the CLI equivalent of the MCP `get_figure` tool.
    """
    from datasheet_rag.backend import RagServerError

    be = _backend_for(db_path)
    full_id = _resolve_chunk_id(be, chunk_id)

    try:
        fig = be.get_figure_bytes(full_id)
    except RagServerError as e:
        raise _friendly_server_error(e) from e
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e

    default_name = _short_chunk_id(fig.chunk_id, fig.doc_id).replace(":", "_") + f".{fig.format}"
    if output_path is None:
        dest = Path(default_name)
    elif output_path.is_dir() or str(output_path).endswith(("/", os.sep)):
        dest = output_path / default_name
    else:
        dest = output_path

    data = fig.image_bytes()
    dest.write_bytes(data)
    console.print(f"[green]Saved[/] {len(data):,} bytes → [cyan]{dest}[/]")

    if fig.caption:
        console.print(f"  caption: {fig.caption}")
    if fig.description:
        console.print(f"  description: {fig.description}")

    citation = fig.citation
    loc_bits = [f"doc={citation.doc_id[:SHORT_DOC_ID_LEN]}"]
    if citation.page:
        loc_bits.append(f"page={citation.page}")
    if citation.section:
        loc_bits.append(f"section={citation.section}")
    console.print(f"  [dim]{' '.join(loc_bits)}[/]")


# ---------------------------------------------------------------------------
# describe-figures (Bedrock Claude vision → figure_description)
# ---------------------------------------------------------------------------


@repair_group.command("figures", short_help="Describe figures with a vision LLM.")
@click.option("--doc-id", default=None, help="Restrict to a single document.")
@click.option("--project-id", default=None, help="Restrict to a single project.")
@click.option(
    "--missing-only/--all", default=True, help="Skip figures that already have a description."
)
@click.option("--limit", default=None, type=int, help="Stop after this many figures (cost guard).")
@click.option(
    "--model", "model_id", default=None, help="Override settings.description_model_id for this run."
)
@click.option(
    "--dry-run", is_flag=True, help="Generate descriptions and print them but do not persist."
)
@_db_option
@click.option("--verbose/--quiet", default=True, help="Print per-figure progress.")
def describe_figures_cmd(
    doc_id: str | None,
    project_id: str | None,
    missing_only: bool,
    limit: int | None,
    model_id: str | None,
    dry_run: bool,
    db_path: Path | None,
    verbose: bool,
) -> None:
    """Generate vision-LLM descriptions for figure chunks and persist them.

    Walks `chunks WHERE layout_type='figure'` (optionally filtered by
    doc/project, skipping those that already have a description), sends
    each image + caption + neighbour text to Bedrock Claude vision, and
    folds the response into the chunk row + context_text.

    Descriptions are folded into the existing chunk rows, so no re-chunk is
    needed — just re-embed the affected document so they show up in vector
    search:

    \b
        rag repair figures --doc-id <doc>
        rag repair embeddings <doc> --project-id <p>
    """
    be = _backend_for(db_path)
    if doc_id:
        doc_id = _backend_resolve(be, doc_id)

    console.print("Describing figures via the backend (vision runs server-side in remote mode)…")
    descriptions, s = be.describe_figures(
        doc_id=doc_id,
        project_id=project_id,
        missing_only=missing_only,
        limit=limit,
        model_id=model_id,
        dry_run=dry_run,
    )

    console.print(
        f"  [green]{len(descriptions)}[/] descriptions · "
        f"in={s.get('total_input_tokens', 0)} tok · "
        f"out={s.get('total_output_tokens', 0)} tok · "
        f"errors={s.get('total_errors', 0)}"
    )

    if dry_run and descriptions:
        console.print("\n[yellow]Dry run — not persisted.[/]\n")
        for chunk_id, desc in descriptions.items():
            console.print(f"[cyan]{chunk_id}[/]")
            console.print(f"  {desc}\n")
    elif descriptions:
        console.print("[green]Descriptions written to chunks + context_text.[/]")
        _next_steps(doc_id, rechunk=False)


# ---------------------------------------------------------------------------
# repair figure-links (reattach cropped images to figure chunks)
# ---------------------------------------------------------------------------


def _relink_plan(
    conn: sqlite3.Connection, doc_id: str
) -> tuple[list[tuple[str, Path, str]], list[str]]:
    """Match a document's cropped figures on disk to its figure chunks.

    Returns ``(plan, notes)``: ``plan`` is ``(chunk_id, image_path, caption)``
    for each source-less figure chunk that a crop can be reattached to, and
    ``notes`` explains every page that was left alone.

    Matching is per page, in reading order: the manifest lists crops in the
    order they were cut from the page and the chunks were inserted in the same
    order, so an equal count on a page pairs them off unambiguously. A page
    whose counts disagree (a crop dropped for being logo-sized, say) is
    skipped rather than guessed at, and so is any pair whose captions are both
    present and different.
    """
    import json as _json

    from datasheet_rag.models.chunk import ChunkLevel, LayoutType
    from datasheet_rag.store import resolve_figure_path

    settings = get_settings()
    orphans = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ? AND layout_type = ? "
        "AND level = ? AND COALESCE(figure_image_path, '') = '' "
        "AND COALESCE(figure_s3_key, '') = ''",
        (doc_id, LayoutType.FIGURE.value, int(ChunkLevel.MICRO)),
    ).fetchone()["n"]
    if not orphans:
        # Coarser levels inherit their image from the MICRO chunk they wrap,
        # which the chunker now does at ingest — a re-chunk carries it up.
        return [], []

    manifest_path = settings.figures_dir / doc_id / "manifest.json"
    if not manifest_path.is_file():
        return [], [f"no manifest at {manifest_path} — re-ingest the PDF to re-crop"]

    manifest = _json.loads(manifest_path.read_text())
    by_page: dict[int, list[dict[str, Any]]] = {}
    for fig in manifest.get("figures", []):
        # Formulas are their own element type and never become figure chunks.
        if "formula" in (fig.get("block_id") or ""):
            continue
        path = resolve_figure_path(fig.get("image_path"))
        if path is None or not path.is_file():
            # A manifest written on another machine: fall back to the crop
            # sitting next to the manifest under this host's figures_dir.
            if fig.get("image_path"):
                local = manifest_path.parent / Path(fig["image_path"]).name
                if local.is_file():
                    path = local
                else:
                    continue
            else:
                continue
        by_page.setdefault(int(fig.get("page") or 1), []).append({**fig, "_path": path})

    chunks_by_page: dict[int, list[Any]] = {}
    for row in conn.execute(
        "SELECT * FROM chunks WHERE doc_id = ? AND layout_type = ? AND level = ? ORDER BY rowid",
        (doc_id, LayoutType.FIGURE.value, int(ChunkLevel.MICRO)),
    ).fetchall():
        pages = json.loads(row["page_numbers"] or "[]")
        chunks_by_page.setdefault(pages[0] if pages else 1, []).append(row)

    plan: list[tuple[str, Path, str]] = []
    notes: list[str] = []
    for page in sorted(set(by_page) | set(chunks_by_page)):
        crops = by_page.get(page, [])
        rows = chunks_by_page.get(page, [])
        # Only pages that actually need repair are worth reporting on.
        if not any(not (r["figure_image_path"] or r["figure_s3_key"]) for r in rows):
            continue
        if len(crops) != len(rows):
            notes.append(
                f"page {page}: {len(crops)} crop(s) vs {len(rows)} figure chunk(s) "
                f"— skipped, cannot pair them unambiguously"
            )
            continue
        for crop, row in zip(crops, rows):
            caption = (crop.get("caption") or "").strip()
            stored_caption = (row["figure_caption"] or "").strip()
            if caption and stored_caption and caption != stored_caption:
                notes.append(
                    f"page {page}: caption mismatch ({stored_caption[:30]!r} vs "
                    f"{caption[:30]!r}) — skipped"
                )
                continue
            if row["figure_image_path"] or row["figure_s3_key"]:
                continue  # already has a source; leave it alone
            plan.append((row["id"], crop["_path"], caption))
    return plan, notes


@repair_group.command("figure-links", short_help="Reattach cropped images to figure chunks.")
@click.option("--doc-id", default=None, help="Restrict to a single document.")
@click.option(
    "--apply", "do_apply", is_flag=True, help="Write the links (default: report what would change)."
)
@_db_option
def relink_figures_cmd(doc_id: str | None, do_apply: bool, db_path: Path | None) -> None:
    """Reattach cropped figure images to figure chunks that lost their link.

    A document ingested with `--skip-figures`, or restored from a backup
    written before figure links existed, leaves chunks that search can find
    and `show_figure` cannot serve. When the crops are still on disk under
    `figures_dir/<doc_id>/` with their manifest, this pairs them back up per
    page in reading order and writes the link.

    Runs against a local store only — it reads the figures directory and
    writes chunk rows directly. On a server deployment, run it there (or
    against the mounted database file with --db).

    \b
        rag repair figure-links                 # report across every document
        rag repair figure-links --doc-id ab12 --apply

    Documents with no crops on disk cannot be repaired this way — re-ingest
    the PDF (without --skip-figures) instead.
    """
    from datasheet_rag.models.chunk import LayoutType
    from datasheet_rag.store import connect, set_figure_source

    path = _require_local_db(db_path)
    conn = connect(path)

    if doc_id:
        doc_id = _resolve_doc_id(conn, doc_id)
        doc_ids = [doc_id]
    else:
        doc_ids = [
            r["doc_id"]
            for r in conn.execute(
                "SELECT DISTINCT doc_id FROM chunks WHERE layout_type = ? "
                "AND COALESCE(figure_image_path, '') = '' "
                "AND COALESCE(figure_s3_key, '') = '' ORDER BY doc_id",
                (LayoutType.FIGURE.value,),
            ).fetchall()
        ]

    if not doc_ids:
        console.print("[green]Every figure chunk already has an image source.[/]")
        return

    total_linked = 0
    quiet = 0
    for did in doc_ids:
        plan, notes = _relink_plan(conn, did)
        head = f"[cyan]{did[:SHORT_DOC_ID_LEN]}[/]"
        if not plan and not notes:
            quiet += 1
            continue
        console.print(f"{head}: {len(plan)} figure chunk(s) can be relinked")
        for note in notes:
            console.print(f"  [yellow]{note}[/]")
        if not do_apply:
            for chunk_id, image_path, caption in plan[:5]:
                console.print(
                    f"  [dim]{_short_chunk_id(chunk_id, did)} → {image_path.name}"
                    + (f" ({caption[:40]})" if caption else "")
                    + "[/]"
                )
            if len(plan) > 5:
                console.print(f"  [dim]… and {len(plan) - 5} more[/]")
            continue
        for chunk_id, image_path, caption in plan:
            if set_figure_source(conn, chunk_id, image_path=image_path, caption=caption):
                total_linked += 1
        conn.commit()

    if quiet:
        console.print(
            f"[dim]{quiet} document(s) only lack images on coarser (MESO) "
            f"figure chunks — re-chunk and re-embed them to carry each "
            f"figure's image up a level.[/]"
        )

    if do_apply:
        console.print(f"[green]Relinked[/] {total_linked} figure chunk(s).")
        if total_linked:
            console.print(
                "  Run [cyan]rag repair figures[/] to describe the newly "
                "visible images, then [cyan]rag repair embeddings[/] to fold "
                "those descriptions into search."
            )
    else:
        console.print("[yellow]Dry run — pass --apply to write the links.[/]")


@repair_group.command("fts", short_help="Rebuild the keyword (BM25) search index.")
@click.option("--check", is_flag=True, help="Report coverage and exit without writing.")
@_db_option
def repair_fts_cmd(check: bool, db_path: Path | None) -> None:
    """Rebuild `chunk_fts`, the FTS5 index behind keyword and hybrid search.

    The index is kept in step with `chunks` by triggers, so it is normally
    correct. When it is not — a store restored from a partial backup, or one
    whose chunks were written before the index existed — the failure is
    silent: `rag search --mode keyword` returns nothing and `--mode hybrid`
    quietly degrades to vector-only, losing the half that matches exact part
    numbers, register names and signal names (GH #23).

    Rebuilding re-derives the whole index from `chunks`. It reads no PDFs,
    calls no LLM, costs nothing, and is safe to run at any time.

    Runs against a local store only — it rewrites an index in the database
    file. On a server deployment, run it there (or against the mounted
    database file with --db).

    \b
        rag repair fts --check      # report coverage, write nothing
        rag repair fts              # rebuild
    """
    from datasheet_rag.store import connect, fts_status, rebuild_fts

    path = _require_local_db(db_path)
    conn = connect(path)
    before = fts_status(conn)

    if before.indexed is None:
        console.print(
            "[yellow]Cannot measure the keyword index[/] on this store — it "
            "was built with a table shape this command does not recognise."
        )
        if check:
            return

    if check:
        if before.healthy:
            console.print(
                f"[green]Keyword index is in sync[/] — "
                f"{before.indexed} of {before.chunks} chunks indexed."
            )
        else:
            console.print(
                f"[yellow]Keyword index is out of sync[/] — "
                f"{before.indexed} of {before.chunks} chunks indexed "
                f"({before.missing} missing). Run [cyan]rag repair fts[/] "
                f"to rebuild."
            )
        return

    console.print(f"Rebuilding the keyword index over {before.chunks} chunk(s)…")
    after = rebuild_fts(conn)
    if after.indexed is None:
        console.print("[green]Rebuilt[/] the keyword index (coverage unmeasurable).")
        return
    console.print(f"[green]Indexed[/] {after.indexed} of {after.chunks} chunks.")
    if before.healthy:
        console.print("  [dim]It was already in sync — nothing was broken.[/]")
    else:
        console.print(
            f"  [dim]{before.missing} chunk(s) were missing before the "
            f"rebuild — keyword and hybrid search can see them now.[/]"
        )


# ---------------------------------------------------------------------------
# Ingest (full pipeline: parse → figures → chunk → embed)
# ---------------------------------------------------------------------------


def _print_cost_table(cost: CostEstimate, heading: str = "Estimated AWS cost") -> None:
    console.rule(f"[bold cyan]{heading}[/]")
    table = Table()
    table.add_column("Item", style="cyan")
    table.add_column("Detail")
    table.add_column("Est. USD", justify="right")
    for item in cost.items:
        table.add_row(item.label, item.detail, f"${item.usd:.5f}")
    if cost.items:
        table.add_row("[bold]Total[/]", "", f"[bold]${cost.total_usd:.5f}[/]")
    console.print(table)
    for note in cost.notes:
        console.print(f"  [yellow]Note:[/] {note}")
    console.print(
        "  [dim]Reference pricing only — hard-coded snapshots (see "
        "datasheet_rag.costs), not live lookups. Verify in your AWS console "
        "before budgeting at volume.[/]"
    )


@cli.command(short_help="Ingest a PDF end to end.")
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--doc-id", default=None, help="Explicit document ID (default: content hash).")
@click.option("--project-id", default=None, help="Project ID attached to all chunks.")
@click.option("--group", "group_name", default=None, help="Group name attached to all chunks.")
@click.option("--mpn", default=None, help="Manufacturer part number, e.g. STM32H743VIT6.")
@click.option("--manufacturer", default=None, help="Manufacturer name, e.g. STMicroelectronics.")
@click.option("--subsystem", default=None, help="e.g. power, rf, mcu.")
@click.option(
    "--doc-type", default=None, help="datasheet | reference-manual | errata | app-note | …"
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Free-text tag for this document, e.g. --tag mcu --tag "
    "reviewed (repeatable). Sets the sidecar's whole tag list "
    "for this ingest. Change it later without a re-ingest via "
    "`rag metadata <doc-id> --tag ...`; for arbitrary "
    "key=value tagging use `rag metadata <doc-id> --attr key=value` "
    "instead.",
)
@click.option(
    "--skip-figures",
    is_flag=True,
    help="Skip figure extraction and description. Figures still "
    "become (caption-only) chunks that `show_figure` cannot "
    "serve — see `rag repair figure-links`.",
)
@click.option(
    "--upload-figures/--no-upload-figures",
    default=False,
    help="Also upload extracted figures to S3. Figures live locally "
    "under ~/.rag/figures/, which is what MCP reads from; "
    "uploading is only useful for sharing a store across machines.",
)
@click.option(
    "--skip-describe", is_flag=True, help="Skip AI figure description (but still extract)."
)
@click.option(
    "--infer-title",
    is_flag=True,
    help="If the document has no usable title after chunking, infer one "
    "with a small Bedrock Claude call against the first page "
    "(one extra LLM call; off by default — see `rag repair titles` "
    "to backfill existing documents).",
)
@click.option("--dpi", default=300, type=int, help="Render DPI for figure extraction.")
@click.option("--micro-tokens", default=128, type=int, help="Max tokens per MICRO chunk.")
@click.option("--meso-tokens", default=512, type=int, help="Max tokens per MESO chunk.")
@_db_option
@click.option("--dry-run", is_flag=True, help="Run all steps but do not write to the store.")
@click.option(
    "--show-cost",
    is_flag=True,
    help=(
        "Estimate AWS costs for this run instead of making the priced Bedrock/"
        "Textract calls — a true dry run for budgeting before ingesting in "
        "volume. Implies --dry-run. Skips embedding and figure-description "
        "API calls (estimated from local chunk/figure counts instead); for "
        "scanned PDFs without cached OCR results, only the Textract line item "
        "can be estimated, since the rest of the pipeline depends on its output."
    ),
)
@click.option(
    "--force", is_flag=True, help="Ignore cached blocks/chunks/rendered pages and redo all steps."
)
@click.option(
    "--local-parse",
    is_flag=True,
    help=(
        "In remote mode, parse the PDF on this client and ship the finished "
        "chunk graph, instead of uploading the raw PDF for the server to parse "
        "(the default). Needs the full Docling/Textract stack locally; useful "
        "for advanced/offline-parse workflows. Ignored in local mode; implied "
        "by --dry-run and --show-cost (both are client-side estimation)."
    ),
)
@click.option(
    "--backend",
    type=click.Choice(["auto", "docling", "textract"], case_sensitive=False),
    default="docling",
    help=(
        "Layout extraction backend. "
        "'docling' handles native PDFs for free and fails verbosely "
        "on scanned PDFs, telling you to pass --backend textract if you want "
        "to pay for AWS OCR. "
        "'auto' detects native vs scanned and silently routes scanned PDFs to "
        "Textract. "
        "'textract' forces AWS OCR for any PDF."
    ),
)
@click.option(
    "--accurate-tables/--fast-tables",
    "accurate_tables",
    is_flag=True,
    default=None,
    help=(
        "Override the global table_structure_mode setting for this ingest "
        "(Docling backend only). FAST is ~2.4x faster; ACCURATE does more "
        "precise cell-boundary detection but — empirically — still doesn't "
        "fully fix complex misparsed/garbled headers (see "
        "table_structure_mode in `rag config` / the README). Defaults to "
        "whatever table_structure_mode is set to in the global config."
    ),
)
def ingest(
    pdf_path: Path,
    doc_id: str | None,
    project_id: str | None,
    group_name: str | None,
    mpn: str | None,
    manufacturer: str | None,
    subsystem: str | None,
    doc_type: str | None,
    tags: tuple[str, ...],
    skip_figures: bool,
    upload_figures: bool,
    skip_describe: bool,
    infer_title: bool,
    dpi: int,
    micro_tokens: int,
    meso_tokens: int,
    db_path: Path | None,
    dry_run: bool,
    show_cost: bool,
    force: bool,
    local_parse: bool,
    backend: str,
    accurate_tables: bool | None,
) -> None:
    """Full ingestion pipeline: parse → figures → chunk → embed.

    Defaults to Docling (free, fast, handles tables/formulas/figures on
    native PDFs) and fails verbosely on scanned PDFs rather than silently
    incurring AWS Textract OCR costs. Pass --backend textract to OCR a
    scanned PDF, or --backend auto to route automatically between the two.
    Intermediate artefacts (blocks/chunk graph) are cached in output_dir;
    pass --force to ignore the cache and redo all steps.

    PDF_PATH may also be a directory, in which case every *.pdf found
    under it (recursively) is ingested one by one with the same options.
    Documents already in the store under their content-hash doc_id are
    skipped unless --force is given, and --show-cost prints a combined
    estimate across all of them in addition to each document's breakdown.
    """
    common: dict[str, Any] = dict(
        project_id=project_id,
        group_name=group_name,
        mpn=mpn,
        manufacturer=manufacturer,
        subsystem=subsystem,
        doc_type=doc_type,
        tags=tags,
        skip_figures=skip_figures,
        upload_figures=upload_figures,
        skip_describe=skip_describe,
        infer_title=infer_title,
        dpi=dpi,
        micro_tokens=micro_tokens,
        meso_tokens=meso_tokens,
        db_path=db_path,
        dry_run=dry_run,
        show_cost=show_cost,
        force=force,
        local_parse=local_parse,
        backend=backend,
        accurate_tables=accurate_tables,
    )

    if not pdf_path.is_dir():
        _ingest_one(pdf_path, doc_id=doc_id, **common)
        return

    if doc_id:
        raise click.ClickException(
            "--doc-id can't be used with a directory — each PDF gets its own "
            "content-hash doc_id. Drop --doc-id for bulk ingestion."
        )

    from datasheet_rag.costs import CostEstimate, CostLineItem
    from datasheet_rag.docling_parser import content_hash

    pdf_files = sorted(p for p in pdf_path.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdf_files:
        raise click.ClickException(f"No PDFs found under {pdf_path}")

    # Skip docs already in the store (local or remote, via the backend).
    ingested_ids: set[str] = set()
    if not show_cost:
        try:
            ingested_ids = {d.doc_id for d in _backend_for(db_path).get_ingested_docs()}
        except Exception:
            ingested_ids = set()

    console.rule(f"[bold magenta]Bulk ingest — {len(pdf_files)} PDFs under {pdf_path}[/]")
    total_cost = CostEstimate()
    skipped = 0
    failed: list[tuple[Path, str]] = []
    for i, pdf in enumerate(pdf_files, 1):
        console.rule(f"[bold cyan]({i}/{len(pdf_files)}) {pdf.relative_to(pdf_path)}[/]")
        did = content_hash(pdf)
        if did in ingested_ids and not force:
            console.print(
                f"  [dim]Already ingested and up to date (doc_id={did}) — "
                "skipping. Pass --force to re-ingest.[/]"
            )
            skipped += 1
            continue
        try:
            result = _ingest_one(pdf, doc_id=None, **common)
        except click.ClickException as e:
            console.print(f"  [red]Failed:[/] {e.format_message()}")
            failed.append((pdf, e.format_message()))
            continue
        if result is not None:
            total_cost.items.extend(result.items)
            total_cost.notes.extend(result.notes)

    if show_cost:
        by_label: dict[str, list[CostLineItem]] = {}
        for item in total_cost.items:
            by_label.setdefault(item.label, []).append(item)
        merged = CostEstimate(notes=total_cost.notes)
        for label, line_items in by_label.items():
            merged.items.append(
                CostLineItem(
                    label=label,
                    detail=f"summed across {len(line_items)} documents",
                    usd=sum(li.usd for li in line_items),
                )
            )
        _print_cost_table(
            merged,
            heading=(
                f"Estimated AWS cost — combined across "
                f"{len(pdf_files) - skipped} of {len(pdf_files)} documents"
            ),
        )
    processed = len(pdf_files) - skipped - len(failed)
    summary_color = "yellow" if failed else "green"
    console.rule(
        f"[bold {summary_color}]Bulk ingest done[/] — {len(pdf_files)} PDFs, "
        f"{processed} processed, {skipped} skipped, {len(failed)} failed"
    )
    if failed:
        console.print("[bold yellow]Failed documents:[/]")
        for pdf, msg in failed:
            console.print(f"  [red]{pdf.relative_to(pdf_path)}[/]: {msg.splitlines()[0]}")


def _ingest_one(
    pdf_path: Path,
    *,
    doc_id: str | None,
    project_id: str | None,
    group_name: str | None,
    mpn: str | None,
    manufacturer: str | None,
    subsystem: str | None,
    doc_type: str | None,
    tags: tuple[str, ...],
    skip_figures: bool,
    upload_figures: bool,
    skip_describe: bool,
    infer_title: bool,
    dpi: int,
    micro_tokens: int,
    meso_tokens: int,
    db_path: Path | None,
    dry_run: bool,
    show_cost: bool,
    force: bool,
    local_parse: bool,
    backend: str,
    accurate_tables: bool | None,
) -> CostEstimate | None:
    """Ingest a single PDF; returns the cost estimate when --show-cost is set."""
    import time

    from datasheet_rag.costs import (
        CostEstimate,
        estimate_embedding_cost,
        estimate_figure_description_cost,
        estimate_textract_cost,
        estimate_title_inference_cost,
    )
    from datasheet_rag.project_config import get_project_config_for

    proj_cfg = get_project_config_for(pdf_path.parent)
    if proj_cfg is not None:
        project_id = project_id or proj_cfg.project_id
        group_name = group_name or proj_cfg.group
        mpn = mpn or proj_cfg.mpn
        manufacturer = manufacturer or proj_cfg.manufacturer
        subsystem = subsystem or proj_cfg.subsystem
        if not tags and proj_cfg.tags:
            tags = tuple(proj_cfg.tags)

    if show_cost:
        dry_run = True
        console.print(
            "[yellow]--show-cost: estimating AWS spend, skipping priced Bedrock/Textract calls.[/]"
        )

    settings = get_settings()
    if accurate_tables is None:
        accurate_tables = settings.table_structure_mode == "accurate"
    t0 = time.monotonic()
    cost = CostEstimate()

    from datasheet_rag.backend import backend_mode
    from datasheet_rag.ingest_pipeline import (
        OcrRequiredError,
        ScannedPdfError,
        parse_pdf_to_graph,
    )

    # Render pipeline progress to the console — used both for a local parse and
    # for the streamed progress of a remote raw-PDF upload.
    step_state = {"n": 0}

    def _progress(ev: ProgressEvent) -> None:
        if ev.kind == "step":
            step_state["n"] = ev.step
            console.rule(f"[bold cyan]Step {ev.step} — {ev.text}[/]")
        else:
            console.print(f"  {ev.text}")

    # ── Remote raw-PDF upload (thin client — GH #16) ─────────────────────────
    # In remote mode the server runs the whole pipeline; the client just ships
    # the PDF. --local-parse (and --dry-run/--show-cost, which are inherently
    # client-side estimation) fall through to the local parse path below.
    if (
        backend_mode() == "remote"
        and db_path is None
        and not local_parse
        and not dry_run
        and not show_cost
    ):
        from datasheet_rag.backend import MetadataPatch, RagServerError, get_backend

        meta_patch = MetadataPatch(
            mpn=mpn or None,
            manufacturer=manufacturer or None,
            subsystem=subsystem or None,
            doc_type=doc_type or None,
            tags=list(tags) if tags else None,
        )
        try:
            result = get_backend().ingest_pdf(
                pdf_path,
                doc_id=doc_id,
                project_id=project_id,
                group_name=group_name,
                metadata=meta_patch,
                backend=backend,
                skip_figures=skip_figures,
                upload_figures=upload_figures,
                skip_describe=skip_describe,
                infer_title=infer_title,
                dpi=dpi,
                micro_tokens=micro_tokens,
                meso_tokens=meso_tokens,
                accurate_tables=accurate_tables,
                force=force,
                progress=_progress,
            )
        except RagServerError as e:
            raise _friendly_server_error(e) from e
        console.print(f"  [green]Upserted[/] {result.inserted} chunks")
        _report_pruned(result.pruned)
        if result.described:
            console.print(f"  [green]{result.described}[/] figure descriptions generated")
        if result.title:
            console.print(f"  [green]Inferred title:[/] {result.title}")
        elapsed = time.monotonic() - t0
        console.rule(f"[bold green]Done[/] — {elapsed:.0f}s")
        console.print(f"  doc_id = [cyan]{result.doc_id}[/]")
        return None

    # ── Local parse (local mode, --local-parse, --dry-run, --show-cost) ──────
    try:
        parsed = parse_pdf_to_graph(
            pdf_path,
            doc_id=doc_id,
            backend=backend,
            skip_figures=skip_figures,
            upload_figures=upload_figures,
            dpi=dpi,
            micro_tokens=micro_tokens,
            meso_tokens=meso_tokens,
            accurate_tables=accurate_tables,
            force=force,
            allow_ocr=not show_cost,
            progress=_progress,
        )
    except ScannedPdfError as e:
        raise click.ClickException(str(e)) from e
    except OcrRequiredError as e:
        # Cost estimation on a scanned PDF with no cached OCR: only the OCR
        # line item can be priced (the rest depends on its output).
        console.print(f"  doc_id = [cyan]{e.doc_id}[/]")
        cost.items.append(estimate_textract_cost(e.pages))
        cost.notes.append(
            "No cached Textract OCR results for this PDF — only the "
            "OCR line item could be estimated. Embedding/description "
            "costs depend on its output; run a real ingest once (or "
            "ingest with cached blocks present) to estimate the full "
            "pipeline without re-paying for OCR on every estimate."
        )
        _print_cost_table(cost)
        return cost

    did = parsed.doc_id
    graph = parsed.graph
    figure_count = parsed.figure_count

    # ── Describe, embed & store (via the backend) ───────────────────────────
    # Parsing/chunking/figure-cropping above ran client-side. Embedding,
    # figure description (vision LLM) and the DB write all happen through the
    # backend — locally against sqlite, or server-side over HTTP in remote
    # mode (where the embedder/vision models live).
    do_describe = not skip_figures and not skip_describe

    if show_cost:
        if do_describe:
            if figure_count == 0:
                figure_count = sum(1 for c in graph.chunks.values() if c.figure_image_path)
            cost.items.append(estimate_figure_description_cost(figure_count))
        embed_item = estimate_embedding_cost(graph)
        cost.items.append(embed_item)
        console.print(f"  [yellow]Estimating only — {embed_item.detail}, no Bedrock calls made.[/]")
        if infer_title:
            cost.items.append(estimate_title_inference_cost())
        _print_cost_table(cost)
        return cost

    if dry_run:
        console.print("[yellow]Dry run — not writing to the store.[/]")
        elapsed = time.monotonic() - t0
        console.rule(f"[bold green]Done (dry run)[/] — {elapsed:.0f}s")
        console.print(f"  doc_id = [cyan]{did}[/]")
        return None

    step_state["n"] += 1
    console.rule(f"[bold cyan]Step {step_state['n']} — Embed & store[/]")
    from datasheet_rag.backend import MetadataPatch, get_backend

    # `rag ingest --db` targets a specific local file; honor it by building a
    # LocalBackend on that path rather than the configured backend.
    if db_path is not None:
        from datasheet_rag.backend import LocalBackend

        backend_obj: RagBackend = LocalBackend(db_path)
    else:
        backend_obj = get_backend()

    # In remote mode the cropped figure images live only on this client —
    # ship their bytes so the server stores them and rewrites the host-local
    # figure_image_path before inserting.
    figures_upload: dict[str, tuple[bytes, str]] | None = None
    if not skip_figures and backend_mode() == "remote" and db_path is None:
        figures_upload, missing = _collect_figure_uploads(graph)
        _warn_missing_figure_crops(missing, len(figures_upload))

    meta_patch = MetadataPatch(
        mpn=mpn or None,
        manufacturer=manufacturer or None,
        subsystem=subsystem or None,
        doc_type=doc_type or None,
        tags=list(tags) if tags else None,
    )
    title_hints = dict(parsed.title_hints)

    from datasheet_rag.backend import RagServerError

    try:
        result = backend_obj.ingest_chunk_graph(
            graph,
            figures=figures_upload,
            project_id=project_id,
            group_name=group_name,
            metadata=meta_patch,
            embed=True,
            describe_figures=do_describe,
            infer_title=infer_title,
            title_hints=title_hints or None,
        )
    except RagServerError as e:
        raise _friendly_server_error(e) from e
    console.print(f"  [green]Upserted[/] {result.inserted} chunks")
    _report_pruned(result.pruned)
    if result.described:
        console.print(f"  [green]{result.described}[/] figure descriptions generated")
    if result.title:
        console.print(f"  [green]Inferred title:[/] {result.title}")

    elapsed = time.monotonic() - t0
    console.rule(f"[bold green]Done[/] — {elapsed:.0f}s")
    console.print(f"  doc_id = [cyan]{did}[/]")
    return None


def _report_pruned(pruned: int, indent: str = "  ") -> None:
    """Say so when an ingest deleted stale chunks from a previous graph.

    Re-chunking shifts positional ids, so a document that chunks differently
    leaves rows behind that the upsert never touches; ingest drops them (GH
    #44). The user did not ask for a delete, so it is never silent.
    """
    if pruned:
        console.print(
            f"{indent}[yellow]Pruned[/] {pruned} stale chunk(s) left by the "
            "previous version of this document"
        )


def _parse_page_range(spec: str) -> tuple[int, int]:
    """Parse '36' or '36-40' into a 1-based inclusive (start, end) pair."""
    spec = spec.strip()
    if "-" in spec:
        start_s, _, end_s = spec.partition("-")
        start, end = int(start_s), int(end_s)
    else:
        start = end = int(spec)
    if start < 1 or end < start:
        raise click.BadParameter(f"invalid page range {spec!r} (expected e.g. '36' or '36-40')")
    return start, end


@repair_group.command("reconvert", short_help="Re-run Docling table recognition (Docling only).")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--pages",
    "pages_spec",
    required=True,
    help="1-based inclusive page range to re-run, e.g. '36' or '36-40'.",
)
@click.option("--doc-id", default=None, help="Override the content-hash doc_id.")
@click.option(
    "--accurate-tables/--fast-tables",
    default=True,
    help="Table mode for the re-run (accurate is the point of this command).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would change without patching the cached outline or invalidating chunks.",
)
def reconvert_tables_cmd(
    pdf_path: Path,
    pages_spec: str,
    doc_id: str | None,
    accurate_tables: bool,
    dry_run: bool,
) -> None:
    """Selectively re-run table-structure recognition for a page range.

    Re-running Docling layout analysis on an entire multi-thousand-page PDF
    just to fix one or two misparsed tables (see the "Table parsing warning"
    printouts during `rag ingest`) is wasteful — this instead converts only
    the given --pages with the chosen TableFormer mode, geometrically matches
    the resulting tables to their counterparts already in the cached layout
    outline (by page number + bounding-box overlap — tables are leaf elements
    with a clear geometric identity, unlike sections/headings which can't be
    safely tree-merged across two independent conversions), and patches just
    those table cells/text in place.

    Requires the document to already be ingested with the Docling backend
    (so a cached `<doc_id>_outline.json` exists) — Textract exposes no
    equivalent re-run. On success it deletes the cached chunk graph; follow
    with `rag repair chunks <doc-id>` and `rag repair embeddings <doc-id>` to
    re-derive from the patched outline. Docling layout analysis itself is NOT
    re-run for the rest of the document.
    """
    from datasheet_rag.chunking.layout_parser import DocumentOutline
    from datasheet_rag.docling_parser import content_hash, reconvert_tables_in_range

    page_range = _parse_page_range(pages_spec)
    settings = get_settings()
    did = doc_id or content_hash(pdf_path)
    outline_path = settings.output_dir / f"{did}_outline.json"
    chunks_path = settings.output_dir / f"{did}_chunks.json"

    if not outline_path.exists():
        raise click.ClickException(
            f"No cached layout outline for doc_id={did} at {outline_path}. "
            "`repair reconvert` is Docling-only and patches an existing "
            "outline — it does not do a first-time conversion. Run "
            "`rag ingest --backend docling` on this PDF first."
        )

    with open(outline_path) as f:
        cached = json.load(f)
    outline = DocumentOutline.from_dict(cached["outline"])

    mode_label = "accurate" if accurate_tables else "fast"
    console.print(
        f"Re-running pages {page_range[0]}-{page_range[1]} of {pdf_path.name} "
        f"with TableFormer [cyan]{mode_label}[/] mode…"
    )
    report = reconvert_tables_in_range(
        pdf_path, outline, doc_id=did, page_range=page_range, accurate_tables=accurate_tables
    )

    if not report:
        console.print(
            f"[yellow]No cached tables found on pages "
            f"{page_range[0]}-{page_range[1]}.[/] Nothing to do."
        )
        return

    table = Table(title=f"Tables on pages {page_range[0]}-{page_range[1]}")
    table.add_column("Page", justify="right")
    table.add_column("Caption")
    table.add_column("Matched", justify="center")
    table.add_column("Chars (old → new)", justify="right")
    table.add_column("Garbled (old → new)", justify="center")
    for entry in report:
        caption = entry["caption"][:50] or "—"
        if entry["matched"]:
            chars = f"{entry['old_chars']:,} → {entry['new_chars']:,}"
            garbled = (
                f"{'yes' if entry['old_garbled'] else 'no'} → "
                f"{'yes' if entry['new_garbled'] else 'no'}"
            )
            matched = f"yes ({entry['overlap']:.0%} overlap)"
        else:
            chars = f"{entry['old_chars']:,} → —"
            garbled = f"{'yes' if entry['old_garbled'] else 'no'} → —"
            matched = "[red]no[/]"
        table.add_row(str(entry["page"]), caption, matched, chars, garbled)
    console.print(table)

    unmatched = [e for e in report if not e["matched"]]
    if unmatched:
        console.print(
            f"[yellow]{len(unmatched)} cached table(s) had no geometric "
            f"counterpart in the re-run output[/] — left untouched (could "
            f"mean the table moved across the page boundary, or Docling "
            f"placed it on a neighbouring page in this mode; widen --pages "
            f"and retry if that looks wrong)."
        )

    matched_entries = [e for e in report if e["matched"]]
    fixed_garbled = [e for e in matched_entries if e["old_garbled"] and not e["new_garbled"]]
    still_garbled = [e for e in matched_entries if e["new_garbled"]]
    if fixed_garbled:
        console.print(f"[green]{len(fixed_garbled)} garbled header(s) fixed by this re-run.[/]")
    if still_garbled:
        console.print(
            f"[yellow]{len(still_garbled)} table(s) still have a garbled "
            f"header after the re-run[/] — _detect_garbled_header will keep "
            f"dropping them from the embedded text regardless."
        )

    if dry_run:
        console.print("[yellow]--dry-run: outline and chunk cache left untouched.[/]")
        return

    if not matched:
        console.print("[yellow]Nothing matched — leaving the cached outline untouched.[/]")
        return

    with open(outline_path, "w") as f:
        json.dump({"outline": outline.to_dict(), "figure_regions": cached["figure_regions"]}, f)
    console.print(f"[green]Patched outline saved[/] → [cyan]{outline_path}[/]")

    if chunks_path.exists():
        chunks_path.unlink()
        console.print(f"[green]Invalidated cached chunk graph[/] → removed [cyan]{chunks_path}[/]")
    _next_steps(did, rechunk=True)


@inspect_group.command("tables", short_help="Report tables that parsed badly.")
@click.argument("doc_id", type=str)
@click.option(
    "--list-flagged",
    is_flag=True,
    help="Print every flagged table (page, caption, reason) instead of just the summary counts.",
)
@click.option(
    "--sample",
    "sample_n",
    type=int,
    default=0,
    help=(
        "Print the rendered text of N randomly-sampled flagged tables AND N "
        "randomly-sampled non-flagged tables, for manual eyeballing — a "
        "zero-cost spot-check of detector accuracy (false positives among "
        "the flagged sample, false negatives among the non-flagged one) "
        "before spending on `rag repair tables`. Sampling uses a fixed seed "
        "so repeated runs show the same tables."
    ),
)
def table_structure_sweep_cmd(doc_id: str, list_flagged: bool, sample_n: int) -> None:
    """Report how many of a document's tables parsed with untrustworthy structure.

    Sweeps every table in the cached layout outline and counts the ones whose
    header band looks wrong, in two independently-detected ways: a header
    whose text is repeated across its cells, and a header the parser fused
    with the first row of data. Both produce a table that reads as clean but
    asserts the wrong column meanings — the failure worth catching before it
    reaches search results.

    Costs nothing: pure Python over `<doc_id>_outline.json`, no Docling re-run
    and no AWS calls. Run it before `rag repair tables`, which does spend
    money, to see how much there is to fix and to spot-check (with --sample)
    whether the flags are accurate on this document.

    DOC_ID is a full hash or unambiguous prefix, looked up in the layout cache
    rather than the ingest registry — like `repair reconvert`, this works on a
    document whose layout was converted but whose ingest never completed.
    Docling-only; see `rag repair tables` for what to do about what it finds.
    """
    from datasheet_rag.chunking.layout_parser import ContentElement, DocumentOutline, ElementType
    from datasheet_rag.docling_parser import (
        _detect_fused_header_row,
        _detect_garbled_header,
        _table_cells_to_compact_text,
    )

    did = _require_docling_outline(doc_id)
    outline_path = _cache_path(did, "_outline.json")

    with open(outline_path) as f:
        cached = json.load(f)
    outline = DocumentOutline.from_dict(cached["outline"])

    tables = [
        el
        for section in outline.all_sections_flat
        for el in section.elements
        if el.element_type == ElementType.TABLE
    ]
    if not tables:
        console.print(f"[yellow]No cached tables found for doc_id={did}.[/]")
        return

    flagged: list[_FlaggedTable] = []
    # Parallel (element, reason | None) record — feeds --sample below without
    # re-running the detectors over the whole corpus a second time.
    judged: list[tuple[ContentElement, str | None]] = []
    garbled_count = fused_count = both_count = 0
    for el in tables:
        garbled = _detect_garbled_header(el.table_cells)
        fused = _detect_fused_header_row(el.table_cells) if garbled is None else None
        if garbled is not None:
            garbled_count += 1
            if _detect_fused_header_row(el.table_cells) is not None:
                both_count += 1
        if fused is not None:
            fused_count += 1
        reason: str | None = None
        if garbled is not None:
            reason = f"garbled ({garbled[:40]!r}…)"
        elif fused is not None:
            reason = f"fused ({fused})"
        if reason is not None:
            flagged.append(_FlaggedTable(el.page, el.table_title, reason))
        judged.append((el, reason))

    total = len(tables)
    flagged_count = len(flagged)
    summary = Table(
        title=f"Table-structure sweep — {did[:SHORT_DOC_ID_LEN]} ({total:,} cached tables)"
    )
    summary.add_column("Detector")
    summary.add_column("Flagged", justify="right")
    summary.add_column("% of corpus", justify="right")

    def _row(label: str, count: int) -> None:
        summary.add_row(label, f"{count:,}", f"{count / total:.1%}")

    _row("Repeated text across header cells", garbled_count)
    _row("Data row fused into the header", fused_count)
    _row("caught by both", both_count)
    _row("[bold]flagged untrustworthy (either)[/]", flagged_count)
    console.print(summary)

    if list_flagged and flagged:
        detail = Table(title="Flagged tables")
        detail.add_column("Page", justify="right")
        detail.add_column("Caption")
        detail.add_column("Reason")
        for entry in flagged:
            detail.add_row(str(entry.page), entry.caption[:60] or "—", entry.reason)
        console.print(detail)
    elif flagged:
        console.print(
            "Pass [cyan]--list-flagged[/] to see page numbers and captions for each flagged table."
        )

    if sample_n > 0:
        flagged_els = [(el, reason) for el, reason in judged if reason is not None]
        clean_els = [(el, reason) for el, reason in judged if reason is None]
        rng = random.Random(0)  # fixed seed: repeated runs sample the same tables
        flagged_sample = rng.sample(flagged_els, min(sample_n, len(flagged_els)))
        clean_sample = rng.sample(clean_els, min(sample_n, len(clean_els)))

        console.print(
            "\n[bold]Spot-check[/] — eyeball whether the detector's calls look "
            "right before spending on `rag repair tables` (zero AWS cost; "
            "renders straight from the cached outline). For "
            "[yellow]flagged[/] tables, does the asserted structure below "
            "actually look broken? For [green]not flagged[/] tables, does it "
            "look correct (no missed fusion/garbling)?"
        )
        for label, sample in (("FLAGGED", flagged_sample), ("NOT FLAGGED", clean_sample)):
            for el, reason in sample:
                where = el.table_title or f"page {el.page}"
                header = f"[bold]{label}[/] — {where} (p.{el.page})"
                if reason is not None:
                    header += f" — {reason}"
                console.print(f"\n{header}")
                rendered = _table_cells_to_compact_text(el.table_cells)
                console.print(rendered[:1500] + ("…" if len(rendered) > 1500 else ""))


# Sanity cap for repair-tables: if Docling's is_header tagging is implausibly
# large, it's too unreliable to use as a header-band crop boundary — skip
# repair and keep the existing Stage-2 reading-order fallback.
_MAX_HEADER_ROWS = 6
_MAX_HEADER_FRACTION = 0.3


@repair_group.command("tables", short_help="LLM-repair tables flagged untrustworthy.")
@click.argument("doc_id", type=str)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Repair at most N flagged tables (omit to repair all of them).",
)
@click.option(
    "--model-id",
    default=None,
    help=(
        "Override the Bedrock model ID for this run. Defaults to "
        "table_repair_model_id, falling back to description_model_id "
        "(Haiku — cheap, and validation rejects anything structurally "
        "inconsistent, so a weaker model fails safe rather than silently)."
    ),
)
@click.option("--dpi", type=int, default=200, help="Render DPI for table crops.")
@click.option(
    "--force",
    is_flag=True,
    help="Re-repair tables that already have a cached table_repaired_cells.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="List the tables that would be repaired without calling Bedrock or touching the cache.",
)
@click.option("-v", "--verbose", is_flag=True, help="Print rejection reasons as they happen.")
def repair_tables_cmd(
    doc_id: str,
    limit: int | None,
    model_id: str | None,
    dpi: int,
    force: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    """LLM-assisted repair of tables flagged untrustworthy (Stage 3).

    For each cached TABLE that ``table_structure_untrustworthy`` flags (and
    doesn't already carry a cached repair, unless --force): renders the
    table's source page, crops to its header band (plus one data row of
    visual context), and asks a vision-capable Claude on Bedrock to
    re-transcribe the header band as an H×C grid — ``H`` header rows (from
    Docling's ``is_header`` tagging) by ``C`` columns (from the table's
    trusted data-row grid). The proposal is validated against threshold-free
    structural invariants (the H×C band is tiled exactly once, the proposed
    header isn't itself garbled or a recreation of a fused data row — see
    table_repair.validate_header_grid). A validated proposal replaces the
    header band wholesale (data rows are never touched), is cached as
    ``table_repaired_cells``, the chunk text is re-rendered through the same
    trusted compact-grid path a correctly-parsed table gets, and
    ``table_structure_warning`` is cleared. An unparseable or structurally-
    inconsistent response is rejected outright, as is a table whose header
    band is implausibly large to crop reliably — repair is additive, never a
    regression; the existing structure-free rendering is kept.

    Like `repair reconvert`, this patches the cached `<doc_id>_outline.json`
    in place and invalidates the cached chunk graph; follow with
    `rag repair chunks <doc_id>` and `rag repair embeddings <doc_id>` to
    re-derive from the repaired structure. Run `rag inspect tables <doc_id>`
    first to see what would be touched and at roughly what volume.
    """
    import io

    from pdf2image import convert_from_bytes

    from datasheet_rag.chunking.layout_parser import ContentElement, DocumentOutline, ElementType
    from datasheet_rag.docling_parser import (
        _table_column_count,
        _table_header_row_count,
        table_structure_untrustworthy,
    )
    from datasheet_rag.pdf_render import load_pdf_bytes
    from datasheet_rag.table_repair import (
        TableRepairer,
        apply_repaired_structure,
        splice_header_band,
    )

    def _skip_reason(cells: list[dict[str, Any]]) -> str | None:
        """Why header-band repair can't safely run on this table, or None."""
        header_rows = _table_header_row_count(cells)
        total_rows = max((c["row"] for c in cells), default=0)
        if header_rows == 0:
            return "no header rows tagged"
        if header_rows > _MAX_HEADER_ROWS or (
            total_rows and header_rows > _MAX_HEADER_FRACTION * total_rows
        ):
            return "header band too large to crop reliably"
        if _table_column_count(cells) == 0:
            return "data rows disagree on column count — can't trust C"
        return None

    did = _require_docling_outline(doc_id)
    outline_path = _cache_path(did, "_outline.json")
    chunks_path = _cache_path(did, "_chunks.json")

    with open(outline_path) as f:
        cached = json.load(f)
    outline = DocumentOutline.from_dict(cached["outline"])

    tables: list[ContentElement] = [
        el
        for section in outline.all_sections_flat
        for el in section.elements
        if el.element_type == ElementType.TABLE
    ]

    candidates: list[tuple[ContentElement, str]] = []
    for el in tables:
        if el.table_repaired_cells is not None and not force:
            continue
        reason = table_structure_untrustworthy(el.table_cells)
        if reason is not None:
            candidates.append((el, reason))

    if limit is not None:
        candidates = candidates[:limit]

    if not candidates:
        hint = "pass --force to re-repair cached ones" if not force else "all are already repaired"
        console.print(
            f"[green]Nothing to repair[/] for doc_id={did[:SHORT_DOC_ID_LEN]} — "
            f"no untrustworthy tables without a cached repair ({hint}). "
            "Run `rag inspect tables` to confirm the flagged count."
        )
        return

    repairer = TableRepairer(model_id=model_id, verbose=verbose)
    console.print(
        f"Repairing up to [cyan]{len(candidates)}[/] flagged table(s) of "
        f"{len(tables):,} via header-band re-transcription with "
        f"[cyan]{repairer.model_id}[/]…"
    )
    if dry_run:
        for el, reason in candidates:
            skip = _skip_reason(el.table_cells)
            tag = f"skip — {skip}" if skip is not None else "LLM (header band)"
            console.print(
                f"  [yellow]would repair ({tag})[/] p.{el.page} "
                f"{(el.table_title or '(untitled)')[:60]} — {reason}"
            )
        console.print("[yellow]--dry-run: no Bedrock calls made, cache untouched.[/]")
        return

    by_page: dict[int, list[tuple[ContentElement, str]]] = {}
    for el, reason in candidates:
        by_page.setdefault(el.page, []).append((el, reason))

    pdf_bytes = load_pdf_bytes(did)

    repaired_count = 0
    llm_count = 0
    for page in sorted(by_page):
        page_image = None  # lazy: render only if a table on this page needs repair
        for el, reason in by_page[page]:
            where = (el.table_title or "(untitled)")[:60]
            cells = el.table_cells

            skip = _skip_reason(cells)
            if skip is not None:
                console.print(f"  [yellow]skipped ({skip})[/] p.{el.page} {where} — {reason}")
                continue

            header_rows = _table_header_row_count(cells)
            total_rows = max((c["row"] for c in cells), default=0)
            column_count = _table_column_count(cells)

            # Crop: header band + one data row for visual alignment context
            crop_end_row = min(header_rows + 1, total_rows) if total_rows else header_rows

            if page_image is None:
                page_images = convert_from_bytes(
                    pdf_bytes, first_page=page, last_page=page, dpi=dpi
                )
                if not page_images:
                    console.print(
                        f"[yellow]Could not render page {page}[/] — skipping its "
                        f"{len(by_page[page])} flagged table(s)."
                    )
                    break
                page_image = page_images[0]

            crop = repairer.crop_table(
                page_image,
                bbox=el.bbox,
                row_range=(1, crop_end_row),
                total_rows=total_rows,
            )
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            llm_count += 1

            proposed = repairer.repair_header_band(
                image_bytes=buf.getvalue(),
                image_format="png",
                cells=cells,
                caption=el.table_title,
                header_rows=header_rows,
                column_count=column_count,
            )
            if proposed is None:
                console.print(f"  [yellow]rejected[/] p.{el.page} {where} — {reason}")
                continue

            merged = splice_header_band(
                cells, proposed, header_rows=header_rows, column_count=column_count
            )
            apply_repaired_structure(el, merged)
            repaired_count += 1
            console.print(f"  [green]repaired (LLM)[/] p.{el.page} {where} — was {reason}")

    s = repairer.stats()
    console.print(
        f"\n[bold]{repaired_count}/{len(candidates)}[/] repaired "
        f"(LLM calls: {llm_count}) · "
        f"rejected={s['total_rejected']} · errors={s['total_errors']} · "
        f"in={s['total_input_tokens']:,} tok · out={s['total_output_tokens']:,} tok"
    )

    if repaired_count == 0:
        console.print("[yellow]Nothing repaired — leaving the cached outline untouched.[/]")
        return

    with open(outline_path, "w") as f:
        json.dump({"outline": outline.to_dict(), "figure_regions": cached["figure_regions"]}, f)
    console.print(f"[green]Patched outline saved[/] → [cyan]{outline_path}[/]")

    if chunks_path.exists():
        chunks_path.unlink()
        console.print(f"[green]Invalidated cached chunk graph[/] → removed [cyan]{chunks_path}[/]")
    _next_steps(did, rechunk=True)


# ---------------------------------------------------------------------------
# Document metadata (sidecar — separate from chunks, no re-ingest required)
# ---------------------------------------------------------------------------


@cli.command("metadata", short_help="Show or set doc-level metadata.")
@click.argument("doc_id", type=str)
@click.option(
    "--title",
    "doc_title",
    default=None,
    help="Override doc_title on every chunk row. Recorded as a manual "
    "title, so re-ingesting the document won't overwrite it.",
)
@click.option("--project-id", default=None, help="Set the document's project ID.")
@click.option("--group", "group_name", default=None, help="Set the document's group name.")
@click.option(
    "--mpn",
    default=None,
    help="Manufacturer part number, e.g. STM32H743VIT6. Replaces the current value.",
)
@click.option(
    "--mpn-alias",
    "mpn_aliases",
    multiple=True,
    help="Add an MPN alias without replacing existing ones (repeatable).",
)
@click.option(
    "--manufacturer",
    default=None,
    help="Manufacturer name, e.g. STMicroelectronics. Replaces the current value.",
)
@click.option("--subsystem", default=None, help="e.g. power, rf, mcu.")
@click.option(
    "--doc-type", default=None, help="datasheet | reference-manual | errata | app-note | …"
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Repeatable — e.g. --tag mcu --tag reviewed. REPLACES the "
    "document's whole tag list wholesale (not additive): the "
    "set of --tag flags you pass *becomes* the full list. "
    "Omit --tag to leave existing tags untouched; pass "
    "--clear-tags to wipe them without setting new ones.",
)
@click.option("--clear-tags", is_flag=True, help="Remove all tags. Ignored if --tag is also given.")
@click.option(
    "--attr",
    "attrs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Arbitrary key=value tag, repeatable, e.g. --attr "
    "revision=B --attr reviewed_by=hector. Unlike --tag, "
    "attributes are merged key-by-key: existing keys not "
    "mentioned are left alone. Use --unset-attr to remove one.",
)
@click.option(
    "--unset-attr",
    "unset_attrs",
    multiple=True,
    metavar="KEY",
    help="Remove a single attribute key (repeatable).",
)
@_db_option
@click.option(
    "--apply-to-chunks/--no-apply-to-chunks",
    default=True,
    help="Propagate project_id and group_name into the chunks table.",
)
def metadata_cmd(
    doc_id: str,
    doc_title: str | None,
    project_id: str | None,
    group_name: str | None,
    mpn: str | None,
    mpn_aliases: tuple[str, ...],
    manufacturer: str | None,
    subsystem: str | None,
    doc_type: str | None,
    tags: tuple[str, ...],
    clear_tags: bool,
    attrs: tuple[str, ...],
    unset_attrs: tuple[str, ...],
    db_path: Path | None,
    apply_to_chunks: bool,
) -> None:
    """Show a document's metadata sidecar row, or update it.

    With no options this prints the document's current metadata. Pass any
    field to update instead — only the fields you pass are touched:

    \b
        rag metadata <doc_id>                 # show
        rag metadata <doc_id> --mpn INA226    # set

    Multiple MPNs (variants/aliases) can be stored on the same document.
    Use --mpn to set the primary MPN and --mpn-alias to append extras:

        rag metadata <doc_id> --mpn INA226 --mpn-alias INA226A --mpn-alias INA226B

    Filtering with --mpn on `rag list` or via the MCP tools matches any
    token in the comma-separated list.
    """
    from datasheet_rag.backend import MetadataPatch
    from datasheet_rag.project_config import get_project_config

    # Decide read-vs-write from what the user actually typed, before any
    # .rag.toml defaults get merged in below — otherwise a project config
    # would silently turn every `rag metadata <doc>` into a write.
    is_read = not any(
        (
            doc_title,
            project_id,
            group_name,
            mpn,
            mpn_aliases,
            manufacturer,
            subsystem,
            doc_type,
            tags,
            clear_tags,
            attrs,
            unset_attrs,
        )
    )
    if is_read:
        be = _backend_for(db_path)
        existing = be.get_metadata(be.resolve_doc_id(doc_id))
        if existing is None:
            console.print(f"[yellow]No metadata recorded for[/] {doc_id}")
            return
        console.print(existing.model_dump_json(indent=2, exclude_none=True))
        return

    proj_cfg = get_project_config()
    if proj_cfg is not None:
        project_id = project_id or proj_cfg.project_id
        group_name = group_name or proj_cfg.group
        mpn = mpn or proj_cfg.mpn
        manufacturer = manufacturer or proj_cfg.manufacturer
        subsystem = subsystem or proj_cfg.subsystem
        if not tags and proj_cfg.tags:
            tags = tuple(proj_cfg.tags)

    attributes: dict[str, Any] = {}
    for item in attrs:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise click.BadParameter(f"--attr expects KEY=VALUE, got {item!r}", param_hint="--attr")
        attributes[key] = value
    for key in unset_attrs:
        attributes[key] = None

    be = _backend_for(db_path)
    doc_id = be.resolve_doc_id(doc_id)

    # Merge --mpn-alias values into the mpn field as a comma-separated list.
    if mpn_aliases:
        if mpn is None:
            # No --mpn given; read existing to append aliases without clobbering
            existing_meta = be.get_metadata(doc_id)
            base = existing_meta.mpn if existing_meta else None
        else:
            base = mpn
        all_mpns: list[str] = []
        if base:
            all_mpns = [t.strip() for t in base.split(",") if t.strip()]
        for alias in mpn_aliases:
            if alias not in all_mpns:
                all_mpns.append(alias)
        mpn = ",".join(all_mpns)

    if doc_title is not None:
        updated = be.set_doc_title(doc_id, doc_title)
        console.print(f"[green]Title set[/] on {updated} chunk rows: {doc_title!r}")

    if tags:
        new_tags: list[str] | None = list(tags)
    elif clear_tags:
        new_tags = []
    else:
        new_tags = None

    meta = be.set_metadata(
        doc_id,
        MetadataPatch(
            project_id=project_id,
            group_name=group_name,
            mpn=mpn,
            manufacturer=manufacturer,
            subsystem=subsystem,
            doc_type=doc_type,
            tags=new_tags,
            attributes=attributes or None,
        ),
    )
    console.print(f"[green]Saved metadata for[/] {doc_id}")
    console.print(meta.model_dump_json(indent=2, exclude_none=True))

    if apply_to_chunks:
        updated = be.apply_metadata_to_chunks(doc_id)
        console.print(f"  Propagated to {updated} chunk rows.")


# ---------------------------------------------------------------------------
# fix-titles (AI-inferred document titles for poorly-titled documents)
# ---------------------------------------------------------------------------

_BLANK_TITLES = (None, "", "—")


@repair_group.command("titles", short_help="Backfill missing document titles.")
@click.option("--doc-id", default=None, help="Restrict to a single document.")
@click.option(
    "--force",
    is_flag=True,
    help="Re-infer even for documents that already have a title "
    "(needed to replace generic titles like 'Contents').",
)
@click.option(
    "--model", "model_id", default=None, help="Override settings.description_model_id for this run."
)
@click.option("--dry-run", is_flag=True, help="Infer and print titles but do not persist them.")
@_db_option
def fix_titles_cmd(
    doc_id: str | None,
    force: bool,
    model_id: str | None,
    dry_run: bool,
    db_path: Path | None,
) -> None:
    """Infer and backfill document titles with a small Bedrock Claude call.

    Without --doc-id, scans every ingested document and infers a title for
    those showing as "—" (blank) in `rag list`. Pass --doc-id to target one
    document, and add --force to replace a generic-but-present title (e.g.
    "Contents", "Disclaimer") that the heuristic above wouldn't flag.

    Inferred titles are written to every chunk row for the document and
    marked `title_source: inferred` in the metadata sidecar (`rag metadata
    <doc_id>`) so they're distinguishable from titles Docling extracted
    directly, and so a later re-ingest can't demote them. A title set by
    hand outranks an inferred one and is left alone — pass --force to
    replace it anyway, or to overwrite an existing inferred title.
    """
    from datasheet_rag.store.metadata import title_rank, title_source_of

    be = _backend_for(db_path)

    if doc_id:
        doc_id = _backend_resolve(be, doc_id)
        docs = [d for d in be.get_ingested_docs() if d.doc_id == doc_id]
    else:
        docs = be.get_ingested_docs()

    if not force:
        skipped = [d for d in docs if d.doc_title not in _BLANK_TITLES]
        docs = [d for d in docs if d.doc_title in _BLANK_TITLES]
        if skipped and len(skipped) == 1 and not docs:
            console.print(
                f"[yellow]{skipped[0].doc_id[:SHORT_DOC_ID_LEN]}[/] already has a title "
                f"({skipped[0].doc_title!r}). Pass --force to re-infer it anyway."
            )

    if not docs:
        console.print("[yellow]No documents need a title fix.[/]")
        return

    console.print(
        f"Inferring titles for {len(docs)} document(s) (LLM runs server-side in remote mode)…"
    )

    for d in docs:
        short_id = d.doc_id[:SHORT_DOC_ID_LEN]
        current = d.doc_title
        md = be.get_metadata(d.doc_id)
        source = title_source_of(md.attributes if md is not None else {})
        # Say why we're leaving a hand-set title alone, rather than letting
        # the store's refusal surface as an indistinguishable "could not
        # infer" further down.
        if not force and title_rank(source) > title_rank("inferred"):
            console.print(
                f"  [yellow]skipped[/] {short_id} — title was set by hand "
                f"({current!r}); pass --force to replace it"
            )
            continue
        title = be.infer_title(d.doc_id, model_id=model_id, dry_run=dry_run, force=force)
        if title is None:
            console.print(f"  [yellow]could not infer[/] {short_id} (was: {current!r})")
            continue
        verb = "would set" if dry_run else "set"
        console.print(f"  [green]{verb}[/] {short_id}: {current!r} → {title!r}")


# ---------------------------------------------------------------------------
# Eval (retrieval-layer evaluation)
# ---------------------------------------------------------------------------


@cli.group("eval", hidden=True, short_help="Evaluate retrieval quality.")
def eval_group() -> None:
    """Retrieval-layer evaluation: golden set, metrics, ablations."""


_CAT_ORDER = ["identifier", "conceptual", "figure", "table_spec", "synthesis", "overall"]


def _render_report_table(report: object) -> None:
    """Print one RunReport's per-category metrics."""
    from datasheet_rag.eval.harness import RunReport

    assert isinstance(report, RunReport)
    hk = report.config.k
    table = Table(
        title=f"Retrieval eval · {report.config.describe()} "
        f"(hit@k = strict lineage; pg@{hk} = loose page upper bound)"
    )
    table.add_column("category", style="magenta")
    table.add_column("n", justify="right", style="dim")
    for k in report.config.ks:
        table.add_column(f"hit@{k}", justify="right", style="cyan")
    table.add_column(f"pg@{hk}", justify="right", style="dim")
    table.add_column("MRR", justify="right", style="green")
    table.add_column("nDCG", justify="right", style="green")

    for cat in _CAT_ORDER:
        m = report.by_category.get(cat)
        if m is None or m.n == 0:
            continue
        row = [cat, str(m.n)]
        row += [f"{m.hit_rate_at_k.get(k, 0.0):.2f}" for k in report.config.ks]
        row += [f"{m.hit_rate_at_k_loose.get(hk, 0.0):.2f}"]
        row += [f"{m.mrr:.3f}", f"{m.ndcg:.3f}"]
        table.add_row(*row, end_section=(cat == "overall"))
    console.print(table)


def _render_matrix_table(reports: list[RunReport], headline_k: int) -> None:
    """Print a comparison across configs: overall + per-category hit@k."""
    from datasheet_rag.eval.dataset import CATEGORIES

    table = Table(title=f"Ablation comparison · hit@{headline_k} by category")
    table.add_column("config", style="yellow")
    table.add_column("overall", justify="right", style="cyan")
    table.add_column("MRR", justify="right", style="green")
    table.add_column("nDCG", justify="right", style="green")
    for cat in CATEGORIES:
        table.add_column(cat[:5], justify="right")

    for rep in reports:
        overall = rep.by_category.get("overall")
        if overall is None:
            continue
        row = [
            rep.config.describe(),
            f"{overall.hit_rate_at_k.get(headline_k, 0.0):.2f}",
            f"{overall.mrr:.3f}",
            f"{overall.ndcg:.3f}",
        ]
        for cat in CATEGORIES:
            m = rep.by_category.get(cat)
            row.append(f"{m.hit_rate_at_k.get(headline_k, 0.0):.2f}" if m else "—")
        table.add_row(*row)
    console.print(table)


@eval_group.command("generate", short_help="Generate a golden set from the corpus.")
@_db_option
@click.option("--per-category", default=4, type=int, help="Items to generate per category.")
@click.option("--doc-id", default=None, help="Restrict sampling to one document.")
@click.option("--project-id", default=None, help="Restrict sampling to one project.")
@click.option("--model", "model_id", default=None, help="Bedrock model ID for generation.")
@click.option("--seed", default=0, type=int, help="Sampling seed (reproducible).")
@click.option(
    "--output",
    "-o",
    "out_path",
    type=click.Path(path_type=Path),
    default=Path("eval/golden.jsonl"),
    help="Output JSONL path.",
)
@click.option("--append", is_flag=True, help="Append to the output file instead of overwriting.")
@click.option("--verbose/--quiet", default=True, help="Print per-item progress.")
def eval_generate(
    db_path: Path | None,
    per_category: int,
    doc_id: str | None,
    project_id: str | None,
    model_id: str | None,
    seed: int,
    out_path: Path,
    append: bool,
    verbose: bool,
) -> None:
    """Generate a reviewable golden set from the corpus (LLM-assisted)."""
    from datasheet_rag.eval.generate import generate_golden_set
    from datasheet_rag.store import connect

    conn = connect(_require_local_db(db_path))
    if doc_id:
        doc_id = _resolve_doc_id(conn, doc_id)
    eval_set = generate_golden_set(
        conn,
        per_category=per_category,
        doc_id=doc_id,
        project_id=project_id,
        model_id=model_id,
        seed=seed,
        verbose=verbose,
    )
    conn.close()

    if append:
        eval_set.append_jsonl(out_path)
    else:
        eval_set.save(out_path)
    console.print(
        f"[green]Wrote[/] {len(eval_set)} items to {out_path} "
        f"(source='auto' — review before trusting as ground truth)."
    )


@eval_group.command("run", short_help="Score one search config against the set.")
@_db_option
@click.option(
    "--set",
    "set_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("eval/golden.jsonl"),
    help="Golden set JSONL.",
)
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "vector", "keyword"]),
    default="hybrid",
    help="Retrieval mode to score.",
)
@click.option("-k", "top_k", default=5, type=int, help="Headline k (nDCG cutoff).")
@click.option(
    "--level",
    type=click.Choice(["macro", "meso", "micro"]),
    default=None,
    help="Restrict retrieval to one zoom level (default: all three).",
)
@click.option(
    "--rrf-k",
    default=60,
    type=int,
    help="Reciprocal-rank-fusion constant merging the two rankings.",
)
@click.option(
    "--vector-weight", default=1.0, type=float, help="Weight on the vector ranking during fusion."
)
@click.option(
    "--keyword-weight", default=1.0, type=float, help="Weight on the keyword ranking during fusion."
)
@click.option(
    "--trace",
    "trace_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Append per-query JSONL traces here.",
)
@click.option(
    "--json-out",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the full report JSON here.",
)
def eval_run(
    db_path: Path | None,
    set_path: Path,
    mode: str,
    top_k: int,
    level: str | None,
    rrf_k: int,
    vector_weight: float,
    keyword_weight: float,
    trace_path: Path | None,
    json_out: Path | None,
) -> None:
    """Run the golden set through one search config and print metrics."""
    from datasheet_rag.eval.dataset import EvalSet
    from datasheet_rag.eval.harness import RunConfig, run_eval
    from datasheet_rag.store import connect

    conn = connect(_require_local_db(db_path))
    eval_set = EvalSet.load(set_path)

    embedder = None
    if mode in ("vector", "hybrid"):
        from datasheet_rag.embedding import get_embedder

        embedder = get_embedder()

    config = RunConfig(
        mode=mode,  # type: ignore[arg-type]
        k=top_k,
        level=level,  # type: ignore[arg-type]
        rrf_k=rrf_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
    report = run_eval(conn, eval_set, config, embedder=embedder, trace_path=trace_path)
    conn.close()

    _render_report_table(report)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Report JSON →[/] {json_out}")


@eval_group.command("ablate", short_help="Run the ablation matrix.")
@_db_option
@click.option(
    "--set",
    "set_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("eval/golden.jsonl"),
    help="Golden set JSONL.",
)
@click.option("-k", "top_k", default=5, type=int, help="Headline k for the comparison.")
@click.option(
    "--trace",
    "trace_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Append per-query JSONL traces here.",
)
@click.option(
    "--json-out",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the full report matrix here as JSON.",
)
@click.option(
    "--index-ablation",
    type=click.Choice(["context-vs-raw", "figure-desc", "macro-summarizer"]),
    default=None,
    help="Heavy re-embedding ablation (incurs Bedrock cost).",
)
@click.option(
    "--variant-db",
    type=click.Path(path_type=Path),
    default=Path("test-project/output/rag-variant.sqlite"),
    help="Where to build the variant store for an index ablation.",
)
@click.option("--limit", default=None, type=int, help="Cap chunks re-embedded (index ablation).")
@click.option(
    "--doc-id",
    default=None,
    help="Document to re-summarize (required for --index-ablation macro-summarizer).",
)
@click.option(
    "--summarizer-model",
    default="anthropic.claude-3-haiku-20240307-v1:0",
    help="Bedrock model id for the macro-summarizer ablation.",
)
@click.option("--verbose/--quiet", default=True, help="Print per-config progress.")
def eval_ablate(
    db_path: Path | None,
    set_path: Path,
    top_k: int,
    trace_path: Path | None,
    json_out: Path | None,
    index_ablation: str | None,
    variant_db: Path,
    limit: int | None,
    doc_id: str | None,
    summarizer_model: str,
    verbose: bool,
) -> None:
    """Run the ablation matrix and print which concepts move the needle."""
    from datasheet_rag.embedding import get_embedder
    from datasheet_rag.eval.ablation import (
        build_macro_summarizer_variant_store,
        build_variant_store,
        default_matrix,
        run_matrix,
    )
    from datasheet_rag.eval.dataset import EvalSet
    from datasheet_rag.eval.harness import RunConfig, run_eval
    from datasheet_rag.store import connect

    conn = connect(_require_local_db(db_path))
    eval_set = EvalSet.load(set_path)
    embedder = get_embedder()

    if index_ablation is None:
        reports = run_matrix(
            conn,
            eval_set,
            default_matrix(base_k=top_k),
            embedder=embedder,
            trace_path=trace_path,
        )
        conn.close()
        _render_matrix_table(reports, headline_k=top_k)
        if json_out:
            _dump_reports_json(reports, json_out)
        return

    if index_ablation == "macro-summarizer":
        if not doc_id:
            raise click.ClickException(
                "--index-ablation macro-summarizer requires --doc-id "
                "(it re-summarizes one document's chapters with Bedrock Claude)."
            )
        from datasheet_rag.chunking.summarizer import AbstractiveSummarizer
        from datasheet_rag.config import get_settings as _get_settings

        console.print(
            f"[yellow]Index ablation[/] 'macro-summarizer': re-summarizing "
            f"{doc_id} chapters abstractively (model={summarizer_model}) "
            f"and re-embedding — this calls Bedrock Claude per chapter."
        )
        summarizer = AbstractiveSummarizer(
            model_id=summarizer_model,
            region=_get_settings().aws_region,
        )
        variant_conn = build_macro_summarizer_variant_store(
            conn,
            variant_db,
            doc_id,
            summarizer,
            embedder,
            verbose=verbose,
        )

        # The variant store only contains doc_id's chunks (by design — see
        # build_macro_summarizer_variant_store), so items targeting other
        # docs would score 0 by construction. Scope the eval set to match.
        scoped_set = EvalSet(items=[i for i in eval_set.items if i.doc_id == doc_id])

        base_cfg = RunConfig(
            mode="hybrid", k=top_k, level="macro", label="baseline (extractive macro)"
        )
        var_cfg = RunConfig(
            mode="hybrid", k=top_k, level="macro", label="variant (abstractive macro)"
        )

        base_report = run_eval(conn, scoped_set, base_cfg, embedder=embedder, trace_path=trace_path)
        var_report = run_eval(
            variant_conn, scoped_set, var_cfg, embedder=embedder, trace_path=trace_path
        )
        conn.close()
        variant_conn.close()

        reports = [base_report, var_report]
        _render_matrix_table(reports, headline_k=top_k)
        stats = summarizer.stats
        console.print(
            f"\n[bold]Cost/latency (abstractive macro re-summarization):[/] "
            f"{stats.calls} Bedrock calls over {stats.chapters} chapters "
            f"({stats.avg_calls_per_chapter:.1f} calls/chapter, "
            f"{stats.avg_latency_ms_per_chapter / 1000:.1f} s/chapter, "
            f"{stats.total_latency_ms / 1000:.1f} s total)"
        )
        if json_out:
            _dump_reports_json(reports, json_out)
        return

    # Index ablation: baseline (current store) vs variant (re-embedded).
    variant: IndexVariant = "raw_text" if index_ablation == "context-vs-raw" else "no_figure_desc"
    console.print(
        f"[yellow]Index ablation[/] '{index_ablation}': building variant store "
        f"(variant={variant}) — this re-embeds and incurs Bedrock cost."
    )
    variant_conn = build_variant_store(
        conn,
        variant_db,
        variant,
        embedder,
        limit=limit,
        verbose=verbose,
    )

    base_cfg = RunConfig(mode="hybrid", k=top_k, label="baseline (context_text)")
    var_label = "raw text" if variant == "raw_text" else "no figure desc"
    var_cfg = RunConfig(mode="hybrid", k=top_k, label=f"variant ({var_label})")

    base_report = run_eval(conn, eval_set, base_cfg, embedder=embedder, trace_path=trace_path)
    var_report = run_eval(variant_conn, eval_set, var_cfg, embedder=embedder, trace_path=trace_path)
    conn.close()
    variant_conn.close()

    reports = [base_report, var_report]
    _render_matrix_table(reports, headline_k=top_k)
    if json_out:
        _dump_reports_json(reports, json_out)


def _dump_reports_json(reports: list[RunReport], path: Path) -> None:
    import json as _json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _json.dumps([r.model_dump() for r in reports], indent=2, default=str),
        encoding="utf-8",
    )
    console.print(f"[green]Reports JSON →[/] {path}")


@eval_group.command("review", short_help="Hand-review the golden set in a web app.")
@_db_option
@click.option(
    "--set",
    "set_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("eval/golden.jsonl"),
    help="Golden set JSONL to review.",
)
@click.option("--port", default=0, type=int, help="Port (0 = pick a free one).")
@click.option("-k", "top_k", default=5, type=int, help="Retrieval results to preview per item.")
@click.option(
    "--open/--no-open", "open_browser", default=True, help="Open the review page in a browser."
)
def eval_review(
    db_path: Path | None,
    set_path: Path,
    port: int,
    top_k: int,
    open_browser: bool,
) -> None:
    """Hand-review the golden set in a local web app (PDF page + page/chunk labels)."""
    from datasheet_rag.eval.review import serve

    serve(set_path, db_path=db_path, port=port, k=top_k, open_browser=open_browser)


# ---------------------------------------------------------------------------
# Remote admin — manage API keys + read the audit log over HTTP.
# ---------------------------------------------------------------------------


def _admin_request(method: str, path: str, *, token: str | None, **kwargs: Any) -> dict[str, Any]:
    """Call an admin endpoint on the configured remote server.

    Admin operates over HTTP against RAG_SERVER_URL (managing the *server's*
    keys), so it requires remote mode. The bearer token must carry the
    ``admin`` scope; default is RAG_SERVER_TOKEN, overridable with --token.
    """
    import httpx

    settings = get_settings()
    base = settings.server_url
    if not base:
        raise click.ClickException(
            "admin commands talk to a remote server — set RAG_SERVER_URL "
            "(and use an admin-scoped token). To manage a local DB instead, "
            "run `rag-server create-key` on the server host."
        )
    bearer = token or settings.server_token
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    try:
        resp = httpx.request(
            method,
            base.rstrip("/") + path,
            headers=headers,
            timeout=settings.server_timeout,
            **kwargs,
        )
    except httpx.HTTPError as exc:
        raise click.ClickException(f"could not reach {base}: {exc}") from exc
    if resp.status_code == 401:
        raise click.ClickException("401: missing/invalid token (need an admin key).")
    if resp.status_code == 403:
        raise click.ClickException("403: this token lacks the 'admin' scope.")
    if resp.status_code == 404:
        raise click.ClickException(f"404: {resp.json().get('detail', resp.text)}")
    if resp.status_code >= 400:
        raise click.ClickException(f"{resp.status_code}: {resp.text}")
    body: dict[str, Any] = resp.json()
    return body


@cli.group(short_help="Administer a remote RAG server.")
def admin() -> None:
    """Administer a remote RAG server: API keys and audit log.

    Talks to RAG_SERVER_URL with an admin-scoped bearer token (RAG_SERVER_TOKEN
    or --token). Bootstrap the very first admin key on the server host with
    `rag-server create-key --label bootstrap --scope admin`.
    """


@admin.group(short_help="Create, list and revoke API keys.")
def key() -> None:
    """Create, list and revoke per-client API keys."""


@key.command("create", short_help="Mint a key (token shown once).")
@click.option("--label", required=True, help="Client identity for the key (shown in audit).")
@click.option(
    "--scope",
    "scopes",
    multiple=True,
    type=click.Choice(["read", "ingest", "admin"]),
    default=("ingest",),
    help="Scope(s) for the key (repeatable). Default: ingest.",
)
@click.option("--token", default=None, help="Admin token (default: RAG_SERVER_TOKEN).")
def key_create(label: str, scopes: tuple[str, ...], token: str | None) -> None:
    """Mint a key. The plaintext token is shown ONCE — copy it now."""
    data = _admin_request(
        "POST",
        "/admin/keys",
        token=token,
        json={"label": label, "scopes": list(scopes)},
    )
    console.print(
        f"[green]Created[/] key '{data['label']}' (id={data['id']}, scopes={data['scopes']})"
    )
    console.print("\n[bold]Token (shown once — store it now):[/]\n")
    console.print(f"  {data['token']}\n")


@key.command("list", short_help="List API keys (never shows secrets).")
@click.option("--token", default=None, help="Admin token (default: RAG_SERVER_TOKEN).")
def key_list(token: str | None) -> None:
    """List API keys (never shows secrets)."""
    data = _admin_request("GET", "/admin/keys", token=token)
    keys = data.get("keys", [])
    if not keys:
        console.print("[yellow]No API keys.[/]")
        return
    table = Table(title=f"API keys · {len(keys)}")
    for col in ("id", "label", "scopes", "created_at", "revoked_at"):
        table.add_column(col)
    for k in keys:
        table.add_row(
            k["id"],
            k["label"],
            ",".join(k["scopes"]),
            k.get("created_at") or "—",
            "[red]revoked[/]" if k.get("revoked_at") else "—",
        )
    console.print(table)


@key.command("revoke", short_help="Revoke a key by id.")
@click.argument("key_id", type=str)
@click.option("--token", default=None, help="Admin token (default: RAG_SERVER_TOKEN).")
def key_revoke(key_id: str, token: str | None) -> None:
    """Revoke a key by id (takes effect immediately, no restart)."""
    _admin_request("DELETE", f"/admin/keys/{key_id}", token=token)
    console.print(f"[green]Revoked[/] key {key_id}")


@admin.command("audit", short_help="Show the ingest-path audit trail.")
@click.option("--doc-id", default=None, help="Filter to one document.")
@click.option("--since", default=None, help="ISO timestamp lower bound (e.g. 2026-06-01).")
@click.option("--limit", default=50, type=int, help="Max rows (newest first).")
@click.option("--token", default=None, help="Admin token (default: RAG_SERVER_TOKEN).")
def audit_cmd(doc_id: str | None, since: str | None, limit: int, token: str | None) -> None:
    """Show the ingest-path audit trail."""
    params: dict[str, Any] = {"limit": limit}
    if doc_id:
        params["doc_id"] = doc_id
    if since:
        params["since"] = since
    data = _admin_request("GET", "/audit", token=token, params=params)
    entries = data.get("entries", [])
    if not entries:
        console.print("[yellow]No audit entries.[/]")
        return
    table = Table(title=f"Audit · {len(entries)} entries")
    for col in ("ts", "action", "status", "key_label", "client_ip", "doc_id", "detail"):
        table.add_column(col, overflow="fold")
    for e in entries:
        table.add_row(
            (e.get("ts") or "")[:19],
            e.get("action") or "",
            e.get("status") or "",
            e.get("key_label") or "—",
            e.get("client_ip") or "—",
            (e.get("doc_id") or "—")[:12],
            e.get("detail_json") or (e.get("error") or "—"),
        )
    console.print(table)
