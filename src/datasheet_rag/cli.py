"""CLI for the RAG pipeline."""

from __future__ import annotations

import glob
import json
import os
import random
import re
import socket
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import click
from rich.console import Console
from rich.table import Table

from datasheet_rag import cache_files
from datasheet_rag.config import get_settings
from datasheet_rag.store.documents import parse_document_ref

if TYPE_CHECKING:
    import sqlite3

    from datasheet_rag.backend import RagBackend
    from datasheet_rag.backend.base import RagServerError, SearchMode
    from datasheet_rag.backend.models import DocumentRevisions, IngestedDoc, IngestResult
    from datasheet_rag.chunking.layout_parser import DocumentOutline
    from datasheet_rag.costs import CostEstimate
    from datasheet_rag.eval.ablation import IndexVariant
    from datasheet_rag.eval.answers import AnswerRecord, AnswerReport, RunComparison
    from datasheet_rag.eval.harness import RunReport
    from datasheet_rag.eval.metrics import PairedComparison
    from datasheet_rag.eval.stats import Interval, PairedDiff
    from datasheet_rag.ingest_pipeline import ProgressEvent
    from datasheet_rag.models.chunk import Chunk
    from datasheet_rag.project_config import ProjectConfig
    from datasheet_rag.store.fields import FieldScope, FieldSpec
    from datasheet_rag.store.metadata import DocMetadata

console = Console()

# rev_ids are full SHA-256 content hashes (64 hex chars). We display and
# accept abbreviated forms — like `git log --oneline` short SHAs — and
# resolve unambiguous prefixes to the full hash via `resolve_rev_id`.
SHORT_REV_ID_LEN = 12


def _resolve_rev_id(conn: sqlite3.Connection, rev_id: str) -> str:
    """CLI wrapper around the **store-backed** `resolve_rev_id`: turns
    ambiguity/misses into ClickException.

    Domain: the sqlite index — only resolves docs that have been *embedded*.
    For commands that operate on on-disk cache artifacts (blocks/chunks/outline)
    before a doc is embedded, use `_resolve_cached_rev_id` instead.
    """
    from datasheet_rag.store import resolve_rev_id

    try:
        return resolve_rev_id(conn, rev_id)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _cache_path(rev_id: str, suffix: str) -> Path:
    """Return the conventional cache-artifact path for *rev_id*.

    All local pipeline artifacts live under ``settings.output_dir`` keyed by
    rev_id, e.g. ``{rev_id}_blocks.json`` / ``{rev_id}_chunks.json`` /
    ``{rev_id}_outline.json``.
    """
    return get_settings().output_dir / f"{rev_id}{suffix}"


def _resolve_store_first(arg: str) -> str | None:
    """Expand a rev_id prefix via the store, or return None if it isn't there.

    The store is the authoritative namespace: a rev_id printed by `rag list`
    or `rag search` should resolve the same way everywhere. Cache-reading
    commands consult this first and only fall back to prefix-matching the
    local cache, which is what lets them still work on a PDF that was parsed
    but never ingested.

    Never raises — an unreachable server or an unknown prefix both mean "not
    in the store", and the caller falls back to the cache.
    """
    try:
        return _backend_for().resolve_rev_id(arg)
    except Exception:
        return None


def _resolve_cached_rev_id(rev_id: str, suffix: str = "_outline.json") -> str:
    """Resolve a possibly-abbreviated rev_id against the local cache dir.

    Domain: the filesystem cache (``settings.output_dir``). Globs
    ``{output_dir}/{rev_id}*{suffix}`` and returns the single full rev_id that
    matches. Unlike the store-backed `_resolve_rev_id`, this sees documents
    that have a cached artifact but have not been embedded yet — which is the
    normal state for `repair chunks` / `repair embeddings` inputs.

    Raises ClickException on zero or ambiguous matches (consistent message).
    """
    settings = get_settings()

    # Store first, so an abbreviation resolves the same way it does in
    # `rag list` / `rag search`; fall back to the cache for docs not (yet)
    # ingested.
    full = _resolve_store_first(rev_id)
    if full is not None and (settings.output_dir / f"{full}{suffix}").is_file():
        return full

    # The prefix is user input: `*`, `?` and `[` must not act as wildcards.
    matches = sorted(settings.output_dir.glob(f"{glob.escape(rev_id)}*{suffix}"))
    if not matches:
        raise click.ClickException(
            f"No cached {suffix.lstrip('_')} matching rev_id {rev_id!r} in "
            f"{settings.output_dir}. Run the earlier pipeline stage for this "
            "document first."
        )
    resolved = {p.name.removesuffix(suffix) for p in matches}
    if len(resolved) > 1:
        names = ", ".join(sorted(resolved))
        raise click.ClickException(f"rev_id {rev_id!r} is ambiguous — matches: {names}")
    return next(iter(resolved))


def _require_docling_outline(rev_id: str) -> str:
    """Resolve *rev_id* to a cached Docling outline, or explain why there isn't one.

    The table commands read Docling's own structure confidence signals, which
    Textract's output has no equivalent of — so they are Docling-only by
    nature, not by oversight. Detect the Textract case explicitly rather than
    letting `_resolve_cached_rev_id` report a generic missing-artifact error
    that reads like the document was never ingested.
    """
    settings = get_settings()
    prefix = glob.escape(rev_id)
    if not sorted(settings.output_dir.glob(f"{prefix}*_outline.json")):
        if sorted(settings.output_dir.glob(f"{prefix}*_blocks.json")):
            raise click.ClickException(
                f"Revision {rev_id!r} was ingested with the Textract backend, and "
                "the table commands are Docling-only — they read Docling's "
                "table-structure confidence signals, which Textract's output "
                "does not provide. Re-ingest with `rag ingest --backend docling` "
                "to use them (see GitHub issue #31)."
            )
    return _resolve_cached_rev_id(rev_id, "_outline.json")


def _doc_input(arg: str, suffix: str) -> tuple[str, Path]:
    """Resolve a smart positional that is *either* a rev_id or an explicit file path.

    If *arg* is an existing file, use it directly and infer the rev_id from its
    stem (dropping the artifact suffix, e.g. ``_blocks``). Otherwise treat *arg*
    as a rev_id prefix, resolve it against the cache (`_resolve_cached_rev_id`),
    and return the conventional ``{rev_id}{suffix}`` path.

    Returns ``(rev_id, path)``.
    """
    p = Path(arg)
    if p.is_file():
        stem_suffix = suffix.removesuffix(".json")  # e.g. "_blocks"
        rev_id = p.stem.removesuffix(stem_suffix)
        return rev_id, p
    rev_id = _resolve_cached_rev_id(arg, suffix)
    return rev_id, _cache_path(rev_id, suffix)


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
    import httpx

    code = getattr(e, "status_code", None)
    if code == 0 and isinstance(e.__cause__, httpx.HTTPError):
        # The request never got an answer. Status 0 alone is not enough: the
        # embedding-dimension checks also use it for an answer they did get.
        return click.ClickException(
            f"Could not get an answer from the RAG server ({e.detail}). Check that "
            "it is running and that RAG_SERVER_URL points at it."
        )
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


def _backend_resolve(be: RagBackend, rev_id: str) -> str:
    """Resolve a rev_id prefix via the backend, turning a miss into ClickException.

    A server error is left to the root group, which reports it the same way.
    """
    try:
        return be.resolve_rev_id(rev_id)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _short_chunk_id(chunk_id: str, rev_id: str) -> str:
    """Abbreviate a chunk_id's rev_id portion the same way rev_ids are
    displayed elsewhere (`ab12cd34ef56:L2:143`), keeping the
    `:L{level}:{index}` suffix intact so it round-trips through
    `_resolve_chunk_id`.
    """
    return rev_id[:SHORT_REV_ID_LEN] + chunk_id[len(rev_id) :]


def _resolve_chunk_id(be: RagBackend, chunk_id: str) -> str:
    """Resolve a chunk_id whose rev_id portion may be abbreviated (as
    printed by `rag search` / `rag inspect figures`) to its full form.

    Chunk IDs are `{rev_id}:L{level}:{index}`; the rev_id may be given in
    full or as an unambiguous prefix, resolved the same way `rev_id`
    arguments are elsewhere in this CLI.
    """
    if ":" not in chunk_id:
        raise click.ClickException(
            f"{chunk_id!r} doesn't look like a chunk ID — expected "
            "'<rev_id>:L<level>:<index>' (see `rag search` output)."
        )
    doc_part, suffix = chunk_id.split(":", 1)
    full_rev_id = _backend_resolve(be, doc_part)
    return f"{full_rev_id}:{suffix}"


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

    def invoke(self, ctx: click.Context) -> Any:
        # Errors a user can cause and fix — a broken .rag.toml, an unreachable
        # or rejecting server — can surface from almost any command, so they
        # are turned into a plain CLI message here once rather than in every
        # command (GH #36). Anything else is a bug: its traceback is kept, with
        # a line saying so, because that is what the developer needs to see.
        from datasheet_rag.backend.base import RagServerError
        from datasheet_rag.project_config import ProjectConfigError

        try:
            return super().invoke(ctx)
        except (ProjectConfigError, cache_files.StaleCacheError) as e:
            raise click.ClickException(str(e)) from e
        except RagServerError as e:
            raise _friendly_server_error(e) from e
        except (click.ClickException, click.exceptions.Exit, click.Abort, EOFError):
            # Click's own control flow; it reports these itself.
            raise
        except Exception:
            click.echo(
                "An unexpected error happened. Please report the following to the developer:",
                err=True,
            )
            raise


def _parse_field_items(items: tuple[str, ...], *, option: str = "--field") -> dict[str, Any]:
    """Parse repeated ``--field NAME=VALUE`` options into a field patch.

    A name given once maps to its string; a name given more than once maps
    to the list of its values, in order — which is how a list field such as
    ``mpn`` gets several values. The store coerces a lone string to a
    one-entry list for a list field, and rejects a list for a single field.
    """
    parsed: dict[str, Any] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise click.BadParameter(
                f"{option} expects NAME=VALUE, got {item!r}", param_hint=option
            )
        if name not in parsed:
            parsed[name] = value
        elif isinstance(parsed[name], list):
            parsed[name].append(value)
        else:
            parsed[name] = [parsed[name], value]
    return parsed


def _shortcut_fields(
    *,
    mpn: tuple[str, ...],
    manufacturer: str | None,
    doc_type: str | None,
    subsystem: str | None,
) -> dict[str, Any]:
    """The built-in-field shortcut options as a field patch (unset ones omitted).

    ``--mpn`` is repeatable, one part number each. An empty string is kept:
    it means "clear this field", the same as for any other metadata option.
    """
    out: dict[str, Any] = {}
    if mpn:
        out["mpn"] = list(mpn)
    if manufacturer is not None:
        out["manufacturer"] = manufacturer
    if doc_type is not None:
        out["doc_type"] = doc_type
    if subsystem is not None:
        out["subsystem"] = subsystem
    return out


class _FieldRegistry:
    """The field registry, read at most once per command and only when asked for.

    A bulk ingest shares one, so 300 PDFs cost one ``list_fields`` call.
    """

    def __init__(self, backend: RagBackend) -> None:
        self._backend = backend
        self._specs: list[FieldSpec] | None = None

    def specs(self) -> list[FieldSpec]:
        if self._specs is None:
            from datasheet_rag.backend import RagServerError

            try:
                self._specs = self._backend.list_fields()
            except RagServerError as e:
                raise _friendly_server_error(e) from e
        return self._specs


def _merge_config_fields(
    proj_cfg: ProjectConfig | None,
    fields: dict[str, Any],
    *,
    scopes: Mapping[str, FieldScope] | None = None,
    in_config_project: bool = True,
    why_not: str | None = None,
) -> dict[str, Any]:
    """Fill ``.rag.toml`` ``[fields]`` in under the user's options.

    ``fields`` is what the user typed, an unset as ``None``. A name the user
    gave, to set or to unset, is not filled from the config.

    A project-scoped config field (``subsystem``) belongs to the config's
    ``project_id``. It is filled in only when ``in_config_project``: the
    write goes to that project and the document is in it after the write
    (always true for a config with no ``project_id``). Otherwise it is left
    out, and a note says so with ``why_not`` as the reason; ``why_not=None``
    leaves it out without a note (the document is leaving that project).
    ``scopes`` is the field registry's scopes; ``None`` (no registry read, as
    on a dry run, which writes nothing) fills every field.
    """
    if proj_cfg is None or not proj_cfg.fields:
        return fields
    cfg_fields = dict(proj_cfg.fields)
    if scopes is not None and not in_config_project:
        left_out = [k for k in cfg_fields if scopes.get(k) == "project" and k not in fields]
        cfg_fields = {k: v for k, v in cfg_fields.items() if scopes.get(k) != "project"}
        if left_out and why_not:
            console.print(
                f"[yellow]Note:[/] .rag.toml sets {', '.join(left_out)} for project "
                f"{proj_cfg.project_id!r}; left out because {why_not}."
            )
    return {**cfg_fields, **fields}


def _check_field_patch(
    registered: Iterable[FieldSpec], fields: dict[str, Any], *, project: str | None
) -> None:
    """Reject a bad field patch before any slow work (parsing a PDF).

    ``project`` is where the patch's project-scoped fields would be written.
    """
    from datasheet_rag.store.fields import scopes_of, shapes_of
    from datasheet_rag.store.metadata import check_patch_for

    specs = list(registered)
    try:
        check_patch_for(shapes_of(specs), scopes_of(specs), fields=fields, project=project)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _merge_shortcuts(fields: dict[str, Any], shortcuts: dict[str, Any]) -> dict[str, Any]:
    """Fold shortcut options into ``--field`` items; naming a field both ways is an error."""
    both = sorted(set(fields) & set(shortcuts))
    if both:
        raise click.UsageError(
            f"{', '.join(both)} given both as --{both[0].replace('_', '-')} and --field; use one"
        )
    return {**fields, **shortcuts}


@click.group(
    cls=OrderedGroup,
    context_settings={"show_default": True},
    order=[
        "ingest",
        "metadata",
        "fields",
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
@click.option(
    "--compute",
    type=click.Choice(["server", "client"]),
    default=None,
    help=(
        "Override RAG_COMPUTE for this command: where the models run in "
        "remote mode. 'server' (the default) lets the RAG server parse, embed, "
        "describe figures and infer titles; 'client' runs all of it here and "
        "uses the server purely as a vector store — for a server host without "
        "a GPU. No effect in local mode."
    ),
)
@click.pass_context
def cli(ctx: click.Context, compute: str | None) -> None:
    """Datasheet RAG Pipeline — electronics datasheet ingestion."""
    # Applied through the environment rather than passed down: every command
    # resolves the backend from settings, so this is the one place that
    # reaches all of them. Must land before anything builds a backend.
    if compute is not None:
        os.environ["RAG_COMPUTE"] = compute
        get_settings.cache_clear()

    # Remind the user when they're using the local sqlite file rather than a
    # shared server, or when a remote session is running the models locally
    # (both printed to stderr, non-failing). Skipped for `config`, which is
    # the group that sets the server up.
    if ctx.invoked_subcommand != "config":
        from datasheet_rag.backend import emit_client_compute_notice, emit_local_notice

        emit_local_notice()
        emit_client_compute_notice()


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
            "\n[dim]By default the server runs the models — parsing, embeddings, "
            "figure descriptions and titles — so this machine needs no model "
            "config at all. Choose 'client' instead if the server host has no "
            "GPU and this one does: the server then only stores what you send "
            "it, and this machine needs the model extras installed.[/]"
        )
        compute = click.prompt(
            "Where should the models run",
            type=click.Choice(["server", "client"]),
            default="server",
        )
        chosen["RAG_COMPUTE"] = compute
        if compute == "client":
            backend = click.prompt(
                "Embedding backend (must match the server's store)",
                type=click.Choice(["local", "bedrock"]),
                default="local",
            )
            chosen["RAG_EMBEDDING_BACKEND"] = backend
            if backend == "bedrock":
                chosen["AWS_REGION"] = click.prompt("AWS region", default=settings.aws_region)
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
        console.print(f"  Models run: [cyan]{chosen.get('RAG_COMPUTE', 'server')}[/]-side")
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
    `figures`, `chunks`) needs `rag repair embeddings <rev-id>` afterwards
    before the change shows up in `rag search`. `titles` is the exception —
    it writes doc_title directly and is visible immediately.
    """


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------

# The column vocabulary `rag list --columns` understands. Split by where the
# value lives: the store's own row (`IngestedDoc`) or the metadata sidecar
# (`DocMetadata`). Anything *not* named here must be a registered field
# (mpn, manufacturer, doc_type and user fields — GH #120); each field is a
# column of its own.
_DOC_COLUMNS: dict[str, Callable[[IngestedDoc], Any]] = {
    "rev_id": lambda d: d.rev_id[:SHORT_REV_ID_LEN],
    "title": lambda d: d.doc_title,
    "chunks": lambda d: d.chunk_count,
    "pages": lambda d: d.page_count,
    "ingested": lambda d: d.ingested_at,
}

_META_COLUMNS: dict[str, Callable[[DocMetadata], Any]] = {
    "document": lambda m: m.document_id,
    "revision": lambda m: m.revision,
    "revisions": lambda m: len(m.revisions) or None,
    "projects": lambda m: m.projects,
    "tags": lambda m: m.tags,
    "updated": lambda m: m.updated_at,
    "stale": lambda m: "[yellow]re-embed[/]" if m.needs_reembed else "",
}

# Model attribute names that are not column names. Anyone who knows the
# model or the JSON `rag metadata` prints reaches for these spellings, so map
# them rather than rejecting them as unknown.
_LIST_COLUMN_ALIASES = {
    "id": "rev_id",
    "doc_title": "title",
    "chunk_count": "chunks",
    "page_count": "pages",
    "ingested_at": "ingested",
    "project": "projects",
    "project_id": "projects",
    "updated_at": "updated",
}

_LIST_RIGHT_ALIGNED = {"chunks", "pages", "revisions"}

#: Shown after a table that marks values set on one revision only (GH #124).
_OVERRIDE_NOTE = (
    "  * set on this revision only (rag metadata <rev_id> --this-revision ...); "
    "the document's other revisions hold the document's value"
)

# The two built-in views. They answer different questions — ingest health vs.
# cataloguing — so --wide swaps the store-stat columns for the sidecar ones
# rather than appending them. Showing all eleven at once is unreadable on an
# 80-column terminal, where Rich squeezes every cell down to nothing.
_LIST_DEFAULT_COLUMNS = ("revision", "revisions", "title", "chunks", "pages", "ingested")
_LIST_WIDE_COLUMNS = ("projects", "mpn", "manufacturer", "subsystem", "tags")


def _parse_list_columns(columns: tuple[str, ...]) -> list[str]:
    """Flatten repeated/comma-separated --columns into a spec list."""
    parsed = [name.strip() for spec in columns for name in spec.split(",") if name.strip()]
    if not parsed:
        raise click.BadParameter("expects at least one column name", param_hint="--columns")
    return [_LIST_COLUMN_ALIASES.get(name, name) for name in parsed]


def _list_cell(
    name: str, doc: IngestedDoc, meta: DocMetadata | None, project_id: str | None = None
) -> Any:
    """Value of column *name* for one row, or None when it has none.

    A name that is not a built-in column is a registered field. A
    project-scoped field shows its value in ``project_id``; in an unscoped
    listing it shows every project's value, ``project: value``.
    """
    if (doc_getter := _DOC_COLUMNS.get(name)) is not None:
        return doc_getter(doc)
    if meta is None:
        return None
    if (meta_getter := _META_COLUMNS.get(name)) is not None:
        return meta_getter(meta)
    if name in meta.fields:
        return meta.fields[name]
    if project_id is not None:
        return meta.project_fields.get(project_id, {}).get(name)
    from datasheet_rag.store.fields import display_value

    per_project = [
        f"{project}: {display_value(values[name])}"
        for project, values in sorted(meta.project_fields.items())
        if name in values
    ]
    return "; ".join(per_project) or None


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
@click.option("--mpn", default=None, help="Only show documents with this part number.")
@click.option(
    "--subsystem",
    default=None,
    help="Only show documents with this subsystem in the listed project "
    "(in any project with --global).",
)
@click.option(
    "--field",
    "field_items",
    multiple=True,
    metavar="NAME=VALUE",
    help="Only show documents whose registered field NAME holds VALUE "
    "(repeatable, all must match; a list field matches any entry; "
    "case-insensitive; a project-scoped field matches in the listed "
    "project). See `rag fields list`.",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Only show documents that have this tag (repeatable — "
    "with multiple --tag, a document must have all of them).",
)
@click.option(
    "--columns",
    "-c",
    "columns",
    multiple=True,
    metavar="COL,COL,...",
    help="Choose the columns to show (comma-separated, repeatable). "
    "Built-ins: revision, revisions, title, chunks, pages, ingested, "
    "projects, mpn, manufacturer, subsystem, doc_type, tags, updated, stale, "
    "and any registered field (`rag fields list`). document and rev_id are "
    "always shown.",
)
@click.option(
    "--wide",
    "-w",
    is_flag=True,
    help="Show the sidecar columns (projects, mpn, manufacturer, "
    "subsystem, tags, then each user-registered field) in place of the "
    "chunk/page/ingest columns. Implied whenever a filter is given.",
)
@click.option(
    "--all-revisions",
    "-a",
    "all_revisions",
    is_flag=True,
    help="One row per revision (PDF), not just each document's newest.",
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
    mpn: str | None,
    subsystem: str | None,
    field_items: tuple[str, ...],
    tags: tuple[str, ...],
    columns: tuple[str, ...],
    wide: bool,
    all_revisions: bool,
    show_s3: bool,
) -> None:
    """List ingested documents (searchable in the store).

    One row per document (D12), showing its newest revision; --all-revisions
    shows every revision. `rag revisions D12` lists one document's revisions.

    Shows the store's own view — title, chunk and page counts, ingest time.
    Pass --wide, or any sidecar filter, to swap those for the metadata columns
    (projects, mpn, manufacturer, subsystem, tags) instead; filters narrow the
    listing to documents whose sidecar row matches.

    subsystem holds one value per project. A listing scoped to a project
    shows that project's value; with --global it shows each project's.

    --columns replaces both views with exactly the columns you name. Each
    registered field is a column of its own:

      rag list --columns mpn,acme-id
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
        table.add_column("rev_id", style="cyan")
        table.add_column("S3 Prefix")
        for s3_doc in s3_docs:
            table.add_row(s3_doc["rev_id"], s3_doc["prefix"])
        console.print(table)
        return

    field_filters: dict[str, str] = {}
    for name, value in _parse_field_items(field_items).items():
        if isinstance(value, list):
            raise click.BadParameter(
                f"--field {name} given more than once; a filter takes one value per field",
                param_hint="--field",
            )
        field_filters[name] = value
    if mpn is not None:
        if "mpn" in field_filters:
            raise click.UsageError("mpn given both as --mpn and --field; use one")
        field_filters["mpn"] = mpn
    if subsystem is not None:
        if "subsystem" in field_filters:
            raise click.UsageError("subsystem given both as --subsystem and --field; use one")
        field_filters["subsystem"] = subsystem

    filtering = bool(field_filters or tags)
    be = _backend_for(db_path)
    # The default view shows no field columns, so it needs no registry read
    # (one fewer round trip in remote mode).
    registered = be.list_fields() if (filtering or wide or chosen) else []
    field_names = {f.name for f in registered}
    unknown = sorted(set(field_filters) - field_names)
    if unknown:
        raise click.BadParameter(
            f"unregistered field(s) {', '.join(unknown)} (see `rag fields list`)",
            param_hint="--field",
        )
    unknown_cols = sorted(
        n
        for n in chosen or []
        if n not in _DOC_COLUMNS and n not in _META_COLUMNS and n not in field_names
    )
    if unknown_cols:
        raise click.BadParameter(
            f"unknown column(s) {', '.join(unknown_cols)}: not a built-in column or a "
            "registered field (see `rag fields list`)",
            param_hint="--columns",
        )
    docs = be.get_ingested_docs(project_id=project_id)

    # The sidecar is a separate table from the chunk store, so join by rev_id
    # rather than assuming every ingested document has a row in it.
    # Always fetched, not just when filtering: the stale marker and the
    # document a revision belongs to live there. Every revision is read, so a
    # PDF missing here is one without a sidecar row, not an older revision.
    meta_by_id = {
        m.rev_id: m
        for m in be.list_docs(
            project_id=project_id, fields=field_filters or None, all_revisions=True
        )
    }
    if not all_revisions:
        docs = [d for d in docs if (m := meta_by_id.get(d.rev_id)) is None or m.is_newest]

    if filtering:
        wanted_tags = set(tags)
        docs = [
            d
            for d in docs
            if (m := meta_by_id.get(d.rev_id)) is not None and wanted_tags.issubset(set(m.tags))
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
        d.rev_id for d in docs if (m := meta_by_id.get(d.rev_id)) is not None and m.needs_reembed
    }

    if chosen is not None:
        shown = list(chosen)
    else:
        shown = list(_LIST_WIDE_COLUMNS if (wide or filtering) else _LIST_DEFAULT_COLUMNS)
        if wide or filtering:
            shown.extend(f.name for f in registered if not f.builtin)
        # Revision columns only earn their width when a row has something in
        # them: most stores hold one unlabelled PDF per document.
        metas = [m for d in docs if (m := meta_by_id.get(d.rev_id)) is not None]
        if not any(m.revision for m in metas) and "revision" in shown:
            shown.remove("revision")
        if not any(len(m.revisions) > 1 for m in metas) and "revisions" in shown:
            shown.remove("revisions")
    # rev_id is the handle every other `rag` command takes, so a listing
    # without it is a dead end; the document id names what metadata is set
    # on. The stale marker is a health flag rather than metadata, and is
    # worth showing whatever view was asked for.
    if "rev_id" not in shown:
        shown.insert(0, "rev_id")
    if "document" not in shown:
        shown.insert(0, "document")
    if stale_ids and "stale" not in shown:
        shown.insert(shown.index("rev_id") + 1, "stale")

    table = Table(title="Ingested Documents")
    for name in shown:
        # The two handles every other command takes are never squeezed.
        handle = name in ("rev_id", "document")
        table.add_column(
            name,
            style="cyan" if name == "rev_id" else None,
            justify="right" if name in _LIST_RIGHT_ALIGNED else "left",
            no_wrap=handle,
            min_width=SHORT_REV_ID_LEN if name == "rev_id" else None,
        )

    marked = False
    for doc in docs:
        meta = meta_by_id.get(doc.rev_id)
        cells = []
        for name in shown:
            cell = _fmt_list_cell(_list_cell(name, doc, meta, project_id))
            if meta is not None and name in meta.overrides:
                cell += "*"
                marked = True
            cells.append(cell)
        table.add_row(*cells)

    console.print(table)
    if marked:
        console.print(_OVERRIDE_NOTE)
    if stale_ids:
        console.print(
            f"  [yellow]{len(stale_ids)} document(s) were repaired since they were "
            "last embedded[/] — their stored text has moved on from their vectors, "
            "so `rag search` still returns the old wording. Run "
            "[cyan]rag repair embeddings <rev-id>[/] on each to catch them up."
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
@click.option("--rev-id", default=None, help="Restrict to one revision (rev_id).")
@_db_option
def stats_cmd(
    project_id: str | None,
    is_global: bool,
    rev_id: str | None,
    db_path: Path | None,
) -> None:
    """Show chunk counts — total and by zoom level — for a scope.

    A quick sanity check on corpus size (e.g. after ingesting: does the
    chunk count look right?). Same rollup as the MCP `stats` tool.
    """
    from datasheet_rag.project_config import resolve_cli_project_id

    project_id = resolve_cli_project_id(project_id, is_global=is_global)
    be = _backend_for(db_path)
    if rev_id:
        rev_id = _backend_resolve(be, rev_id)

    result = be.stats(project_id=project_id, rev_id=rev_id)

    scope_bits = []
    if result.project_id:
        scope_bits.append(f"project={result.project_id}")
    if result.rev_id:
        scope_bits.append(f"rev={result.rev_id[:SHORT_REV_ID_LEN]}")
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
@click.argument("rev_id", type=str)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Destination file or directory (default: ./<short_rev_id>.pdf). Ignored with --host.",
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
    rev_id: str,
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
    rev_id = _backend_resolve(be, rev_id)

    if host:
        import time
        import webbrowser

        from datasheet_rag import pdf_viewer

        try:
            # Fetch via the backend (HTTP in remote mode) and prime the
            # viewer cache so the loopback server can serve it locally.
            pdf_viewer.prime_pdf_cache(rev_id, be.get_pdf_bytes(rev_id))
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc

        port = pdf_viewer.ensure_pdf_server()
        local_url = f"http://127.0.0.1:{port}/viewer/{rev_id}#page={page}"

        console.print("[green]PDF viewer running — pick whichever URL your browser can reach:[/]")
        for ip in _local_ips():
            console.print(f"  http://{ip}:{port}/viewer/{rev_id}#page={page}", soft_wrap=True)

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

    title = be.get_doc_titles().get(rev_id)

    try:
        data = be.get_pdf_bytes(rev_id)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    short_id = rev_id[:SHORT_REV_ID_LEN]
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
@click.argument("rev_id", type=str)
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
    "rev_id and page in the current directory. If a directory, the "
    "default filename is placed inside it.",
)
@click.option("--dpi", default=150, type=int, help="Render DPI for the page image.")
@_db_option
def get_page_cmd(
    rev_id: str,
    page_arg: int | None,
    page_opt: int | None,
    output_path: Path | None,
    dpi: int,
    db_path: Path | None,
) -> None:
    """Render a single PDF page to a PNG and save it to disk.

    PAGE may be given positionally (``rag get page <rev_id> 5``) or via
    ``--page`` (``rag get page <rev_id> --page 5``) — pass exactly one. This
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
    rev_id = _backend_resolve(be, rev_id)

    try:
        pdf_bytes = be.get_pdf_bytes(rev_id)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        images = convert_from_bytes(pdf_bytes, first_page=page, last_page=page, dpi=dpi)
    except Exception as exc:
        raise click.ClickException(f"Failed to render page {page}: {exc}") from exc
    if not images:
        short_id = rev_id[:SHORT_REV_ID_LEN]
        raise click.ClickException(f"Page {page} not found in document {short_id}.")

    default_name = f"{rev_id[:SHORT_REV_ID_LEN]}_p{page}.png"
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


@cli.command("delete", short_help="Permanently delete a revision or a document.")
@click.argument("rev_id", type=str)
@_db_option
@click.option(
    "--dry-run", is_flag=True, help="Show what would be deleted without deleting anything."
)
@click.option("-y", "--yes", "assume_yes", is_flag=True, help="Skip the confirmation prompt.")
def delete_doc_cmd(rev_id: str, db_path: Path | None, dry_run: bool, assume_yes: bool) -> None:
    """Permanently delete one revision (a rev_id) or a whole document (D12).

    Removes the chunks, vectors and revision metadata from the store, the
    local PDF and figure-crop files, cached pipeline artifacts, and (if S3
    is configured) the matching S3 objects. A document's own metadata
    (projects, tags, fields) goes with its last revision. There is no undo —
    a revision must be re-ingested from the source PDF to get it back.
    """
    be = _backend_for(db_path)
    targets = _rev_ids_for(be, rev_id, all_revisions=True)
    document = rev_id if parse_document_ref(rev_id) is not None else None
    titles = be.get_doc_titles()
    chunk_count = sum(be.count_chunks(rev_id=d) for d in targets)
    if document is not None:
        label = f"{document} ({len(targets)} revision(s))"
    else:
        did = targets[0]
        title = titles.get(did)
        label = f"{title!r} ({did[:SHORT_REV_ID_LEN]})" if title else did[:SHORT_REV_ID_LEN]
        meta = be.get_metadata(did)
        if meta is not None and len(meta.revisions) > 1:
            label += (
                f", one of {len(meta.revisions)} revisions of {meta.document_id}; "
                "the others are kept"
            )

    if dry_run:
        console.print(f"[yellow]--dry-run:[/] would delete {chunk_count} chunk(s) for {label}")
        for did in targets:
            console.print(
                "  Also removes: local PDF, figure-crop directory, cached pipeline "
                "artifacts, and any S3 objects under "
                f"[cyan]{get_settings().s3_pdf_prefix}{did}/[/] and "
                f"[cyan]figures/{did}/[/] (if S3 is configured)."
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

    deleted = sum(be.delete_doc(did) for did in targets)

    console.print(f"[green]Deleted[/] {deleted} chunk(s) for {label}.")


# ---------------------------------------------------------------------------
# revisions (GH #124)
# ---------------------------------------------------------------------------


def _rev_ids_for(be: RagBackend, ref: str, *, all_revisions: bool) -> list[str]:
    """``RagBackend.rev_ids_for``, a miss turned into a ClickException."""
    try:
        return be.rev_ids_for(ref, all_revisions=all_revisions)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _list_revisions(be: RagBackend, ref: str) -> DocumentRevisions:
    try:
        return be.list_revisions(ref)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _print_revisions(revs: DocumentRevisions) -> None:
    table = Table(title=f"{revs.document_id} · {len(revs.revisions)} revision(s), newest first")
    for name in ("", "rev_id", "revision", "ingested", "chunks", "title"):
        table.add_column(
            name,
            style="cyan" if name == "rev_id" else None,
            justify="right" if name == "chunks" else "left",
        )
    marked = False
    for r in revs.revisions:
        doc = r.rev_id[:SHORT_REV_ID_LEN]
        if r.overrides:
            doc += "*"
            marked = True
        table.add_row(
            "newest" if r.newest else "",
            doc,
            _fmt_list_cell(r.revision),
            _fmt_list_cell(r.ingested_at),
            str(r.chunk_count),
            _fmt_list_cell(r.doc_title),
        )
    console.print(table)
    if marked:
        console.print(
            "  * has values set on this revision only; `rag metadata <rev_id>` shows them"
        )


class _RevisionsGroup(click.Group):
    """`rag revisions D12` lists; `rag revisions <command> ...` runs a command."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["list", *args]
        return super().parse_args(ctx, args)


@cli.group(
    "revisions",
    cls=_RevisionsGroup,
    short_help="List and arrange the revisions of a document.",
)
def revisions_group() -> None:
    """List and arrange the revisions of a document.

    A document (D12) groups the revisions of one datasheet; each revision is
    one PDF with its own rev_id. Metadata belongs to the document. Search and
    `rag list` show the newest revision unless asked for all. The order is
    the one you set; revision labels ("Rev C") are never parsed:

    \b
        rag revisions D12                       # list, newest first
        rag ingest new.pdf --new-revision-of D12
        rag revisions add D12 <rev_id>          # an already stored PDF on top
        rag revisions move <rev_id> --below <rev_id>
        rag revisions unlink <rev_id>           # into a document of its own
        rag revisions rule set "intel-id OR (acme-pn AND doc_type)"
    """


@revisions_group.command("list", short_help="List a document's revisions.")
@click.argument("ref", type=str)
@_db_option
def revisions_list_cmd(ref: str, db_path: Path | None) -> None:
    """List the revisions of REF (a D12 or any revision's rev_id), newest first."""
    _print_revisions(_list_revisions(_backend_for(db_path), ref))


@revisions_group.command("add", short_help="Make a stored PDF the newest revision of a document.")
@click.argument("document", type=str)
@click.argument("rev_id", type=str)
@_db_option
def revisions_add_cmd(document: str, rev_id: str, db_path: Path | None) -> None:
    """Put stored PDF REV_ID on top of DOCUMENT's revisions, as its newest.

    DOCUMENT is a D12 or any of its revisions' rev_ids. A PDF that is alone
    in its own document is merged in: projects and tags are joined, and the
    two documents must not hold different values for a field. A PDF that is
    one of several revisions of another document must be unlinked first.
    To ingest a new PDF as a revision, use `rag ingest --new-revision-of`.
    """
    be = _backend_for(db_path)
    try:
        revs = be.add_revision(document, _backend_resolve(be, rev_id))
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    console.print(
        f"[green]{rev_id[:SHORT_REV_ID_LEN]} is the newest revision of[/] {revs.document_id}"
    )
    _print_revisions(revs)


@revisions_group.command("move", short_help="Change a revision's place in its document.")
@click.argument("rev_id", type=str)
@click.option(
    "--above",
    default=None,
    metavar="REV_ID",
    help="Put it directly above (newer than) this revision.",
)
@click.option(
    "--below",
    default=None,
    metavar="REV_ID",
    help="Put it directly below (older than) this revision.",
)
@_db_option
def revisions_move_cmd(
    rev_id: str, above: str | None, below: str | None, db_path: Path | None
) -> None:
    """Move revision REV_ID directly above or below another revision of the
    same document. The top one is the newest: the one search uses."""
    if (above is None) == (below is None):
        raise click.UsageError("give exactly one of --above / --below")
    be = _backend_for(db_path)
    try:
        revs = be.move_revision(
            _backend_resolve(be, rev_id),
            above=_backend_resolve(be, above) if above else None,
            below=_backend_resolve(be, below) if below else None,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    _print_revisions(revs)


@revisions_group.command("unlink", short_help="Take a revision out into a document of its own.")
@click.argument("rev_id", type=str)
@_db_option
def revisions_unlink_cmd(rev_id: str, db_path: Path | None) -> None:
    """Take revision REV_ID out of its document into a new one.

    The new document starts with a copy of the old one's projects, tags and
    field values, so the PDF stays where it was in every listing and filter.
    """
    be = _backend_for(db_path)
    full = _backend_resolve(be, rev_id)
    try:
        before = be.get_metadata(full)
        revs = be.unlink_revision(full)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    old = before.document_id if before is not None else "its document"
    console.print(
        f"[green]{full[:SHORT_REV_ID_LEN]} is now[/] {revs.document_id} "
        f"[green](was a revision of[/] {old}[green])[/]"
    )
    _print_revisions(revs)


@revisions_group.group("rule", short_help="Show or set the identity rule.")
def revisions_rule_group() -> None:
    """The identity rule links a newly ingested PDF to a stored document.

    It is an expression over registered fields, with AND, OR and
    parentheses (AND binds tighter):

        rag revisions rule set "intel-id OR ti-id OR (acme-pn AND doc_type)"

    When a PDF is ingested without --new-revision-of, the rule is checked
    against every document using the field values the ingest sets. A field
    matches only when both have it and the values are equal; for a list
    field, equal as a set. One match: the PDF becomes that document's newest
    revision. None: a new document. More than one: the ingest stops and
    names them. Without a rule nothing links on its own.
    """


@revisions_rule_group.command("show")
@_db_option
def revisions_rule_show_cmd(db_path: Path | None) -> None:
    """Print the identity rule."""
    rule = _backend_for(db_path).get_identity_rule()
    if rule is None:
        console.print(
            "No identity rule: revisions link only by hand (--new-revision-of, rag revisions add)."
        )
    else:
        console.print(rule)


@revisions_rule_group.command("set")
@click.argument("expression", type=str)
@_db_option
def revisions_rule_set_cmd(expression: str, db_path: Path | None) -> None:
    """Set the identity rule to EXPRESSION."""
    try:
        rule = _backend_for(db_path).set_identity_rule(expression)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    console.print(f"[green]Identity rule set:[/] {rule}")
    console.print(
        "  It applies to PDFs ingested from now on. Documents already stored are "
        "not relinked; arrange those with rag revisions add / unlink."
    )


@revisions_rule_group.command("clear")
@_db_option
def revisions_rule_clear_cmd(db_path: Path | None) -> None:
    """Remove the identity rule. Stored links are kept."""
    _backend_for(db_path).set_identity_rule(None)
    console.print("[green]Identity rule cleared.[/] Revisions now link only by hand.")


@get_group.command("text", short_help="Dump a document's extracted text.")
@click.argument("rev_id", type=str)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the text to this file (default: print it to stdout).",
)
def get_text_cmd(rev_id: str, output: Path | None) -> None:
    """Dump a document's extracted text in reading order.

    REV_ID is a rev_id (full hash or unambiguous prefix), resolved against
    whichever layout artifact the ingest backend cached — Docling's
    ``{rev_id}_outline.json`` or Textract's ``{rev_id}_blocks.json``. An
    explicit path to either is also accepted in place of a rev_id.

    Section titles are emitted as headings so the dump keeps the document's
    structure rather than running together as one wall of prose.
    """
    _, _, outline = _load_outline(rev_id)

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
    ``{rev_id}_blocks.json`` (raw blocks) and Docling writes
    ``{rev_id}_outline.json`` (a serialized `DocumentOutline`). *arg* is either
    an explicit path to one of those, or a rev_id (full hash or unambiguous
    prefix) resolved against the cache — preferring the Docling outline when
    both exist, since Docling is the default backend.

    Returns ``(rev_id, backend, path)`` where *backend* is ``"docling"`` or
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
        if sorted(settings.output_dir.glob(f"{glob.escape(arg)}*{suffix}")):
            rev_id = _resolve_cached_rev_id(arg, suffix)
            return rev_id, backend, _cache_path(rev_id, suffix)

    raise click.ClickException(
        f"No cached layout artifact matching rev_id {arg!r} in "
        f"{settings.output_dir}. Run `rag ingest` for this document first "
        "(Docling writes '_outline.json', Textract writes '_blocks.json')."
    )


def _load_outline(arg: str) -> tuple[str, str, DocumentOutline]:
    """Load a cached layout artifact as a `DocumentOutline`, whichever backend
    produced it. Returns ``(rev_id, backend, outline)``.

    `DocumentOutline` is the backend-neutral representation the whole chunking
    pipeline already runs on — Docling builds one directly, and Textract blocks
    are converted by `parse_textract_blocks`. Resolving to it here is what lets
    the post-ingest commands behave identically on both backends instead of
    only working on whichever one happened to write their input file.
    """
    from datasheet_rag.chunking.layout_parser import DocumentOutline, parse_textract_blocks

    rev_id, backend, path = _resolve_layout_input(arg)
    data = cache_files.load(path)

    if backend == "docling":
        # The cache wraps the outline alongside its figure regions.
        return rev_id, backend, DocumentOutline.from_dict(data["outline"])
    return rev_id, backend, parse_textract_blocks(data, rev_id=rev_id)


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
@click.argument("rev_id", type=str)
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Show the entire section/layout listing instead of the first "
    f"{_INSPECT_LAYOUT_PREVIEW} rows.",
)
def inspect_layout(rev_id: str, full: bool) -> None:
    """Show a summary of the parsed layout structure for a document.

    Works with either ingestion backend: the cached Docling
    ``{rev_id}_outline.json`` (element types + section hierarchy) or the
    Textract ``{rev_id}_blocks.json`` (block types + layout blocks). REV_ID is
    a rev_id (full hash or unambiguous prefix) resolved against the cache, or an
    explicit path to one of those artifacts.
    """
    resolved_id, backend, path = _resolve_layout_input(rev_id)
    data = cache_files.load(path)

    console.print(f"rev_id [cyan]{resolved_id}[/] — [cyan]{backend}[/] backend")

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
@click.argument("rev_id", type=str)
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
    help="Where to write the chunk graph (default: the cached {rev_id}_chunks.json).",
)
def chunk_cmd(
    rev_id: str,
    figures_manifest: Path | None,
    micro_tokens: int,
    meso_tokens: int,
    summarizer: str,
    output: Path | None,
) -> None:
    """Re-run multi-scale chunking against a document's cached layout.

    REV_ID is a rev_id (full hash or unambiguous prefix), resolved against
    whichever layout artifact the ingest backend cached — Docling's
    ``{rev_id}_outline.json`` or Textract's ``{rev_id}_blocks.json``. An
    explicit path to either is also accepted in place of a rev_id. Chunks are
    written to ``{rev_id}_chunks.json`` unless --output is given.

    This is the cheap way to retune --micro-tokens / --meso-tokens: it reuses
    the cached layout analysis, where `rag ingest --force` would redo it.
    Follow with `rag repair embeddings <rev-id>` to re-index the result.

    Produces a hierarchical chunk graph at three levels (MACRO/MESO/MICRO)
    with navigation links, context enrichment, and chapter summaries.
    """
    from datasheet_rag.chunking.pipeline import (
        run_chunking_pipeline_from_outline,
        save_chunk_graph,
    )
    from datasheet_rag.chunking.splitter import SplitterConfig

    rev_id, backend, outline = _load_outline(rev_id)

    # Default to the manifest ingest wrote for this document — without it a
    # re-chunk would silently drop every figure from the graph.
    if figures_manifest is None:
        cached_manifest = get_settings().figures_dir / rev_id / "manifest.json"
        if cached_manifest.is_file():
            figures_manifest = cached_manifest

    figure_manifest = None
    if figures_manifest:
        figure_manifest = cache_files.load(Path(figures_manifest))

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
    # Only a Docling outline has title hints; Textract has none to report.
    graph.title_hints = outline.title_hints() if backend == "docling" else None

    # Save
    settings = get_settings()
    if output is None:
        output = settings.output_dir / f"{rev_id}_chunks.json"

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
    if output == settings.output_dir / f"{rev_id}_chunks.json":
        _next_steps(rev_id, rechunk=False)


# ---------------------------------------------------------------------------
# Embed (Bedrock Titan + SQLite store)
# ---------------------------------------------------------------------------


@repair_group.command("embeddings", short_help="Re-embed a document into the store.")
@click.argument("rev_id", type=str)
@_db_option
@click.option(
    "--project-id",
    default=None,
    help="Add the document to this project. Projects it is already in are kept.",
)
@click.option("--verbose/--quiet", default=True, help="Print per-batch progress.")
def embed(
    rev_id: str,
    db_path: Path | None,
    project_id: str | None,
    verbose: bool,
) -> None:
    """Embed a document's chunk graph and store it.

    REV_ID is a rev_id (full hash or unambiguous prefix); the chunk graph is
    read from the cached ``{rev_id}_chunks.json``. An explicit chunks-JSON path
    is also accepted in place of a rev_id.

    Embedding + insert run through the backend, so this writes to the remote
    server when RAG_SERVER_URL is set, or the local sqlite store otherwise.
    The embedding model itself runs on the server unless RAG_COMPUTE=client,
    in which case it runs here and only the vectors are uploaded. There is no
    dry run — use `rag ingest --show-cost` to price a run without writing.
    """
    from datasheet_rag.backend import MetadataPatch, backend_mode, compute_mode
    from datasheet_rag.chunking.pipeline import load_chunk_graph
    from datasheet_rag.ingest_pipeline import collect_figure_uploads
    from datasheet_rag.project_config import get_project_config

    _, chunks_json = _doc_input(rev_id, "_chunks.json")

    proj_cfg = get_project_config()
    if proj_cfg is not None:
        project_id = project_id or proj_cfg.project_id

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
    local_pdf = get_settings().pdf_dir / f"{graph.rev_id}.pdf"

    # Remote: ship figure images so the server stores them and the embedded
    # context_text can fold in any server-side descriptions.
    figures_upload: dict[str, tuple[bytes, str]] | None = None
    if backend_mode() == "remote" and db_path is None:
        figures_upload, missing = collect_figure_uploads(graph)
        _warn_missing_figure_crops(missing, len(figures_upload))

    console.print(
        "Embedding & writing via the backend "
        f"(the embedding model runs {'here' if compute_mode() == 'client' else 'on the server'})…"
    )
    result = be.ingest_chunk_graph(
        graph,
        figures=figures_upload,
        project_id=project_id,
        metadata=MetadataPatch(),
        embed=True,
        describe_figures=False,
        # The parse that produced this cached graph ran here, so the PDF
        # is in this machine's pdf_dir and a remote store has never seen
        # it. Only offered if it is actually there — a graph can outlive
        # the file it came from (GH #43).
        source_pdf=local_pdf if local_pdf.is_file() else None,
    )
    console.print(f"[green]Inserted[/] {result.inserted} chunks.")
    _report_pruned(result.pruned, indent="")
    # Vectors now match the stored text again.
    _set_stale(result.rev_id or rev_id, False)


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
@click.option(
    "--subsystem",
    default=None,
    help="Restrict to documents with this subsystem in the searched project "
    "(in any project with --global).",
)
@click.option(
    "--rev-id",
    "rev_ids",
    multiple=True,
    help="Restrict to one or more revisions (rev_ids) or documents (D12: its "
    "newest revision, or every one with --all-revisions).",
)
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
@click.option(
    "--all-revisions",
    "-a",
    "all_revisions",
    is_flag=True,
    help="Search every revision of each document, not just the newest.",
)
def search(
    query: str,
    mode: SearchMode,
    top_k: int,
    db_path: Path | None,
    project_id: str | None,
    is_global: bool,
    subsystem: str | None,
    rev_ids: tuple[str, ...],
    level: str | None,
    show_context: bool,
    all_revisions: bool,
) -> None:
    """Search the RAG store (local sqlite or remote server) with hybrid /
    vector / keyword retrieval. The query is embedded by the backend.

    Only the newest revision of each document is searched unless you pass
    --all-revisions (see `rag revisions`)."""
    from datasheet_rag.backend import SEARCH_POOL
    from datasheet_rag.models.chunk import ChunkLevel

    if not 1 <= top_k <= SEARCH_POOL:
        raise click.BadParameter(f"must be 1 to {SEARCH_POOL}", param_hint="-k")
    from datasheet_rag.project_config import resolve_cli_project_id
    from datasheet_rag.store import SearchFilters

    project_id = resolve_cli_project_id(project_id, is_global=is_global)
    be = _backend_for(db_path)

    resolved_rev_ids: list[str] = []
    for d in rev_ids:
        resolved_rev_ids.extend(_rev_ids_for(be, d, all_revisions=all_revisions))

    level_enum = None
    if level:
        level_enum = {
            "macro": ChunkLevel.MACRO,
            "meso": ChunkLevel.MESO,
            "micro": ChunkLevel.MICRO,
        }[level]

    filters = SearchFilters(
        rev_ids=resolved_rev_ids if resolved_rev_ids else None,
        all_revisions=all_revisions,
        project_id=project_id,
        fields={"subsystem": subsystem} if subsystem is not None else None,
        level=level_enum,
    )

    results = be.search(query, mode=mode, k=top_k, filters=filters)

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
            _short_chunk_id(r.chunk.id, r.chunk.rev_id),
            (r.chunk.metadata.section_title or r.chunk.metadata.chapter_title or "")[:40],
            preview,
        )

    console.print(table)
    console.print("[dim]Fetch a full chunk with: rag get chunk <chunk_id>[/]")


def _set_stale(rev_id: str | None, stale: bool) -> None:
    """Best-effort flip of the stale marker on *rev_id*'s sidecar row.

    The marker (``needs_reembed``) says the document's stored text has moved
    on from its vectors. Set by the repairs that change embedded text, cleared
    by `rag repair embeddings`, and surfaced as a column in `rag list`.

    Deliberately swallows every failure: a repair that already succeeded must
    not be reported as failed because a bookkeeping write didn't land, and
    these commands legitimately run against documents that were parsed but
    never ingested, which have no sidecar row to write to.
    """
    if not rev_id:
        return
    try:
        be = _backend_for()
        be.set_needs_reembed(be.resolve_rev_id(rev_id), stale)
    except Exception:
        pass


def _next_steps(rev_id: str | None, *, rechunk: bool) -> None:
    """Print what still has to happen before a repair shows up in `rag search`.

    Every `repair` subcommand edits something upstream of the vectors, so none
    of them take effect on their own. Repairs that patch the cached layout
    outline need the chunk graph rebuilt first; repairs that edit chunk text
    in place only need re-embedding.
    """
    _set_stale(rev_id, True)
    target = rev_id[:SHORT_REV_ID_LEN] if rev_id else "<rev-id>"
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
    from datasheet_rag.models.chunk import IMAGE_LAYOUT_TYPES

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

    if chunk.metadata.layout_type in IMAGE_LAYOUT_TYPES:
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

    CHUNK_ID accepts the full id or an abbreviated form using a rev_id
    prefix, e.g. ``ab12cd34ef56:L2:143`` as printed by `rag search` /
    `rag inspect figures`.
    """
    be = _backend_for(db_path)
    full_id = _resolve_chunk_id(be, chunk_id)

    chunk = be.get_chunk(full_id)
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
            nchunk = be.get_chunk(nid)
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
    docs = {c.rev_id for c in unusable}
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


@inspect_group.command("figures", short_help="List figure and formula chunks in the store.")
@click.option("--rev-id", default=None, help="Restrict to one revision (rev_id).")
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
    rev_id: str | None,
    project_id: str | None,
    is_global: bool,
    db_path: Path | None,
    missing_description_only: bool,
    include_unusable: bool,
) -> None:
    """List figure and formula chunks in the store.

    Shows the ones with a usable image by default. Pass --include-unusable to
    see figure chunks whose image is missing — search will not offer those,
    and `rag repair figure-links` may be able to reattach their crops.
    """
    from datasheet_rag.project_config import resolve_cli_project_id

    project_id = resolve_cli_project_id(project_id, is_global=is_global)

    be = _backend_for(db_path)
    if rev_id:
        rev_id = _backend_resolve(be, rev_id)
    figs = be.list_figure_chunks(rev_id=rev_id, project_id=project_id, only_with_image=False)
    unusable = [c for c in figs if not c.figure_available]
    if not include_unusable:
        figs = [c for c in figs if c.figure_available]
    if missing_description_only:
        figs = [c for c in figs if not c.figure_description]

    if not figs:
        console.print("[yellow]No figure chunks match.[/]")
        _warn_unusable_figures(unusable, include_unusable)
        return

    table = Table(title=f"Figure and formula chunks ({len(figs)})")
    table.add_column("chunk_id", style="cyan", no_wrap=True)
    table.add_column("kind")
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
            _short_chunk_id(c.id, c.rev_id),
            c.metadata.layout_type.value,
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

    CHUNK_ID accepts the full id or an abbreviated form using a rev_id
    prefix, e.g. ``ab12cd34ef56:L2:143`` as printed by `rag inspect figures` /
    `rag search`. This is the CLI equivalent of the MCP `get_figure` tool.
    """
    be = _backend_for(db_path)
    full_id = _resolve_chunk_id(be, chunk_id)

    try:
        fig = be.get_figure_bytes(full_id)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e

    default_name = _short_chunk_id(fig.chunk_id, fig.rev_id).replace(":", "_") + f".{fig.format}"
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
    loc_bits = [f"rev={citation.rev_id[:SHORT_REV_ID_LEN]}"]
    if citation.page:
        loc_bits.append(f"page={citation.page}")
    if citation.section:
        loc_bits.append(f"section={citation.section}")
    console.print(f"  [dim]{' '.join(loc_bits)}[/]")


# ---------------------------------------------------------------------------
# describe-figures (Bedrock Claude vision → figure_description)
# ---------------------------------------------------------------------------


@repair_group.command("figures", short_help="Describe figures and formulas with a vision LLM.")
@click.option("--rev-id", default=None, help="Restrict to one revision (rev_id).")
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
    rev_id: str | None,
    project_id: str | None,
    missing_only: bool,
    limit: int | None,
    model_id: str | None,
    dry_run: bool,
    db_path: Path | None,
    verbose: bool,
) -> None:
    """Generate vision-LLM descriptions for figure and formula chunks.

    Walks the figure and formula chunks (optionally filtered by
    doc/project, skipping those that already have a description), sends
    each image + caption + neighbour text to Bedrock Claude vision, and
    folds the response into the chunk row + context_text. A figure gets a
    short prose description; a formula gets a LaTeX transcription and one
    sentence on what it computes.

    Descriptions are folded into the existing chunk rows, so no re-chunk is
    needed — just re-embed the affected document so they show up in vector
    search:

    \b
        rag repair figures --rev-id <doc>
        rag repair embeddings <doc>
    """
    be = _backend_for(db_path)
    if rev_id:
        rev_id = _backend_resolve(be, rev_id)

    from datasheet_rag.backend import compute_mode

    where = "on this machine" if compute_mode() == "client" else "on the server"
    console.print(f"Describing figures via the backend (the vision model runs {where})…")
    descriptions, s = be.describe_figures(
        rev_id=rev_id,
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
        _next_steps(rev_id, rechunk=False)


# ---------------------------------------------------------------------------
# repair figure-links (reattach cropped images to figure chunks)
# ---------------------------------------------------------------------------


def _relink_plan(
    conn: sqlite3.Connection, rev_id: str
) -> tuple[list[tuple[str, Path, str]], list[str]]:
    """Match a document's cropped figures on disk to its figure chunks.

    Returns ``(plan, notes)``: ``plan`` is ``(chunk_id, image_path, caption)``
    for each source-less figure chunk that a crop can be reattached to, and
    ``notes`` explains every page that was left alone.

    Matching is per page and per kind (figure or formula), in reading order:
    the manifest lists crops in the order they were cut from the page and the
    chunks were inserted in the same order, so an equal count on a page pairs
    them off unambiguously. A page whose counts disagree (a crop dropped for
    being logo-sized, say) is skipped rather than guessed at, and so is any
    pair whose captions are both present and different.
    """

    from datasheet_rag.models.chunk import IMAGE_LAYOUT_TYPES, ChunkLevel, LayoutType
    from datasheet_rag.store import resolve_figure_path

    settings = get_settings()
    image_types = sorted(t.value for t in IMAGE_LAYOUT_TYPES)
    type_marks = ",".join("?" * len(image_types))
    orphans = conn.execute(
        f"SELECT COUNT(*) AS n FROM chunks WHERE rev_id = ? AND layout_type IN ({type_marks}) "
        "AND level = ? AND COALESCE(figure_image_path, '') = '' "
        "AND COALESCE(figure_s3_key, '') = ''",
        (rev_id, *image_types, int(ChunkLevel.MICRO)),
    ).fetchone()["n"]
    if not orphans:
        # Coarser levels inherit their image from the MICRO chunk they wrap,
        # which the chunker now does at ingest — a re-chunk carries it up.
        return [], []

    manifest_path = settings.figures_dir / rev_id / "manifest.json"
    if not manifest_path.is_file():
        return [], [f"no manifest at {manifest_path} — re-ingest the PDF to re-crop"]

    manifest = cache_files.load(manifest_path)
    # Keyed by (layout_type, page): a formula crop only pairs with a formula
    # chunk, so a page holding both kinds still lines up.
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for fig in manifest.get("figures", []):
        crop_kind = (
            LayoutType.FORMULA if "formula" in (fig.get("block_id") or "") else LayoutType.FIGURE
        )
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
        by_page.setdefault((crop_kind.value, int(fig.get("page") or 1)), []).append(
            {**fig, "_path": path}
        )

    chunks_by_page: dict[tuple[str, int], list[Any]] = {}
    for row in conn.execute(
        f"SELECT * FROM chunks WHERE rev_id = ? AND layout_type IN ({type_marks}) "
        "AND level = ? ORDER BY rowid",
        (rev_id, *image_types, int(ChunkLevel.MICRO)),
    ).fetchall():
        pages = json.loads(row["page_numbers"] or "[]")
        key = (row["layout_type"], pages[0] if pages else 1)
        chunks_by_page.setdefault(key, []).append(row)

    plan: list[tuple[str, Path, str]] = []
    notes: list[str] = []
    for kind, page in sorted(set(by_page) | set(chunks_by_page), key=lambda k: (k[1], k[0])):
        crops = by_page.get((kind, page), [])
        rows = chunks_by_page.get((kind, page), [])
        # Only pages that actually need repair are worth reporting on.
        if not any(not (r["figure_image_path"] or r["figure_s3_key"]) for r in rows):
            continue
        if len(crops) != len(rows):
            notes.append(
                f"page {page}: {len(crops)} {kind} crop(s) vs {len(rows)} {kind} chunk(s) "
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
@click.option("--rev-id", default=None, help="Restrict to one revision (rev_id).")
@click.option(
    "--apply", "do_apply", is_flag=True, help="Write the links (default: report what would change)."
)
@_db_option
def relink_figures_cmd(rev_id: str | None, do_apply: bool, db_path: Path | None) -> None:
    """Reattach cropped figure images to figure chunks that lost their link.

    Formula chunks are repaired the same way, from the formula crops.

    A document ingested with `--skip-figures`, or restored from a backup
    written before figure links existed, leaves chunks that search can find
    and `show_figure` cannot serve. When the crops are still on disk under
    `figures_dir/<rev_id>/` with their manifest, this pairs them back up per
    page in reading order and writes the link.

    Runs against a local store only — it reads the figures directory and
    writes chunk rows directly. On a server deployment, run it there (or
    against the mounted database file with --db).

    \b
        rag repair figure-links                 # report across every document
        rag repair figure-links --rev-id ab12 --apply

    Documents with no crops on disk cannot be repaired this way — re-ingest
    the PDF (without --skip-figures) instead.
    """
    from datasheet_rag.models.chunk import IMAGE_LAYOUT_TYPES
    from datasheet_rag.store import connect, set_figure_source

    path = _require_local_db(db_path)
    conn = connect(path)

    if rev_id:
        rev_id = _resolve_rev_id(conn, rev_id)
        rev_ids = [rev_id]
    else:
        image_types = sorted(t.value for t in IMAGE_LAYOUT_TYPES)
        rev_ids = [
            r["rev_id"]
            for r in conn.execute(
                "SELECT DISTINCT rev_id FROM chunks "
                f"WHERE layout_type IN ({','.join('?' * len(image_types))}) "
                "AND COALESCE(figure_image_path, '') = '' "
                "AND COALESCE(figure_s3_key, '') = '' ORDER BY rev_id",
                image_types,
            ).fetchall()
        ]

    if not rev_ids:
        console.print("[green]Every figure chunk already has an image source.[/]")
        return

    total_linked = 0
    quiet = 0
    for did in rev_ids:
        plan, notes = _relink_plan(conn, did)
        head = f"[cyan]{did[:SHORT_REV_ID_LEN]}[/]"
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


@repair_group.command("caches", short_help="Update cached files to the rev_id name.")
@click.option("--apply", is_flag=True, help="Rewrite the files. Without it, only list them.")
def repair_caches_cmd(apply: bool) -> None:
    """Rewrite cached pipeline files that still say `doc_id` (GH #124).

    The chunk graphs and layouts under the output directory and the figure
    manifests under the figures directory name a revision by id. Files
    written before the id was renamed `rev_id` say `doc_id`, and every
    command that reads one stops and points here. This renames the key in
    place. It reads no PDFs and touches no store; it is safe to run again.

    Caches live on the machine that ran the pipeline, so run it there: on a
    client that computes locally as well as on the server.

    \b
        rag repair caches           # list the files that need it
        rag repair caches --apply   # rewrite them
    """
    settings = get_settings()
    stale = cache_files.repair_caches(settings.output_dir, settings.figures_dir, apply=apply)
    if not stale:
        console.print("[green]All cached files use rev_id[/] — nothing to do.")
        return
    for path in stale:
        console.print(f"  {path}")
    if apply:
        console.print(f"[green]Rewrote {len(stale)} cached file(s)[/] to use rev_id.")
    else:
        console.print(
            f"[yellow]{len(stale)} cached file(s) still say doc_id.[/] "
            "Run [cyan]rag repair caches --apply[/] to rewrite them."
        )


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
@click.option("--rev-id", default=None, help="Explicit document ID (default: content hash).")
@click.option(
    "--project-id",
    default=None,
    help="Add the document to this project (default: from .rag.toml). "
    "Projects it is already in are kept; leave one with "
    "`rag metadata <doc> --remove-project`.",
)
@click.option(
    "--mpn",
    multiple=True,
    help="Manufacturer part number, e.g. STM32H743VIT6. Repeat for several.",
)
@click.option("--manufacturer", default=None, help="Manufacturer name, e.g. STMicroelectronics.")
@click.option(
    "--subsystem",
    default=None,
    help="The document's subsystem in the ingest project, e.g. power, rf, mcu. "
    "Needs a project (--project-id or .rag.toml).",
)
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
    "`rag metadata <rev-id> --tag ...`; for name=value metadata "
    "use --field.",
)
@click.option(
    "--field",
    "field_items",
    multiple=True,
    metavar="NAME=VALUE",
    help="Value of a registered field (see `rag fields list`), repeatable; "
    "repeat a list field to give several values. Merged over any [fields] "
    "from .rag.toml, field by field.",
)
@click.option(
    "--new-revision-of",
    "new_revision_of",
    default=None,
    metavar="DOC",
    help="Make the PDF the newest revision of this document (a D12 or any of "
    "its revisions' rev_ids). Without it the identity rule decides, if the "
    "store has one (`rag revisions rule`); else the PDF is a new document.",
)
@click.option(
    "--revision",
    "revision_label",
    default=None,
    metavar="LABEL",
    help="The PDF's revision label, e.g. 'Rev C' (free text, never parsed).",
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
        "by --dry-run and --show-cost (both are client-side estimation) and by "
        "RAG_COMPUTE=client. On its own it moves only the parse — embedding, "
        "figure description and titling still run on the server; use "
        "--compute client to move those too."
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
    rev_id: str | None,
    project_id: str | None,
    mpn: tuple[str, ...],
    manufacturer: str | None,
    subsystem: str | None,
    doc_type: str | None,
    tags: tuple[str, ...],
    field_items: tuple[str, ...],
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
    new_revision_of: str | None,
    revision_label: str | None,
) -> None:
    """Full ingestion pipeline: parse → figures → chunk → embed.

    A new PDF becomes a revision of a document (D12): of the one named by
    --new-revision-of, else of the one the identity rule matches, else of a
    new document of its own. The output says which. Metadata given here is
    set on that document, so it applies to all its revisions.

    Defaults to Docling (free, fast, handles tables/formulas/figures on
    native PDFs) and fails verbosely on scanned PDFs rather than silently
    incurring AWS Textract OCR costs. Pass --backend textract to OCR a
    scanned PDF, or --backend auto to route automatically between the two.
    Intermediate artefacts (blocks/chunk graph) are cached in output_dir;
    pass --force to ignore the cache and redo all steps.

    PDF_PATH may also be a directory, in which case every *.pdf found
    under it (recursively) is ingested one by one with the same options.
    Documents already in the store under their content-hash rev_id are
    skipped unless --force is given, and --show-cost prints a combined
    estimate across all of them in addition to each document's breakdown.
    """
    # A store write follows unless this is a dry run. Built once, so a bulk
    # ingest shares one backend and one read of the field registry.
    store_backend = None if dry_run or show_cost else _backend_for(db_path)
    common: dict[str, Any] = dict(
        store_backend=store_backend,
        field_registry=_FieldRegistry(store_backend) if store_backend is not None else None,
        project_id=project_id,
        tags=tags,
        fields=_merge_shortcuts(
            _parse_field_items(field_items),
            # Empty shortcut values were never a "clear" on ingest (there is
            # nothing to clear yet), so drop them rather than send "".
            _shortcut_fields(
                mpn=tuple(m for m in mpn if m),
                manufacturer=manufacturer or None,
                doc_type=doc_type or None,
                subsystem=subsystem or None,
            ),
        ),
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
        new_revision_of=new_revision_of,
        revision_label=revision_label,
    )

    if not pdf_path.is_dir():
        _ingest_one(pdf_path, rev_id=rev_id, **common)
        return

    if rev_id:
        raise click.ClickException(
            "--rev-id can't be used with a directory — each PDF gets its own "
            "content-hash rev_id. Drop --rev-id for bulk ingestion."
        )
    if new_revision_of is not None or revision_label is not None:
        raise click.ClickException(
            "--new-revision-of and --revision name one PDF's place; ingest the "
            "revisions one file at a time, oldest first."
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
            ingested_ids = {
                d.rev_id for d in (store_backend or _backend_for(db_path)).get_ingested_docs()
            }
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
                f"  [dim]Already ingested and up to date (rev_id={did}) — "
                "skipping. Pass --force to re-ingest.[/]"
            )
            skipped += 1
            continue
        try:
            result = _ingest_one(pdf, rev_id=None, **common)
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
    rev_id: str | None,
    store_backend: RagBackend | None,
    field_registry: _FieldRegistry | None,
    project_id: str | None,
    tags: tuple[str, ...],
    fields: dict[str, Any],
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
    new_revision_of: str | None = None,
    revision_label: str | None = None,
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
    from datasheet_rag.project_config import ProjectConfigError, get_project_config_for

    # Raised as a ClickException here, not left to the root group, so a bulk
    # ingest records the one bad directory as a failed document and goes on.
    try:
        proj_cfg = get_project_config_for(pdf_path.parent)
    except ProjectConfigError as e:
        raise click.ClickException(str(e)) from e
    # A blank --project-id is no project (the store treats it so): the
    # .rag.toml project_id still applies.
    project_id = (project_id.strip() or None) if project_id else None
    if proj_cfg is not None:
        project_id = project_id or proj_cfg.project_id
        if not tags and proj_cfg.tags:
            tags = tuple(proj_cfg.tags)

    if show_cost:
        dry_run = True
        console.print(
            "[yellow]--show-cost: estimating AWS spend, skipping priced Bedrock/Textract calls.[/]"
        )

    # A store write follows (store_backend is None on a dry run): read the
    # field registry now, so a bad field fails before the PDF is parsed, not
    # after. A patch with no fields has nothing to check. Project-scoped
    # fields (subsystem) are written in the ingest project; the config's
    # only when that is the config's project.
    backend_obj = store_backend
    if proj_cfg is not None and proj_cfg.fields:
        from datasheet_rag.store.fields import scopes_of

        fields = _merge_config_fields(
            proj_cfg,
            fields,
            scopes=scopes_of(field_registry.specs()) if field_registry is not None else None,
            in_config_project=proj_cfg.project_id in (None, project_id),
            why_not=f"this ingest goes to project {project_id!r}",
        )
    if field_registry is not None and fields:
        _check_field_patch(field_registry.specs(), fields, project=project_id)

    from datasheet_rag.backend import MetadataPatch

    meta_patch = MetadataPatch(
        tags=list(tags) if tags else None,
        fields=fields or None,
        revision=revision_label,
    )
    # Which document the PDF goes into, decided now so an unknown
    # --new-revision-of or an ambiguous rule match fails before the parse.
    if backend_obj is not None:
        from datasheet_rag.docling_parser import content_hash

        try:
            planned = backend_obj.plan_ingest_link(
                rev_id=rev_id or content_hash(pdf_path),
                metadata=meta_patch,
                new_revision_of=new_revision_of,
            )
        except ValueError as e:
            raise click.ClickException(str(e)) from e
        if planned.action in ("explicit", "rule"):
            console.print(f"  [cyan]Revision:[/] {planned.describe()}")

    settings = get_settings()
    if accurate_tables is None:
        accurate_tables = settings.table_structure_mode == "accurate"
    t0 = time.monotonic()
    cost = CostEstimate()

    from datasheet_rag.backend import backend_mode, compute_mode
    from datasheet_rag.ingest_pipeline import (
        OcrRequiredError,
        ScannedPdfError,
        collect_figure_uploads,
        parse_pdf_to_graph,
    )

    # RAG_COMPUTE=client means nothing about this document should run on the
    # server, parsing included — so it takes the same route as --local-parse
    # and the backend handles the rest (GH #43).
    local_parse = local_parse or compute_mode() == "client"

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
    # the PDF. --local-parse / RAG_COMPUTE=client (and --dry-run/--show-cost,
    # which are inherently client-side estimation) fall through to the local
    # parse path below.
    if (
        backend_mode() == "remote"
        and db_path is None
        and not local_parse
        and not dry_run
        and not show_cost
    ):
        from datasheet_rag.backend import RagServerError

        assert backend_obj is not None  # built above for every non-dry run
        try:
            result = backend_obj.ingest_pdf(
                pdf_path,
                rev_id=rev_id,
                project_id=project_id,
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
                new_revision_of=new_revision_of,
            )
        except RagServerError as e:
            raise _friendly_server_error(e) from e
        _report_ingest(result, t0)
        return None

    # ── Local parse (local mode, --local-parse, --dry-run, --show-cost) ──────
    try:
        parsed = parse_pdf_to_graph(
            pdf_path,
            rev_id=rev_id,
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
        console.print(f"  rev_id = [cyan]{e.rev_id}[/]")
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

    did = parsed.rev_id
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
        console.print(f"  rev_id = [cyan]{did}[/]")
        return None

    step_state["n"] += 1
    where = "here" if backend_mode() == "local" or compute_mode() == "client" else "on the server"
    console.rule(f"[bold cyan]Step {step_state['n']} — Embed & store ({where})[/]")
    assert backend_obj is not None  # built above for every non-dry run

    # In remote mode the cropped figure images live only on this client —
    # ship their bytes so the server stores them and rewrites the host-local
    # figure_image_path before inserting.
    figures_upload: dict[str, tuple[bytes, str]] | None = None
    if not skip_figures and backend_mode() == "remote" and db_path is None:
        figures_upload, missing = collect_figure_uploads(graph)
        _warn_missing_figure_crops(missing, len(figures_upload))

    from datasheet_rag.backend import RagServerError

    try:
        result = backend_obj.ingest_chunk_graph(
            graph,
            figures=figures_upload,
            project_id=project_id,
            metadata=meta_patch,
            embed=True,
            describe_figures=do_describe,
            infer_title=infer_title,
            title_hints=parsed.title_hints,
            # This path parsed the PDF here — that is what --local-parse and
            # RAG_COMPUTE=client both mean. A remote backend has therefore
            # never seen the file and uploads it; a local one already has it
            # from the parse and ignores this (GH #43).
            source_pdf=pdf_path,
            new_revision_of=new_revision_of,
        )
    except RagServerError as e:
        raise _friendly_server_error(e) from e
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    _report_ingest(result, t0)
    return None


def _report_ingest(result: IngestResult, t0: float) -> None:
    """The lines every successful ingest ends with."""
    import time

    console.print(f"  [green]Upserted[/] {result.inserted} chunks")
    _report_pruned(result.pruned)
    if result.described:
        console.print(f"  [green]{result.described}[/] figure descriptions generated")
    if result.title:
        console.print(f"  [green]Inferred title:[/] {result.title}")
    elapsed = time.monotonic() - t0
    console.rule(f"[bold green]Done[/] — {elapsed:.0f}s")
    console.print(f"  rev_id = [cyan]{result.rev_id}[/]")
    if result.link is not None:
        console.print(f"  {result.link.describe()}")


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
    bad = click.BadParameter(
        f"invalid page range {spec!r} (expected e.g. '36' or '36-40')", param_hint="--pages"
    )
    try:
        if "-" in spec:
            start_s, _, end_s = spec.partition("-")
            start, end = int(start_s), int(end_s)
        else:
            start = end = int(spec)
    except ValueError:
        raise bad from None
    if start < 1 or end < start:
        raise bad
    return start, end


@repair_group.command("reconvert", short_help="Re-run Docling table recognition (Docling only).")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--pages",
    "pages_spec",
    required=True,
    help="1-based inclusive page range to re-run, e.g. '36' or '36-40'.",
)
@click.option("--rev-id", default=None, help="Override the content-hash rev_id.")
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
    rev_id: str | None,
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
    (so a cached `<rev_id>_outline.json` exists) — Textract exposes no
    equivalent re-run. On success it deletes the cached chunk graph; follow
    with `rag repair chunks <rev-id>` and `rag repair embeddings <rev-id>` to
    re-derive from the patched outline. Docling layout analysis itself is NOT
    re-run for the rest of the document.
    """
    from datasheet_rag.chunking.layout_parser import DocumentOutline
    from datasheet_rag.docling_parser import content_hash, reconvert_tables_in_range

    page_range = _parse_page_range(pages_spec)
    settings = get_settings()
    did = rev_id or content_hash(pdf_path)
    outline_path = settings.output_dir / f"{did}_outline.json"
    chunks_path = settings.output_dir / f"{did}_chunks.json"

    if not outline_path.exists():
        raise click.ClickException(
            f"No cached layout outline for rev_id={did} at {outline_path}. "
            "`repair reconvert` is Docling-only and patches an existing "
            "outline — it does not do a first-time conversion. Run "
            "`rag ingest --backend docling` on this PDF first."
        )

    cached = cache_files.load(outline_path)
    outline = DocumentOutline.from_dict(cached["outline"])

    mode_label = "accurate" if accurate_tables else "fast"
    console.print(
        f"Re-running pages {page_range[0]}-{page_range[1]} of {pdf_path.name} "
        f"with TableFormer [cyan]{mode_label}[/] mode…"
    )
    report = reconvert_tables_in_range(
        pdf_path, outline, rev_id=did, page_range=page_range, accurate_tables=accurate_tables
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
@click.argument("rev_id", type=str)
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
def table_structure_sweep_cmd(rev_id: str, list_flagged: bool, sample_n: int) -> None:
    """Report how many of a document's tables parsed with untrustworthy structure.

    Sweeps every table in the cached layout outline and counts the ones whose
    header band looks wrong, in two independently-detected ways: a header
    whose text is repeated across its cells, and a header the parser fused
    with the first row of data. Both produce a table that reads as clean but
    asserts the wrong column meanings — the failure worth catching before it
    reaches search results.

    Costs nothing: pure Python over `<rev_id>_outline.json`, no Docling re-run
    and no AWS calls. Run it before `rag repair tables`, which does spend
    money, to see how much there is to fix and to spot-check (with --sample)
    whether the flags are accurate on this document.

    REV_ID is a full hash or unambiguous prefix, looked up in the layout cache
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

    did = _require_docling_outline(rev_id)
    outline_path = _cache_path(did, "_outline.json")

    cached = cache_files.load(outline_path)
    outline = DocumentOutline.from_dict(cached["outline"])

    tables = [
        el
        for section in outline.all_sections_flat
        for el in section.elements
        if el.element_type == ElementType.TABLE
    ]
    if not tables:
        console.print(f"[yellow]No cached tables found for rev_id={did}.[/]")
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
        title=f"Table-structure sweep — {did[:SHORT_REV_ID_LEN]} ({total:,} cached tables)"
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
@click.argument("rev_id", type=str)
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
    rev_id: str,
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

    Like `repair reconvert`, this patches the cached `<rev_id>_outline.json`
    in place and invalidates the cached chunk graph; follow with
    `rag repair chunks <rev_id>` and `rag repair embeddings <rev_id>` to
    re-derive from the repaired structure. Run `rag inspect tables <rev_id>`
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

    did = _require_docling_outline(rev_id)
    outline_path = _cache_path(did, "_outline.json")
    chunks_path = _cache_path(did, "_chunks.json")

    cached = cache_files.load(outline_path)
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
            f"[green]Nothing to repair[/] for rev_id={did[:SHORT_REV_ID_LEN]} — "
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

    try:
        pdf_bytes = load_pdf_bytes(did)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

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


@cli.command("metadata", short_help="Show or set a document's metadata.")
@click.argument("rev_id", type=str)
@click.option(
    "--title",
    "doc_title",
    default=None,
    help="Override doc_title on every chunk row. Recorded as a manual "
    "title, so re-ingesting the document won't overwrite it.",
)
@click.option(
    "--add-project",
    "add_projects",
    multiple=True,
    metavar="PROJECT",
    help="Add the document to this project (repeatable). Its other projects are kept.",
)
@click.option(
    "--remove-project",
    "remove_projects",
    multiple=True,
    metavar="PROJECT",
    help="Remove the document from this project (repeatable). Its "
    "per-project fields there (subsystem) are deleted with it.",
)
@click.option(
    "--project",
    "fields_project",
    default=None,
    metavar="PROJECT",
    help="The project to set per-project fields such as --subsystem in "
    "(default: project_id from .rag.toml). The document must be in it.",
)
@click.option(
    "--mpn",
    multiple=True,
    help="Manufacturer part number, e.g. STM32H743VIT6. Repeat for several. "
    "Replaces the whole mpn list; --mpn '' clears it.",
)
@click.option(
    "--mpn-alias",
    "mpn_aliases",
    multiple=True,
    help="Append an MPN to the list without replacing existing ones (repeatable).",
)
@click.option(
    "--manufacturer",
    default=None,
    help="Manufacturer name, e.g. STMicroelectronics. Replaces the current value.",
)
@click.option(
    "--subsystem",
    default=None,
    help="The document's subsystem in one project, e.g. power, rf, mcu "
    "(see --project). --subsystem '' clears it there.",
)
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
    "--field",
    "field_items",
    multiple=True,
    metavar="NAME=VALUE",
    help="Set a registered field (see `rag fields list`), repeatable. Each field "
    "named is replaced; repeat a list field to give several values, e.g. "
    "--field acme-pn=P1 --field acme-pn=P2. Fields not named are left alone.",
)
@click.option(
    "--unset-field",
    "unset_fields",
    multiple=True,
    metavar="NAME",
    help="Remove a field's value from the document (repeatable).",
)
@click.option(
    "--this-revision",
    is_flag=True,
    help="Write --tag/--clear-tags and the fields on this one revision (a rev_id) "
    "instead of on its document. Clearing a field or the tags then removes the "
    "override, so the revision holds the document's value again.",
)
@click.option(
    "--revision",
    "revision_label",
    default=None,
    metavar="LABEL",
    help="Set this revision's label, e.g. 'Rev C' (free text; '' clears it). "
    "Needs a rev_id, not a D12.",
)
@_db_option
def metadata_cmd(
    rev_id: str,
    doc_title: str | None,
    add_projects: tuple[str, ...],
    remove_projects: tuple[str, ...],
    fields_project: str | None,
    mpn: tuple[str, ...],
    mpn_aliases: tuple[str, ...],
    manufacturer: str | None,
    subsystem: str | None,
    doc_type: str | None,
    tags: tuple[str, ...],
    clear_tags: bool,
    field_items: tuple[str, ...],
    unset_fields: tuple[str, ...],
    this_revision: bool,
    revision_label: str | None,
    db_path: Path | None,
) -> None:
    """Show a document's metadata, or update it.

    REV_ID is a document (D12) or any of its revisions' rev_ids (prefix ok).
    Metadata belongs to the document, so a write through any revision's
    rev_id changes it for every revision. With no options this prints the
    current metadata. Pass any option to update instead — only what you pass
    is touched:

    \b
        rag metadata D12                      # show
        rag metadata D12 --mpn INA226         # set, on every revision
        rag metadata <rev_id> --revision 'Rev C'

    To give one revision a different value, write it with --this-revision.
    Listings mark such values with *:

        rag metadata <rev_id> --this-revision --field doc_type=errata

    A document's mpn is a list (variants/aliases). --mpn replaces the list,
    --mpn-alias appends to it:

        rag metadata <rev_id> --mpn INA226 --mpn-alias INA226A --mpn-alias INA226B

    Filtering with --mpn on `rag list` or via the MCP tools matches any
    entry in the list. --mpn, --manufacturer and --doc-type are shortcuts
    for --field on the built-in fields; `rag fields list` shows every field.

    A document can belong to several projects. --add-project and
    --remove-project change one membership each and keep the rest:

        rag metadata <rev_id> --add-project board-b --remove-project board-a

    In a folder with a .rag.toml, a write with neither option adds the
    document to the .rag.toml project.

    subsystem holds one value per project: a datasheet can be "power" on
    one board and "aux" on another. --project names the project to set it
    in; without it the .rag.toml project is used. The project is never
    guessed from the document's memberships:

        rag metadata <rev_id> --project board-b --subsystem aux
    """
    from datasheet_rag.backend import MetadataPatch
    from datasheet_rag.project_config import get_project_config

    # Decide read-vs-write from what the user actually typed, before any
    # .rag.toml defaults get merged in below — otherwise a project config
    # would silently turn every `rag metadata <doc>` into a write. The scalar
    # options are tested against None, not truthiness: `--title ''` is an
    # explicit request to clear the field, not an absent option.
    #
    # document_write: what the user typed that lands on the whole document
    # unless --this-revision says otherwise (the title and the revision label
    # are always the revision's own).
    document_write = any(
        (
            add_projects,
            remove_projects,
            mpn,
            mpn_aliases,
            tags,
            clear_tags,
            field_items,
            unset_fields,
            manufacturer is not None,
            subsystem is not None,
            doc_type is not None,
        )
    )
    is_read = not document_write and doc_title is None and revision_label is None
    if is_read:
        if fields_project is not None:
            raise click.UsageError("--project names where to write a per-project field; set one")
        if this_revision:
            raise click.UsageError("--this-revision says where a write goes; give something to set")
        be = _backend_for(db_path)
        existing = be.get_metadata(_metadata_target(be, rev_id))
        if existing is None:
            console.print(f"[yellow]No metadata recorded for[/] {rev_id}")
            return
        console.print(existing.model_dump_json(indent=2, exclude_none=True))
        _note_overrides(existing)
        return
    if this_revision and (add_projects or remove_projects):
        raise click.UsageError(
            "projects belong to the whole document; drop --this-revision to change them"
        )
    fields: dict[str, Any] = _merge_shortcuts(
        _parse_field_items(field_items),
        _shortcut_fields(
            mpn=mpn, manufacturer=manufacturer, doc_type=doc_type, subsystem=subsystem
        ),
    )
    both = sorted(set(fields) & set(unset_fields))
    if both:
        raise click.UsageError(f"{', '.join(both)} given both to set and to --unset-field")
    for name in unset_fields:
        fields[name] = None

    # .rag.toml fills only what the user did not pass; an explicit blank
    # beats a config default. Its project is added only when the user named
    # no project change: `--remove-project <toml project>` must not be undone.
    # Strip the project names as the store does, so the .rag.toml checks
    # below compare the same names the store will write.
    from datasheet_rag.store.projects import normalize_projects

    try:
        add_projects = tuple(normalize_projects(add_projects))
        remove_projects = tuple(normalize_projects(remove_projects))
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    # A blank --project is no project (the store treats it so): the
    # .rag.toml project_id below still applies.
    if fields_project is not None:
        fields_project = fields_project.strip() or None

    # A per-revision write is an exception to the document's values; the
    # .rag.toml defaults describe the document and stay out of it.
    per_revision = this_revision
    be = _backend_for(db_path)
    rev_id = _metadata_target(be, rev_id)
    if parse_document_ref(rev_id) is not None:
        what = next(
            (
                flag
                for flag, given in (
                    ("--this-revision", this_revision),
                    ("--revision", revision_label is not None),
                    ("--title", doc_title is not None),
                )
                if given
            ),
            None,
        )
        if what is not None:
            raise click.UsageError(
                f"{rev_id} is a whole document; {what} needs one revision's rev_id "
                f"(see rag revisions {rev_id})"
            )
    elif not this_revision:
        # A rev_id may mean just that PDF; writing to every revision of its
        # document needs the D12 (GH #124). A --title / --revision write on
        # such a rev_id is the revision's own, so .rag.toml stays out of it.
        current = be.get_metadata(rev_id)
        if current is not None and len(current.revisions) > 1:
            if document_write:
                raise click.UsageError(
                    f"{rev_id[:SHORT_REV_ID_LEN]} is one of {len(current.revisions)} revisions "
                    f"of {current.document_id}. Use {current.document_id} to change all of "
                    "them, or --this-revision for this one (fields and tags only)."
                )
            per_revision = True

    proj_cfg = None if per_revision else get_project_config()
    if proj_cfg is not None:
        if proj_cfg.project_id and not (add_projects or remove_projects):
            add_projects = (proj_cfg.project_id,)
        if fields_project is None:
            fields_project = proj_cfg.project_id
        if not tags and proj_cfg.tags:
            tags = tuple(proj_cfg.tags)

    scopes: dict[str, FieldScope] | None = None
    if proj_cfg is not None and proj_cfg.fields:
        from datasheet_rag.store.fields import project_scoped_names, scopes_of

        scopes = scopes_of(be.list_fields())
        # The config's project-scoped fields belong to its project_id. A write
        # that cannot put them there leaves them out with a note; leaving that
        # project leaves them out silently (the values go with the
        # membership). A config with no project_id ties its fields to no
        # project: the user's --project decides, or the write fails for want
        # of one.
        cfg_project = proj_cfg.project_id
        why_not: str | None = None
        in_cfg = True
        from_cfg = [n for n in project_scoped_names(scopes, proj_cfg.fields) if n not in fields]
        if cfg_project is not None and from_cfg:
            if cfg_project in remove_projects:
                in_cfg = False
            elif fields_project != cfg_project:
                in_cfg, why_not = False, f"this write goes to project {fields_project!r}"
            elif cfg_project not in add_projects:
                stored = be.get_metadata(rev_id)
                if stored is None or cfg_project not in stored.projects:
                    in_cfg, why_not = False, f"{rev_id} is not in that project"
        fields = _merge_config_fields(
            proj_cfg, fields, scopes=scopes, in_config_project=in_cfg, why_not=why_not
        )

    # --mpn-alias appends to the mpn list: the one --mpn (or .rag.toml) just
    # gave, else the one already stored.
    if mpn_aliases:
        if fields.get("mpn") is not None:
            base = fields["mpn"]
        else:
            existing_meta = be.get_metadata(rev_id)
            # A whole-document write builds on the document's list, not on
            # one revision's override of it.
            if (
                existing_meta is not None
                and not this_revision
                and "mpn" in existing_meta.overrides
                and existing_meta.document_id
            ):
                existing_meta = be.get_metadata(existing_meta.document_id)
            base = existing_meta.fields.get("mpn") if existing_meta else None
        all_mpns: list[str] = [base] if isinstance(base, str) else list(base or [])
        for alias in mpn_aliases:
            if alias not in all_mpns:
                all_mpns.append(alias)
        fields["mpn"] = all_mpns

    if doc_title is not None:
        updated = be.set_doc_title(rev_id, doc_title)
        console.print(f"[green]Title set[/] on {updated} chunk rows: {doc_title!r}")

    if tags:
        new_tags: list[str] | None = list(tags)
    elif clear_tags:
        new_tags = []
    else:
        new_tags = None

    try:
        meta = be.set_metadata(
            rev_id,
            MetadataPatch(
                add_projects=list(add_projects) or None,
                remove_projects=list(remove_projects) or None,
                tags=new_tags,
                fields=fields or None,
                project=fields_project,
                this_revision=this_revision or None,
                revision=revision_label,
            ),
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    if this_revision:
        where = f"revision {rev_id[:SHORT_REV_ID_LEN]} only"
    elif len(meta.revisions) > 1:
        where = f"{meta.document_id} (all {len(meta.revisions)} revisions)"
    else:
        where = meta.document_id or rev_id
    console.print(f"[green]Saved metadata for[/] {where}")
    _note_clears_outside_project(be, meta, fields, fields_project, remove_projects)
    console.print(meta.model_dump_json(indent=2, exclude_none=True))
    _note_overrides(meta)


def _metadata_target(be: RagBackend, ref: str) -> str:
    """What `rag metadata` reads or writes (see ``RagBackend.resolve_ref``)."""
    try:
        return be.resolve_ref(ref)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _note_overrides(meta: DocMetadata) -> None:
    """Name the values set on this one revision, after a metadata print."""
    if meta.overrides:
        console.print(
            f"  * set on this revision only: {', '.join(meta.overrides)} "
            f"(the other revisions of {meta.document_id} hold the document's value)"
        )


def _note_clears_outside_project(
    be: RagBackend,
    meta: DocMetadata,
    fields: Mapping[str, Any],
    project: str | None,
    remove_projects: Iterable[str],
) -> None:
    """Say so when a project-scoped field was cleared in a project the
    document is not in: the store does nothing then, and the user may have
    meant another project. Leaving that project in the same call is not
    worth a note; its values go with the membership. ``project`` is
    already stripped."""
    if not fields or project is None or project in meta.projects or project in remove_projects:
        return
    from datasheet_rag.store.fields import (
        normalize_field_patch_for,
        project_scoped_names,
        scopes_of,
        shapes_of,
    )

    specs = be.list_fields()
    # The store's own rule for what counts as a clear: an empty normalized value.
    patch = normalize_field_patch_for(shapes_of(specs), fields)
    cleared = [n for n, v in patch.items() if not v]
    names = project_scoped_names(scopes_of(specs), cleared)
    if names:
        console.print(
            f"[yellow]Note:[/] {meta.rev_id} is not in project {project!r}, so it holds "
            f"no {', '.join(names)} there; nothing was cleared."
        )


# ---------------------------------------------------------------------------
# fields (registered metadata fields — GH #120)
# ---------------------------------------------------------------------------


@cli.group("fields", short_help="Register the metadata fields documents can carry.")
def fields_group() -> None:
    """Register, list and remove metadata fields.

    A field is a named, typed, filterable piece of document metadata — e.g.
    an internal document number, a vendor's document number, an internal
    part number. A field must be registered before a document can hold a
    value for it. Then set it with `rag metadata <doc> --field NAME=VALUE`
    or a `[fields]` table in .rag.toml, filter with `rag list --field`, and
    agents can filter `search` on it. mpn, manufacturer and doc_type are
    built in.
    """


@fields_group.command("list", short_help="Show every registered field.")
@_db_option
def fields_list_cmd(db_path: Path | None) -> None:
    """Show every registered field: name, shape, scope, roles, description."""
    table = Table(title="Metadata fields")
    for col in ("name", "shape", "scope", "roles", "built in", "description"):
        table.add_column(col, style="cyan" if col == "name" else None)
    for f in _backend_for(db_path).list_fields():
        table.add_row(
            f.name,
            f.shape,
            f.scope,
            ", ".join(f.roles) or "—",
            "yes" if f.builtin else "",
            f.description,
        )
    console.print(table)


@fields_group.command("add", short_help="Register a new field.")
@click.argument("name")
@click.option(
    "--shape",
    type=click.Choice(["single", "list"]),
    default="single",
    help="single: one value per document. list: several, in order.",
)
@click.option(
    "--scope",
    type=click.Choice(["document", "project"]),
    default="document",
    help="document: one value per document, the same in every project. "
    "project: one value per document in each project (like subsystem).",
)
@click.option("--description", default="", help="Shown to agents and in `rag fields list`.")
@click.option(
    "--role",
    "roles",
    multiple=True,
    type=click.Choice(["part"]),
    help="part: the values are part numbers (used to match a BOM).",
)
@_db_option
def fields_add_cmd(
    name: str,
    shape: str,
    scope: str,
    description: str,
    roles: tuple[str, ...],
    db_path: Path | None,
) -> None:
    """Register field NAME (lower case, digits, '-' and '_').

    \b
        rag fields add acme-id --description "ACME document number"
        rag fields add acme-pn --shape list --role part
        rag fields add board-ref --scope project
    """
    try:
        spec = _backend_for(db_path).register_field(
            name,
            shape=shape,  # type: ignore[arg-type]  # click.Choice guarantees the literal
            scope=scope,  # type: ignore[arg-type]  # click.Choice guarantees the literal
            description=description,
            roles=roles,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    console.print(f"[green]Registered[/] {spec.name} ({spec.shape}, per {spec.scope})")


@fields_group.command("remove", short_help="Unregister a field.")
@click.argument("name")
@click.option(
    "--clear",
    is_flag=True,
    help="Delete the field's values from every document with it, instead of "
    "refusing while any document has one.",
)
@_db_option
def fields_remove_cmd(name: str, clear: bool, db_path: Path | None) -> None:
    """Unregister field NAME. Built-in fields cannot be removed."""
    try:
        held = _backend_for(db_path).remove_field(name, clear=clear)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    console.print(f"[green]Removed[/] {name}")
    if held:
        console.print(f"  Deleted its values on {held} document(s).")


# ---------------------------------------------------------------------------
# fix-titles (AI-inferred document titles for poorly-titled documents)
# ---------------------------------------------------------------------------

_BLANK_TITLES = (None, "", "—")


@repair_group.command("titles", short_help="Backfill missing document titles.")
@click.option("--rev-id", default=None, help="Restrict to one revision (rev_id).")
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
    rev_id: str | None,
    force: bool,
    model_id: str | None,
    dry_run: bool,
    db_path: Path | None,
) -> None:
    """Infer and backfill document titles with a small Bedrock Claude call.

    Without --rev-id, scans every ingested document and infers a title for
    those showing as "—" (blank) in `rag list`. Pass --rev-id to target one
    document, and add --force to replace a generic-but-present title (e.g.
    "Contents", "Disclaimer") that the heuristic above wouldn't flag.

    Inferred titles are written to every chunk row for the document and
    marked `title_source: inferred` in the metadata sidecar (`rag metadata
    <rev_id>`) so they're distinguishable from titles Docling extracted
    directly, and so a later re-ingest can't demote them. A title set by
    hand outranks an inferred one and is left alone — pass --force to
    replace it anyway, or to overwrite an existing inferred title.
    """
    from datasheet_rag.store.metadata import title_rank

    be = _backend_for(db_path)

    if rev_id:
        rev_id = _backend_resolve(be, rev_id)
        docs = [d for d in be.get_ingested_docs() if d.rev_id == rev_id]
    else:
        docs = be.get_ingested_docs()

    if not force:
        skipped = [d for d in docs if d.doc_title not in _BLANK_TITLES]
        docs = [d for d in docs if d.doc_title in _BLANK_TITLES]
        if skipped and len(skipped) == 1 and not docs:
            console.print(
                f"[yellow]{skipped[0].rev_id[:SHORT_REV_ID_LEN]}[/] already has a title "
                f"({skipped[0].doc_title!r}). Pass --force to re-infer it anyway."
            )

    if not docs:
        console.print("[yellow]No documents need a title fix.[/]")
        return

    from datasheet_rag.backend import compute_mode

    where = "on this machine" if compute_mode() == "client" else "on the server"
    console.print(f"Inferring titles for {len(docs)} document(s) (the LLM runs {where})…")

    for d in docs:
        short_id = d.rev_id[:SHORT_REV_ID_LEN]
        current = d.doc_title
        md = be.get_metadata(d.rev_id)
        source = md.title_source if md is not None else "auto"
        # Say why we're leaving a hand-set title alone, rather than letting
        # the store's refusal surface as an indistinguishable "could not
        # infer" further down.
        if not force and title_rank(source) > title_rank("inferred"):
            console.print(
                f"  [yellow]skipped[/] {short_id} — title was set by hand "
                f"({current!r}); pass --force to replace it"
            )
            continue
        title = be.infer_title(d.rev_id, model_id=model_id, dry_run=dry_run, force=force)
        if title is None:
            console.print(f"  [yellow]could not infer[/] {short_id} (was: {current!r})")
            continue
        verb = "would set" if dry_run else "set"
        console.print(f"  [green]{verb}[/] {short_id}: {current!r} → {title!r}")


# ---------------------------------------------------------------------------
# Eval (retrieval-layer evaluation)
# ---------------------------------------------------------------------------


class _EvalGroup(click.Group):
    """Reports a broken keyword index as one line, like the root group's
    user errors (GH #36). Kept here, not in the root group, so every other
    command does not pay for importing the store."""

    def invoke(self, ctx: click.Context) -> Any:
        from datasheet_rag.store.schema import FtsOutOfSyncError

        try:
            return super().invoke(ctx)
        except FtsOutOfSyncError as e:
            raise click.ClickException(str(e)) from e


@cli.group("eval", cls=_EvalGroup, hidden=True, short_help="Evaluate retrieval quality.")
def eval_group() -> None:
    """Retrieval-layer evaluation: golden set, metrics, ablations."""


_CAT_ORDER = ["identifier", "conceptual", "figure", "table_spec", "synthesis", "overall"]


def _mean_cell(value: float, ci: Interval | None, fmt: str = ".2f") -> str:
    """A mean over its 95% interval; a group of one need has none."""
    # The space lets a narrow terminal wrap the interval instead of cutting it.
    band = f"\\[{ci.lo:{fmt}}, {ci.hi:{fmt}}]" if ci is not None else "one need"
    return f"{value:{fmt}}\n[dim]{band}[/]"


def _diff_cell(d: PairedDiff, fmt: str = "+.2f") -> str:
    """A paired difference, or 'n too small' when its interval spans zero."""
    band = f"\\[{d.ci.lo:{fmt}}, {d.ci.hi:{fmt}}]" if d.ci is not None else "one need"
    if d.verdict == "n too small":
        return f"[yellow]n too small[/]\n[dim]{band}[/]"
    if d.verdict == "no change":
        return "[dim]no change[/]"
    colour = "green" if d.verdict == "better" else "red"
    return f"[{colour}]{d.diff:{fmt}}[/]\n[dim]{band}[/]"


def _render_report_table(report: object) -> None:
    """Print one RunReport's per-category metrics."""
    from datasheet_rag.eval.harness import RunReport

    assert isinstance(report, RunReport)
    hk = report.config.k
    table = Table(
        title=f"Retrieval eval · {report.config.describe()} "
        f"(hit@k = strict lineage; pg@{hk} = loose page upper bound; "
        f"dim = 95% bootstrap interval over needs)"
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
        row += [
            _mean_cell(m.hit_rate_at_k.get(k, 0.0), m.hit_rate_at_k_ci.get(k))
            for k in report.config.ks
        ]
        row += [_mean_cell(m.hit_rate_at_k_loose.get(hk, 0.0), m.hit_rate_at_k_loose_ci.get(hk))]
        row += [_mean_cell(m.mrr, m.mrr_ci, ".3f"), _mean_cell(m.ndcg, m.ndcg_ci, ".3f")]
        table.add_row(*row, end_section=True)
    console.print(table)
    if report.unresolved_gold:
        console.print(
            f"[yellow]{len(report.unresolved_gold)} item(s) score zero because no chunk in "
            f"the store holds their evidence[/] (a parser gap, not a ranking miss):"
        )
        for q in report.unresolved_gold:
            console.print(f"  - {q}")


def _render_matrix_table(reports: list[RunReport], headline_k: int) -> None:
    """Print a comparison across configs: overall + per-category hit@k,
    then each config's paired difference from the first (the baseline)."""
    from datasheet_rag.eval.dataset import CATEGORIES

    table = Table(
        title=f"Ablation comparison · hit@{headline_k} by category "
        f"(dim = 95% bootstrap interval over needs)"
    )
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
            _mean_cell(
                overall.hit_rate_at_k.get(headline_k, 0.0),
                overall.hit_rate_at_k_ci.get(headline_k),
            ),
            _mean_cell(overall.mrr, overall.mrr_ci, ".3f"),
            _mean_cell(overall.ndcg, overall.ndcg_ci, ".3f"),
        ]
        for cat in CATEGORIES:
            m = rep.by_category.get(cat)
            row.append(
                _mean_cell(m.hit_rate_at_k.get(headline_k, 0.0), m.hit_rate_at_k_ci.get(headline_k))
                if m
                else "—"
            )
        table.add_row(*row, end_section=True)
    console.print(table)

    if len(reports) < 2:
        return
    base = reports[0]
    diffs = Table(
        title=f"Paired difference from '{base.config.describe()}' "
        f"(variant − baseline, same questions; 95% bootstrap interval over needs)"
    )
    diffs.add_column("config", style="yellow")
    diffs.add_column(f"Δhit@{headline_k}", justify="right")
    diffs.add_column("ΔMRR", justify="right")
    diffs.add_column("ΔnDCG", justify="right")
    for cat in CATEGORIES:
        diffs.add_column(f"Δ{cat[:5]}", justify="right")
    for rep, cmp in zip(reports[1:], _paired_vs_baseline(reports, headline_k)):
        whole = cmp["overall"]
        row = [
            rep.config.describe(),
            _diff_cell(whole.hit_rate),
            _diff_cell(whole.mrr, "+.3f"),
            _diff_cell(whole.ndcg, "+.3f"),
        ]
        row += [_diff_cell(cmp[cat].hit_rate) if cat in cmp else "—" for cat in CATEGORIES]
        diffs.add_row(*row, end_section=True)
    console.print(diffs)


def _paired_vs_baseline(
    reports: list[RunReport], headline_k: int
) -> list[dict[str, PairedComparison]]:
    """Each report after the first, paired against the first."""
    from datasheet_rag.eval.metrics import compare_by_category

    base = reports[0]
    return [compare_by_category(base.outcomes, r.outcomes, k=headline_k) for r in reports[1:]]


@eval_group.command("generate", short_help="Generate a golden set from the corpus.")
@_db_option
@click.option("--per-category", default=4, type=int, help="Items to generate per category.")
@click.option("--rev-id", default=None, help="Restrict sampling to one document.")
@click.option("--project-id", default=None, help="Restrict sampling to one project.")
@click.option("--model", "model_id", default=None, help="Bedrock model ID for generation.")
@click.option("--seed", default=0, type=int, help="Sampling seed (reproducible).")
@click.option(
    "--output",
    "-o",
    "out_path",
    type=click.Path(path_type=Path),
    # Not golden-mined.jsonl: generated items are written from the chunk they
    # are scored against, so they never replace the mined set (GH #92).
    default=Path("eval/golden-auto.jsonl"),
    help="Output JSONL path.",
)
@click.option("--append", is_flag=True, help="Append to the output file instead of overwriting.")
@click.option("--verbose/--quiet", default=True, help="Print per-item progress.")
def eval_generate(
    db_path: Path | None,
    per_category: int,
    rev_id: str | None,
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
    if rev_id:
        rev_id = _resolve_rev_id(conn, rev_id)
    eval_set = generate_golden_set(
        conn,
        per_category=per_category,
        rev_id=rev_id,
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
    default=Path("eval/golden-mined.jsonl"),
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
    from datasheet_rag.backend import SEARCH_POOL

    if not 1 <= top_k <= SEARCH_POOL:
        raise click.BadParameter(f"must be 1 to {SEARCH_POOL}", param_hint="-k")
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
    default=Path("eval/golden-mined.jsonl"),
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
    "--rev-id",
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
    rev_id: str | None,
    summarizer_model: str,
    verbose: bool,
) -> None:
    """Run the ablation matrix and print which concepts move the needle."""
    from datasheet_rag.backend import SEARCH_POOL

    if not 1 <= top_k <= SEARCH_POOL:
        raise click.BadParameter(f"must be 1 to {SEARCH_POOL}", param_hint="-k")
    from datasheet_rag.embedding import get_embedder
    from datasheet_rag.eval.ablation import (
        build_macro_summarizer_variant_store,
        build_variant_store,
        default_matrix,
        run_matrix,
    )
    from datasheet_rag.eval.dataset import EvalSet
    from datasheet_rag.eval.harness import RunConfig, run_eval
    from datasheet_rag.store import connect, require_fts_in_sync

    conn = connect(_require_local_db(db_path))
    # run_eval checks this too, but an index ablation would first pay Bedrock
    # to re-embed the whole store (GH #91).
    require_fts_in_sync(conn)
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
            _dump_reports_json(reports, json_out, top_k)
        return

    if index_ablation == "macro-summarizer":
        if not rev_id:
            raise click.ClickException(
                "--index-ablation macro-summarizer requires --rev-id "
                "(it re-summarizes one document's chapters with Bedrock Claude)."
            )
        from datasheet_rag.chunking.summarizer import AbstractiveSummarizer
        from datasheet_rag.config import get_settings as _get_settings

        console.print(
            f"[yellow]Index ablation[/] 'macro-summarizer': re-summarizing "
            f"{rev_id} chapters abstractively (model={summarizer_model}) "
            f"and re-embedding — this calls Bedrock Claude per chapter."
        )
        summarizer = AbstractiveSummarizer(
            model_id=summarizer_model,
            region=_get_settings().aws_region,
        )
        variant_conn = build_macro_summarizer_variant_store(
            conn,
            variant_db,
            rev_id,
            summarizer,
            embedder,
            verbose=verbose,
        )

        # The variant store only contains rev_id's chunks (by design — see
        # build_macro_summarizer_variant_store), so items targeting other
        # docs would score 0 by construction. Scope the eval set to match.
        scoped_set = EvalSet(items=[i for i in eval_set.items if i.rev_id == rev_id])

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
            _dump_reports_json(reports, json_out, top_k)
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
        _dump_reports_json(reports, json_out, top_k)


def _dump_reports_json(reports: list[RunReport], path: Path, headline_k: int) -> None:
    import json as _json

    paired = [
        {"config": r.config.describe(), "by_category": {c: m.model_dump() for c, m in cmp.items()}}
        for r, cmp in zip(reports[1:], _paired_vs_baseline(reports, headline_k))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _json.dumps(
            {
                "reports": [r.model_dump() for r in reports],
                "baseline": reports[0].config.describe() if reports else None,
                "headline_k": headline_k,
                "paired_vs_baseline": paired,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]Reports JSON →[/] {path}")


#: Agent and judge model for the answer eval. The global profile bills about
#: 10% below the eu. one for the same model (see costs.CLAUDE_TOKEN_PRICES).
_ANSWER_MODEL = "global.anthropic.claude-sonnet-4-6"


@eval_group.command("answer", short_help="Grounded answer rate: no tool vs raw PDF vs RAG.")
@_db_option
@click.option(
    "--needs",
    "needs_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("eval/needs-mined.jsonl"),
    help="Information needs with answer keys (JSONL).",
)
@click.option(
    "--condition",
    "conditions",
    type=click.Choice(["A", "B", "C"]),
    multiple=True,
    help="A = no tool, B = raw PDF, C = datasheet-rag. Repeat; default all three.",
)
@click.option("--model", "model_id", default=_ANSWER_MODEL, help="Bedrock model for the agent.")
@click.option("--judge-model", default=None, help="Bedrock model for the judge (default: --model).")
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=Path("eval/answers.jsonl"),
    help="Graded runs are appended here; a rerun resumes from it.",
)
@click.option("--need-id", "need_ids", multiple=True, help="Run only these needs. Repeatable.")
@click.option(
    "--need-list",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Run only the needs listed in this file (one id per line), e.g. eval/needs-ab.txt.",
)
@click.option("--limit", default=None, type=int, help="Run only the first N needs.")
@click.option("--workers", default=4, type=int, help="Agent runs in parallel.")
@click.option(
    "--max-usd",
    default=5.0,
    type=float,
    show_default=True,
    help="Stop starting new runs once this invocation has spent this much (list price). "
    "Low by default so a full run is a deliberate choice.",
)
@click.option(
    "--agent-temperature",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="The agent's sampling temperature (default: the model's own, as a real client). "
    "Use 0 for A/B runs, on both sides, to cut run-to-run noise.",
)
@click.option("--report-only", is_flag=True, help="Report on --out without running anything.")
@click.option(
    "--regrade",
    is_flag=True,
    help="Grade the answers saved in --out again (judge calls only, no agent runs), then report.",
)
@click.option(
    "--compare",
    "compare_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="A/B: pair the runs in this baseline file with --out, per condition (default C). "
    "Runs nothing.",
)
@click.option(
    "--json-out",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the report JSON here.",
)
def eval_answer(
    db_path: Path | None,
    needs_path: Path,
    conditions: tuple[str, ...],
    model_id: str,
    judge_model: str | None,
    out_path: Path,
    need_ids: tuple[str, ...],
    need_list: Path | None,
    limit: int | None,
    workers: int,
    max_usd: float,
    agent_temperature: float | None,
    report_only: bool,
    regrade: bool,
    compare_path: Path | None,
    json_out: Path | None,
) -> None:
    """Ask each need's question under three conditions and grade the answers.

    The headline is the grounded answer rate: correct and citing a gold page.
    Runs call Bedrock (agent and judge); PDFs come from settings.pdf_dir.

    A/B testing a datasheet-rag change needs only condition C: run it into a
    new --out on the variant, then --compare the baseline file against it.
    Day to day, and sparingly: --condition C --need-list eval/needs-ab.txt
    --model global.anthropic.claude-haiku-4-5-20251001-v1:0
    --agent-temperature 0 (about $3 a side). A baseline serves later
    variants while the model, store and code under test stay the same.
    At the default temperature, two identical Haiku runs flipped 7 of 47
    needs. For big changes, measure with the default Sonnet on all needs.
    """
    from datasheet_rag.eval.agent import Condition
    from datasheet_rag.eval.answers import build_report, load_records

    needs = _select_needs(needs_path, need_ids, need_list, limit)

    if compare_path is not None:
        if report_only or regrade:
            raise click.UsageError("--compare runs nothing; drop --report-only / --regrade")
        from datasheet_rag.eval.answers import compare_runs

        ids = {n.need_id for n in needs}
        base = [r for r in load_records(compare_path) if r.need_id in ids]
        variant = [r for r in load_records(out_path) if r.need_id in ids]
        results = []
        for cond in conditions or ("C",):
            try:
                cmp = compare_runs(base, variant, cond)  # type: ignore[arg-type]
            except ValueError as e:
                raise click.ClickException(str(e)) from e
            _render_run_comparison(cmp, compare_path, out_path)
            results.append(cmp)
        if json_out:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                "[" + ",".join(c.model_dump_json(indent=2) for c in results) + "]",
                encoding="utf-8",
            )
            console.print(f"[green]Report JSON →[/] {json_out}")
        return

    conds: list[Condition] = list(conditions or ("A", "B", "C"))  # type: ignore[arg-type]

    if report_only and regrade:
        raise click.UsageError("--report-only and --regrade exclude each other")
    if regrade:
        from datasheet_rag.aws import bedrock_runtime_client
        from datasheet_rag.eval.answers import regrade_records

        judge = judge_model or model_id
        n, failed = regrade_records(
            out_path,
            needs,
            client=bedrock_runtime_client(read_timeout=300, max_attempts=10),
            judge_model=judge,
            workers=workers,
        )
        console.print(f"[cyan]Regraded[/] {n} answer(s) in {out_path} with {judge}.")
        for f in failed:
            console.print(f"  [red]judge failed, old grade (if any) kept[/] {f}")
    elif not report_only:
        from datasheet_rag.costs import CLAUDE_TOKEN_PRICES

        unpriced = {model_id, judge_model or model_id} - set(CLAUDE_TOKEN_PRICES)
        if unpriced:
            raise click.ClickException(
                f"no price for {', '.join(sorted(unpriced))}, so --max-usd cannot be "
                "enforced; add it to costs.CLAUDE_TOKEN_PRICES"
            )
        _run_answer_eval(
            db_path,
            needs,
            conds,
            model_id,
            judge_model or model_id,
            out_path,
            workers,
            max_usd,
            agent_temperature,
        )

    wanted = {n.need_id for n in needs}
    records = [r for r in load_records(out_path) if r.need_id in wanted]
    report = build_report(records, conds, model_id)
    _render_answer_report(report)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Report JSON →[/] {json_out}")


def _shared_project(backend: Any, needs: list[Any]) -> str:
    """The one project every needed document belongs to.

    A document can be in several projects (GH #121), so this is the
    intersection of their projects. It must hold exactly one project, or the
    eval cannot tell which project's catalogue to hand the agent. A needed
    document missing from the store is skipped.
    """
    found = (backend.get_document(i) for i in sorted({n.rev_id for n in needs if n.rev_id}))
    project_lists: list[list[str]] = [d.projects for d in found if d is not None]
    shared: set[str] = set(project_lists[0]) if project_lists else set()
    for projects in project_lists[1:]:
        shared &= set(projects)
    if len(shared) != 1:
        raise click.ClickException(
            "the needs' documents should share exactly one project, found "
            f"{sorted(shared) or 'none'}"
        )
    return shared.pop()


def _select_needs(
    needs_path: Path, need_ids: tuple[str, ...], need_list: Path | None, limit: int | None
) -> list[Any]:
    """The needs a run covers: all, or those named by --need-id / --need-list."""
    from datasheet_rag.eval.dataset import load_need_ids, load_needs

    needs = load_needs(needs_path)
    wanted = set(need_ids)
    if need_list is not None:
        wanted |= set(load_need_ids(need_list))
    if wanted:
        unknown = wanted - {n.need_id for n in needs}
        if unknown:
            raise click.ClickException(f"unknown need id(s): {', '.join(sorted(unknown))}")
        needs = [n for n in needs if n.need_id in wanted]
    if limit is not None:
        needs = needs[:limit]
    if not needs:
        raise click.ClickException("no needs selected")
    return needs


def _print_answer_progress(rec: AnswerRecord) -> None:
    """One line per finished run."""
    if rec.error is not None:
        console.print(f"  [red]error[/] {rec.need_id} {rec.condition}: {rec.error.splitlines()[0]}")
        return
    assert rec.run is not None
    if rec.grade is None:
        console.print(
            f"  [yellow]judge failed[/] {rec.need_id} {rec.condition}: {rec.grade_error} "
            "(answer kept; the next run grades it)"
        )
        return
    mark = (
        "[green]grounded[/]"
        if rec.grade.grounded
        else ("[yellow]correct[/]" if rec.grade.correct else "[red]wrong[/]")
    )
    console.print(
        f"  {rec.condition} {rec.need_id}: {mark} ({rec.grade.grader}, "
        f"{len(rec.run.tool_calls)} calls, {rec.run.wall_s:.0f} s, "
        f"${rec.cost_usd() or 0.0:.3f})"
    )


def _run_answer_eval(
    db_path: Path | None,
    needs: list[Any],
    conds: list[Any],
    model_id: str,
    judge_model: str,
    out_path: Path,
    workers: int,
    max_usd: float,
    agent_temperature: float | None,
) -> None:
    import logging

    from datasheet_rag import pdf_viewer
    from datasheet_rag.aws import bedrock_runtime_client
    from datasheet_rag.backend import LocalBackend
    from datasheet_rag.eval.agent import (
        CONDITION_NAMES,
        NoTools,
        PdfText,
        PdfTools,
        RagTools,
        ToolSet,
    )
    from datasheet_rag.eval.answers import LockedEmbedder, run_answers
    from datasheet_rag.store import connect, require_fts_in_sync

    # Building the MCP server turns on INFO logging, and the model download
    # checks then bury the per-run progress lines.
    for name in ("httpx", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    db = _require_local_db(db_path)
    conn = connect(db)
    local = LocalBackend(conn=conn)
    project = _shared_project(local, needs)
    from datasheet_rag.store.fields import display_value

    listing = local.list_documents(project_id=project).documents
    docs = [
        {
            "rev_id": d.rev_id,
            "title": d.doc_title,
            "mpn": display_value(d.fields.get("mpn")),
            "manufacturer": display_value(d.fields.get("manufacturer")),
            "pages": d.page_count,
        }
        for d in listing
    ]

    embedder = None
    if "C" in conds:
        # A desynced keyword index would quietly turn hybrid search into
        # vector-only search for the agent (GH #91).
        require_fts_in_sync(conn)
        from datasheet_rag.embedding import get_embedder

        embedder = LockedEmbedder(get_embedder())
    pdf_text = PdfText(pdf_viewer.load_pdf_bytes)

    def toolset_for(cond: Any) -> ToolSet:
        if cond == "A":
            return NoTools()
        if cond == "B":
            return PdfTools(docs, pdf_text)
        from datasheet_rag.mcp.server import build_server

        backend = LocalBackend(conn=connect(db), embedder=embedder)
        return RagTools(build_server(backend, local_client=False), project)

    # An agent turn can write thousands of tokens; the 60 s default is short.
    client = bedrock_runtime_client(read_timeout=300, max_attempts=10)
    names = ", ".join(f"{c} = {CONDITION_NAMES[c]}" for c in conds)
    console.print(
        f"[cyan]Answer eval[/]: {len(needs)} needs × {len(conds)} conditions ({names}), "
        f"project {project}, model {model_id}"
        + (f" at temperature {agent_temperature}" if agent_temperature is not None else "")
        + f", judge {judge_model}, cap ${max_usd:.2f} → {out_path}"
    )

    try:
        run_answers(
            needs,
            conds,
            model_id=model_id,
            judge_model=judge_model,
            client=client,
            toolset_for=toolset_for,
            out_path=out_path,
            workers=workers,
            progress=_print_answer_progress,
            max_usd=max_usd,
            temperature=agent_temperature,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    conn.close()


@eval_group.command(
    "claude-code", short_help="The answer eval through Claude Code, paced on the usage meter."
)
@click.option(
    "--needs",
    "needs_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("eval/needs-mined.jsonl"),
    help="Information needs with answer keys (JSONL).",
)
@click.option(
    "--condition",
    "conditions",
    type=click.Choice(["A", "B", "C"]),
    multiple=True,
    help="A = no tool, B = raw PDF, C = datasheet-rag. Repeat; default all three.",
)
@click.option(
    "--model",
    "model_alias",
    default="sonnet",
    show_default=True,
    help="Agent model, as Claude Code's --model takes it.",
)
@click.option(
    "--judge-model",
    "judge_alias",
    default="opus",
    show_default=True,
    help="Judge model, as Claude Code's --model takes it.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=Path("eval/answers-claude-code.jsonl"),
    show_default=True,
    help="Graded runs are appended here; every invocation resumes from it.",
)
@click.option("--need-id", "need_ids", multiple=True, help="Run only these needs. Repeatable.")
@click.option(
    "--need-list",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Run only the needs listed in this file, e.g. eval/needs-ab.txt.",
)
@click.option("--limit", default=None, type=int, help="Run only the first N needs.")
@click.option(
    "--stop-at",
    type=click.IntRange(1, 100),
    default=85,
    show_default=True,
    help="Start no new run once the 5-hour usage meter is at this percent.",
)
@click.option(
    "--weekly-ceiling",
    type=click.IntRange(1, 100),
    default=80,
    show_default=True,
    help="Start no new run once the weekly usage meter is at this percent.",
)
@click.option(
    "--workers",
    default=2,
    type=int,
    show_default=True,
    help="Runs in parallel. The meter can pass --stop-at by up to this many runs.",
)
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=None,
    help="Where runs start: PDF copies and the MCP config (default: <rag_home>/eval-claude-code).",
)
@click.option("--report-only", is_flag=True, help="Report on --out without running anything.")
@click.option(
    "--regrade",
    is_flag=True,
    help=(
        "Grade the saved answers again with --judge-model (no agent runs), then report. "
        "Answers that judge already graded are skipped, so a stopped regrade resumes."
    ),
)
@click.option("--json-out", type=click.Path(path_type=Path), default=None, help="Report JSON.")
def eval_claude_code(
    needs_path: Path,
    conditions: tuple[str, ...],
    model_alias: str,
    judge_alias: str,
    out_path: Path,
    need_ids: tuple[str, ...],
    need_list: Path | None,
    limit: int | None,
    stop_at: int,
    weekly_ceiling: int,
    workers: int,
    workspace: Path | None,
    report_only: bool,
    regrade: bool,
    json_out: Path | None,
) -> None:
    """Run the answer eval as Claude Code runs on your subscription.

    Each agent run and each judge call is a `claude -p` process, so it uses
    your Claude usage limits instead of Bedrock money. Every run reports the
    5-hour and weekly meters; no new run starts once either reaches its
    limit. Run the same command again in a later window: it resumes where
    it stopped and says how much is left.

    \b
    Condition C starts your own `rag-mcp` against the RAG server, as your
    Claude Code sessions do. B gets real copies of the project's PDFs.
    Your CLAUDE.md, settings and other MCP servers are not loaded.
    """
    from datasheet_rag.eval.agent import CONDITION_NAMES, Condition
    from datasheet_rag.eval.answers import (
        GATE_STOPPED,
        build_report,
        load_records,
        regrade_records,
        run_answers,
    )
    from datasheet_rag.eval.claude_code import (
        MODEL_PREFIX,
        ClaudeJudge,
        Meter,
        Workspace,
        probe,
        run_condition,
    )

    needs = _select_needs(needs_path, need_ids, need_list, limit)
    conds: list[Condition] = list(conditions or ("A", "B", "C"))  # type: ignore[arg-type]
    if report_only and regrade:
        raise click.UsageError("--report-only and --regrade exclude each other")
    ids = {n.need_id for n in needs}

    def report(model_id: str | None) -> None:
        records = [r for r in load_records(out_path) if r.need_id in ids]
        models = {r.model for r in records if r.model.startswith(MODEL_PREFIX)}
        if model_id is None:
            if len(models) != 1:
                raise click.ClickException(
                    f"{out_path} holds {len(models)} Claude Code models; nothing to report"
                    if not models
                    else f"{out_path} holds several models ({', '.join(sorted(models))})"
                )
            model_id = models.pop()
        graded = sum(
            1
            for r in records
            if r.model == model_id and r.grade is not None and r.condition in conds
        )
        console.print(f"[cyan]Progress[/]: {graded} of {len(needs) * len(conds)} runs graded.")
        rep = build_report(records, conds, model_id)
        _render_answer_report(rep)
        if json_out:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(rep.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"[green]Report JSON →[/] {json_out}")

    if report_only:
        report(None)
        return

    from datasheet_rag.config import get_settings

    ws = Workspace(workspace or get_settings().rag_home / "eval-claude-code")
    ws.root.mkdir(parents=True, exist_ok=True)
    meter = Meter(stop_at / 100, weekly_ceiling / 100)
    # Resolve the aliases once: they move to newer models over time, and the
    # records must say which model answered. Also gives the first meter reading.
    agent_model = probe(model_alias, workdir=ws.root, meter=meter)
    judge_model = probe(judge_alias, workdir=ws.root, meter=meter)
    model_id, judge_id = MODEL_PREFIX + agent_model, MODEL_PREFIX + judge_model
    console.print(f"[cyan]Usage now[/]: {meter.describe()}")
    earlier = {r.model for r in load_records(out_path) if r.model.startswith(MODEL_PREFIX)}
    if earlier - {model_id}:
        # Resume keys on the model: a newer model behind the same alias would
        # quietly start the whole set again and mix two models in one file.
        old = ", ".join(sorted(m.removeprefix(MODEL_PREFIX) for m in earlier - {model_id}))
        raise click.ClickException(
            f"{out_path} holds runs of {old}, but --model {model_alias} is now {agent_model}. "
            f"Pass --model {old} to finish that run, or use a new --out."
        )
    judge = ClaudeJudge(judge_model, workdir=ws.root, meter=meter)

    if regrade:
        # Skip what this judge already graded, so a regrade the meter stopped
        # resumes in the next window instead of starting over.
        n, failed = regrade_records(
            out_path,
            needs,
            client=judge,
            judge_model=judge_id,
            workers=workers,
            gate=meter.allows,
            skip_same_judge=True,
        )
        console.print(f"[cyan]Regraded[/] {n} answer(s) with {judge_id}.")
        stopped = [f for f in failed if f.endswith(GATE_STOPPED)]
        for f in failed:
            if f not in stopped:
                console.print(f"  [red]judge failed, old grade (if any) kept[/] {f}")
        if stopped:
            console.print(
                f"[yellow]Stopped[/]: {meter.stop_reason()}; "
                f"{len(stopped)} answer(s) keep their old grade."
            )
        report(model_id)
        return

    if not meter.allows():
        console.print(f"[yellow]Not starting[/]: {meter.stop_reason()}.")
        if any(r.model == model_id for r in load_records(out_path)):
            report(model_id)
        return

    if any(c in conds for c in ("B", "C")):
        from datasheet_rag.backend import get_backend

        backend = get_backend()
        project = _shared_project(backend, needs)
        from datasheet_rag.store.fields import display_value

        listing = backend.list_documents(project_id=project).documents
        docs = [
            {"rev_id": d.rev_id, "title": d.doc_title, "mpn": display_value(d.fields.get("mpn"))}
            for d in listing
        ]
        copied = ws.prepare(docs, backend.get_pdf_bytes, project)
        console.print(
            f"[cyan]Workspace[/] {ws.root}: {len(docs)} PDFs of project {project}"
            + (f" ({copied} copied now)" if copied else "")
        )

    names = ", ".join(f"{c} = {CONDITION_NAMES[c]}" for c in conds)
    console.print(
        f"[cyan]Answer eval via Claude Code[/]: {len(needs)} needs × {len(conds)} conditions "
        f"({names}), agent {agent_model}, judge {judge_model}; stop at {stop_at}% of the "
        f"5-hour meter or {weekly_ceiling}% of the week → {out_path}"
    )
    try:
        run_answers(
            needs,
            conds,
            model_id=model_id,
            judge_model=judge_id,
            client=judge,
            agent=lambda need, cond, run: run_condition(
                need, cond, run, model=agent_model, workspace=ws, meter=meter
            ),
            gate=meter.allows,
            out_path=out_path,
            workers=workers,
            progress=_print_answer_progress,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    except KeyboardInterrupt:
        done = sum(1 for r in load_records(out_path) if r.model == model_id and r.grade)
        console.print(
            f"\n[yellow]Interrupted.[/] {done} graded runs are saved in {out_path} and will not "
            "run again. Runs that were in progress were stopped with you and run again next "
            f"time. Usage now: {meter.describe()}"
        )
        raise SystemExit(130) from None
    reason = meter.stop_reason()
    console.print(
        f"[cyan]Usage now[/]: {meter.describe()}"
        + (f". Stopped: {reason}; run again later to continue." if reason else "")
    )
    report(model_id)


def _render_answer_report(report: AnswerReport) -> None:
    from datasheet_rag.eval.agent import CONDITION_NAMES
    from datasheet_rag.eval.dataset import CATEGORIES

    if not report.conditions or report.conditions[0].n == 0:
        console.print("[yellow]No graded runs to report.[/]")
        return
    table = Table(
        title=f"Answer eval · {report.model} · n = {report.conditions[0].n} needs "
        f"(dim = 95% bootstrap interval over needs)"
    )
    table.add_column("condition", style="yellow")
    table.add_column("grounded", justify="right", style="cyan")
    table.add_column("correct", justify="right", style="green")
    table.add_column("context tok\n[dim]cached[/]", justify="right")
    table.add_column("out tok", justify="right")
    table.add_column("tool calls", justify="right")
    table.add_column("wall s", justify="right")
    table.add_column("$ / need", justify="right")
    table.add_column("graded by", style="dim")
    for s in report.conditions:
        table.add_row(
            f"{s.condition} · {CONDITION_NAMES[s.condition]}",
            _mean_cell(s.grounded.value, s.grounded.ci),
            _mean_cell(s.correct.value, s.correct.ci),
            f"{s.mean_context_tokens:,.0f}\n[dim]{s.cache_read_share:.0%}[/]",
            f"{s.mean_output_tokens:,.0f}",
            f"{s.mean_tool_calls:.1f}",
            f"{s.mean_wall_s:.0f}",
            "—" if s.mean_cost_usd is None else f"{s.mean_cost_usd:.3f}",
            ", ".join(f"{k} {v}" for k, v in sorted(s.graders.items()))
            + (f"; {s.not_submitted} never submitted" if s.not_submitted else ""),
            end_section=True,
        )
    console.print(table)

    if report.comparisons:
        diffs = Table(title="Paired difference (same needs; 95% bootstrap interval over needs)")
        diffs.add_column("comparison", style="yellow")
        diffs.add_column("Δgrounded", justify="right")
        diffs.add_column("Δcorrect", justify="right")
        diffs.add_column("Δcontext tok", justify="right")
        diffs.add_column("Δwall s", justify="right")
        diffs.add_column("Δ$ / need", justify="right")
        for cmp in report.comparisons:
            diffs.add_row(
                f"{cmp.later} − {cmp.earlier}",
                _diff_cell(cmp.grounded),
                _diff_cell(cmp.correct),
                f"{cmp.context_tokens:+,.0f}",
                f"{cmp.wall_s:+.0f}",
                "—" if cmp.cost_usd is None else f"{cmp.cost_usd:+.3f}",
                end_section=True,
            )
        console.print(diffs)

    cats = Table(title="Grounded answer rate by category")
    cats.add_column("category", style="magenta")
    for s in report.conditions:
        cats.add_column(s.condition, justify="right")
    for cat in CATEGORIES:
        rates = [s.grounded_by_category.get(cat) for s in report.conditions]
        if all(r is None for r in rates):
            continue
        cats.add_row(
            cat,
            *[_mean_cell(r.value, r.ci) if r else "—" for r in rates],
            end_section=True,
        )
    console.print(cats)

    if report.incomplete:
        console.print(
            f"[yellow]{len(report.incomplete)} need(s) left out[/]: not graded under every "
            "condition yet — rerun to fill them in."
        )
    if report.failed:
        console.print(f"[red]{len(report.failed)} run(s) failed[/] (rerun retries them):")
        for f in report.failed:
            console.print(f"  - {f}")
    if report.ungraded:
        console.print(
            f"[yellow]{len(report.ungraded)} answer(s) not graded[/] (the judge call failed; "
            "the next run grades them without rerunning the agent):"
        )
        for f in report.ungraded:
            console.print(f"  - {f}")
    if report.total_cost_usd is not None:
        console.print(
            f"Spent on these runs: [bold]${report.total_cost_usd:.2f}[/] at list price "
            "(agent + judge, failed runs included)."
        )


def _render_run_comparison(cmp: RunComparison, base_path: Path, variant_path: Path) -> None:
    from datasheet_rag.eval.agent import CONDITION_NAMES

    table = Table(
        title=f"A/B · condition {cmp.condition} ({CONDITION_NAMES[cmp.condition]}) · "
        f"n = {cmp.n} needs (dim = 95% bootstrap interval over needs)"
    )
    table.add_column("run", style="yellow")
    table.add_column("model", style="dim")
    table.add_column("grounded", justify="right", style="cyan")
    table.add_column("correct", justify="right", style="green")
    table.add_column("context tok", justify="right")
    table.add_column("$ / need", justify="right")
    for label, path, s in (("base", base_path, cmp.base), ("variant", variant_path, cmp.variant)):
        table.add_row(
            f"{label}\n[dim]{path.name}[/]",
            s.model + f"\ntemperature {'default' if s.temperature is None else s.temperature}",
            _mean_cell(s.grounded.value, s.grounded.ci),
            _mean_cell(s.correct.value, s.correct.ci),
            f"{s.mean_context_tokens:,.0f}",
            "—" if s.mean_cost_usd is None else f"{s.mean_cost_usd:.3f}",
            end_section=True,
        )
    table.add_row(
        "variant − base",
        "",
        _diff_cell(cmp.grounded),
        _diff_cell(cmp.correct),
        f"{cmp.variant.mean_context_tokens - cmp.base.mean_context_tokens:+,.0f}",
        "—"
        if cmp.base.mean_cost_usd is None or cmp.variant.mean_cost_usd is None
        else f"{cmp.variant.mean_cost_usd - cmp.base.mean_cost_usd:+.3f}",
    )
    console.print(table)
    if cmp.base.temperature != cmp.variant.temperature:
        console.print(
            "[yellow]The two runs used different agent temperatures[/]: the difference "
            "mixes that with whatever else changed."
        )
    if cmp.flipped:
        console.print(f"{len(cmp.flipped)} need(s) changed grounded outcome:")
        for f in cmp.flipped:
            arrow = "[green]gained[/]" if f.variant_grounded else "[red]lost[/]"
            console.print(f"  {arrow} {f.need_id}")
    for label, ids in (("base", cmp.only_base), ("variant", cmp.only_variant)):
        if ids:
            console.print(f"[yellow]{len(ids)} need(s) graded only in {label}[/], left out.")


@eval_group.command("review", short_help="Hand-review the golden set in a web app.")
@_db_option
@click.option(
    "--set",
    "set_path",
    type=click.Path(exists=True, path_type=Path),
    # Not golden-mined.jsonl: its gold comes from evidence quotes, which this
    # app cannot edit, and Accept would relabel its items source="human".
    default=Path("eval/golden-auto.jsonl"),
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
    try:
        body: dict[str, Any] = resp.json()
    except ValueError:
        body = {}
        if resp.status_code < 400:
            raise click.ClickException(
                f"{base} did not return JSON — is RAG_SERVER_URL pointing at a RAG server?"
            ) from None
    if resp.status_code == 404:
        raise click.ClickException(f"404: {body.get('detail', resp.text)}")
    if resp.status_code >= 400:
        raise click.ClickException(f"{resp.status_code}: {resp.text}")
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
@click.option("--rev-id", default=None, help="Filter to one document.")
@click.option("--since", default=None, help="ISO timestamp lower bound (e.g. 2026-06-01).")
@click.option("--limit", default=50, type=int, help="Max rows (newest first).")
@click.option("--token", default=None, help="Admin token (default: RAG_SERVER_TOKEN).")
def audit_cmd(rev_id: str | None, since: str | None, limit: int, token: str | None) -> None:
    """Show the ingest-path audit trail."""
    params: dict[str, Any] = {"limit": limit}
    if rev_id:
        params["rev_id"] = rev_id
    if since:
        params["since"] = since
    data = _admin_request("GET", "/audit", token=token, params=params)
    entries = data.get("entries", [])
    if not entries:
        console.print("[yellow]No audit entries.[/]")
        return
    table = Table(title=f"Audit · {len(entries)} entries")
    for col in ("ts", "action", "status", "key_label", "client_ip", "rev_id", "detail"):
        table.add_column(col, overflow="fold")
    for e in entries:
        table.add_row(
            (e.get("ts") or "")[:19],
            e.get("action") or "",
            e.get("status") or "",
            e.get("key_label") or "—",
            e.get("client_ip") or "—",
            (e.get("rev_id") or "—")[:12],
            e.get("detail_json") or (e.get("error") or "—"),
        )
    console.print(table)
