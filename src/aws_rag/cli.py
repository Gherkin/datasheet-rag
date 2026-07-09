"""CLI for the RAG pipeline."""

from __future__ import annotations

import json
import os
import random
import re
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console
from rich.table import Table

from aws_rag.config import get_settings

if TYPE_CHECKING:
    from aws_rag.costs import CostEstimate

console = Console()

# doc_ids are full SHA-256 content hashes (64 hex chars). We display and
# accept abbreviated forms — like `git log --oneline` short SHAs — and
# resolve unambiguous prefixes to the full hash via `resolve_doc_id`.
SHORT_DOC_ID_LEN = 12


def _resolve_doc_id(conn, doc_id: str) -> str:
    """CLI wrapper around the **store-backed** `resolve_doc_id`: turns
    ambiguity/misses into ClickException.

    Domain: the sqlite index — only resolves docs that have been *embedded*.
    For commands that operate on on-disk cache artifacts (blocks/chunks/outline)
    before a doc is embedded, use `_resolve_cached_doc_id` instead.
    """
    from aws_rag.store import resolve_doc_id

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


def _resolve_cached_doc_id(doc_id: str, suffix: str = "_outline.json") -> str:
    """Resolve a possibly-abbreviated doc_id against the local cache dir.

    Domain: the filesystem cache (``settings.output_dir``). Globs
    ``{output_dir}/{doc_id}*{suffix}`` and returns the single full doc_id that
    matches. Unlike the store-backed `_resolve_doc_id`, this sees documents
    that have a cached artifact but have not been embedded yet — which is the
    normal state for `chunk`/`embed`/`extract-*` inputs.

    Raises ClickException on zero or ambiguous matches (consistent message).
    """
    settings = get_settings()
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


def _backend_for(db_path: Path | None = None):
    """Return the backend a command should use.

    ``--db <path>`` always means "this specific local sqlite file" so it
    builds a LocalBackend on that path; otherwise the configured backend
    (remote when RAG_SERVER_URL is set, else local).
    """
    if db_path is not None:
        from aws_rag.backend import LocalBackend

        return LocalBackend(db_path)
    from aws_rag.backend import get_backend

    return get_backend()


def _require_local_db(db_path: Path | None) -> Path:
    """Return a concrete local sqlite path for commands that need raw index
    access (the eval harness: tunable RRF weights, variant-store builds).

    In remote mode these can't run against the HTTP API, so require an
    explicit local --db pointing at a copy of the corpus.
    """
    from aws_rag.backend import backend_mode

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


def _friendly_server_error(e) -> click.ClickException:
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


def _backend_resolve(be, doc_id: str) -> str:
    """Resolve a doc_id prefix via the backend, turning errors into ClickException."""
    from aws_rag.backend import RagServerError

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
    return doc_id[:SHORT_DOC_ID_LEN] + chunk_id[len(doc_id):]


def _resolve_chunk_id(be, chunk_id: str) -> str:
    """Resolve a chunk_id whose doc_id portion may be abbreviated (as
    printed by `rag search` / `rag list-figures`) to its full form.

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


@click.group()
@click.option("--bucket", envvar="RAG_S3_BUCKET", default=None, help="Override S3 bucket name.")
@click.pass_context
def cli(ctx: click.Context, bucket: str | None) -> None:
    """AWS RAG Pipeline — electronics datasheet ingestion."""
    if bucket:
        import os
        os.environ["RAG_S3_BUCKET"] = bucket
    # Remind the user when they're using the local sqlite file rather than a
    # shared server (printed to stderr, non-failing). Skipped for `init`,
    # which is the command that sets the server up.
    if ctx.invoked_subcommand != "init":
        from aws_rag.backend import emit_local_notice

        emit_local_notice()


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def _config_env_lines() -> list[str]:
    """Render every Settings field as a commented-out config.env line.

    Schema-driven so the template never drifts from the model: each field
    becomes ``# RAG_X=<default>   # <terse description>``. The caller fills
    in (uncomments) the handful of critical fields it prompted for.
    """
    from aws_rag.config import Settings

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


@cli.command()
@click.option("--force", is_flag=True, help="Overwrite an existing config.env without prompting.")
def init(force: bool) -> None:
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

    console.rule("[bold magenta]Configure aws-rag[/]")
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
            "[dim]Embeddings run on the server in remote mode — no local "
            "model config needed.[/]"
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
        "# aws-rag configuration — generated by `rag init`.",
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
# Upload
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("pdf_paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--doc-id", default=None, help="Explicit document ID (default: content hash).")
def upload(pdf_paths: tuple[Path, ...], doc_id: str | None) -> None:
    """Upload one or more PDFs to S3."""
    from aws_rag.storage import upload_pdf

    for pdf_path in pdf_paths:
        if not pdf_path.suffix.lower() == ".pdf":
            console.print(f"[red]Skipping non-PDF:[/] {pdf_path}")
            continue
        did, key = upload_pdf(pdf_path, doc_id=doc_id)
        console.print(f"  doc_id = {did}")
        console.print(f"  s3_key = {key}")
        console.print()


# ---------------------------------------------------------------------------
# Analyze (Textract)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("target", type=str)
@click.option(
    "--mode",
    type=click.Choice(["sync", "async"]),
    default="async",
    help="sync = local single-page PDF, async = S3 multi-page.",
)
@click.option("--wait/--no-wait", default=True, help="Wait for async job to complete.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def analyze(target: str, mode: str, wait: bool, output: Path | None) -> None:
    """Run Textract analysis on a PDF.

    TARGET's meaning follows --mode:
      • sync  — TARGET is a local PDF path; its doc_id is its content hash.
      • async — TARGET is a doc_id (full hash or unambiguous prefix) of a PDF
                already uploaded to S3 with `rag upload`.

    Either way the blocks are saved to the cached ``{doc_id}_blocks.json`` (or
    --output).
    """
    from aws_rag.textract import (
        analyze_document_sync,
        get_job_results,
        save_blocks,
        start_analysis,
        wait_for_job,
    )

    settings = get_settings()

    if mode == "sync":
        from aws_rag.docling_parser import content_hash

        pdf_path = Path(target)
        if not pdf_path.is_file():
            raise click.BadParameter(f"File not found: {pdf_path}")
        doc_id = content_hash(pdf_path)
        response = analyze_document_sync(pdf_path)
        blocks = response.get("Blocks", [])
    else:
        # async — target is a doc_id (prefix-resolved against S3 uploads).
        from aws_rag.storage import list_documents

        docs = list_documents()
        matches = [d for d in docs if d["doc_id"].startswith(target)]
        if not matches:
            raise click.BadParameter(
                f"doc_id '{target}' not found. Upload the PDF first with `rag upload`."
            )
        if len({d["doc_id"] for d in matches}) > 1:
            names = ", ".join(sorted({d["doc_id"][:SHORT_DOC_ID_LEN] for d in matches}))
            raise click.BadParameter(f"doc_id '{target}' is ambiguous — matches: {names}")
        doc_id = matches[0]["doc_id"]

        # Find the actual PDF key under the prefix
        from aws_rag.aws import s3_client

        client = s3_client()
        prefix = matches[0]["prefix"]
        resp = client.list_objects_v2(Bucket=settings.s3_bucket, Prefix=prefix)
        pdf_keys = [
            obj["Key"]
            for obj in resp.get("Contents", [])
            if obj["Key"].lower().endswith(".pdf")
        ]
        if not pdf_keys:
            raise click.ClickException(f"No PDF found under s3://{settings.s3_bucket}/{prefix}")

        s3_key = pdf_keys[0]
        job_id = start_analysis(doc_id, s3_key)

        if not wait:
            console.print(f"Job ID: {job_id}")
            console.print("Use `rag job-status` to check progress.")
            return

        status = wait_for_job(job_id)
        if status != "SUCCEEDED":
            raise click.ClickException(f"Textract job failed with status: {status}")

        blocks = get_job_results(job_id)

    # Save output
    if output is None:
        output = _cache_path(doc_id, "_blocks.json")

    save_blocks(blocks, output)


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------


@cli.command("list")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--project-id", default=None,
              help="Restrict to a project (default: scoped by .rag.toml if present).")
@click.option("--global", "-g", "is_global", is_flag=True,
              help="Show every project, ignoring any .rag.toml scoping.")
@click.option("--s3", "show_s3", is_flag=True,
              help="List raw S3 uploads instead (debug — includes documents not yet ingested).")
def list_docs(db_path: Path | None, project_id: str | None, is_global: bool, show_s3: bool) -> None:
    """List ingested documents (searchable in the store)."""
    from aws_rag.project_config import resolve_cli_project_id
    project_id = resolve_cli_project_id(project_id, is_global=is_global)

    if show_s3:
        from aws_rag.storage import list_documents

        docs = list_documents()
        if not docs:
            console.print("[yellow]No documents found in S3.[/]")
            return

        table = Table(title="S3 Uploads")
        table.add_column("doc_id", style="cyan")
        table.add_column("S3 Prefix")
        for doc in docs:
            table.add_row(doc["doc_id"], doc["prefix"])
        console.print(table)
        return

    docs = _backend_for(db_path).get_ingested_docs(project_id=project_id)

    if not docs:
        console.print("[yellow]No ingested documents found.[/] Run [cyan]rag ingest[/] first "
                      "(or pass --s3 to see raw uploads).")
        return

    table = Table(title="Ingested Documents")
    table.add_column("doc_id", style="cyan")
    table.add_column("title")
    table.add_column("chunks", justify="right")
    table.add_column("pages", justify="right")
    table.add_column("ingested")

    for doc in docs:
        table.add_row(
            doc.doc_id[:SHORT_DOC_ID_LEN],
            doc.doc_title or "—",
            str(doc.chunk_count),
            str(doc.page_count) if doc.page_count is not None else "—",
            doc.ingested_at or "—",
        )

    console.print(table)


@cli.command("stats")
@click.option("--project-id", default=None,
              help="Restrict to a project (default: scoped by .rag.toml if present).")
@click.option("--global", "-g", "is_global", is_flag=True,
              help="Show stats across every project, ignoring any .rag.toml scoping.")
@click.option("--doc-id", default=None, help="Restrict to a single document.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
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
    from aws_rag.backend import RagServerError
    from aws_rag.project_config import resolve_cli_project_id

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


@cli.group("get", cls=AliasedGroup)
def get_group() -> None:
    """Fetch a document, page, chunk, or figure and save/show it.

    Subcommands: doc (or document), page, chunk, fig (or figure).
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
            capture_output=True, text=True, timeout=2,
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if "inet" in parts:
                ips.add(parts[parts.index("inet") + 1].split("/")[0])
    except (OSError, ValueError):
        pass

    ips.discard("127.0.0.1")
    return ["127.0.0.1", *sorted(ips)]


@get_group.command("doc")
@click.argument("doc_id", type=str)
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None,
              help="Destination file or directory (default: ./<short_doc_id>.pdf). "
                   "Ignored with --host.")
@click.option("--host", is_flag=True,
              help="Serve the document instead of downloading it — starts the local "
                   "PDF.js viewer and prints browser URLs (see below).")
@click.option("--page", default=1, type=int, help="1-based page to open to (--host only).")
@click.option("--launch/--no-launch", default=True,
              help="With --host, open the 127.0.0.1 URL in your default browser "
                   "(default on — skip this if you're connecting from a different "
                   "machine). No effect without --host.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
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

        from aws_rag import pdf_viewer

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
            console.print(
                f"  http://{ip}:{port}/viewer/{doc_id}#page={page}", soft_wrap=True
            )

        if launch:
            webbrowser.open(local_url)
            console.print("[dim]Opened the 127.0.0.1 link in your default browser "
                          "(use --no-launch to skip this if you're connecting remotely).[/]")
        console.print("[dim]Serving from this process — keep it running to keep the link alive. Ctrl+C to stop.[/]")

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


@get_group.command("page")
@click.argument("doc_id", type=str)
@click.argument("page_arg", metavar="PAGE", type=int, required=False)
@click.option("--page", "page_opt", type=int, default=None,
              help="1-based page number (alternative to the positional PAGE argument).")
@click.option("--output", "-o", "output_path", type=click.Path(path_type=Path), default=None,
              help="Where to save the image. Defaults to a name derived from the "
                   "doc_id and page in the current directory. If a directory, the "
                   "default filename is placed inside it.")
@click.option("--dpi", default=150, type=int, help="Render DPI.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
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
        raise click.UsageError(
            "Pass PAGE either positionally or via --page, not both."
        )
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


@cli.command("delete")
@click.argument("doc_id", type=str)
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting anything.")
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


# ---------------------------------------------------------------------------
# Extract text (from saved Textract JSON)
# ---------------------------------------------------------------------------


@cli.command("extract-text")
@click.argument("doc_id", type=str)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def extract_text(doc_id: str, output: Path | None) -> None:
    """Extract readable text from a document's Textract blocks, preserving layout order.

    DOC_ID is a doc_id (full hash or unambiguous prefix); the blocks are read
    from the cached ``{doc_id}_blocks.json``. An explicit blocks-JSON path is
    also accepted in place of a doc_id.
    """
    from aws_rag.textract import build_text_from_layout

    _, blocks_json = _doc_input(doc_id, "_blocks.json")
    with open(blocks_json) as f:
        blocks = json.load(f)

    text = build_text_from_layout(blocks)

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
    for suffix, backend in (("_outline.json", "docling"), ("_blocks.json", "textract")):
        if sorted(settings.output_dir.glob(f"{arg}*{suffix}")):
            doc_id = _resolve_cached_doc_id(arg, suffix)
            return doc_id, backend, _cache_path(doc_id, suffix)

    raise click.ClickException(
        f"No cached layout artifact matching doc_id {arg!r} in "
        f"{settings.output_dir}. Run `rag ingest` for this document first "
        "(Docling writes '_outline.json', Textract writes '_blocks.json')."
    )


# Rows of the section/layout listing to show before truncating (unless --full).
_INSPECT_LAYOUT_PREVIEW = 30


def _inspect_textract_layout(blocks: list, full: bool = False) -> None:
    """Render the block-type summary and layout hierarchy for Textract output."""
    from aws_rag.textract import extract_layout_elements

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


def _inspect_docling_layout(outline, full: bool = False) -> None:
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
            console.print(
                f"{indent}{title}  [dim]{pages}, {section.element_count} elements[/]"
            )
        if len(flat) > len(shown):
            console.print(f"  … and {len(flat) - len(shown)} more (pass --full to show all)")


@cli.command("inspect-layout")
@click.argument("doc_id", type=str)
@click.option("--full", is_flag=True, default=False,
              help="Show the entire section/layout listing instead of the first "
                   f"{_INSPECT_LAYOUT_PREVIEW} rows.")
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
        from aws_rag.chunking.layout_parser import DocumentOutline

        _inspect_docling_layout(DocumentOutline.from_dict(data["outline"]), full=full)
    else:
        # Accept both the plain list `save_blocks` writes and a raw Textract
        # API response dict (e.g. a file fetched independently of `rag ingest`).
        blocks = data.get("Blocks", []) if isinstance(data, dict) else data
        _inspect_textract_layout(blocks, full=full)


# ---------------------------------------------------------------------------
# Extract figures
# ---------------------------------------------------------------------------


@cli.command("extract-figures")
@click.argument("doc_id", type=str)
@click.option("--pdf", "pdf_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="Source PDF path (defaults to the cached {pdf_dir}/{doc_id}.pdf).")
@click.option("--dpi", default=300, type=int, help="Render DPI for PDF pages.")
@click.option("--format", "image_format", default="png", type=click.Choice(["png", "jpg", "webp"]))
@click.option("--padding", default=0.02, type=float, help="Padding around figures (fraction of page).")
@click.option("--upload/--no-upload", default=False,
              help="Also upload figures to S3 (opt-in — the local store under "
                   "~/.rag/figures/ is the default and MCP reads from it directly).")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None,
              help="Output directory for figure images (defaults to the doc's figures cache).")
def extract_figures_cmd(
    doc_id: str,
    pdf_path: Path | None,
    dpi: int,
    image_format: str,
    padding: float,
    upload: bool,
    output_dir: Path | None,
) -> None:
    """Extract figure images from a PDF using Textract layout detection.

    DOC_ID is a doc_id (full hash or unambiguous prefix); the blocks are read
    from the cached ``{doc_id}_blocks.json`` and the source PDF from
    ``{pdf_dir}/{doc_id}.pdf`` (override with --pdf). An explicit blocks-JSON
    path is also accepted in place of a doc_id.

    Crops each LAYOUT_FIGURE region and saves as individual images.
    Generates a manifest JSON with metadata, captions, and context.
    """
    from aws_rag.figures import extract_figures, upload_figures_to_s3

    doc_id, blocks_json = _doc_input(doc_id, "_blocks.json")
    with open(blocks_json) as f:
        blocks = json.load(f)

    settings = get_settings()
    if pdf_path is None:
        pdf_path = settings.pdf_dir / f"{doc_id}.pdf"
        if not pdf_path.is_file():
            raise click.ClickException(
                f"No cached PDF for doc_id={doc_id} at {pdf_path}. Pass --pdf "
                "with the source PDF, or run `rag ingest`/`rag upload` first."
            )

    manifest = extract_figures(
        pdf_path=pdf_path,
        blocks=blocks,
        doc_id=doc_id,
        output_dir=output_dir,
        dpi=dpi,
        image_format=image_format,
        padding_pct=padding,
    )

    if upload and manifest.figures:
        manifest = upload_figures_to_s3(manifest)

    # Save manifest
    manifest_dir = output_dir or settings.output_dir / "figures" / doc_id
    manifest.save(manifest_dir / "manifest.json")

    # Summary table
    if manifest.figures:
        table = Table(title=f"Extracted Figures ({len(manifest.figures)})")
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Page")
        table.add_column("Size")
        table.add_column("Caption")
        table.add_column("Section")

        for i, fig in enumerate(manifest.figures):
            table.add_row(
                str(i),
                str(fig.region.page),
                f"{fig.width_px}×{fig.height_px}",
                (fig.region.caption[:60] + "…") if len(fig.region.caption) > 60 else fig.region.caption or "—",
                fig.region.section_header[:40] or "—",
            )

        console.print(table)


# ---------------------------------------------------------------------------
# Chunk (multi-scale chunking pipeline)
# ---------------------------------------------------------------------------


@cli.command("chunk")
@click.argument("doc_id", type=str)
@click.option(
    "--figures-manifest",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to figure manifest JSON from extract-figures.",
)
@click.option("--micro-tokens", default=128, type=int, help="Max tokens per MICRO chunk.")
@click.option("--meso-tokens", default=512, type=int, help="Max tokens per MESO chunk.")
@click.option(
    "--summarizer",
    type=click.Choice(["extractive", "abstractive"]),
    default="extractive",
    help="Summarization mode for MACRO chunks.",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
def chunk_cmd(
    doc_id: str,
    figures_manifest: Path | None,
    micro_tokens: int,
    meso_tokens: int,
    summarizer: str,
    output: Path | None,
) -> None:
    """Run the multi-scale chunking pipeline on a document's Textract blocks.

    DOC_ID is a doc_id (full hash or unambiguous prefix); the blocks are read
    from the cached ``{doc_id}_blocks.json`` and chunks are written to
    ``{doc_id}_chunks.json`` unless --output is given. An explicit blocks-JSON
    path is also accepted in place of a doc_id.

    Produces a hierarchical chunk graph at three levels (MACRO/MESO/MICRO)
    with navigation links, context enrichment, and chapter summaries.
    """
    from aws_rag.chunking.pipeline import run_chunking_pipeline, save_chunk_graph
    from aws_rag.chunking.splitter import SplitterConfig

    doc_id, blocks_json = _doc_input(doc_id, "_blocks.json")
    with open(blocks_json) as f:
        blocks = json.load(f)

    # Load figure manifest if provided
    figure_manifest = None
    if figures_manifest:
        with open(figures_manifest) as f:
            figure_manifest = json.load(f)

    config = SplitterConfig(
        micro_max_tokens=micro_tokens,
        meso_max_tokens=meso_tokens,
    )

    graph = run_chunking_pipeline(
        blocks,
        doc_id=doc_id,
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
    from aws_rag.models.chunk import ChunkLevel

    macros = graph.by_level(ChunkLevel.MACRO)
    if macros:
        console.print("\n[bold]Chapter Summaries:[/]")
        for m in macros:
            console.print(f"\n[cyan]{m.metadata.chapter_title}[/] (pages {m.metadata.page_numbers})")
            if m.text:
                preview = m.text[:300] + "…" if len(m.text) > 300 else m.text
                console.print(f"  {preview}")
            else:
                console.print("  [yellow](no summary generated)[/]")


# ---------------------------------------------------------------------------
# Embed (Bedrock Titan + SQLite store)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("doc_id", type=str)
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None,
              help="SQLite DB path. Defaults to settings.sqlite_db_path.")
@click.option("--project-id", default=None, help="Project ID to attach to every chunk.")
@click.option("--group", "group_name", default=None, help="Group name to attach to every chunk.")
@click.option("--verbose/--quiet", default=True, help="Print per-batch progress.")
@click.option("--dry-run", is_flag=True, help="Embed but do not write to the store.")
def embed(
    doc_id: str,
    db_path: Path | None,
    project_id: str | None,
    group_name: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Embed a document's chunk graph (produced by `rag chunk`) and store it.

    DOC_ID is a doc_id (full hash or unambiguous prefix); the chunk graph is
    read from the cached ``{doc_id}_chunks.json``. An explicit chunks-JSON path
    is also accepted in place of a doc_id.

    Embedding + insert run through the backend, so this writes to the remote
    server (which embeds) when RAG_SERVER_URL is set, or the local sqlite
    store otherwise.
    """
    from aws_rag.backend import MetadataPatch, backend_mode
    from aws_rag.chunking.pipeline import load_chunk_graph
    from aws_rag.project_config import get_project_config

    _, chunks_json = _doc_input(doc_id, "_chunks.json")

    proj_cfg = get_project_config()
    if proj_cfg is not None:
        project_id = project_id or proj_cfg.project_id
        group_name = group_name or proj_cfg.group

    console.print(f"Loading chunk graph from [cyan]{chunks_json}[/]…")
    graph = load_chunk_graph(chunks_json)
    stats = graph.stats()
    console.print(f"  {stats['total_chunks']} chunks "
                  f"(MACRO {stats['by_level']['MACRO']}, "
                  f"MESO {stats['by_level']['MESO']}, "
                  f"MICRO {stats['by_level']['MICRO']})")

    if dry_run:
        raise click.ClickException(
            "embed --dry-run is no longer supported (embedding now happens via "
            "the backend, server-side in remote mode). Use `rag ingest --dry-run "
            "--show-cost` for an estimate without writing."
        )

    be = _backend_for(db_path)

    # Remote: ship figure images so the server stores them and the embedded
    # context_text can fold in any server-side descriptions.
    figures_upload: dict[str, tuple[bytes, str]] | None = None
    if backend_mode() == "remote" and db_path is None:
        figures_upload = {}
        for c in graph.chunks.values():
            if c.figure_image_path:
                p = Path(c.figure_image_path)
                if p.is_file():
                    figures_upload[c.id] = (p.read_bytes(), p.suffix.lstrip(".") or "png")

    console.print("Embedding & writing via the backend…")
    from aws_rag.backend import RagServerError

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


# ---------------------------------------------------------------------------
# Search (hybrid / vector / keyword)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("query", type=str)
@click.option("--mode", type=click.Choice(["hybrid", "vector", "keyword"]),
              default="hybrid", help="Retrieval mode.")
@click.option("-k", "top_k", default=10, type=int, help="Number of results.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None,
              help="SQLite DB path. Defaults to settings.sqlite_db_path.")
@click.option("--project-id", default=None,
              help="Restrict to a project (default: scoped by .rag.toml if present).")
@click.option("--global", "-g", "is_global", is_flag=True,
              help="Search every project, ignoring any .rag.toml scoping.")
@click.option("--group", "group_name", default=None)
@click.option("--doc-id", "doc_ids", multiple=True, help="Restrict to one or more doc IDs.")
@click.option("--level", type=click.Choice(["macro", "meso", "micro"]),
              default=None, help="Restrict to a single zoom level.")
@click.option("--show-context/--no-show-context", default=False,
              help="Show context_text (full embedding-ready blob) instead of raw text.")
def search(
    query: str,
    mode: str,
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
    from aws_rag.backend import RagServerError
    from aws_rag.models.chunk import ChunkLevel
    from aws_rag.project_config import resolve_cli_project_id
    from aws_rag.store import SearchFilters

    project_id = resolve_cli_project_id(project_id, is_global=is_global)
    be = _backend_for(db_path)

    resolved_doc_ids = [be.resolve_doc_id(d) for d in doc_ids]

    level_enum = None
    if level:
        level_enum = {"macro": ChunkLevel.MACRO, "meso": ChunkLevel.MESO,
                      "micro": ChunkLevel.MICRO}[level]

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


def _print_chunk_detail(chunk, *, show_context: bool = False) -> None:
    """Render one chunk's metadata + text — the CLI counterpart of the MCP
    server's ``_shape_chunk``."""
    from aws_rag.models.chunk import LayoutType

    pages = chunk.metadata.page_numbers
    page = (str(pages[0]) if len(pages) == 1
            else f"{pages[0]}-{pages[-1]}" if pages else "—")

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


@get_group.command("chunk")
@click.argument("chunk_id", type=str)
@click.option("--neighbors/--no-neighbors", default=False,
              help="Also print the parent/prev/next chunks (mirrors the MCP "
                   "get_chunk tool's include_neighbors option).")
@click.option("--show-context/--no-show-context", default=False,
              help="Print context_text (embedding-ready blob) instead of raw text.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
def get_chunk_cmd(
    chunk_id: str,
    neighbors: bool,
    show_context: bool,
    db_path: Path | None,
) -> None:
    """Fetch one chunk by ID — the CLI equivalent of the MCP `get_chunk` tool.

    CHUNK_ID accepts the full id or an abbreviated form using a doc_id
    prefix, e.g. ``ab12cd34ef56:L2:143`` as printed by `rag search` /
    `rag list-figures`.
    """
    from aws_rag.backend import RagServerError

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


@cli.command("list-figures")
@click.option("--doc-id", default=None)
@click.option("--project-id", default=None,
              help="Restrict to a project (default: scoped by .rag.toml if present).")
@click.option("--global", "-g", "is_global", is_flag=True,
              help="List figures across every project, ignoring any .rag.toml scoping.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--missing-description-only", is_flag=True,
              help="Only show figure chunks whose figure_description is empty.")
def list_figures_cmd(
    doc_id: str | None,
    project_id: str | None,
    is_global: bool,
    db_path: Path | None,
    missing_description_only: bool,
) -> None:
    """List figure chunks in the store (those with a usable image)."""
    from aws_rag.project_config import resolve_cli_project_id

    project_id = resolve_cli_project_id(project_id, is_global=is_global)

    be = _backend_for(db_path)
    if doc_id:
        doc_id = _backend_resolve(be, doc_id)
    figs = be.list_figure_chunks(doc_id=doc_id, project_id=project_id)
    if missing_description_only:
        figs = [c for c in figs if not c.figure_description]

    if not figs:
        console.print("[yellow]No figure chunks match.[/]")
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
        page = (str(pages[0]) if len(pages) == 1
                else f"{pages[0]}-{pages[-1]}" if pages else "")
        src = "local" if c.figure_image_path else ("s3" if c.figure_s3_key else "—")
        table.add_row(
            _short_chunk_id(c.id, c.doc_id),
            page,
            (c.metadata.section_title or "")[:30],
            (c.figure_caption or "")[:40],
            "[green]Y[/]" if c.figure_description else "[red]N[/]",
            src,
        )
    console.print(table)


@get_group.command("fig")
@click.argument("chunk_id", type=str)
@click.option("--output", "-o", "output_path", type=click.Path(path_type=Path), default=None,
              help="Where to save the image. Defaults to a name derived from the "
                   "chunk_id in the current directory. If a directory, the default "
                   "filename is placed inside it.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
def get_figure_cmd(chunk_id: str, output_path: Path | None, db_path: Path | None) -> None:
    """Fetch a figure chunk's image and save it to disk.

    CHUNK_ID accepts the full id or an abbreviated form using a doc_id
    prefix, e.g. ``ab12cd34ef56:L2:143`` as printed by `rag list-figures` /
    `rag search`. This is the CLI equivalent of the MCP `get_figure` tool.
    """
    from aws_rag.backend import RagServerError

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


@cli.command("describe-figures")
@click.option("--doc-id", default=None, help="Restrict to a single document.")
@click.option("--project-id", default=None, help="Restrict to a single project.")
@click.option("--missing-only/--all", default=True,
              help="Skip figures that already have a description (default on).")
@click.option("--limit", default=None, type=int,
              help="Stop after this many figures (cost guard).")
@click.option("--model", "model_id", default=None,
              help="Override settings.description_model_id for this run.")
@click.option("--dry-run", is_flag=True,
              help="Generate descriptions and print them but do not persist.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--verbose/--quiet", default=True)
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

    After running, re-embed the affected document so the new
    descriptions show up in vector search:

        rag describe-figures --doc-id <doc>
        rag chunk <doc> --figures-manifest ...   # if needed
        rag embed <doc> --project-id <p>
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
        console.print(
            "  Re-run `rag embed <doc-id>` (or implement an "
            "updated-only re-embed) to refresh the vectors."
        )


# ---------------------------------------------------------------------------
# Ingest (full pipeline: upload → analyze → extract-figures → chunk → embed)
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
        "aws_rag.costs), not live lookups. Verify in your AWS console "
        "before budgeting at volume.[/]"
    )


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--doc-id", default=None, help="Explicit document ID (default: content hash).")
@click.option("--project-id", default=None, help="Project ID attached to all chunks.")
@click.option("--group", "group_name", default=None, help="Group name attached to all chunks.")
@click.option("--mpn", default=None, help="Manufacturer part number, e.g. STM32H743VIT6.")
@click.option("--manufacturer", default=None)
@click.option("--subsystem", default=None, help="e.g. power, rf, mcu.")
@click.option("--doc-type", default=None,
              help="datasheet | reference-manual | errata | app-note | …")
@click.option("--tag", "tags", multiple=True, help="Repeatable --tag flag.")
@click.option("--skip-figures", is_flag=True, help="Skip figure extraction and description steps.")
@click.option("--upload-figures/--no-upload-figures", default=False,
              help="Also upload extracted figures to S3 (opt-in — figures live "
                   "locally under ~/.rag/figures/ by default, which is what MCP "
                   "reads from; only useful for sharing a store across machines).")
@click.option("--skip-describe", is_flag=True, help="Skip AI figure description (but still extract).")
@click.option("--infer-title", is_flag=True,
              help="If the document has no usable title after chunking, infer one "
                   "with a small Bedrock Claude call against the first page "
                   "(one extra LLM call; off by default — see `rag fix-titles` "
                   "to backfill existing documents).")
@click.option("--dpi", default=300, type=int, help="Render DPI for figure extraction.")
@click.option("--micro-tokens", default=128, type=int, help="Max tokens per MICRO chunk.")
@click.option("--meso-tokens", default=512, type=int, help="Max tokens per MESO chunk.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None,
              help="SQLite DB path. Defaults to settings.sqlite_db_path.")
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
@click.option("--force", is_flag=True, help="Ignore cached blocks/chunks and redo all steps.")
@click.option(
    "--backend",
    type=click.Choice(["auto", "docling", "textract"], case_sensitive=False),
    default="docling",
    help=(
        "Layout extraction backend. "
        "'docling' (default) handles native PDFs for free and fails verbosely "
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
    backend: str,
    accurate_tables: bool | None,
) -> None:
    """Full ingestion pipeline: analyse → figures → chunk → embed.

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
    common = dict(
        project_id=project_id, group_name=group_name, mpn=mpn, manufacturer=manufacturer,
        subsystem=subsystem, doc_type=doc_type, tags=tags, skip_figures=skip_figures,
        upload_figures=upload_figures, skip_describe=skip_describe, infer_title=infer_title,
        dpi=dpi, micro_tokens=micro_tokens, meso_tokens=meso_tokens, db_path=db_path,
        dry_run=dry_run, show_cost=show_cost, force=force, backend=backend,
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

    from aws_rag.costs import CostEstimate, CostLineItem
    from aws_rag.docling_parser import content_hash

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
        result = _ingest_one(pdf, doc_id=None, **common)
        if result is not None:
            total_cost.items.extend(result.items)
            total_cost.notes.extend(result.notes)

    if show_cost:
        by_label: dict[str, list[CostLineItem]] = {}
        for item in total_cost.items:
            by_label.setdefault(item.label, []).append(item)
        merged = CostEstimate(notes=total_cost.notes)
        for label, line_items in by_label.items():
            merged.items.append(CostLineItem(
                label=label,
                detail=f"summed across {len(line_items)} documents",
                usd=sum(li.usd for li in line_items),
            ))
        _print_cost_table(
            merged,
            heading=(
                f"Estimated AWS cost — combined across "
                f"{len(pdf_files) - skipped} of {len(pdf_files)} documents"
            ),
        )
    console.rule(
        f"[bold green]Bulk ingest done[/] — {len(pdf_files)} PDFs, "
        f"{len(pdf_files) - skipped} processed, {skipped} skipped"
    )


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
    backend: str,
    accurate_tables: bool | None,
) -> CostEstimate | None:
    """Ingest a single PDF; returns the cost estimate when --show-cost is set."""
    import time

    from aws_rag.chunking.pipeline import (
        load_chunk_graph,
        run_chunking_pipeline,
        run_chunking_pipeline_from_outline,
        save_chunk_graph,
    )
    from aws_rag.chunking.splitter import SplitterConfig
    from aws_rag.costs import (
        CostEstimate,
        estimate_embedding_cost,
        estimate_figure_description_cost,
        estimate_textract_cost,
        estimate_title_inference_cost,
        pdf_page_count,
    )
    from aws_rag.figures import extract_figures, extract_figures_from_regions, upload_figures_to_s3
    from aws_rag.project_config import get_project_config_for
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
            "[yellow]--show-cost: estimating AWS spend, skipping priced "
            "Bedrock/Textract calls.[/]"
        )

    settings = get_settings()
    if accurate_tables is None:
        accurate_tables = settings.table_structure_mode == "accurate"
    t0 = time.monotonic()
    step_n = 0
    cost = CostEstimate()
    figure_count = 0
    # Set in the docling path; the textract path leaves them empty.
    running_header = ""
    pdf_meta_title = ""

    def _step(label: str) -> None:
        nonlocal step_n
        step_n += 1
        console.rule(f"[bold cyan]Step {step_n} — {label}[/]")

    # ── 1. Detect backend ────────────────────────────────────────────────────
    _step("Detect PDF type")
    if backend in ("auto", "docling"):
        from aws_rag.docling_parser import is_native_pdf
        native = is_native_pdf(pdf_path)
        if native:
            resolved_backend = "docling"
            console.print("  Native PDF detected → using [cyan]docling[/] backend")
        elif backend == "auto":
            resolved_backend = "textract"
            console.print("  Scanned PDF detected → using [cyan]textract[/] backend")
        else:
            raise click.ClickException(
                f"{pdf_path.name} looks like a scanned PDF — Docling needs a "
                "native text layer and cannot OCR it.\n"
                "  Re-run with --backend textract to use AWS Textract OCR "
                "instead (this incurs AWS costs), or --backend auto to route "
                "automatically based on PDF type."
            )
    else:
        resolved_backend = backend
        console.print(f"  Backend forced to [cyan]{resolved_backend}[/]")

    # ── 2a. Docling path (native PDFs) ───────────────────────────────────────
    if resolved_backend == "docling":
        import dataclasses

        from aws_rag.chunking.layout_parser import DocumentOutline
        from aws_rag.docling_parser import content_hash, convert_pdf
        from aws_rag.figures import FigureRegion

        did = doc_id or content_hash(pdf_path)
        console.print(f"  doc_id = [cyan]{did}[/]")

        from aws_rag.storage import save_pdf_locally
        save_pdf_locally(pdf_path, did)

        chunks_path = settings.output_dir / f"{did}_chunks.json"
        running_header = ""
        pdf_meta_title = ""
        if chunks_path.exists() and not force:
            _step("Multi-scale chunking")
            console.print(f"  [yellow]Resuming — loading cached chunk graph[/] → [cyan]{chunks_path}[/]")
            graph = load_chunk_graph(chunks_path)
            stats = graph.stats()
            console.print(
                f"  {stats['total_chunks']} chunks "
                f"(MACRO {stats['by_level']['MACRO']}, "
                f"MESO {stats['by_level']['MESO']}, "
                f"MICRO {stats['by_level']['MICRO']}) (cached)"
            )
        else:
            _step("Docling layout analysis")
            outline_path = settings.output_dir / f"{did}_outline.json"
            if outline_path.exists() and not force:
                console.print(
                    f"  [yellow]Resuming — loading cached layout analysis[/] → [cyan]{outline_path}[/]"
                )
                with open(outline_path) as f:
                    cached = json.load(f)
                outline = DocumentOutline.from_dict(cached["outline"])
                figure_regions = [FigureRegion(**r) for r in cached["figure_regions"]]
            else:
                outline, figure_regions = convert_pdf(
                    pdf_path, doc_id=did, accurate_tables=accurate_tables
                )
                outline_path.parent.mkdir(parents=True, exist_ok=True)
                with open(outline_path, "w") as f:
                    json.dump(
                        {
                            "outline": outline.to_dict(),
                            "figure_regions": [dataclasses.asdict(r) for r in figure_regions],
                        },
                        f,
                    )
                console.print(f"  Layout analysis cached → [cyan]{outline_path}[/]")

            running_header = outline.running_header
            pdf_meta_title = outline.pdf_meta_title
            summary = outline.summary()
            console.print(
                f"  {summary['top_level_sections']} chapters, "
                f"{summary['total_sections']} sections, "
                f"{summary['total_elements']} elements "
                f"({summary['elements_by_type'].get('formula', 0)} formulas, "
                f"{summary['elements_by_type'].get('table', 0)} tables, "
                f"{summary['elements_by_type'].get('figure', 0)} figures)"
            )

            figure_manifest_dict = None
            if not skip_figures:
                _step("Extract figures & formulas")
                figures_out = settings.figures_dir / did
                manifest = extract_figures_from_regions(
                    pdf_path=pdf_path,
                    regions=figure_regions,
                    doc_id=did,
                    output_dir=figures_out,
                    dpi=dpi,
                    image_format="png",
                    padding_pct=0.02,
                )
                if upload_figures and manifest.figures:
                    manifest = upload_figures_to_s3(manifest)
                manifest_path = figures_out / "manifest.json"
                manifest.save(manifest_path)
                figure_manifest_dict = manifest.to_dict()
                figure_count = len(manifest.figures)
                console.print(f"  {len(manifest.figures)} regions → [cyan]{manifest_path}[/]")

            _step("Multi-scale chunking")
            config = SplitterConfig(micro_max_tokens=micro_tokens, meso_max_tokens=meso_tokens)
            graph = run_chunking_pipeline_from_outline(
                outline,
                figure_manifest=figure_manifest_dict,
                config=config,
                summarizer_mode="extractive",
            )
            save_chunk_graph(graph, chunks_path)
            stats = graph.stats()
            console.print(
                f"  {stats['total_chunks']} chunks "
                f"(MACRO {stats['by_level']['MACRO']}, "
                f"MESO {stats['by_level']['MESO']}, "
                f"MICRO {stats['by_level']['MICRO']}) → [cyan]{chunks_path}[/]"
            )

    # ── 2b. Textract path (scanned PDFs) ─────────────────────────────────────
    else:
        from aws_rag.storage import save_pdf_locally
        from aws_rag.textract import (
            get_job_results,
            load_blocks,
            save_blocks,
            start_analysis,
            wait_for_job,
        )

        if show_cost:
            from aws_rag.docling_parser import content_hash
            did = doc_id or content_hash(pdf_path)
            blocks_path = settings.output_dir / f"{did}_blocks.json"
            cached = blocks_path.exists() and not force
            console.print(f"  doc_id = [cyan]{did}[/]")
            if not cached:
                pages = pdf_page_count(pdf_path)
                cost.items.append(estimate_textract_cost(pages))
                cost.notes.append(
                    "No cached Textract OCR results for this PDF — only the "
                    "OCR line item could be estimated. Embedding/description "
                    "costs depend on its output; run a real ingest once (or "
                    "ingest with cached blocks present) to estimate the full "
                    "pipeline without re-paying for OCR on every estimate."
                )
                _print_cost_table(cost)
                return cost
            console.print("  [yellow]Cached Textract blocks found — estimating full pipeline.[/]")
            blocks = load_blocks(blocks_path)
            console.print(f"  {len(blocks)} blocks (cached)")
        else:
            from aws_rag.storage import upload_pdf

            _step("Upload PDF to S3")
            did, s3_key = upload_pdf(pdf_path, doc_id=doc_id)
            console.print(f"  doc_id = [cyan]{did}[/]")
            console.print(f"  s3_key = {s3_key}")

            blocks_path = settings.output_dir / f"{did}_blocks.json"
            _step("Textract layout analysis (OCR)")
            if blocks_path.exists() and not force:
                console.print(f"  [yellow]Resuming — loading cached blocks[/] → [cyan]{blocks_path}[/]")
                blocks = load_blocks(blocks_path)
                console.print(f"  {len(blocks)} blocks (cached)")
            else:
                job_id = start_analysis(did, s3_key)
                console.print(f"  job_id = {job_id}  (waiting…)")
                status = wait_for_job(job_id)
                if status != "SUCCEEDED":
                    raise click.ClickException(f"Textract job failed with status: {status}")
                blocks = get_job_results(job_id)
                save_blocks(blocks, blocks_path)
                console.print(f"  {len(blocks)} blocks → [cyan]{blocks_path}[/]")

        save_pdf_locally(pdf_path, did)

        figure_manifest_dict = None
        if not skip_figures:
            _step("Extract figures")
            figures_out = settings.figures_dir / did
            manifest = extract_figures(
                pdf_path=pdf_path,
                blocks=blocks,
                doc_id=did,
                output_dir=figures_out,
                dpi=dpi,
                image_format="png",
                padding_pct=0.02,
            )
            if upload_figures and manifest.figures:
                manifest = upload_figures_to_s3(manifest)
            manifest_path = figures_out / "manifest.json"
            manifest.save(manifest_path)
            figure_manifest_dict = manifest.to_dict()
            figure_count = len(manifest.figures)
            console.print(f"  {len(manifest.figures)} figures → [cyan]{manifest_path}[/]")

        chunks_path = settings.output_dir / f"{did}_chunks.json"
        if chunks_path.exists() and not force:
            _step("Multi-scale chunking")
            console.print(f"  [yellow]Resuming — loading cached chunk graph[/] → [cyan]{chunks_path}[/]")
            graph = load_chunk_graph(chunks_path)
            stats = graph.stats()
            console.print(
                f"  {stats['total_chunks']} chunks "
                f"(MACRO {stats['by_level']['MACRO']}, "
                f"MESO {stats['by_level']['MESO']}, "
                f"MICRO {stats['by_level']['MICRO']}) (cached)"
            )
        else:
            _step("Multi-scale chunking")
            config = SplitterConfig(micro_max_tokens=micro_tokens, meso_max_tokens=meso_tokens)
            graph = run_chunking_pipeline(
                blocks,
                doc_id=did,
                figure_manifest=figure_manifest_dict,
                config=config,
                summarizer_mode="extractive",
            )
            save_chunk_graph(graph, chunks_path)
            stats = graph.stats()
            console.print(
                f"  {stats['total_chunks']} chunks "
                f"(MACRO {stats['by_level']['MACRO']}, "
                f"MESO {stats['by_level']['MESO']}, "
                f"MICRO {stats['by_level']['MICRO']}) → [cyan]{chunks_path}[/]"
            )

    # ── 5/6. Describe, embed & store (via the backend) ───────────────────────
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

    _step("Embed & store")
    from aws_rag.backend import MetadataPatch, backend_mode, get_backend

    # `rag ingest --db` targets a specific local file; honor it by building a
    # LocalBackend on that path rather than the configured backend.
    if db_path is not None:
        from aws_rag.backend import LocalBackend

        backend_obj = LocalBackend(db_path)
    else:
        backend_obj = get_backend()

    # In remote mode the cropped figure images live only on this client —
    # ship their bytes so the server stores them and rewrites the host-local
    # figure_image_path before inserting.
    figures_upload: dict[str, tuple[bytes, str]] | None = None
    if not skip_figures and backend_mode() == "remote" and db_path is None:
        figures_upload = {}
        for c in graph.chunks.values():
            if c.figure_image_path:
                p = Path(c.figure_image_path)
                if p.is_file():
                    figures_upload[c.id] = (p.read_bytes(), p.suffix.lstrip(".") or "png")

    meta_patch = MetadataPatch(
        mpn=mpn or None,
        manufacturer=manufacturer or None,
        subsystem=subsystem or None,
        doc_type=doc_type or None,
        tags=list(tags) if tags else None,
    )
    title_hints: dict[str, str] = {}
    if running_header:
        title_hints["running_header"] = running_header
    if pdf_meta_title:
        title_hints["pdf_meta_title"] = pdf_meta_title

    from aws_rag.backend import RagServerError

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
    if result.described:
        console.print(f"  [green]{result.described}[/] figure descriptions generated")
    if result.title:
        console.print(f"  [green]Inferred title:[/] {result.title}")

    elapsed = time.monotonic() - t0
    console.rule(f"[bold green]Done[/] — {elapsed:.0f}s")
    console.print(f"  doc_id = [cyan]{did}[/]")
    return None


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


@cli.command("reconvert-tables")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--pages", "pages_spec", required=True,
    help="1-based inclusive page range to re-run, e.g. '36' or '36-40'.",
)
@click.option("--doc-id", default=None, help="Override the content-hash doc_id.")
@click.option(
    "--accurate-tables/--fast-tables", default=True,
    help="Table mode for the re-run (default: accurate — that's the point of this command).",
)
@click.option(
    "--dry-run", is_flag=True,
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
    (so a cached `<doc_id>_outline.json` exists). On success it deletes the
    cached chunk graph so the next `rag ingest` re-derives chunks (and
    embeddings) from the patched outline — Docling layout analysis itself is
    NOT re-run for the rest of the document.
    """
    from aws_rag.chunking.layout_parser import DocumentOutline
    from aws_rag.docling_parser import content_hash, reconvert_tables_in_range

    page_range = _parse_page_range(pages_spec)
    settings = get_settings()
    did = doc_id or content_hash(pdf_path)
    outline_path = settings.output_dir / f"{did}_outline.json"
    chunks_path = settings.output_dir / f"{did}_chunks.json"

    if not outline_path.exists():
        raise click.ClickException(
            f"No cached layout outline for doc_id={did} at {outline_path}. "
            "Run `rag ingest` (Docling backend) on this PDF first — "
            "reconvert-tables only patches an existing outline, it doesn't "
            "do a first-time conversion."
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

    matched = [e for e in report if e["matched"]]
    fixed_garbled = [e for e in matched if e["old_garbled"] and not e["new_garbled"]]
    still_garbled = [e for e in matched if e["new_garbled"]]
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
        console.print(
            f"[green]Invalidated cached chunk graph[/] → removed [cyan]{chunks_path}[/]. "
            f"Re-run `rag ingest {pdf_path}` (without --force) to re-derive "
            f"chunks and embeddings from the patched outline — Docling layout "
            f"analysis will be skipped since the outline cache still exists."
        )
    else:
        console.print("Run `rag ingest` to (re)derive chunks and embeddings from the patched outline.")


@cli.command("table-structure-sweep")
@click.argument("doc_id", type=str)
@click.option(
    "--list-flagged", is_flag=True,
    help="Print every flagged table (page, caption, reason) instead of just the summary counts.",
)
@click.option(
    "--sample", "sample_n", type=int, default=0,
    help=(
        "Print the rendered text of N randomly-sampled flagged tables AND N "
        "randomly-sampled non-flagged tables, for manual eyeballing — a "
        "zero-cost spot-check of detector accuracy (false positives among "
        "the flagged sample, false negatives among the non-flagged one) "
        "before spending on `rag repair-tables`. Sampling uses a fixed seed "
        "so repeated runs show the same tables."
    ),
)
def table_structure_sweep_cmd(doc_id: str, list_flagged: bool, sample_n: int) -> None:
    """Report how many cached tables Docling's structure detectors flag as untrustworthy.

    This is the Phase-0 instrument from docs/table-structure-repair/plan.md —
    a zero-cost (pure Python over the cached `<doc_id>_outline.json`, no
    Docling re-run, no AWS calls) sweep across every TABLE element already in
    the layout outline, reporting how many each of
    docling_parser._detect_garbled_header (failure mode #1: repeated header
    text) and _detect_fused_header_row (failure mode #3: data leaked into the
    header band) independently catches, and their overlap.

    The resulting "fraction flagged" number is what gates whether an
    LLM-assisted repair pipeline (Stage 3 of that plan) belongs in the ingest
    hot path, a lazy on-demand path, or an explicit opt-in maintenance
    command — don't guess the cost shape, measure it.

    Looks up the cache directly by doc_id (full hash or unambiguous prefix —
    like `reconvert-tables`, this works straight off `<doc_id>_outline.json`,
    not the sqlite ingest registry, since a layout conversion can be cached
    without a completed ingest).
    """
    from aws_rag.chunking.layout_parser import ContentElement, DocumentOutline, ElementType
    from aws_rag.docling_parser import (
        _detect_fused_header_row,
        _detect_garbled_header,
        _table_cells_to_compact_text,
    )

    did = _resolve_cached_doc_id(doc_id, "_outline.json")
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

    flagged: list[dict[str, object]] = []
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
            flagged.append({"page": el.page, "caption": el.table_title, "reason": reason})
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

    _row("_detect_garbled_header (mode #1: repeated text)", garbled_count)
    _row("_detect_fused_header_row (mode #3: data-in-header)", fused_count)
    _row("caught by both", both_count)
    _row("[bold]flagged untrustworthy (either)[/]", flagged_count)
    console.print(summary)

    if list_flagged and flagged:
        detail = Table(title="Flagged tables")
        detail.add_column("Page", justify="right")
        detail.add_column("Caption")
        detail.add_column("Reason")
        for entry in flagged:
            detail.add_row(str(entry["page"]), (entry["caption"][:60] or "—"), entry["reason"])
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
            "right before spending on `rag repair-tables` (zero AWS cost; "
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


@cli.command("repair-tables")
@click.argument("doc_id", type=str)
@click.option(
    "--limit", type=int, default=None,
    help="Repair at most N flagged tables (omit to repair all of them).",
)
@click.option(
    "--model-id", default=None,
    help=(
        "Override the Bedrock model ID for this run. Defaults to "
        "table_repair_model_id, falling back to description_model_id "
        "(Haiku — cheap, and validation rejects anything structurally "
        "inconsistent, so a weaker model fails safe rather than silently)."
    ),
)
@click.option("--dpi", type=int, default=200, help="Render DPI for table crops (default: 200).")
@click.option(
    "--force", is_flag=True,
    help="Re-repair tables that already have a cached table_repaired_cells.",
)
@click.option(
    "--dry-run", is_flag=True,
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

    Like `reconvert-tables`, this patches the cached `<doc_id>_outline.json`
    in place and invalidates the cached chunk graph so the next `rag ingest`
    re-derives chunks/embeddings from the repaired structure. Run
    `rag table-structure-sweep <doc_id>` first to see what would be touched
    and at roughly what volume.
    """
    import io

    from pdf2image import convert_from_bytes

    from aws_rag.chunking.layout_parser import ContentElement, DocumentOutline, ElementType
    from aws_rag.docling_parser import (
        _table_column_count,
        _table_header_row_count,
        table_structure_untrustworthy,
    )
    from aws_rag.pdf_render import load_pdf_bytes
    from aws_rag.table_repair import (
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

    did = _resolve_cached_doc_id(doc_id, "_outline.json")
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
            "Run `rag table-structure-sweep` to confirm the flagged count."
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
        console.print(
            f"[green]Invalidated cached chunk graph[/] → removed [cyan]{chunks_path}[/]. "
            f"Re-run `rag ingest` (without --force) to re-derive chunks and "
            f"embeddings from the repaired structure — Docling layout "
            f"analysis will be skipped since the outline cache still exists."
        )
    else:
        console.print(
            "Run `rag ingest` to (re)derive chunks and embeddings from the repaired outline."
        )


# ---------------------------------------------------------------------------
# Document metadata (sidecar — separate from chunks, no re-ingest required)
# ---------------------------------------------------------------------------


@cli.group()
def metadata() -> None:
    """Manage the doc-level metadata sidecar (project, mpn, manufacturer, …)."""


@metadata.command("set")
@click.argument("doc_id", type=str)
@click.option("--title", "doc_title", default=None,
              help="Override doc_title on every chunk row (manual title fix).")
@click.option("--project-id", default=None)
@click.option("--group", "group_name", default=None)
@click.option("--mpn", default=None, help="Manufacturer part number, e.g. STM32H743VIT6.")
@click.option("--manufacturer", default=None)
@click.option("--subsystem", default=None, help="e.g. power, rf, mcu.")
@click.option("--doc-type", default=None,
              help="datasheet | reference-manual | errata | app-note | …")
@click.option("--tag", "tags", multiple=True, help="Repeatable --tag flag.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--apply-to-chunks/--no-apply-to-chunks", default=True,
              help="Propagate project_id and group_name into the chunks table.")
def metadata_set(
    doc_id: str,
    doc_title: str | None,
    project_id: str | None,
    group_name: str | None,
    mpn: str | None,
    manufacturer: str | None,
    subsystem: str | None,
    doc_type: str | None,
    tags: tuple[str, ...],
    db_path: Path | None,
    apply_to_chunks: bool,
) -> None:
    """Upsert document metadata. Only fields you pass are updated."""
    from aws_rag.backend import MetadataPatch
    from aws_rag.project_config import get_project_config

    proj_cfg = get_project_config()
    if proj_cfg is not None:
        project_id = project_id or proj_cfg.project_id
        group_name = group_name or proj_cfg.group
        mpn = mpn or proj_cfg.mpn
        manufacturer = manufacturer or proj_cfg.manufacturer
        subsystem = subsystem or proj_cfg.subsystem
        if not tags and proj_cfg.tags:
            tags = tuple(proj_cfg.tags)

    be = _backend_for(db_path)
    doc_id = be.resolve_doc_id(doc_id)

    if doc_title is not None:
        updated = be.set_doc_title(doc_id, doc_title)
        console.print(f"[green]Title set[/] on {updated} chunk rows: {doc_title!r}")

    meta = be.set_metadata(
        doc_id,
        MetadataPatch(
            project_id=project_id, group_name=group_name,
            mpn=mpn, manufacturer=manufacturer, subsystem=subsystem,
            doc_type=doc_type,
            tags=list(tags) if tags else None,
        ),
    )
    console.print(f"[green]Saved metadata for[/] {doc_id}")
    console.print(meta.model_dump_json(indent=2, exclude_none=True))

    if apply_to_chunks:
        updated = be.apply_metadata_to_chunks(doc_id)
        console.print(f"  Propagated to {updated} chunk rows.")


@metadata.command("get")
@click.argument("doc_id", type=str)
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
def metadata_get(doc_id: str, db_path: Path | None) -> None:
    """Show the sidecar metadata row for a document."""
    be = _backend_for(db_path)
    doc_id = be.resolve_doc_id(doc_id)
    meta = be.get_metadata(doc_id)
    if meta is None:
        console.print(f"[yellow]No metadata recorded for[/] {doc_id}")
        return
    console.print(meta.model_dump_json(indent=2, exclude_none=True))


@metadata.command("list")
@click.option("--project-id", default=None,
              help="Restrict to a project (default: scoped by .rag.toml if present).")
@click.option("--global", "-g", "is_global", is_flag=True,
              help="List documents across every project, ignoring any .rag.toml scoping.")
@click.option("--group", "group_name", default=None)
@click.option("--mpn", default=None)
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
def metadata_list(
    project_id: str | None,
    is_global: bool,
    group_name: str | None,
    mpn: str | None,
    db_path: Path | None,
) -> None:
    """List documents in the sidecar, optionally filtered."""
    from aws_rag.project_config import resolve_cli_project_id

    project_id = resolve_cli_project_id(project_id, is_global=is_global)

    be = _backend_for(db_path)
    docs = be.list_docs(project_id=project_id, group_name=group_name, mpn=mpn)

    if not docs:
        console.print("[yellow]No documents match.[/]")
        return

    table = Table(title=f"Documents ({len(docs)})")
    table.add_column("doc_id", style="cyan")
    table.add_column("project")
    table.add_column("group")
    table.add_column("mpn")
    table.add_column("manufacturer")
    table.add_column("subsystem")

    for d in docs:
        table.add_row(
            d.doc_id[:SHORT_DOC_ID_LEN], d.project_id or "—",
            d.group_name or "—", d.mpn or "—",
            d.manufacturer or "—", d.subsystem or "—",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# fix-titles (AI-inferred document titles for poorly-titled documents)
# ---------------------------------------------------------------------------

_BLANK_TITLES = (None, "", "—")


@cli.command("fix-titles")
@click.option("--doc-id", default=None, help="Restrict to a single document.")
@click.option("--force", is_flag=True,
              help="Re-infer even for documents that already have a title "
                   "(needed to replace generic titles like 'Contents').")
@click.option("--model", "model_id", default=None,
              help="Override settings.description_model_id for this run.")
@click.option("--dry-run", is_flag=True,
              help="Infer and print titles but do not persist them.")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
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
    marked `title_inferred: true` in the metadata sidecar (`rag metadata get
    <doc_id>`) so they're distinguishable from titles Docling extracted
    directly. Re-run with --doc-id --force to overwrite an inferred title.
    """
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

    console.print(f"Inferring titles for {len(docs)} document(s) (LLM runs server-side in remote mode)…")

    for d in docs:
        short_id = d.doc_id[:SHORT_DOC_ID_LEN]
        current = d.doc_title
        title = be.infer_title(d.doc_id, model_id=model_id, dry_run=dry_run)
        if title is None:
            console.print(f"  [yellow]could not infer[/] {short_id} (was: {current!r})")
            continue
        verb = "would set" if dry_run else "set"
        console.print(f"  [green]{verb}[/] {short_id}: {current!r} → {title!r}")


# ---------------------------------------------------------------------------
# Eval (retrieval-layer evaluation)
# ---------------------------------------------------------------------------


@cli.group("eval")
def eval_group() -> None:
    """Retrieval-layer evaluation: golden set, metrics, ablations."""


_CAT_ORDER = ["identifier", "conceptual", "figure", "table_spec", "synthesis", "overall"]


def _render_report_table(report: object) -> None:
    """Print one RunReport's per-category metrics."""
    from aws_rag.eval.harness import RunReport

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


def _render_matrix_table(reports: list, headline_k: int) -> None:
    """Print a comparison across configs: overall + per-category hit@k."""
    from aws_rag.eval.dataset import CATEGORIES

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


@eval_group.command("generate")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None,
              help="SQLite DB path. Defaults to settings.sqlite_db_path.")
@click.option("--per-category", default=4, type=int, help="Items to generate per category.")
@click.option("--doc-id", default=None, help="Restrict sampling to one document.")
@click.option("--project-id", default=None, help="Restrict sampling to one project.")
@click.option("--model", "model_id", default=None, help="Bedrock model ID for generation.")
@click.option("--seed", default=0, type=int, help="Sampling seed (reproducible).")
@click.option("--output", "-o", "out_path", type=click.Path(path_type=Path),
              default=Path("eval/golden.jsonl"), help="Output JSONL path.")
@click.option("--append", is_flag=True, help="Append to the output file instead of overwriting.")
@click.option("--verbose/--quiet", default=True)
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
    from aws_rag.eval.generate import generate_golden_set
    from aws_rag.store import connect

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


@eval_group.command("run")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--set", "set_path", type=click.Path(exists=True, path_type=Path),
              default=Path("eval/golden.jsonl"), help="Golden set JSONL.")
@click.option("--mode", type=click.Choice(["hybrid", "vector", "keyword"]), default="hybrid")
@click.option("-k", "top_k", default=5, type=int, help="Headline k (nDCG cutoff).")
@click.option("--level", type=click.Choice(["macro", "meso", "micro"]), default=None)
@click.option("--rrf-k", default=60, type=int)
@click.option("--vector-weight", default=1.0, type=float)
@click.option("--keyword-weight", default=1.0, type=float)
@click.option("--trace", "trace_path", type=click.Path(path_type=Path), default=None,
              help="Append per-query JSONL traces here.")
@click.option("--json-out", type=click.Path(path_type=Path), default=None,
              help="Write the full report JSON here.")
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
    from aws_rag.eval.dataset import EvalSet
    from aws_rag.eval.harness import RunConfig, run_eval
    from aws_rag.store import connect

    conn = connect(_require_local_db(db_path))
    eval_set = EvalSet.load(set_path)

    embedder = None
    if mode in ("vector", "hybrid"):
        from aws_rag.embedding import get_embedder

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


@eval_group.command("ablate")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--set", "set_path", type=click.Path(exists=True, path_type=Path),
              default=Path("eval/golden.jsonl"))
@click.option("-k", "top_k", default=5, type=int, help="Headline k for the comparison.")
@click.option("--trace", "trace_path", type=click.Path(path_type=Path), default=None)
@click.option("--json-out", type=click.Path(path_type=Path), default=None)
@click.option("--index-ablation",
              type=click.Choice(["context-vs-raw", "figure-desc", "macro-summarizer"]),
              default=None, help="Heavy re-embedding ablation (incurs Bedrock cost).")
@click.option("--variant-db", type=click.Path(path_type=Path),
              default=Path("test-project/output/rag-variant.sqlite"),
              help="Where to build the variant store for an index ablation.")
@click.option("--limit", default=None, type=int, help="Cap chunks re-embedded (index ablation).")
@click.option("--doc-id", default=None,
              help="Document to re-summarize (required for --index-ablation macro-summarizer).")
@click.option("--summarizer-model", default="anthropic.claude-3-haiku-20240307-v1:0",
              help="Bedrock model id for the macro-summarizer ablation.")
@click.option("--verbose/--quiet", default=True)
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
    from aws_rag.embedding import get_embedder
    from aws_rag.eval.ablation import (
        build_macro_summarizer_variant_store,
        build_variant_store,
        default_matrix,
        run_matrix,
    )
    from aws_rag.eval.dataset import EvalSet
    from aws_rag.eval.harness import RunConfig, run_eval
    from aws_rag.store import connect

    conn = connect(_require_local_db(db_path))
    eval_set = EvalSet.load(set_path)
    embedder = get_embedder()

    if index_ablation is None:
        reports = run_matrix(
            conn, eval_set, default_matrix(base_k=top_k),
            embedder=embedder, trace_path=trace_path,
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
        from aws_rag.chunking.summarizer import AbstractiveSummarizer
        from aws_rag.config import get_settings as _get_settings

        console.print(
            f"[yellow]Index ablation[/] 'macro-summarizer': re-summarizing "
            f"{doc_id} chapters abstractively (model={summarizer_model}) "
            f"and re-embedding — this calls Bedrock Claude per chapter."
        )
        summarizer = AbstractiveSummarizer(
            model_id=summarizer_model, region=_get_settings().aws_region,
        )
        variant_conn = build_macro_summarizer_variant_store(
            conn, variant_db, doc_id, summarizer, embedder, verbose=verbose,
        )

        # The variant store only contains doc_id's chunks (by design — see
        # build_macro_summarizer_variant_store), so items targeting other
        # docs would score 0 by construction. Scope the eval set to match.
        scoped_set = EvalSet(items=[i for i in eval_set.items if i.doc_id == doc_id])

        base_cfg = RunConfig(mode="hybrid", k=top_k, level="macro", label="baseline (extractive macro)")
        var_cfg = RunConfig(mode="hybrid", k=top_k, level="macro", label="variant (abstractive macro)")

        base_report = run_eval(conn, scoped_set, base_cfg, embedder=embedder, trace_path=trace_path)
        var_report = run_eval(variant_conn, scoped_set, var_cfg, embedder=embedder, trace_path=trace_path)
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
    variant = "raw_text" if index_ablation == "context-vs-raw" else "no_figure_desc"
    console.print(
        f"[yellow]Index ablation[/] '{index_ablation}': building variant store "
        f"(variant={variant}) — this re-embeds and incurs Bedrock cost."
    )
    variant_conn = build_variant_store(
        conn, variant_db, variant, embedder,  # type: ignore[arg-type]
        limit=limit, verbose=verbose,
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


def _dump_reports_json(reports: list, path: Path) -> None:
    import json as _json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _json.dumps([r.model_dump() for r in reports], indent=2, default=str),
        encoding="utf-8",
    )
    console.print(f"[green]Reports JSON →[/] {path}")


@eval_group.command("review")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--set", "set_path", type=click.Path(exists=True, path_type=Path),
              default=Path("eval/golden.jsonl"), help="Golden set JSONL to review.")
@click.option("--port", default=0, type=int, help="Port (0 = pick a free one).")
@click.option("-k", "top_k", default=5, type=int, help="Retrieval results to preview per item.")
@click.option("--open/--no-open", "open_browser", default=True,
              help="Open the review page in a browser.")
def eval_review(
    db_path: Path | None,
    set_path: Path,
    port: int,
    top_k: int,
    open_browser: bool,
) -> None:
    """Hand-review the golden set in a local web app (PDF page + page/chunk labels)."""
    from aws_rag.eval.review import serve

    serve(set_path, db_path=db_path, port=port, k=top_k, open_browser=open_browser)


# ---------------------------------------------------------------------------
# Remote admin — manage API keys + read the audit log over HTTP.
# ---------------------------------------------------------------------------


def _admin_request(method: str, path: str, *, token: str | None, **kwargs):
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
            method, base.rstrip("/") + path, headers=headers,
            timeout=settings.server_timeout, **kwargs,
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
    return resp.json()


@cli.group()
def admin() -> None:
    """Administer a remote RAG server: API keys and audit log.

    Talks to RAG_SERVER_URL with an admin-scoped bearer token (RAG_SERVER_TOKEN
    or --token). Bootstrap the very first admin key on the server host with
    `rag-server create-key --label bootstrap --scope admin`.
    """


@admin.group()
def key() -> None:
    """Create, list and revoke per-client API keys."""


@key.command("create")
@click.option("--label", required=True, help="Client identity for the key (shown in audit).")
@click.option(
    "--scope", "scopes", multiple=True,
    type=click.Choice(["read", "ingest", "admin"]), default=("ingest",),
    help="Scope(s) for the key (repeatable). Default: ingest.",
)
@click.option("--token", default=None, help="Admin token (default: RAG_SERVER_TOKEN).")
def key_create(label: str, scopes: tuple[str, ...], token: str | None) -> None:
    """Mint a key. The plaintext token is shown ONCE — copy it now."""
    data = _admin_request(
        "POST", "/admin/keys", token=token,
        json={"label": label, "scopes": list(scopes)},
    )
    console.print(f"[green]Created[/] key '{data['label']}' "
                  f"(id={data['id']}, scopes={data['scopes']})")
    console.print("\n[bold]Token (shown once — store it now):[/]\n")
    console.print(f"  {data['token']}\n")


@key.command("list")
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
            k["id"], k["label"], ",".join(k["scopes"]),
            k.get("created_at") or "—",
            "[red]revoked[/]" if k.get("revoked_at") else "—",
        )
    console.print(table)


@key.command("revoke")
@click.argument("key_id", type=str)
@click.option("--token", default=None, help="Admin token (default: RAG_SERVER_TOKEN).")
def key_revoke(key_id: str, token: str | None) -> None:
    """Revoke a key by id (takes effect immediately, no restart)."""
    _admin_request("DELETE", f"/admin/keys/{key_id}", token=token)
    console.print(f"[green]Revoked[/] key {key_id}")


@admin.command("audit")
@click.option("--doc-id", default=None, help="Filter to one document.")
@click.option("--since", default=None, help="ISO timestamp lower bound (e.g. 2026-06-01).")
@click.option("--limit", default=50, type=int, help="Max rows (newest first).")
@click.option("--token", default=None, help="Admin token (default: RAG_SERVER_TOKEN).")
def audit_cmd(doc_id: str | None, since: str | None, limit: int, token: str | None) -> None:
    """Show the ingest-path audit trail."""
    params = {"limit": limit}
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
            (e.get("ts") or "")[:19], e.get("action") or "", e.get("status") or "",
            e.get("key_label") or "—", e.get("client_ip") or "—",
            (e.get("doc_id") or "—")[:12],
            e.get("detail_json") or (e.get("error") or "—"),
        )
    console.print(table)
