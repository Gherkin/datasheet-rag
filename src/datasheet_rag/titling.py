"""AI-inferred document titles for poorly-titled documents.

Some PDFs come through ingestion without a usable `doc_title` — Docling
sometimes picks up a generic heading ("Contents", "Disclaimer", "Technical
manual") as the title instead of the document's actual name. This module
asks a small Claude model to read the first page and infer a real title,
then backfills it onto every chunk for that document plus the
`doc_metadata` sidecar (marked `title_source: inferred` so it can be
reviewed or overridden later via `rag metadata`, and so a later
re-ingest cannot demote it back to the parser's guess).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from typing import Any

from rich.console import Console

from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkGraph, ChunkLevel

console = Console()

_FIRST_PAGE_CHAR_LIMIT = 4000

_SYSTEM_PROMPT = (
    "You read the first page of technical datasheets and reference manuals "
    "and infer the document's real title — the specific product or part "
    "name plus document type, e.g. 'STM32H743 Reference Manual' or "
    "'LM358 Low-Power Dual Operational Amplifier Datasheet'. "
    "Respond with ONLY the title itself — no preamble, quotes, or trailing "
    "punctuation. If the text does not contain enough information to infer "
    "a specific title, respond with exactly: UNKNOWN"
)


class TitleInferer:
    """Wrap a single Bedrock Claude text call: first-page text → title."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        max_tokens: int = 60,
        region: str | None = None,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.model_id = model_id or settings.description_model_id
        self.max_tokens = max_tokens
        self.region = region or settings.aws_region
        self.client = client

    def _get_client(self) -> Any:
        if self.client is None:
            from datasheet_rag.local_models import get_chat_client

            self.client = get_chat_client(kind="text", region=self.region)
        return self.client

    def infer(self, first_page_text: str) -> str | None:
        """Return an inferred title, or None if the model declined / errored."""
        text = first_page_text.strip()[:_FIRST_PAGE_CHAR_LIMIT]
        if not text:
            return None

        client = self._get_client()
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.0,
        }
        response = client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        title = (
            "".join(
                block.get("text", "")
                for block in payload.get("content", [])
                if isinstance(block, dict) and block.get("type") == "text"
            )
            .strip()
            .strip('"')
            .strip()
        )

        if not title or title.upper() == "UNKNOWN":
            return None
        return title


def first_page_text(conn: sqlite3.Connection, doc_id: str) -> str:
    """Concatenate MICRO-level chunk text from page 1, in document order."""
    rows = conn.execute(
        "SELECT text, page_numbers FROM chunks WHERE doc_id = ? AND level = ? ORDER BY rowid",
        (doc_id, int(ChunkLevel.MICRO)),
    ).fetchall()

    fragments: list[str] = []
    total = 0
    for row in rows:
        try:
            pages = json.loads(row["page_numbers"] or "[]")
        except (TypeError, ValueError):
            pages = []
        if 1 not in pages:
            continue
        text = (row["text"] or "").strip()
        if not text:
            continue
        fragments.append(text)
        total += len(text)
        if total >= _FIRST_PAGE_CHAR_LIMIT:
            break

    return "\n".join(fragments)


def infer_and_backfill_title(
    conn: sqlite3.Connection,
    doc_id: str,
    *,
    inferer: TitleInferer | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> str | None:
    """Infer a title for `doc_id` from its first page and persist it.

    Persists the inferred title onto every `chunks` row for the document
    and records `title_source: inferred` in the `doc_metadata` sidecar
    attributes. Returns the inferred title, or None if no title could be
    inferred (in which case nothing is written).

    Refuses to overwrite a title a human set by hand, returning None
    without spending an LLM call — `inferred` ranks below `manual` (see
    :mod:`datasheet_rag.store.metadata`). Pass `force=True` to overrule
    that, which is what `rag repair titles --force` does.

    Two extra hints captured at ingest time (`doc_metadata.attributes`,
    see `ingest --infer-title`/the Docling path) are passed alongside the
    first-page text when present, since cover pages are often mostly
    imagery and leave the model with very little to go on:

    - `running_header`: text repeated at the top of every page — often
      carries the title even when the cover page itself doesn't.
    - `pdf_meta_title`: the PDF's own embedded `/Title` + `/Subject`
      metadata — publishers sometimes split the real title across these
      two fields (e.g. Title="Programmer's Guide", Subject="CC Linux"),
      information Docling never sees since it isn't page content.
    """
    from datasheet_rag.store.metadata import get_metadata, get_title_source, title_rank
    from datasheet_rag.store.sqlite import set_doc_title

    if not force and title_rank("inferred") < title_rank(get_title_source(conn, doc_id)):
        return None

    meta = get_metadata(conn, doc_id)
    text = build_title_prompt(
        first_page_text(conn, doc_id),
        meta.attributes if meta else {},
    )
    if not text:
        return None

    inferer = inferer or TitleInferer()
    title = inferer.infer(text)
    if not title:
        return None

    if not dry_run:
        set_doc_title(conn, doc_id, title, source="inferred", force=force)

    return title


def build_title_prompt(first_page_text: str, attributes: dict[str, Any] | None) -> str:
    """Prefix the first page's text with whatever hints the sidecar carries.

    Returns "" when there is no first-page text to work with, which callers
    read as "nothing to infer from" — the hints alone are too thin to spend a
    model call on.
    """
    if not first_page_text:
        return ""
    hints = []
    attributes = attributes or {}
    if attributes.get("running_header"):
        hints.append(f"Running page header (appears on every page): {attributes['running_header']}")
    if attributes.get("pdf_meta_title"):
        hints.append(f"PDF embedded title metadata: {attributes['pdf_meta_title']}")
    if not hints:
        return first_page_text
    return "\n".join(hints) + "\n\n" + first_page_text


def first_page_text_from_chunks(chunks: Iterable[Chunk]) -> str:
    """Concatenate page-1 MICRO chunk text, in the order given.

    The in-memory twin of :func:`first_page_text`, for callers holding a
    freshly parsed graph rather than a store.
    """
    fragments: list[str] = []
    total = 0
    for chunk in chunks:
        if int(chunk.level) != int(ChunkLevel.MICRO):
            continue
        if 1 not in chunk.metadata.page_numbers:
            continue
        text = (chunk.text or "").strip()
        if not text:
            continue
        fragments.append(text)
        total += len(text)
        if total >= _FIRST_PAGE_CHAR_LIMIT:
            break
    return "\n".join(fragments)


def infer_title_from_graph(
    graph: ChunkGraph,
    *,
    title_hints: dict[str, str] | None = None,
    inferer: TitleInferer | None = None,
) -> str | None:
    """Infer a title from a parsed graph, before anything has been stored.

    Used when the model runs on the client but the store is remote
    (``RAG_COMPUTE=client``, GH #43): everything the store-backed path reads
    out of sqlite — page-1 text and the two hints ingest captures — is already
    in hand here, so no provisional insert is needed to spend the call.
    """
    text = build_title_prompt(
        first_page_text_from_chunks(graph.chunks.values()), dict(title_hints or {})
    )
    if not text:
        return None
    return (inferer or TitleInferer()).infer(text)


def infer_title_via_backend(
    backend: Any,
    doc_id: str,
    *,
    inferer: TitleInferer | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> str | None:
    """:func:`infer_and_backfill_title`, driven through a backend.

    The provenance check and the page-1 text both come from the store over
    ``get_title_context``; only the model call happens here. Returns None
    without spending it when a hand-set title outranks an inferred one, matching
    the store-backed path.
    """
    from datasheet_rag.store.metadata import title_rank

    context = backend.get_title_context(doc_id)
    if not force and title_rank("inferred") < title_rank(context.title_source):
        return None
    text = build_title_prompt(context.first_page_text, context.attributes)
    if not text:
        return None
    title = (inferer or TitleInferer()).infer(text)
    if not title:
        return None
    if not dry_run:
        backend.set_doc_title(doc_id, title, source="inferred", force=force)
    return title
