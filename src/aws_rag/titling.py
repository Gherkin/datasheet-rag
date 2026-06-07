"""AI-inferred document titles for poorly-titled documents.

Some PDFs come through ingestion without a usable `doc_title` — Docling
sometimes picks up a generic heading ("Contents", "Disclaimer", "Technical
manual") as the title instead of the document's actual name. This module
asks a small Claude model to read the first page and infer a real title,
then backfills it onto every chunk for that document plus the
`doc_metadata` sidecar (marked `title_inferred: true` so it can be
reviewed or overridden later via `rag metadata set`).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from rich.console import Console

from aws_rag.config import get_settings
from aws_rag.models.chunk import ChunkLevel

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
            from aws_rag.aws import _session
            self.client = _session().client("bedrock-runtime", region_name=self.region)
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
        title = "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip().strip('"').strip()

        if not title or title.upper() == "UNKNOWN":
            return None
        return title


def _first_page_text(conn: sqlite3.Connection, doc_id: str) -> str:
    """Concatenate MICRO-level chunk text from page 1, in document order."""
    rows = conn.execute(
        "SELECT text, page_numbers FROM chunks "
        "WHERE doc_id = ? AND level = ? ORDER BY rowid",
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
) -> str | None:
    """Infer a title for `doc_id` from its first page and persist it.

    Persists the inferred title onto every `chunks` row for the document
    and records `title_inferred: true` in the `doc_metadata` sidecar
    attributes. Returns the inferred title, or None if no title could be
    inferred (in which case nothing is written).

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
    from aws_rag.store.metadata import get_metadata, set_metadata

    text = _first_page_text(conn, doc_id)
    if not text:
        return None

    meta = get_metadata(conn, doc_id)
    attributes = meta.attributes if meta else {}
    hints = []
    if attributes.get("running_header"):
        hints.append(f"Running page header (appears on every page): {attributes['running_header']}")
    if attributes.get("pdf_meta_title"):
        hints.append(f"PDF embedded title metadata: {attributes['pdf_meta_title']}")
    if hints:
        text = "\n".join(hints) + "\n\n" + text

    inferer = inferer or TitleInferer()
    title = inferer.infer(text)
    if not title:
        return None

    if not dry_run:
        conn.execute("UPDATE chunks SET doc_title = ? WHERE doc_id = ?", (title, doc_id))
        conn.commit()
        set_metadata(conn, doc_id, attributes={"title_inferred": True})

    return title
