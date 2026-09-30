"""Resolve an item's gold chunks from its evidence quotes (GH #92).

Chunk ids are positional: re-chunking a document shifts them, and a golden
set that stores them goes stale without a sound. A mined item instead stores
verbatim quotes from physical PDF pages. At eval time the gold chunks are the
chunks on that page that hold the quote — found by token coverage rather than
exact substring, because a chunk's text is the parser's rendering of the page
(table cells joined with ``|``, different spacing) and not pdftotext's.

A MACRO chunk is a chapter summary that spans many pages; counting it as gold
would credit "found the right chapter" as "found the fact", so only MESO and
MICRO chunks qualify. Lineage expansion in the metrics still credits a hit on
the parent or a neighbour.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata

from datasheet_rag.eval.dataset import GoldenItem
from datasheet_rag.models.chunk import ChunkLevel

#: Share of a quote's tokens a chunk must contain to count as holding it.
COVERAGE = 0.8

_SUBS = {
    "μ": "µ",
    "–": "-",
    "—": "-",
    "−": "-",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "Ω": "ω",
    "±": "+-",
    "≤": "<=",
    "≥": ">=",
}


def normalize(text: str) -> str:
    """Fold the spellings PDF text extraction varies on, then lowercase and
    collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    # Unmapped glyphs (an Ω in a figure, say) come out as control characters.
    text = "".join(c for c in text if c.isspace() or unicodedata.category(c) != "Cc")
    for a, b in _SUBS.items():
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9µω.+\-]+", normalize(text)))


def resolve_gold_chunk_ids(conn: sqlite3.Connection, item: GoldenItem) -> list[str]:
    """The MESO/MICRO chunks of ``item.doc_id`` that hold an evidence quote.

    Empty when the store lost the text (e.g. a table the parser dropped): the
    item then scores zero, which is the store's failure to report, not a
    labelling error.
    """
    if not item.evidence or item.doc_id is None:
        return []
    gold: list[str] = []
    for ev in item.evidence:
        want = tokens(ev.quote)
        if not want:
            continue
        # Only the evidence page's chunks: a reference manual holds thousands.
        # A figure chunk's text is often just "[Figure]"; what search sees is
        # its caption and vision description, so those count as its content.
        rows = conn.execute(
            "SELECT id, text, figure_caption, figure_description FROM chunks "
            "WHERE doc_id = ? AND level != ? "
            "AND EXISTS (SELECT 1 FROM json_each(page_numbers) WHERE value = ?) "
            "ORDER BY rowid",
            (item.doc_id, ChunkLevel.MACRO.value, ev.page),
        ).fetchall()
        for chunk_id, text, caption, description in rows:
            if chunk_id in gold:
                continue
            have = tokens(" ".join(t for t in (text, caption, description) if t))
            if len(want & have) / len(want) >= COVERAGE:
                gold.append(chunk_id)
    return gold
