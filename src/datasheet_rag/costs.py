"""Reference AWS pricing for cost estimation (informational only).

These are hard-coded snapshots of list prices, not fetched live — verify
against your console/region before budgeting at volume. They power
``rag ingest --show-cost``, which estimates what a run would cost *without*
invoking the priced AWS calls (Bedrock embeddings/vision, Textract OCR), so
you can size up a batch of datasheets before committing to it.

Pricing reference (eu-west-1, May 2026):

* Bedrock Titan Embed Text v2  — ``$0.02 / 1M`` input tokens.
* Bedrock Claude 3 Haiku       — ``$0.25 / 1M`` input, ``$1.25 / 1M`` output.
  Per figure description: roughly ``$0.0005 - $0.001`` including the
  image-token surcharge and ~200 output tokens (see
  :mod:`datasheet_rag.description.describer`).
* AWS Textract AnalyzeDocument (LAYOUT) — billed per page regardless of
  document complexity; this is the line item most likely to dominate cost
  at volume for scanned datasheets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import ChunkGraph

TITAN_V2_USD_PER_1M_INPUT_TOKENS = 0.02

HAIKU_USD_PER_1M_INPUT_TOKENS = 0.25
HAIKU_USD_PER_1M_OUTPUT_TOKENS = 1.25

# Midpoint of the describer's documented $0.0005-$0.001 per-figure range.
HAIKU_VISION_USD_PER_FIGURE = 0.00075

# A single small text completion (a few hundred tokens in and out).
HAIKU_TITLE_INFERENCE_USD = 0.0005


@dataclass(frozen=True)
class TokenPrice:
    """USD per million tokens, split the way Bedrock bills a Converse call."""

    input: float
    output: float
    cache_read: float
    cache_write: float


#: From the Bedrock EU rate cards (``aws bedrock
#: list-foundation-model-agreement-offers``), read 2026-09-30. A ``global.``
#: profile bills the "Global" rows, an ``eu.`` profile the "Regional CRIS"
#: rows, about 10% more. Cache writes are the 5-minute ones.
CLAUDE_TOKEN_PRICES: dict[str, TokenPrice] = {
    "global.anthropic.claude-sonnet-4-6": TokenPrice(3.0, 15.0, 0.3, 3.75),
    "eu.anthropic.claude-sonnet-4-6": TokenPrice(3.3, 16.5, 0.33, 4.125),
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": TokenPrice(1.0, 5.0, 0.1, 1.25),
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0": TokenPrice(1.1, 5.5, 0.11, 1.375),
}


def token_cost_usd(
    model_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """What a model's token use cost, or None for a model not in the table."""
    p = CLAUDE_TOKEN_PRICES.get(model_id)
    if p is None:
        return None
    return (
        input_tokens * p.input
        + output_tokens * p.output
        + cache_read_tokens * p.cache_read
        + cache_write_tokens * p.cache_write
    ) / 1_000_000


# Approximate — verify in your console; Textract does not publish a
# per-feature price for LAYOUT in isolation, so this assumes it's billed
# alongside base AnalyzeDocument page processing.
TEXTRACT_LAYOUT_USD_PER_PAGE = 0.004

_CHARS_PER_TOKEN = 4


@dataclass
class CostLineItem:
    label: str
    detail: str
    usd: float


@dataclass
class CostEstimate:
    items: list[CostLineItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(item.usd for item in self.items)


def pdf_page_count(pdf_path: Path) -> int:
    """Return the page count of ``pdf_path`` using PyMuPDF."""
    import fitz  # pymupdf

    with fitz.open(str(pdf_path)) as doc:
        pages: int = doc.page_count
    return pages


def estimate_textract_cost(page_count: int) -> CostLineItem:
    usd = page_count * TEXTRACT_LAYOUT_USD_PER_PAGE
    return CostLineItem(
        label="Textract OCR",
        detail=f"{page_count} pages × ${TEXTRACT_LAYOUT_USD_PER_PAGE:.4f}/page",
        usd=usd,
    )


def estimate_embedding_cost(graph: ChunkGraph) -> CostLineItem:
    """Estimate Titan v2 embedding cost from chunk text length.

    Mirrors the embedder's own preference for ``context_text`` (falling
    back to ``text``), and the ``chars // 4`` heuristic already used for
    :attr:`Chunk.token_count` — so the estimate lines up with what
    ``rag ingest`` would report for a real run.
    """
    if get_settings().embedding_backend == "local":
        return CostLineItem(
            label="Embeddings (local Ollama)",
            detail="local backend — no AWS cost",
            usd=0.0,
        )
    chars = sum(len(c.context_text or c.text) for c in graph.chunks.values())
    tokens = max(1, chars // _CHARS_PER_TOKEN)
    usd = tokens / 1_000_000 * TITAN_V2_USD_PER_1M_INPUT_TOKENS
    return CostLineItem(
        label="Embeddings (Titan v2)",
        detail=f"~{tokens:,} input tokens × ${TITAN_V2_USD_PER_1M_INPUT_TOKENS}/1M",
        usd=usd,
    )


def estimate_figure_description_cost(figure_count: int) -> CostLineItem:
    if get_settings().vision_backend == "local":
        return CostLineItem(
            label="Figure descriptions (local Ollama vision)",
            detail=f"{figure_count} figures — local backend, no AWS cost",
            usd=0.0,
        )
    usd = figure_count * HAIKU_VISION_USD_PER_FIGURE
    return CostLineItem(
        label="Figure descriptions (Haiku vision)",
        detail=f"{figure_count} figures × ~${HAIKU_VISION_USD_PER_FIGURE:.5f}/figure",
        usd=usd,
    )


def estimate_title_inference_cost() -> CostLineItem:
    if get_settings().text_backend == "local":
        return CostLineItem(
            label="Title inference (local Ollama text)",
            detail="local backend — no AWS cost",
            usd=0.0,
        )
    return CostLineItem(
        label="Title inference (Haiku text)",
        detail="1 completion × ~$0.0005",
        usd=HAIKU_TITLE_INFERENCE_USD,
    )
