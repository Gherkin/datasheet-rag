"""Multi-pass concentrating summarizer for MACRO chunks.

MACRO chunks (Level 0) represent entire chapters/sections, which can be
very large. Instead of embedding the raw text, we produce a concentrated
summary through bottom-up aggregation:

  Pass 1: Summarize each MICRO chunk into a one-line digest + weight
  Pass 2: Group MICRO digests by MESO parent → summarize each MESO group
  Pass 3: Combine weighted MESO summaries → produce the MACRO summary

The weight reflects information density: tables with specifications and
key-value data weigh more than boilerplate text. This ensures the summary
prioritises the most information-rich content.

Supports two modes:
  - extractive: No LLM needed. Uses heuristics to pick representative
    sentences weighted by content type. Fast and free.
  - abstractive: Uses Bedrock Claude to produce quality summaries.
    Better quality but requires API calls.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.progress import track

from aws_rag.chunking.context import build_macro_context
from aws_rag.models.chunk import ChunkGraph, ChunkLevel, LayoutType

console = Console()


# ---------------------------------------------------------------------------
# Content weights — how much a chunk type matters for the summary
# ---------------------------------------------------------------------------

CONTENT_WEIGHTS: dict[LayoutType, float] = {
    LayoutType.TABLE: 1.5,       # Tables with specs are high-value
    LayoutType.KEY_VALUE: 1.4,   # Key-value pairs (specs, params)
    LayoutType.FIGURE: 0.8,      # Figures contribute via captions
    LayoutType.LIST: 1.0,        # Lists (feature lists, etc.)
    LayoutType.TEXT: 1.0,        # Regular text
    LayoutType.HEADER: 0.3,      # Headers are structural, less content
    LayoutType.MIXED: 1.0,       # Mixed content
}


def _content_weight(layout_type: LayoutType) -> float:
    return CONTENT_WEIGHTS.get(layout_type, 1.0)


# ---------------------------------------------------------------------------
# Extractive summarizer (no LLM)
# ---------------------------------------------------------------------------


class ExtractiveSummarizer:
    """Bottom-up extractive summarizer using heuristic sentence selection."""

    def __init__(
        self,
        *,
        micro_digest_max_chars: int = 150,
        meso_summary_max_chars: int = 400,
        macro_summary_max_chars: int = 1500,
    ):
        self.micro_digest_max_chars = micro_digest_max_chars
        self.meso_summary_max_chars = meso_summary_max_chars
        self.macro_summary_max_chars = macro_summary_max_chars

    def summarize_graph(self, graph: ChunkGraph) -> ChunkGraph:
        """Fill in MACRO chunk text via multi-pass extractive summarization."""
        macro_chunks = graph.by_level(ChunkLevel.MACRO)

        for macro in track(macro_chunks, description="Summarizing chapters…"):
            meso_children = graph.children_of(macro.id)
            if not meso_children:
                continue

            # Pass 1 & 2: For each MESO, summarize its MICRO children
            meso_summaries: list[tuple[str, float]] = []  # (summary, weight)
            for meso in meso_children:
                micro_children = graph.children_of(meso.id)
                if not micro_children:
                    # MESO has no micros — use its own text
                    digest = _extractive_digest(meso.text, self.meso_summary_max_chars)
                    weight = _content_weight(meso.metadata.layout_type)
                    meso_summaries.append((digest, weight))
                    continue

                # Pass 1: Digest each MICRO
                micro_digests: list[tuple[str, float]] = []
                for mc in micro_children:
                    digest = _extractive_digest(mc.text, self.micro_digest_max_chars)
                    weight = _content_weight(mc.metadata.layout_type)
                    micro_digests.append((digest, weight))

                # Pass 2: Combine MICRO digests into MESO summary
                meso_summary = _weighted_combine(
                    micro_digests, self.meso_summary_max_chars
                )
                # MESO weight is the average of its micro weights
                avg_weight = (
                    sum(w for _, w in micro_digests) / len(micro_digests)
                    if micro_digests else 1.0
                )
                meso_summaries.append((meso_summary, avg_weight))

            # Pass 3: Combine MESO summaries into MACRO summary
            macro_text = _weighted_combine(
                meso_summaries, self.macro_summary_max_chars
            )

            macro.text = macro_text
            macro.token_count = len(macro_text) // 4
            macro.context_text = build_macro_context(macro, graph)

        return graph


def _extractive_digest(text: str, max_chars: int) -> str:
    """Extract the most representative sentences from text.

    Strategy:
    - For short text: return as-is
    - For tables: take the first row (headers) and a sample of data rows
    - For long text: take the first sentence(s) that fit the budget
    """
    text = text.strip()
    if not text or len(text) <= max_chars:
        return text

    # Try to split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) <= 1:
        # Can't split by sentences — truncate at word boundary
        return _truncate_word_boundary(text, max_chars)

    # Greedily take sentences from the start
    result: list[str] = []
    total = 0
    for s in sentences:
        if total + len(s) + 1 > max_chars and result:
            break
        result.append(s)
        total += len(s) + 1

    return " ".join(result)


# Below this, a per-item slice is too short to read as anything but noise —
# better to drop the item than pad the output with fragments.
_MIN_ITEM_CHARS = 40


def _weighted_combine(
    items: list[tuple[str, float]],
    max_chars: int,
) -> str:
    """Combine text items weighted by importance, fitting within budget.

    Higher-weight items get proportionally more space in the output.

    When there are far more items than the budget can give a meaningful
    slice to (e.g. thousands of MESO summaries feeding one MACRO budget),
    keep only the highest-weight items that each clear _MIN_ITEM_CHARS,
    in their original document order, and drop the rest — a coherent
    summary of the most important pieces beats a wall of sub-word
    fragments. (A previous version floored every item's allocation to
    30-50 chars regardless of count, which guaranteed the combined output
    could blow past max_chars by an order of magnitude — e.g. a single
    MACRO summary came out at 91KB against a 1500-char budget.)
    """
    if not items:
        return ""

    # Single item — just truncate
    if len(items) == 1:
        return _truncate_word_boundary(items[0][0], max_chars)

    max_items = max(1, max_chars // (_MIN_ITEM_CHARS + 1))
    if len(items) > max_items:
        keep = {
            i for i, _ in sorted(enumerate(items), key=lambda p: p[1][1], reverse=True)[:max_items]
        }
        items = [item for i, item in enumerate(items) if i in keep]

    # Calculate space allocation proportional to weight
    total_weight = sum(w for _, w in items)
    if total_weight == 0:
        total_weight = len(items)

    allocations = [
        max(_MIN_ITEM_CHARS, int((weight / total_weight) * max_chars))
        for _, weight in items
    ]

    # Scale down if total exceeds budget — no floor here, since a floor
    # is exactly what let the total run away from max_chars in the first place.
    total_alloc = sum(allocations)
    if total_alloc > max_chars:
        scale = max_chars / total_alloc
        allocations = [max(1, int(a * scale)) for a in allocations]

    parts: list[str] = []
    for (text, _), alloc in zip(items, allocations):
        truncated = _truncate_word_boundary(text.strip(), alloc)
        if truncated:
            parts.append(truncated)

    # Backstop: guarantees the result can never exceed max_chars regardless
    # of rounding/separator drift in the allocation above.
    return _truncate_word_boundary(" ".join(parts), max_chars)


def _truncate_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text at a word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Find last space
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.5:
        truncated = truncated[:last_space]
    return truncated.rstrip() + "…"


# ---------------------------------------------------------------------------
# Abstractive summarizer (LLM-based via Bedrock)
# ---------------------------------------------------------------------------


@dataclass
class SummarizerStats:
    """Cost/latency tally for one ``summarize_graph`` run.

    Lets the eval answer "how many extra Bedrock calls per chapter, and
    what's the latency budget" without instrumenting Bedrock itself.
    """

    calls: int = 0
    total_latency_ms: float = 0.0
    input_chars: int = 0
    output_chars: int = 0
    chapters: int = 0
    per_chapter_calls: list[int] = field(default_factory=list)

    @property
    def avg_calls_per_chapter(self) -> float:
        return self.calls / self.chapters if self.chapters else 0.0

    @property
    def avg_latency_ms_per_chapter(self) -> float:
        return self.total_latency_ms / self.chapters if self.chapters else 0.0


class AbstractiveSummarizer:
    """Bottom-up LLM-based summarizer using Bedrock Claude.

    Uses the same 3-pass structure as extractive, but each pass
    calls Claude for higher quality summaries. Includes content
    weighting in the prompt.
    """

    def __init__(
        self,
        *,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        micro_digest_max_tokens: int = 50,
        meso_summary_max_tokens: int = 150,
        macro_summary_max_tokens: int = 500,
        region: str | None = None,
    ):
        self.model_id = model_id
        self.micro_digest_max_tokens = micro_digest_max_tokens
        self.meso_summary_max_tokens = meso_summary_max_tokens
        self.macro_summary_max_tokens = macro_summary_max_tokens
        self.region = region
        self._client: Any = None
        self.stats = SummarizerStats()

    def _get_client(self) -> Any:
        if self._client is None:
            from aws_rag.local_models import get_chat_client
            self._client = get_chat_client(kind="text", region=self.region)
        return self._client

    def _invoke(self, prompt: str, max_tokens: int) -> str:
        """Call Bedrock Claude and return the response text.

        Tallies call count, latency, and char volume into ``self.stats`` —
        the substrate for the cost/latency side of the extractive-vs-
        abstractive eval (see README "Switch MACRO summaries…" TODO).
        """
        client = self._get_client()

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": (
                "You are a technical indexer for electronics datasheets. "
                "Your job is to describe what a chapter or section CONTAINS — "
                "the actual data, tables, and specifications present in the source text — "
                "not to explain the subject matter in general terms. "
                "A chapter with one feature table should be described in terms of what "
                "that table shows: which devices, which parameters, which value ranges. "
                "A short chapter has a short accurate summary. Never pad with general "
                "background knowledge. "
                "STRICT RULE: Only use facts, values, part numbers, and specifications "
                "that are explicitly stated in the provided source text. "
                "If a value is not in the source, do not include it. "
                "Never infer, extrapolate, or recall facts from your training data."
            ),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        })

        t0 = time.perf_counter()
        response = client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = json.loads(response["body"].read())
        text = result["content"][0]["text"].strip()

        self.stats.calls += 1
        self.stats.total_latency_ms += latency_ms
        self.stats.input_chars += len(prompt)
        self.stats.output_chars += len(text)

        return text

    def summarize_graph(self, graph: ChunkGraph) -> ChunkGraph:
        """Fill in MACRO chunk text via multi-pass LLM summarization."""
        macro_chunks = graph.by_level(ChunkLevel.MACRO)

        for macro in track(macro_chunks, description="Summarizing chapters (LLM)…"):
            meso_children = graph.children_of(macro.id)
            if not meso_children:
                continue

            calls_before = self.stats.calls
            self.stats.chapters += 1
            meso_summaries: list[tuple[str, float]] = []

            for meso in meso_children:
                micro_children = graph.children_of(meso.id)
                if not micro_children:
                    digest = _extractive_digest(meso.text, 600)
                    weight = _content_weight(meso.metadata.layout_type)
                    meso_summaries.append((digest, weight))
                    continue

                # Pass 1: Digest each MICRO chunk
                micro_digests: list[tuple[str, float]] = []
                for mc in micro_children:
                    if mc.token_count < 30:
                        # Very short chunks don't need LLM summarization
                        micro_digests.append((mc.text, _content_weight(mc.metadata.layout_type)))
                        continue

                    prompt = _micro_digest_prompt(
                        mc.text,
                        mc.metadata.layout_type,
                        chapter_title=macro.metadata.chapter_title,
                        section_title=meso.metadata.section_title,
                    )
                    digest = self._invoke(prompt, self.micro_digest_max_tokens)
                    weight = _content_weight(mc.metadata.layout_type)
                    micro_digests.append((digest, weight))

                # Pass 2: Combine micro digests → meso summary
                prompt = _meso_summary_prompt(
                    micro_digests,
                    meso.metadata.section_title,
                    chapter_title=macro.metadata.chapter_title,
                    doc_title=macro.metadata.doc_title,
                )
                meso_summary = self._invoke(prompt, self.meso_summary_max_tokens)
                avg_weight = (
                    sum(w for _, w in micro_digests) / len(micro_digests)
                    if micro_digests else 1.0
                )
                meso_summaries.append((meso_summary, avg_weight))

            # Pass 3: Recursively reduce meso summaries to a small set (bounded
            # input per LLM call regardless of chapter fan-in), then combine
            # the survivors into the final chapter summary.
            reduced = self._reduce_summaries(
                meso_summaries,
                chapter_title=macro.metadata.chapter_title,
                doc_title=macro.metadata.doc_title,
            )
            prompt = _macro_summary_prompt(
                reduced,
                macro.metadata.chapter_title,
                macro.metadata.doc_title,
            )
            macro_text = self._invoke(prompt, self.macro_summary_max_tokens)

            macro.text = macro_text
            macro.token_count = len(macro_text) // 4
            macro.context_text = build_macro_context(macro, graph)

            self.stats.per_chapter_calls.append(self.stats.calls - calls_before)

        return graph

    def _reduce_summaries(
        self,
        items: list[tuple[str, float]],
        *,
        chapter_title: str,
        doc_title: str,
    ) -> list[tuple[str, float]]:
        """Recursively batch-digest items until few enough for a final reduce.

        Each LLM call sees at most _MAX_REDUCE_FANIN items — bounded input
        regardless of how many MESO children a chapter has (one chapter in
        a 2200-page MCU family datasheet had 3949, which would otherwise
        produce a single ~600K-char reduce prompt). Genuinely abstractive
        at every level: each batch becomes a real summary of what it
        collectively covers, not a concatenation of fragments.
        """
        if len(items) <= _MAX_REDUCE_FANIN:
            return items

        digests: list[tuple[str, float]] = []
        for i in range(0, len(items), _MAX_REDUCE_FANIN):
            batch = items[i:i + _MAX_REDUCE_FANIN]
            prompt = _group_digest_prompt(batch, chapter_title, doc_title)
            digest = self._invoke(prompt, self.meso_summary_max_tokens)
            avg_weight = sum(w for _, w in batch) / len(batch)
            digests.append((digest, avg_weight))

        return self._reduce_summaries(digests, chapter_title=chapter_title, doc_title=doc_title)


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

# Maximum number of summaries combined in a single LLM reduce call. Keeps
# every prompt bounded regardless of chapter fan-in — larger sets are
# digested in batches and reduced recursively (see _reduce_summaries).
_MAX_REDUCE_FANIN = 25


def _group_digest_prompt(
    items: list[tuple[str, float]],
    chapter_title: str,
    doc_title: str,
) -> str:
    items_text = "\n".join(f"- [weight={w:.1f}] {text}" for text, w in items)
    return (
        f"Document: {doc_title}\n"
        f"Chapter: {chapter_title}\n\n"
        f"The following are descriptions of a group of consecutive sections "
        f"within this chapter. These descriptions are your ONLY source of "
        f"truth — you do not have access to the original text. Higher-weight "
        f"items are more information-dense.\n\n"
        f"{items_text}\n\n"
        f"Summarize what this group of sections collectively contains in "
        f"2-4 sentences. Only use facts present in the descriptions above — "
        f"no added values or inferences. Preserve exact part numbers, "
        f"parameter names, and value ranges as stated."
    )


def _micro_digest_prompt(
    text: str,
    layout_type: LayoutType,
    chapter_title: str | None = None,
    section_title: str | None = None,
) -> str:
    type_hint = ""
    if layout_type == LayoutType.TABLE:
        type_hint = "This is a specification table. Describe what parameters and value ranges it contains."
    elif layout_type == LayoutType.FIGURE:
        type_hint = "This is a figure or diagram. Describe what it shows."
    elif layout_type == LayoutType.KEY_VALUE:
        type_hint = "This contains key-value specification pairs."

    context = ""
    if chapter_title:
        context += f"Chapter: {chapter_title}\n"
    if section_title:
        context += f"Section: {section_title}\n"
    if context:
        context += "\n"

    return (
        f"{context}"
        f"Describe what the following source text contains in 1-2 sentences. "
        f"{type_hint} "
        f"Only use facts explicitly present in this source text. "
        f"Do not add background knowledge.\n\n"
        f"Source text:\n{text}"
    )


def _meso_summary_prompt(
    micro_digests: list[tuple[str, float]],
    section_title: str,
    chapter_title: str | None = None,
    doc_title: str | None = None,
) -> str:
    digests_text = "\n".join(
        f"- [weight={w:.1f}] {text}" for text, w in micro_digests
    )
    context = ""
    if doc_title:
        context += f"Document: {doc_title}\n"
    if chapter_title:
        context += f"Chapter: {chapter_title}\n"
    context += f"Section: {section_title}\n"

    return (
        f"{context}\n"
        f"The following are descriptions of individual blocks in this section. "
        f"Higher-weight items are more information-dense.\n\n"
        f"{digests_text}\n\n"
        f"Describe what this section contains in 2-4 sentences. "
        f"Only use facts present in the descriptions above — no added values or inferences. "
        f"If the section contains a table, say what the table shows and what value ranges appear. "
        f"Preserve exact part numbers, parameter names, and ranges as stated."
    )


def _macro_summary_prompt(
    meso_summaries: list[tuple[str, float]],
    chapter_title: str,
    doc_title: str,
) -> str:
    summaries_text = "\n".join(
        f"- [weight={w:.1f}] {text}" for text, w in meso_summaries
    )
    return (
        f"Document: {doc_title}\n"
        f"Chapter: {chapter_title}\n\n"
        f"The following are descriptions of the sections within this chapter. "
        f"These section descriptions are your ONLY source of truth — "
        f"you do not have access to the original document text, only these summaries. "
        f"Higher-weight sections are more information-dense.\n\n"
        f"{summaries_text}\n\n"
        f"Open by stating what is DISTINCT or UNIQUE about this chapter — "
        f"the specific topic, procedure, or specification it covers — not by "
        f"restating the document title or device family (the reader already "
        f"knows what document this is; your job is to differentiate this "
        f"chapter from the others). "
        f"Describe what this chapter contains in 4-8 sentences. "
        f"If the chapter is short, a short accurate description is correct — do not pad. "
        f"Only use facts present in the section descriptions above. "
        f"State which device families, parameters, tables, or figures are present, "
        f"and what specific values or ranges appear. "
        f"This will be used for semantic search — include exact part numbers, "
        f"parameter names, and value ranges as stated."
    )
