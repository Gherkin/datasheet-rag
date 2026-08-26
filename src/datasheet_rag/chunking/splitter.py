"""Multi-scale text splitter for hierarchical chunking.

Takes a DocumentOutline (from layout_parser) and produces chunks at three
levels (MICRO, MESO, MACRO), respecting layout boundaries so that tables,
figures, and key-value blocks are never split mid-element.

Level 2 (MICRO, ~128 tokens)  — paragraph / table / figure level
Level 1 (MESO,  ~512 tokens)  — subsection level (groups of micro chunks)
Level 0 (MACRO)                — chapter level (summarised, not raw text)

MACRO chunks do NOT contain the full section text. They will be populated
by the summarizer module with multi-pass concentrated summaries.
"""

from __future__ import annotations

import re
from typing import Any

from datasheet_rag.chunking.layout_parser import (
    ContentElement,
    DocumentOutline,
    DocumentSection,
    ElementType,
)
from datasheet_rag.models.chunk import (
    Chunk,
    ChunkGraph,
    ChunkLevel,
    ChunkMetadata,
    LayoutType,
)

# ---------------------------------------------------------------------------
# Token counting (lightweight approximation, no external dependency needed)
# ---------------------------------------------------------------------------

# Rough ratio: 1 token ≈ 4 characters for English text.
# We use this for splitting decisions. Actual token counts for embedding
# are computed later with the model's tokenizer.
_CHARS_PER_TOKEN = 4

# Bedrock Titan embeddings reject input over 50,000 characters
# (`ValidationException: maxLength: 50000`). Tables are atomic — we never
# split them — so this is the one place a single chunk can still blow past
# that limit despite the compact-rendering and garbled-header fixes (e.g. a
# table whose structure Docling genuinely cannot recover, see
# docling_parser._detect_garbled_header and the "Table parsing warning"
# messages it emits). Rather than silently truncate (and ship a chunk that's
# missing whoever-knows-what) or let Bedrock's stack trace surface deep in
# the embedding pipeline, we fail loudly here with enough context to act on.
_EMBEDDING_MAX_CHARS = 50_000


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SplitterConfig:
    """Configuration for the multi-scale splitter."""

    def __init__(
        self,
        *,
        micro_max_tokens: int = 128,
        meso_max_tokens: int = 512,
        macro_max_tokens: int = 2000,
        # Minimum figure size (pixels) to include as a figure chunk.
        # Smaller figures (logos, icons) are skipped.
        min_figure_px: int = 400,
        # When splitting text, try to break at sentence boundaries
        prefer_sentence_breaks: bool = True,
    ):
        self.micro_max_tokens = micro_max_tokens
        self.meso_max_tokens = meso_max_tokens
        self.macro_max_tokens = macro_max_tokens
        self.min_figure_px = min_figure_px
        self.prefer_sentence_breaks = prefer_sentence_breaks


# ---------------------------------------------------------------------------
# Main splitter
# ---------------------------------------------------------------------------


def split_document(
    outline: DocumentOutline,
    *,
    config: SplitterConfig | None = None,
    figure_manifest: dict[str, Any] | None = None,
) -> ChunkGraph:
    """Split a parsed document outline into a multi-scale chunk graph.

    Returns a ChunkGraph with chunks at all three levels.
    MACRO chunks have placeholder text — they must be filled by the summarizer.

    figure_manifest: optional dict from figures.py FigureManifest.to_dict()
        Used to link figure chunks to their extracted images.
    """
    if config is None:
        config = SplitterConfig()

    doc_id = outline.doc_id
    graph = ChunkGraph(doc_id=doc_id)

    # Build figure lookup: block_id → figure info
    figure_lookup = _build_figure_lookup(figure_manifest)

    micro_counter = 0
    meso_counter = 0
    macro_counter = 0

    for section in outline.sections:
        # --- MACRO chunk (Level 0) ---
        macro_id = f"{doc_id}:L0:{macro_counter}"
        macro_counter += 1

        macro_chunk = Chunk(
            id=macro_id,
            doc_id=doc_id,
            level=ChunkLevel.MACRO,
            text="",  # Placeholder — filled by summarizer
            token_count=0,
            metadata=ChunkMetadata(
                doc_id=doc_id,
                doc_title=outline.title,
                chapter_title=section.title,
                section_title=section.title,
                page_numbers=section.all_pages,
                layout_type=LayoutType.MIXED,
            ),
        )
        graph.add(macro_chunk)

        # Collect all elements for this top-level section (including children)
        all_elements = _collect_section_elements(section)

        # --- MICRO chunks (Level 2) ---
        micro_chunks = _create_micro_chunks(
            elements=all_elements,
            doc_id=doc_id,
            doc_title=outline.title,
            section=section,
            config=config,
            figure_lookup=figure_lookup,
            counter_start=micro_counter,
        )
        micro_counter += len(micro_chunks)

        for mc in micro_chunks:
            graph.add(mc)

        # --- MESO chunks (Level 1) ---
        meso_chunks = _create_meso_chunks(
            micro_chunks=micro_chunks,
            doc_id=doc_id,
            doc_title=outline.title,
            section=section,
            config=config,
            counter_start=meso_counter,
        )
        meso_counter += len(meso_chunks)

        for meso in meso_chunks:
            graph.add(meso)

        # --- Wire up parent/child links ---
        macro_chunk.children_ids = [m.id for m in meso_chunks]
        for meso in meso_chunks:
            meso.parent_id = macro_id
            meso.chapter_root_id = macro_id

        for mc in micro_chunks:
            mc.chapter_root_id = macro_id

    return graph


# ---------------------------------------------------------------------------
# MICRO chunk creation
# ---------------------------------------------------------------------------


def _collect_section_elements(
    section: DocumentSection,
) -> list[tuple[ContentElement, DocumentSection]]:
    """Collect all elements with their immediate section context, depth-first."""
    results: list[tuple[ContentElement, DocumentSection]] = []
    for elem in section.elements:
        results.append((elem, section))
    for child in section.children:
        results.extend(_collect_section_elements(child))
    return results


def _create_micro_chunks(
    *,
    elements: list[tuple[ContentElement, DocumentSection]],
    doc_id: str,
    doc_title: str,
    section: DocumentSection,
    config: SplitterConfig,
    figure_lookup: dict[str, dict[str, Any]],
    counter_start: int,
) -> list[Chunk]:
    """Create MICRO level chunks from content elements.

    - Tables and figures become their own chunks (never split).
    - Text elements are split at sentence boundaries if they exceed the token limit.
    - Small figures (logos) are filtered out.
    """
    chunks: list[Chunk] = []
    counter = counter_start

    for elem, elem_section in elements:
        if elem.element_type == ElementType.FIGURE:
            # Check if figure is large enough to include
            fig_info = figure_lookup.get(elem.figure_block_id)
            if fig_info and (
                fig_info.get("width_px", 0) < config.min_figure_px
                and fig_info.get("height_px", 0) < config.min_figure_px
            ):
                continue  # Skip small figures (logos, icons)

            chunk = _make_chunk(
                doc_id=doc_id,
                level=ChunkLevel.MICRO,
                index=counter,
                text=elem.figure_caption or "[Figure]",
                doc_title=doc_title,
                chapter_title=section.title,
                section_title=elem_section.title,
                page=elem.page,
                layout_type=LayoutType.FIGURE,
            )
            # Attach figure references
            if fig_info:
                chunk.figure_image_path = fig_info.get("image_path")
                chunk.figure_s3_key = fig_info.get("s3_key")
            # The caption is the element's regardless of whether a crop was
            # made: with figures skipped there is no manifest, and folding the
            # caption into the same `if` left those chunks as a bare
            # "[Figure]" with nothing to reason about (GH #41).
            chunk.figure_caption = (
                (fig_info or {}).get("caption") or elem.figure_caption or ""
            ) or None

            chunks.append(chunk)
            counter += 1

        elif elem.element_type == ElementType.TABLE:
            # Tables are atomic — never split today. A table whose structure
            # Docling has genuinely mangled beyond what our rendering/
            # detection safety nets can recover (see
            # docling_parser.table_structure_untrustworthy,
            # _table_cells_to_compact_text, and
            # _table_cells_to_reading_order_text — the latter is the
            # structure-free fallback already baked into elem.text when
            # table_structure_warning is set) can still produce text too
            # large for Bedrock's embedding input limit. Truncating would
            # silently ship an incomplete table — worse than an error, for a
            # RAG system whose whole point is trustworthy answers about these
            # tables. Fail loudly with enough context instead.
            #
            # TODO: there are two genuine fixes here, neither implemented —
            # (a) page-boundary splitting for tables that physically span
            #     multiple PDF pages (see the "single Docling item physically
            #     spanning pages" warning in docling_parser._build_outline) —
            #     blocked on reconciling mismatched coordinate frames (table
            #     provenance bbox is BOTTOMLEFT-origin, cell bboxes are
            #     TOPLEFT-origin and often absent for empty cells), with no
            #     real multi-page+oversized instance yet to validate against;
            # (b) LLM-assisted introspective table repair/splitting (see
            #     docs/table-structure-repair/plan.md, local/untracked) — for
            #     the (hopefully now smaller) set of tables still oversized
            #     even after the structure-free fallback above.
            if len(elem.text) > _EMBEDDING_MAX_CHARS:
                raise ValueError(
                    f"Table on page {elem.page} "
                    f"({elem.table_title or 'untitled'!r}) renders to "
                    f"{len(elem.text):,} characters, which exceeds the "
                    f"{_EMBEDDING_MAX_CHARS:,}-character Bedrock embedding "
                    f"limit even after compact rendering and garbled-header "
                    f"suppression. Docling's table-structure recognition has "
                    f"likely produced a structure too broken to render "
                    f"sensibly (see the 'Table parsing warning' messages "
                    f"logged during conversion for this page). Tables are "
                    f"never split today, and switching table_structure_mode "
                    f"is NOT a fix: empirically, complex tables this badly "
                    f"mangled stay unusable in BOTH fast and accurate modes, "
                    f"just via different (and sometimes more dangerous, "
                    f"silent) failure mechanisms — see README → Table "
                    f"parsing mode. The durable fixes are unimplemented: "
                    f"page-boundary splitting (for tables spanning multiple "
                    f"PDF pages) or LLM-assisted introspective table "
                    f"repair/splitting (docs/table-structure-repair/plan.md). "
                    f"Until one of those lands, inspect page {elem.page} by "
                    f"hand to decide whether to exclude or hand-correct it."
                )

            chunk = _make_chunk(
                doc_id=doc_id,
                level=ChunkLevel.MICRO,
                index=counter,
                text=elem.text,
                doc_title=doc_title,
                chapter_title=section.title,
                section_title=elem_section.title,
                page=elem.page,
                layout_type=LayoutType.TABLE,
                table_structure_warning=elem.table_structure_warning,
            )
            chunks.append(chunk)
            counter += 1

        elif elem.element_type == ElementType.KEY_VALUE:
            chunk = _make_chunk(
                doc_id=doc_id,
                level=ChunkLevel.MICRO,
                index=counter,
                text=f"{elem.kv_key}: {elem.kv_value}" if elem.kv_key else elem.text,
                doc_title=doc_title,
                chapter_title=section.title,
                section_title=elem_section.title,
                page=elem.page,
                layout_type=LayoutType.KEY_VALUE,
            )
            chunks.append(chunk)
            counter += 1

        elif elem.element_type in (ElementType.TEXT, ElementType.LIST):
            # Split long text at sentence boundaries
            text_chunks = _split_text(
                elem.text,
                max_tokens=config.micro_max_tokens,
                prefer_sentences=config.prefer_sentence_breaks,
            )
            layout_type = (
                LayoutType.LIST if elem.element_type == ElementType.LIST else LayoutType.TEXT
            )

            for text_piece in text_chunks:
                chunk = _make_chunk(
                    doc_id=doc_id,
                    level=ChunkLevel.MICRO,
                    index=counter,
                    text=text_piece,
                    doc_title=doc_title,
                    chapter_title=section.title,
                    section_title=elem_section.title,
                    page=elem.page,
                    layout_type=layout_type,
                )
                chunks.append(chunk)
                counter += 1

    return chunks


# ---------------------------------------------------------------------------
# MESO chunk creation
# ---------------------------------------------------------------------------


def _create_meso_chunks(
    *,
    micro_chunks: list[Chunk],
    doc_id: str,
    doc_title: str,
    section: DocumentSection,
    config: SplitterConfig,
    counter_start: int,
) -> list[Chunk]:
    """Group MICRO chunks into MESO chunks respecting the token budget.

    Micro chunks are grouped greedily: keep adding micros to the current
    meso until the token budget would be exceeded, then start a new meso.
    Figures and tables are never merged with adjacent text into the same meso
    (they get their own meso if they're large enough).
    """
    if not micro_chunks:
        return []

    meso_chunks: list[Chunk] = []
    counter = counter_start

    current_group: list[Chunk] = []
    current_tokens = 0

    def _flush_group() -> None:
        nonlocal counter, current_group, current_tokens
        if not current_group:
            return

        combined_text = "\n\n".join(c.text for c in current_group)
        pages = sorted(set(p for c in current_group for p in c.metadata.page_numbers))
        # Determine dominant layout type
        types = [c.metadata.layout_type for c in current_group]
        layout_type = types[0] if len(set(types)) == 1 else LayoutType.MIXED
        # Use the section title from the first chunk's context
        section_title = current_group[0].metadata.section_title

        meso = _make_chunk(
            doc_id=doc_id,
            level=ChunkLevel.MESO,
            index=counter,
            text=combined_text,
            doc_title=doc_title,
            chapter_title=section.title,
            section_title=section_title,
            page=pages[0] if pages else 1,
            layout_type=layout_type,
        )
        meso.metadata.page_numbers = pages
        # A meso wrapping exactly one figure IS that figure — carry its image
        # across so the coarser zoom level is showable too. With more than one
        # there is no single right image to advertise, so it keeps none and
        # search reports it as unshowable rather than promising an image it
        # cannot serve (GH #41).
        if layout_type == LayoutType.FIGURE:
            figures = [c for c in current_group if c.figure_image_path or c.figure_s3_key]
            if len(figures) == 1:
                src = figures[0]
                meso.figure_image_path = src.figure_image_path
                meso.figure_s3_key = src.figure_s3_key
                meso.figure_caption = src.figure_caption
                meso.figure_description = src.figure_description
        meso.children_ids = [c.id for c in current_group]
        for c in current_group:
            c.parent_id = meso.id

        meso_chunks.append(meso)
        counter += 1
        current_group = []
        current_tokens = 0

    for mc in micro_chunks:
        mc_tokens = mc.token_count

        # Tables and figures that are substantial get their own meso
        is_atomic = mc.metadata.layout_type in (LayoutType.TABLE, LayoutType.FIGURE)

        if is_atomic and mc_tokens > config.micro_max_tokens:
            _flush_group()
            current_group = [mc]
            current_tokens = mc_tokens
            _flush_group()
            continue

        # Would adding this micro exceed the meso budget?
        if current_tokens + mc_tokens > config.meso_max_tokens and current_group:
            _flush_group()

        current_group.append(mc)
        current_tokens += mc_tokens

    _flush_group()
    return meso_chunks


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

# Sentence boundary pattern: split after .!? followed by whitespace
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_text(
    text: str,
    *,
    max_tokens: int,
    prefer_sentences: bool = True,
) -> list[str]:
    """Split text into pieces respecting the token budget.

    Tries to break at sentence boundaries. Falls back to word boundaries
    if sentences are too long.
    """
    text = text.strip()
    if not text:
        return []

    if _estimate_tokens(text) <= max_tokens:
        return [text]

    if prefer_sentences:
        sentences = _SENTENCE_RE.split(text)
    else:
        sentences = [text]

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = _estimate_tokens(sentence)

        if s_tokens > max_tokens:
            # Sentence itself is too long — split by words
            if current:
                pieces.append(" ".join(current))
                current = []
                current_tokens = 0
            pieces.extend(_split_by_words(sentence, max_tokens=max_tokens))
            continue

        if current_tokens + s_tokens > max_tokens and current:
            pieces.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += s_tokens

    if current:
        pieces.append(" ".join(current))

    return [p.strip() for p in pieces if p.strip()]


def _split_by_words(text: str, *, max_tokens: int) -> list[str]:
    """Last-resort splitting by word boundaries."""
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for word in words:
        w_tokens = _estimate_tokens(word)
        if current_tokens + w_tokens > max_tokens and current:
            pieces.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += w_tokens

    if current:
        pieces.append(" ".join(current))

    return pieces


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    doc_id: str,
    level: ChunkLevel,
    index: int,
    text: str,
    doc_title: str,
    chapter_title: str,
    section_title: str,
    page: int,
    layout_type: LayoutType,
    table_structure_warning: str | None = None,
) -> Chunk:
    """Create a Chunk with standard metadata."""
    return Chunk(
        id=f"{doc_id}:L{level.value}:{index}",
        doc_id=doc_id,
        level=level,
        text=text,
        token_count=_estimate_tokens(text),
        metadata=ChunkMetadata(
            doc_id=doc_id,
            doc_title=doc_title,
            chapter_title=chapter_title,
            section_title=section_title,
            page_numbers=[page],
            layout_type=layout_type,
            table_structure_warning=table_structure_warning,
        ),
    )


def _build_figure_lookup(
    figure_manifest: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Build a block_id → figure info lookup from the manifest."""
    if not figure_manifest:
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    for fig in figure_manifest.get("figures", []):
        block_id = fig.get("block_id", "")
        if block_id:
            lookup[block_id] = fig
    return lookup
