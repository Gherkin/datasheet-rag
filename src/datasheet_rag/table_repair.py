"""LLM-assisted table-structure repair (Stage 3 — see
docs/table-structure-repair/{problem,plan}.md).

**Header-band re-transcription**: when :func:`docling_parser.
table_structure_untrustworthy` flags a table, Docling's *extracted text* for
the header band cannot be trusted — not just its row/column placement.
Investigation against a real flagged table (PIC32CK1025GC01100, p.16) showed
the raw ``table_cells`` already contained corrupted, *concatenated* header
text (e.g. ``"Device rota e hisp g 90 Program Memory (KB)"`` for a single
column header that should just read ``"Device"``) — a text-extraction
failure, not merely a misplacement of otherwise-correct cell text. A repair
that only repositions existing cell text (the earlier design) cannot fix
this: it just shuffles the same garbled strings into different cells.

The fix instead **re-derives the header band from the page image directly**:
Docling's *data-row* grid (below the header band) is geometrically reliable —
its column count is the trusted dimension. We crop the header band (plus one
data row for visual alignment), tell the model exactly how many columns (C)
and header rows (H) to expect, and ask it to transcribe the H×C grid of
header cells from the image. The proposal must exactly tile the H×C band
(:func:`validate_header_grid`); validation also re-runs the garbled/fused
detectors against the *proposed* header to catch a non-fix or a fusion
recreation. On any rejection, the table keeps its existing structure-free
(reading-order) rendering — never worse than before repair.

Mirrors :class:`datasheet_rag.description.describer.FigureDescriber`'s shape
(lazy client, ``client: Any | None`` for mockability, tenacity retry on
transient ``ModelErrorException``, token-usage stats).
"""

from __future__ import annotations

import base64
import json
import re
from collections import Counter
from typing import Any

from PIL import Image
from rich.console import Console
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from datasheet_rag.chunking.layout_parser import BoundingBox
from datasheet_rag.config import get_settings
from datasheet_rag.docling_parser import (
    _FUSED_HEADER_MIN_CHARS,
    _GARBLED_HEADER_MIN_CHARS,
    _GARBLED_HEADER_MIN_REPEATS,
)
from datasheet_rag.figures import FigureRegion, crop_figure

console = Console()

_MAX_INVOKE_ATTEMPTS = 4
_INVOKE_WAIT = wait_exponential(multiplier=1, min=2, max=8)


def _is_transient_model_error(exc: BaseException) -> bool:
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return False
    return resp.get("Error", {}).get("Code") == "ModelErrorException"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You repair the header of a mis-parsed table from an electronics "
    "datasheet. Docling's text extraction produced corrupted header cell "
    "text (duplicated, concatenated, or misplaced labels) for this table's "
    "header band, but the table's column structure below the header band is "
    "correct and trustworthy. You are given a cropped image showing the "
    "header band plus one data row for visual alignment, and the table's "
    "exact dimensions. Read the header directly from the image and return "
    "its true contents as a grid of cells: "
    '{"cells": [{"row": 1, "col": 1, "row_span": 1, "col_span": 1, '
    '"text": "..."}, ...]}. '
    "row/col are 1-indexed positions within the header band (row 1 = "
    "topmost header row, col 1 = leftmost column); row_span/col_span are "
    "the number of grid positions a cell visually spans (1 for "
    "non-spanning cells). The cells must exactly tile the header band with "
    "no gaps or overlaps: every (row, col) from (1,1) to (H,C) covered "
    "exactly once, accounting for spans. Respond with JSON only, no prose."
)


def _build_user_blocks(
    *,
    image_bytes: bytes,
    image_format: str,
    caption: str,
    header_rows: int,
    column_count: int,
) -> list[dict[str, Any]]:
    """Build the user-message ``content`` array for the Anthropic messages API."""
    media_type = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(image_format.lower(), "image/png")

    text_parts: list[str] = []
    if caption:
        text_parts.append(f"Caption: {caption}")
    text_parts.append(
        f"This table's header band has H={header_rows} row(s) and "
        f"C={column_count} column(s). Return the corrected header grid as "
        f"JSON, tiling all H×C positions exactly once."
    )

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(image_bytes).decode("ascii"),
            },
        },
        {"type": "text", "text": "\n\n".join(text_parts)},
    ]


# ---------------------------------------------------------------------------
# Response parsing + validation
# ---------------------------------------------------------------------------

_REQUIRED_CELL_FIELDS = ("row", "col", "row_span", "col_span", "text")

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Strip a wrapping ```json ... ``` / ``` ... ``` markdown fence, if present.

    Despite the system prompt's "Respond with JSON only, no prose", Claude
    (Haiku especially) routinely wraps structured output in a fenced code
    block anyway — a cosmetic habit, not a malformed response, so unwrap it
    rather than reject it.
    """
    match = _CODE_FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _parse_repair_response(raw_text: str) -> list[dict[str, Any]] | None:
    """Extract the proposed header-cell grid from the model's JSON response.

    Returns ``None`` (rather than raising) on any malformed shape — an
    unparseable response is treated as "repair failed", not a crash; the
    caller falls back to the existing structure-free rendering.
    """
    try:
        payload = json.loads(_strip_code_fence(raw_text))
        proposed = payload["cells"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(proposed, list):
        return None
    for entry in proposed:
        if not isinstance(entry, dict) or any(f not in entry for f in _REQUIRED_CELL_FIELDS):
            return None
    return proposed


def validate_header_grid(
    proposed: list[dict[str, Any]],
    *,
    header_rows: int,
    column_count: int,
    data_cells: list[dict[str, Any]],
) -> str | None:
    """Check a proposed header-band grid for self-consistency before accepting it.

    Returns ``None`` if the proposal is acceptable, else a human-readable
    rejection reason. Every check is a structural invariant or a
    non-regression check — not a content-quality judgement (we have no
    ground truth for what the header *should* say):

    * row/col/spans are positive integers, within the H×C band
    * every (row, col) in the H×C band is covered exactly once (no gaps,
      no overlaps)
    * anti-degenerate: the proposed header doesn't still have the same
      long text repeated across columns (the original garbled-header
      signal — :func:`docling_parser._detect_garbled_header`'s logic,
      re-applied to the proposal)
    * anti-fusion: the proposed header doesn't recreate a fused header/data
      row by duplicating ``data_cells`` values (:func:`docling_parser.
      _detect_fused_header_row`'s logic, re-applied to the proposal)
    """
    occupied: set[tuple[int, int]] = set()
    texts_by_pos: dict[tuple[int, int], str] = {}
    for entry in proposed:
        try:
            row, col, row_span, col_span = (
                int(entry["row"]),
                int(entry["col"]),
                int(entry["row_span"]),
                int(entry["col_span"]),
            )
        except (TypeError, ValueError):
            return f"cell {entry!r} has non-integer row/col/span values"
        if row < 1 or col < 1 or row_span < 1 or col_span < 1:
            return f"cell at (row={row}, col={col}) has a non-positive row/col/span value"
        if row + row_span - 1 > header_rows or col + col_span - 1 > column_count:
            return (
                f"cell at (row={row}, col={col}) span {row_span}x{col_span} "
                f"extends outside the {header_rows}x{column_count} header band"
            )
        for r in range(row, row + row_span):
            for c in range(col, col + col_span):
                if (r, c) in occupied:
                    return f"cell at (row={row}, col={col}) overlaps another cell at ({r},{c})"
                occupied.add((r, c))
        texts_by_pos[(row, col)] = str(entry.get("text", "")).strip()

    full = {(r, c) for r in range(1, header_rows + 1) for c in range(1, column_count + 1)}
    if occupied != full:
        missing = sorted(full - occupied)[:5]
        return f"header band not fully covered — missing positions {missing}"

    # Anti-degenerate: same long text still repeated across distinct columns
    counts = Counter(
        text for text in texts_by_pos.values() if len(text) >= _GARBLED_HEADER_MIN_CHARS
    )
    for text, count in counts.most_common(1):
        if count >= _GARBLED_HEADER_MIN_REPEATS:
            return f"header still garbled — {text[:60]!r} repeated across {count} cells"

    # Anti-fusion: proposed header text recreating a leaked data row
    data_texts_by_col: dict[int, set[str]] = {}
    for c in data_cells:
        if not c.get("is_origin", True):
            continue
        text = c["text"].strip().casefold()
        if len(text) < _FUSED_HEADER_MIN_CHARS:
            continue
        for col in range(c["col"], c["col"] + c["col_span"]):
            data_texts_by_col.setdefault(col, set()).add(text)

    for row in range(1, header_rows + 1):
        overlap: set[str] = set()
        for (r, c), text in texts_by_pos.items():
            if r != row or len(text) < _FUSED_HEADER_MIN_CHARS:
                continue
            if text.casefold() in data_texts_by_col.get(c, set()):
                overlap.add(text)
        if len(overlap) >= 2:
            return f"proposed header row {row} duplicates data values: {sorted(overlap)[:3]}"

    return None


def splice_header_band(
    cells: list[dict[str, Any]],
    proposed: list[dict[str, Any]],
    *,
    header_rows: int,
    column_count: int,
) -> list[dict[str, Any]]:
    """Replace a table's header band (rows ``1..header_rows``) with a
    validated proposal, leaving data rows untouched.

    ``proposed`` (one entry per header cell, already validated by
    :func:`validate_header_grid`) is expanded into Docling's grid-fill
    convention (a spanning cell's text repeated at every covered position,
    ``is_origin=True`` only at the top-left) so the repaired header slots
    into the same shape existing renderers and detectors expect.
    """
    new_header: list[dict[str, Any]] = []
    for entry in proposed:
        row, col = int(entry["row"]), int(entry["col"])
        row_span, col_span = int(entry["row_span"]), int(entry["col_span"])
        text = str(entry.get("text", "")).strip()
        for r in range(row, row + row_span):
            for c in range(col, col + col_span):
                new_header.append(
                    {
                        "row": r,
                        "col": c,
                        "row_span": row_span,
                        "col_span": col_span,
                        "text": text,
                        "is_header": True,
                        "is_origin": (r, c) == (row, col),
                    }
                )

    rest = [c for c in cells if c["row"] > header_rows]
    return sorted(new_header + rest, key=lambda c: (c["row"], c["col"]))


def apply_repaired_structure(element: Any, merged_cells: list[dict[str, Any]]) -> None:
    """Apply a validated, spliced repair onto a TABLE ``ContentElement`` in place.

    Stores the repaired grid in ``table_repaired_cells`` (the cache field),
    re-renders ``element.text`` from it via the same trusted compact-grid path
    a Docling table that needed no repair gets (``_table_cells_to_compact_text``
    — safe now that the structure has passed :func:`validate_header_grid`),
    and clears ``table_structure_warning``: the table has earned trust, so it
    should render and gate identically to one Docling parsed correctly the
    first time. Mirrors the in-place patch convention
    ``reconvert_tables_in_range`` uses for ``cached.text`` / ``table_cells``.
    """
    from datasheet_rag.docling_parser import _table_cells_to_compact_text

    element.table_repaired_cells = merged_cells
    table_text = _table_cells_to_compact_text(merged_cells)
    caption = element.table_title
    element.text = (caption + "\n" + table_text).strip() if caption else table_text.strip()
    element.table_structure_warning = None


# ---------------------------------------------------------------------------
# The repairer
# ---------------------------------------------------------------------------


class TableRepairer:
    """Wrap Bedrock Claude vision for table-header re-transcription.

    Mirrors :class:`datasheet_rag.description.describer.FigureDescriber`'s shape:
    lazy client construction, ``client: Any | None`` for test injection,
    tenacity retry on transient ``ModelErrorException``, token-usage stats.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        max_tokens: int | None = None,
        region: str | None = None,
        profile: str | None = None,
        client: Any | None = None,
        max_concurrency: int | None = None,
        verbose: bool = False,
    ) -> None:
        settings = get_settings()
        self.model_id = model_id or settings.table_repair_model_id or settings.description_model_id
        self.max_tokens = max_tokens or settings.table_repair_max_tokens
        self.max_concurrency = max_concurrency or settings.table_repair_concurrency
        self.verbose = verbose
        self.region = region or settings.aws_region
        self.profile = profile or settings.aws_profile

        self.client: Any = client

        self._total_invocations = 0
        self._total_errors = 0
        self._total_rejected = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _get_client(self) -> Any:
        if self.client is None:
            from datasheet_rag.local_models import get_chat_client

            self.client = get_chat_client(kind="vision", region=self.region, profile=self.profile)
        return self.client

    # ---- public ---------------------------------------------------------

    def crop_table(
        self,
        page_image: Image.Image,
        *,
        bbox: BoundingBox,
        row_range: tuple[int, int] | None = None,
        total_rows: int | None = None,
    ) -> Image.Image:
        """Crop the table region from a rendered page image.

        ``row_range`` (1-indexed inclusive start/end) and ``total_rows``
        allow proportional sub-region cropping for the header-band repair —
        the crop covers just the requested row band rather than the whole
        table. This is only an approximation (per-cell bbox is not currently
        stored), but it is good enough to focus the model's attention and
        keeps the image token cost proportional to the header band size, not
        the table size.

        Adds a small padding margin beyond the crop region (passed through
        to :func:`~datasheet_rag.figures.crop_figure`).
        """
        if row_range is None or total_rows is None or total_rows == 0:
            region = FigureRegion(
                block_id="table-repair-crop",
                page=0,
                left=bbox.left,
                top=bbox.top,
                width=bbox.width,
                height=bbox.height,
            )
            return crop_figure(page_image, region)

        # Proportional vertical slice: [start-1 .. end] / total_rows * bbox_height
        row_start, row_end = row_range
        frac_top = (row_start - 1) / total_rows
        frac_bottom = row_end / total_rows
        sub_top = bbox.top + frac_top * bbox.height
        sub_height = (frac_bottom - frac_top) * bbox.height
        region = FigureRegion(
            block_id="table-repair-crop",
            page=0,
            left=bbox.left,
            top=sub_top,
            width=bbox.width,
            height=sub_height,
        )
        return crop_figure(page_image, region)

    def repair_header_band(
        self,
        *,
        image_bytes: bytes,
        image_format: str,
        cells: list[dict[str, Any]],
        caption: str = "",
        header_rows: int,
        column_count: int,
    ) -> list[dict[str, Any]] | None:
        """Propose a corrected header band for one table.

        ``header_rows``/``column_count`` define the H×C band the model must
        tile exactly. The crop image should already be pre-cropped to the
        header band (plus a data row for context).

        Returns the validated proposal (one entry per header cell, ready for
        :func:`splice_header_band`) on success, or ``None`` if the response
        was unparseable or failed structural validation — ``None`` means
        "keep the existing structure-free rendering".
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": _SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": _build_user_blocks(
                        image_bytes=image_bytes,
                        image_format=image_format,
                        caption=caption,
                        header_rows=header_rows,
                        column_count=column_count,
                    ),
                }
            ],
        }
        raw_text = self._invoke(body)

        proposed = _parse_repair_response(raw_text)
        if proposed is None:
            self._total_rejected += 1
            if self.verbose:
                console.print("[yellow]table repair rejected[/]: unparseable response")
            return None

        data_cells = [c for c in cells if c["row"] > header_rows]
        rejection = validate_header_grid(
            proposed,
            header_rows=header_rows,
            column_count=column_count,
            data_cells=data_cells,
        )
        if rejection is not None:
            self._total_rejected += 1
            if self.verbose:
                console.print(f"[yellow]table repair rejected[/]: {rejection}")
            return None

        return proposed

    def stats(self) -> dict[str, int]:
        return {
            "total_invocations": self._total_invocations,
            "total_errors": self._total_errors,
            "total_rejected": self._total_rejected,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }

    # ---- private --------------------------------------------------------

    def _invoke(self, body: dict[str, Any]) -> str:
        self._get_client()
        try:
            for attempt in Retrying(
                retry=retry_if_exception(_is_transient_model_error),
                stop=stop_after_attempt(_MAX_INVOKE_ATTEMPTS),
                wait=_INVOKE_WAIT,
                reraise=True,
            ):
                with attempt:
                    response = self.client.invoke_model(
                        modelId=self.model_id,
                        body=json.dumps(body),
                        contentType="application/json",
                        accept="application/json",
                    )
        except Exception:
            self._total_errors += 1
            raise
        self._total_invocations += 1

        payload = json.loads(response["body"].read())
        usage = payload.get("usage", {}) or {}
        self._total_input_tokens += int(usage.get("input_tokens", 0))
        self._total_output_tokens += int(usage.get("output_tokens", 0))

        content = payload.get("content", [])
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(t for t in text_parts if t).strip()
