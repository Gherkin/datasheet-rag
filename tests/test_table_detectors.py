"""Tests for table-structure-untrustworthiness detection (docling_parser).

See docs/table-structure-repair/{problem,plan}.md for the investigation this
responds to: Docling/TableFormer reliably mis-parses complex multi-level-header
electronics-datasheet tables in two distinct, increasingly subtle ways —

* failure mode #1 (garbled header): one cell's text stamped across many
  header positions — loud, self-announcing via repetition
  (`_detect_garbled_header`)
* failure mode #3 (fused header/data row): a data row's values tagged as
  header content — looks structurally clean, silently wrong
  (`_detect_fused_header_row`)

Both feed `table_structure_untrustworthy`, which gates whether
`_table_cells_to_compact_text` (asserts a grid) or
`_table_cells_to_reading_order_text` (asserts nothing) is used to render.
"""

from __future__ import annotations

from typing import Any

from datasheet_rag.docling_parser import (
    _detect_fused_header_row,
    _detect_garbled_header,
    _table_cells_to_compact_text,
    _table_cells_to_reading_order_text,
    table_structure_untrustworthy,
)


def _cell(
    row: int,
    col: int,
    text: str,
    *,
    is_header: bool = False,
    is_origin: bool = True,
    row_span: int = 1,
    col_span: int = 1,
) -> dict[str, Any]:
    return {
        "row": row,
        "col": col,
        "row_span": row_span,
        "col_span": col_span,
        "text": text,
        "is_header": is_header,
        "is_origin": is_origin,
    }


# ---------------------------------------------------------------------------
# Control group: a clean, simple table should never be flagged
# ---------------------------------------------------------------------------


def _clean_two_col_table() -> list[dict[str, Any]]:
    """A plain parameter/value table — the kind that already parses fine."""
    return [
        _cell(1, 1, "Parameter", is_header=True),
        _cell(1, 2, "Value", is_header=True),
        _cell(2, 1, "Supply Voltage"),
        _cell(2, 2, "3.3V"),
        _cell(3, 1, "Operating Temperature"),
        _cell(3, 2, "-40C to 85C"),
    ]


def test_clean_table_not_flagged_by_either_detector():
    cells = _clean_two_col_table()
    assert _detect_garbled_header(cells) is None
    assert _detect_fused_header_row(cells) is None
    assert table_structure_untrustworthy(cells) is None


def test_clean_table_renders_with_compact_pipe_structure():
    text = _table_cells_to_compact_text(_clean_two_col_table())
    assert "Parameter | Value" in text
    assert "Supply Voltage | 3.3V" in text


# ---------------------------------------------------------------------------
# Failure mode #1: garbled header (repeated text stamped across columns)
# ---------------------------------------------------------------------------


def _garbled_header_table() -> list[dict[str, Any]]:
    garbled = "Grouped Signal Values 0x0 0x1 0x2 0x3 0x5 0x6"
    return [_cell(1, col, garbled, is_header=True, col_span=15) for col in range(1, 4)] + [
        _cell(2, 1, "30"),
        _cell(2, 2, "RTC"),
        _cell(2, 3, "PZ01"),
    ]


def test_garbled_header_detected_by_repetition():
    cells = _garbled_header_table()
    garbled = _detect_garbled_header(cells)
    assert garbled is not None
    assert garbled.startswith("Grouped Signal Values")
    # Fused-row check should not even need to run for this — garbled wins first.
    assert table_structure_untrustworthy(cells) is not None
    assert "garbled" in table_structure_untrustworthy(cells)


def test_short_repeated_header_tokens_do_not_trigger_garbled_detection():
    """Short tokens ('Pin', '0x0') legitimately repeat across columns."""
    cells = [
        _cell(1, 1, "Pin", is_header=True),
        _cell(1, 2, "Pin", is_header=True),
        _cell(1, 3, "Pin", is_header=True),
        _cell(2, 1, "30"),
        _cell(2, 2, "31"),
        _cell(2, 3, "32"),
    ]
    assert _detect_garbled_header(cells) is None


# ---------------------------------------------------------------------------
# Failure mode #3: fused header/data row (data leaked into header band)
# ---------------------------------------------------------------------------


def _fused_header_table() -> list[dict[str, Any]]:
    """Mirrors the reference fusion case: TableFormer fused pin
    30's data ("30", "RTC", "PZ01", "IFACE1_SIG3") into the header band
    alongside genuine column labels ("Port", "Iface"), while those same
    values also exist as proper data elsewhere in the table.
    """
    return [
        # row 1: genuine header labels
        _cell(1, 1, "Pin", is_header=True),
        _cell(1, 2, "Port", is_header=True),
        _cell(1, 3, "Iface", is_header=True),
        # row 2: TableFormer fused pin 30's data into the header band
        _cell(2, 1, "30", is_header=True),
        _cell(2, 2, "PZ01", is_header=True),
        _cell(2, 3, "IFACE1_SIG3", is_header=True),
        # rows 3+: real data rows — including pin 30's actual values,
        # which is what makes the header-band copies fused, not novel
        _cell(3, 1, "30"),
        _cell(3, 2, "PZ01"),
        _cell(3, 3, "IFACE1_SIG3"),
        _cell(4, 1, "31"),
        _cell(4, 2, "PZ02"),
        _cell(4, 3, "IFACE1_SIG2"),
    ]


def test_fused_header_row_detected_by_text_overlap():
    cells = _fused_header_table()
    # No repetition across many header cells — garbled detector stays quiet.
    assert _detect_garbled_header(cells) is None

    fused = _detect_fused_header_row(cells)
    assert fused is not None
    # The overlapping text is exactly the leaked data values, not the labels.
    for leaked in ("30", "pz01", "iface1_sig3"):
        assert leaked in fused.lower()
    assert "port" not in fused.lower()
    assert "iface" not in fused.lower() or "iface1_sig3" in fused.lower()

    reason = table_structure_untrustworthy(cells)
    assert reason is not None
    assert "fused" in reason


def test_legitimate_value_shaped_header_not_flagged_as_fusion():
    """A genuine hex-code header row (0x0..0xd) is entirely value-shaped —
    shape-based heuristics would false-positive on it. Self-consistency
    (does this text *also* appear as data?) correctly leaves it alone.
    """
    cells = [
        _cell(1, col, text, is_header=True)
        for col, text in enumerate(["0x0", "0x1", "0x2", "0x3"], start=1)
    ] + [
        _cell(2, 1, "EIC"),
        _cell(2, 2, "ADCN"),
        _cell(2, 3, "Iface"),
        _cell(2, 4, "TCC"),
    ]
    assert _detect_fused_header_row(cells) is None
    assert table_structure_untrustworthy(cells) is None


def _multi_block_register_table() -> list[dict[str, Any]]:
    """Mirrors a real false-positive pattern found via `rag inspect tables
    --sample` (a real flagged table — flagged "fused (bit)"): a 32-bit
    register rendered as repeated "Bit / Access / Reset" sub-blocks (one per
    8-bit range). Docling tagged "Bit" is_header=True in the first sub-block's
    row label but is_header=False on its repeats in later sub-blocks — while
    "Access"/"Reset"/"R/W" stayed consistently tagged throughout. So "bit" is
    the *only* overlap candidate, alone in its row each time — nothing
    resembling a leaked data row (several distinct values clustered together,
    cf. pin 30's "30"/"PZ01"/"IFACE1_SIG3") ever appears.
    """
    return [
        # Sub-block 1 (bits 31-24): "Bit" tagged as a header row label
        _cell(1, 1, "Bit", is_header=True),
        _cell(1, 2, "31", is_header=True),
        _cell(1, 3, "30", is_header=True),
        _cell(2, 1, "Access", is_header=True),
        _cell(2, 2, "R/W", is_header=True),
        _cell(2, 3, "R/W", is_header=True),
        _cell(3, 1, "Reset", is_header=True),
        _cell(3, 2, "0", is_header=True),
        _cell(3, 3, "0", is_header=True),
        # Sub-block 2 (bits 23-16): "Bit" repeats, now tagged as plain data —
        # the inconsistency that produces the overlap candidate. The other
        # row labels stay consistently is_header=True, so they never overlap.
        _cell(4, 1, "Bit"),
        _cell(4, 2, "23", is_header=True),
        _cell(4, 3, "22", is_header=True),
        _cell(5, 1, "Access", is_header=True),
        _cell(5, 2, "R/W", is_header=True),
        _cell(5, 3, "R/W", is_header=True),
        _cell(6, 1, "Reset", is_header=True),
        _cell(6, 2, "0", is_header=True),
        _cell(6, 3, "0", is_header=True),
    ]


def test_recurring_row_label_in_register_subblocks_not_flagged_as_fusion():
    """A lone recurring structural label ("Bit") that Docling tags
    inconsistently across repeated sub-blocks must not trip the fusion
    detector — it's the only overlap candidate in its row, never part of a
    multi-value cluster, which is what distinguishes it from a genuinely
    leaked data row (pin 30's "30"/"PZ01"/"IFACE1_SIG3" together)."""
    cells = _multi_block_register_table()
    assert _detect_fused_header_row(cells) is None
    assert table_structure_untrustworthy(cells) is None


def test_single_char_overlap_does_not_trigger_fusion():
    """Single-character placeholder tokens ('-', 'x') legitimately recur in
    both header and data bands (units columns, "not applicable" markers).
    """
    cells = [
        _cell(1, 1, "Pin", is_header=True),
        _cell(1, 2, "-", is_header=True),
        _cell(2, 1, "30"),
        _cell(2, 2, "-"),
    ]
    assert _detect_fused_header_row(cells) is None


# ---------------------------------------------------------------------------
# Span / origin handling — must not double-count or misread spanning cells
# ---------------------------------------------------------------------------


def test_spanning_cell_origin_dedup_in_fusion_check():
    """A col-spanning header cell repeats its text at every covered grid
    position (Docling's grid-fill convention) — only the origin copy should
    be considered, or every span would look like N independent repetitions.
    """
    cells = [
        _cell(1, 1, "MODE=1 Group Values", is_header=True, col_span=3, is_origin=True),
        _cell(1, 2, "MODE=1 Group Values", is_header=True, col_span=3, is_origin=False),
        _cell(1, 3, "MODE=1 Group Values", is_header=True, col_span=3, is_origin=False),
        _cell(2, 1, "0x0", is_header=True),
        _cell(2, 2, "0x1", is_header=True),
        _cell(2, 3, "0x2", is_header=True),
        _cell(3, 1, "30"),
        _cell(3, 2, "PZ01"),
        _cell(3, 3, "IFACE1_SIG3"),
    ]
    assert _detect_garbled_header(cells) is None
    assert _detect_fused_header_row(cells) is None


# ---------------------------------------------------------------------------
# Reading-order rendering — the Stage-2 structure-free fallback
# ---------------------------------------------------------------------------


def test_reading_order_text_emits_each_cell_once_in_geometric_order():
    cells = _fused_header_table()
    text = _table_cells_to_reading_order_text(cells)
    lines = text.split("\n")
    # Row-major order, no pipes, no grid asserted.
    assert lines[0] == "Pin"
    assert "|" not in text
    assert lines.index("Port") < lines.index("30")
    # Spans are collapsed: nothing appears more than its true cell count.
    assert lines.count("30") == 2  # once in the fused header band, once as real data


def test_reading_order_text_collapses_consecutive_garbled_repeats():
    """Failure mode #1's span-detection bug stamps the *same* garbled string
    across many independently-"origin" cells — reading-order rendering must
    not carry that repetition straight through (collapsing it is a general
    text-quality property, not special-cased to this table).
    """
    garbled = "Grouped Signal Values 0x0 0x1 0x2 0x3 0x5 0x6"
    cells = [_cell(1, col, garbled, is_header=True, col_span=15) for col in range(1, 16)]
    cells += [_cell(2, 1, "30"), _cell(2, 2, "PZ01")]
    text = _table_cells_to_reading_order_text(cells)
    assert text.count(garbled) == 1
    assert text == f"{garbled}\n30\nPZ01"


def test_reading_order_text_dedups_spanning_cells():
    cells = [
        _cell(1, 1, "Header spanning two cols", is_header=True, col_span=2, is_origin=True),
        _cell(1, 2, "Header spanning two cols", is_header=True, col_span=2, is_origin=False),
        _cell(2, 1, "a"),
        _cell(2, 2, "b"),
    ]
    text = _table_cells_to_reading_order_text(cells)
    assert text.count("Header spanning two cols") == 1
