"""Tests for TableRepairer (Stage 3 — header-band re-transcription repair).

Bedrock is fully mocked — no network, no AWS spend (mirrors
tests/description/test_describer.py). Tests verify:

* code-fence stripping / response parsing
* validate_header_grid's structural invariants: tiling (no gaps/overlaps,
  in-range), anti-degenerate (proposal still garbled), anti-fusion (proposal
  recreates a leaked data row)
* splice_header_band: header rows replaced wholesale (grid-fill expansion of
  spans), data rows untouched
* TableRepairer.repair_header_band happy/rejected paths against a mocked client
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from datasheet_rag.table_repair import (
    TableRepairer,
    _parse_repair_response,
    splice_header_band,
    validate_header_grid,
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


def _fused_header_cells() -> list[dict[str, Any]]:
    """A 2-row header band over a 2-column table, plus one data row."""
    return [
        _cell(1, 1, "Pin", is_header=True),
        _cell(1, 2, "Port", is_header=True),
        _cell(2, 1, "30", is_header=True),  # fused: should be data, row 3
        _cell(2, 2, "PZ01", is_header=True),
        _cell(3, 1, "30"),
        _cell(3, 2, "PZ01"),
    ]


def _mock_bedrock_response(
    text: str,
    *,
    input_tokens: int = 900,
    output_tokens: int = 120,
) -> Any:
    body = json.dumps(
        {
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    ).encode()
    resp = MagicMock()
    resp.__getitem__.side_effect = lambda k: {"body": MagicMock(read=lambda: body)}[k]
    return resp


_VALID_REPAIR_JSON = json.dumps(
    {
        "cells": [
            {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "Pin"},
            {"row": 1, "col": 2, "row_span": 1, "col_span": 1, "text": "Port"},
        ]
    }
)


@pytest.fixture
def fake_client() -> Any:
    client = MagicMock()
    client.invoke_model.return_value = _mock_bedrock_response(_VALID_REPAIR_JSON)
    return client


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_repair_response_accepts_well_formed_json():
    proposed = _parse_repair_response(_VALID_REPAIR_JSON)
    assert proposed is not None
    assert len(proposed) == 2
    assert proposed[0]["text"] == "Pin"


@pytest.mark.parametrize(
    "wrapped",
    [
        f"```json\n{_VALID_REPAIR_JSON}\n```",
        f"```\n{_VALID_REPAIR_JSON}\n```",
        f"  ```json\n{_VALID_REPAIR_JSON}\n```  ",
    ],
)
def test_parse_repair_response_unwraps_markdown_code_fences(wrapped: str):
    """Despite "Respond with JSON only, no prose", Claude (Haiku especially)
    routinely wraps structured output in a fenced code block — a cosmetic
    habit that shouldn't be treated as a malformed response."""
    proposed = _parse_repair_response(wrapped)
    assert proposed is not None
    assert len(proposed) == 2


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        json.dumps({"wrong_key": []}),
        json.dumps({"cells": "not a list"}),
        json.dumps({"cells": [{"row": 1, "col": 1}]}),  # missing required fields
        json.dumps({"cells": [[1, 1, 1, 1, "Pin"]]}),  # not a dict
    ],
)
def test_parse_repair_response_rejects_malformed_shapes(raw: str):
    assert _parse_repair_response(raw) is None


# ---------------------------------------------------------------------------
# validate_header_grid — threshold-free invariants
# ---------------------------------------------------------------------------


def _data_cells() -> list[dict[str, Any]]:
    return [_cell(2, 1, "30"), _cell(2, 2, "PZ01")]


def test_validate_accepts_exact_tiling():
    proposed = json.loads(_VALID_REPAIR_JSON)["cells"]
    assert (
        validate_header_grid(proposed, header_rows=1, column_count=2, data_cells=_data_cells())
        is None
    )


def test_validate_rejects_gap_in_tiling():
    proposed = [{"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "Pin"}]
    reason = validate_header_grid(proposed, header_rows=1, column_count=2, data_cells=_data_cells())
    assert reason is not None
    assert "not fully covered" in reason


def test_validate_rejects_overlap_in_tiling():
    proposed = [
        {"row": 1, "col": 1, "row_span": 1, "col_span": 2, "text": "Pin"},
        {"row": 1, "col": 2, "row_span": 1, "col_span": 1, "text": "Port"},
    ]
    reason = validate_header_grid(proposed, header_rows=1, column_count=2, data_cells=_data_cells())
    assert reason is not None
    assert "overlaps" in reason


def test_validate_rejects_non_positive_span():
    proposed = [
        {"row": 1, "col": 1, "row_span": 0, "col_span": 1, "text": "Pin"},
        {"row": 1, "col": 2, "row_span": 1, "col_span": 1, "text": "Port"},
    ]
    reason = validate_header_grid(proposed, header_rows=1, column_count=2, data_cells=_data_cells())
    assert reason is not None
    assert "non-positive" in reason


def test_validate_rejects_cell_outside_band():
    proposed = [
        {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "Pin"},
        {"row": 1, "col": 2, "row_span": 1, "col_span": 2, "text": "Port"},
    ]
    reason = validate_header_grid(proposed, header_rows=1, column_count=2, data_cells=_data_cells())
    assert reason is not None
    assert "extends outside" in reason


def test_validate_accepts_legitimate_spans():
    proposed = [
        {"row": 1, "col": 1, "row_span": 1, "col_span": 2, "text": "Header"},
        {"row": 2, "col": 1, "row_span": 1, "col_span": 1, "text": "A"},
        {"row": 2, "col": 2, "row_span": 1, "col_span": 1, "text": "B"},
    ]
    data = [_cell(3, 1, "x"), _cell(3, 2, "y")]
    assert validate_header_grid(proposed, header_rows=2, column_count=2, data_cells=data) is None


def test_validate_rejects_still_garbled_proposal():
    """The model returns a tiled, non-overlapping grid — but it's the same
    long text repeated, i.e. it didn't actually fix anything."""
    garbled = "Grouped Signal Values 0x0 0x1 0x2"
    proposed = [
        {"row": 1, "col": c, "row_span": 1, "col_span": 1, "text": garbled} for c in (1, 2, 3)
    ]
    data = [_cell(2, c, f"v{c}") for c in (1, 2, 3)]
    reason = validate_header_grid(proposed, header_rows=1, column_count=3, data_cells=data)
    assert reason is not None
    assert "still garbled" in reason


def test_validate_rejects_proposal_that_recreates_fused_data_row():
    """The proposed header row duplicates >=2 distinct data-row values —
    same fusion defect, just regenerated by the model."""
    proposed = [
        {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "30"},
        {"row": 1, "col": 2, "row_span": 1, "col_span": 1, "text": "PZ01"},
    ]
    data = [_cell(2, 1, "30"), _cell(2, 2, "PZ01")]
    reason = validate_header_grid(proposed, header_rows=1, column_count=2, data_cells=data)
    assert reason is not None
    assert "duplicates data values" in reason


def test_validate_allows_short_overlap_with_data_row():
    """A single short marker (e.g. "-") legitimately recurring in both header
    and data is not a fusion signal — only >=2 distinct overlaps of
    sufficient length count."""
    proposed = [
        {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "Status"},
        {"row": 1, "col": 2, "row_span": 1, "col_span": 1, "text": "-"},
    ]
    data = [_cell(2, 1, "ok"), _cell(2, 2, "-")]
    assert validate_header_grid(proposed, header_rows=1, column_count=2, data_cells=data) is None


# ---------------------------------------------------------------------------
# splice_header_band
# ---------------------------------------------------------------------------


def test_splice_replaces_header_band_and_preserves_data_rows():
    cells = _fused_header_cells()
    proposed = [
        {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "Pin"},
        {"row": 1, "col": 2, "row_span": 1, "col_span": 1, "text": "Port"},
        {"row": 2, "col": 1, "row_span": 1, "col_span": 1, "text": "Number"},
        {"row": 2, "col": 2, "row_span": 1, "col_span": 1, "text": "Name"},
    ]
    grid = splice_header_band(cells, proposed, header_rows=2, column_count=2)

    header = [c for c in grid if c["row"] <= 2]
    assert len(header) == 4
    assert all(c["is_header"] for c in header)
    assert {c["text"] for c in header} == {"Pin", "Port", "Number", "Name"}

    # Data row (row 3) preserved untouched
    data = [c for c in grid if c["row"] == 3]
    assert len(data) == 2
    assert {c["text"] for c in data} == {"30", "PZ01"}


def test_splice_expands_spans_with_grid_fill_convention():
    cells = [
        _cell(1, 1, "junk", is_header=True, col_span=2),
        _cell(1, 2, "junk", is_header=True, col_span=2, is_origin=False),
        _cell(2, 1, "a"),
        _cell(2, 2, "b"),
    ]
    proposed = [
        {"row": 1, "col": 1, "row_span": 1, "col_span": 2, "text": "Spanning Header"},
    ]
    grid = splice_header_band(cells, proposed, header_rows=1, column_count=2)

    header = [c for c in grid if c["row"] == 1]
    assert len(header) == 2
    origin = next(c for c in header if c["col"] == 1)
    fill = next(c for c in header if c["col"] == 2)
    assert origin["is_origin"] is True
    assert fill["is_origin"] is False
    assert origin["text"] == fill["text"] == "Spanning Header"


def test_splice_round_trips_through_existing_renderers():
    """A repaired grid should render exactly like a trusted Docling grid —
    no special-casing needed downstream."""
    from datasheet_rag.docling_parser import (
        _table_cells_to_compact_text,
        table_structure_untrustworthy,
    )

    cells = _fused_header_cells()
    proposed = [
        {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "Pin"},
        {"row": 1, "col": 2, "row_span": 1, "col_span": 1, "text": "Port"},
        {"row": 2, "col": 1, "row_span": 1, "col_span": 1, "text": "Number"},
        {"row": 2, "col": 2, "row_span": 1, "col_span": 1, "text": "Name"},
    ]
    grid = splice_header_band(cells, proposed, header_rows=2, column_count=2)

    text = _table_cells_to_compact_text(grid)
    assert "Pin | Port" in text
    assert "30 | PZ01" in text
    assert table_structure_untrustworthy(grid) is None


# ---------------------------------------------------------------------------
# TableRepairer.repair_header_band — end-to-end against a mocked client
# ---------------------------------------------------------------------------


def test_repair_header_band_builds_correct_request_and_returns_validated_grid(fake_client: Any):
    repairer = TableRepairer(client=fake_client, max_concurrency=1, model_id="test-model")
    result = repairer.repair_header_band(
        image_bytes=b"fake-png-bytes",
        image_format="png",
        cells=_fused_header_cells(),
        caption="Table 1-1",
        header_rows=1,
        column_count=2,
    )

    assert result is not None
    assert len(result) == 2
    assert {e["text"] for e in result} == {"Pin", "Port"}

    body = json.loads(fake_client.invoke_model.call_args.kwargs["body"])
    assert body["system"].startswith("You repair the header")
    blocks = body["messages"][0]["content"]
    image_block, text_block = blocks
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/png"
    assert base64.b64decode(image_block["source"]["data"]) == b"fake-png-bytes"
    assert "Table 1-1" in text_block["text"]
    assert "H=1" in text_block["text"]
    assert "C=2" in text_block["text"]

    stats = repairer.stats()
    assert stats["total_invocations"] == 1
    assert stats["total_rejected"] == 0
    assert stats["total_input_tokens"] == 900
    assert stats["total_output_tokens"] == 120


def test_repair_header_band_rejects_unparseable_response():
    client = MagicMock()
    client.invoke_model.return_value = _mock_bedrock_response("not json at all")
    repairer = TableRepairer(client=client, max_concurrency=1, model_id="test-model")

    result = repairer.repair_header_band(
        image_bytes=b"x",
        image_format="png",
        cells=_fused_header_cells(),
        header_rows=1,
        column_count=2,
    )
    assert result is None
    assert repairer.stats()["total_rejected"] == 1


def test_repair_header_band_rejects_structurally_inconsistent_response():
    """The model returns well-formed JSON, but it doesn't tile the H×C
    band — repair_header_band must reject it via validate_header_grid."""
    incomplete = json.dumps(
        {
            "cells": [
                {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "Pin"},
            ]
        }
    )
    client = MagicMock()
    client.invoke_model.return_value = _mock_bedrock_response(incomplete)
    repairer = TableRepairer(client=client, max_concurrency=1, model_id="test-model")

    result = repairer.repair_header_band(
        image_bytes=b"x",
        image_format="png",
        cells=_fused_header_cells(),
        header_rows=1,
        column_count=2,
    )
    assert result is None
    assert repairer.stats()["total_rejected"] == 1


def test_repairer_falls_back_to_description_model_id_when_unset(fake_client: Any):
    """table_repair_model_id defaults to None (config.py) — falling back to
    description_model_id keeps the repairer usable out of the box, though the
    config docs note that default is tuned for cheap figure descriptions."""
    from datasheet_rag.config import get_settings

    settings = get_settings()
    assert settings.table_repair_model_id is None  # the documented default

    repairer = TableRepairer(client=fake_client)
    assert repairer.model_id == settings.description_model_id
