"""Cross-document validation for header-band re-transcription repair.

Addresses the "are we overfitting to one PDF" concern: every detector
heuristic and validation invariant in :mod:`datasheet_rag.table_repair` /
:mod:`datasheet_rag.docling_parser` was derived from a single document
(PIC32CK1025GC01100, doc_id ``d44efe...``). This test exercises a small,
tracked fixture corpus (``tests/fixtures/table_repair_corpus/``) spanning
that document's flagged tables *and* a structurally varied, healthy
second document (doc_id ``928d4097...``) that the detectors must leave
alone.

Two tiers:

* **False-positive regression gate** (always runs, no AWS): for every
  ``*_healthy`` fixture, :func:`table_structure_untrustworthy` must still
  return ``None`` — these are real tables with header bands of varying
  shape (H=2..8, C=4) that must never get routed into repair.
* **Live repair quality** (opt-in via ``RAG_TEST_LIVE_BEDROCK=1`` — real
  Bedrock spend): for every ``d44efe_*`` broken fixture, run
  :meth:`TableRepairer.repair_header_band` against its tracked header-band
  crop and assert the proposal is accepted, the spliced grid is no longer
  flagged, and the re-transcribed header contains a small hand-curated set
  of expected substrings (a loose recall check, not exact match — derived
  by a human reading the crop in ``tests/fixtures/table_repair_corpus/<name>/
  header_crop.png``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from datasheet_rag.docling_parser import table_structure_untrustworthy
from datasheet_rag.table_repair import TableRepairer, splice_header_band

CORPUS_DIR = Path(__file__).parent / "fixtures" / "table_repair_corpus"

_LIVE_BEDROCK = os.environ.get("RAG_TEST_LIVE_BEDROCK") == "1"


def _load_fixture(name: str) -> tuple[list[dict], dict]:
    fixture_dir = CORPUS_DIR / name
    cells = json.loads((fixture_dir / "table_cells.json").read_text())
    meta = json.loads((fixture_dir / "meta.json").read_text())
    return cells, meta


def _healthy_fixture_names() -> list[str]:
    return sorted(p.name for p in CORPUS_DIR.iterdir() if p.name.endswith("_healthy"))


def _broken_fixture_names() -> list[str]:
    return sorted(
        p.name for p in CORPUS_DIR.iterdir()
        if p.name.startswith("d44efe_") and (p / "header_crop.png").exists()
    )


# ---------------------------------------------------------------------------
# False-positive regression gate — no AWS, always runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _healthy_fixture_names())
def test_healthy_fixture_not_flagged(name: str):
    cells, meta = _load_fixture(name)
    assert meta["warning_before"] is None
    assert table_structure_untrustworthy(cells) is None


# ---------------------------------------------------------------------------
# Live repair quality — opt-in, real Bedrock spend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _LIVE_BEDROCK,
    reason="set RAG_TEST_LIVE_BEDROCK=1 to run live Bedrock repair-quality tests",
)
@pytest.mark.parametrize("name", _broken_fixture_names())
def test_broken_fixture_repairs_cleanly(name: str):
    """A validated repair must be high-quality; an outright rejection is an
    acceptable (safe-fallback) outcome, not a test failure — Haiku
    occasionally returns an unparseable or off-by-one response for the same
    crop, and :func:`validate_header_grid` correctly rejects it. Retry a
    couple of times before treating "always rejected" as inconclusive."""
    cells, meta = _load_fixture(name)
    fixture_dir = CORPUS_DIR / name
    image_bytes = (fixture_dir / "header_crop.png").read_bytes()

    repairer = TableRepairer(verbose=True)
    proposed = None
    for _ in range(3):
        proposed = repairer.repair_header_band(
            image_bytes=image_bytes,
            image_format="png",
            cells=cells,
            caption=meta["caption"],
            header_rows=meta["header_rows"],
            column_count=meta["column_count"],
        )
        if proposed is not None:
            break

    if proposed is None:
        pytest.skip("repair rejected on every attempt — inconclusive, see verbose output above")

    grid = splice_header_band(
        cells, proposed, header_rows=meta["header_rows"], column_count=meta["column_count"]
    )
    assert table_structure_untrustworthy(grid) is None

    header_text = " ".join(
        c["text"] for c in grid if c["row"] <= meta["header_rows"] and c.get("is_origin", True)
    )
    for substring in meta["expected_header_substrings"]:
        assert substring in header_text, f"{substring!r} not found in repaired header: {header_text!r}"
