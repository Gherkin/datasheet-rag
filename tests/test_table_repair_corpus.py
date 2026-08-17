"""False-positive regression gate for the table-structure detectors.

Every heuristic and validation invariant in :mod:`datasheet_rag.table_repair` /
:mod:`datasheet_rag.docling_parser` was derived from tables Docling
mis-parsed. The risk that creates is the inverse one: detectors tuned on
broken tables firing on *healthy* ones. This test pins that down — for every
fixture in ``tests/fixtures/table_repair_corpus/``,
:func:`table_structure_untrustworthy` must return ``None``, so a real table
with an unusual header band never gets routed into repair.

The fixtures carry header bands of varying shape (H=2, 3, 8 rows; C=4
columns) with the span and empty-cell patterns Docling emits, which is what
the detectors key on structurally.

**The fixture text is synthetic.** The corpus was originally captured from
real manufacturer datasheets; the cell geometry is preserved from those
captures but every text value has been replaced, so nothing here reproduces
third-party document content. Two consequences worth knowing:

* The detectors' *content* signals are only partly exercised. Both key on
  relations between cell texts — repeat counts for the garbled-header check,
  header/data set intersection for the fused-row check — and the substitution
  preserved those relations, but not the realistic strings the thresholds
  were tuned against.
* There is no longer a repair-*quality* test. Checking that a garbled header
  gets re-transcribed correctly needs a real image of a real mis-parsed
  table, which is exactly what cannot be committed here.
  :meth:`TableRepairer.repair_header_band` is still covered by
  ``tests/test_table_repair.py`` against a mocked Bedrock client, which pins
  the parsing, validation and splicing logic — but not the model's
  transcription accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasheet_rag.docling_parser import table_structure_untrustworthy

CORPUS_DIR = Path(__file__).parent / "fixtures" / "table_repair_corpus"


def _load_fixture(name: str) -> tuple[list[dict], dict]:
    fixture_dir = CORPUS_DIR / name
    cells = json.loads((fixture_dir / "table_cells.json").read_text())
    meta = json.loads((fixture_dir / "meta.json").read_text())
    return cells, meta


def _healthy_fixture_names() -> list[str]:
    return sorted(p.name for p in CORPUS_DIR.iterdir() if p.name.endswith("_healthy"))


def test_corpus_is_not_empty():
    """Guard the parametrize below — an empty corpus would pass silently."""
    assert _healthy_fixture_names()


@pytest.mark.parametrize("name", _healthy_fixture_names())
def test_healthy_fixture_not_flagged(name: str):
    cells, meta = _load_fixture(name)
    assert meta["warning_before"] is None
    assert table_structure_untrustworthy(cells) is None
