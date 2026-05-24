"""Golden evaluation set: schema + JSONL persistence.

A :class:`GoldenItem` is one labeled question. Each item carries the
question text, a category (which retrieval concept it stresses), and the
ground-truth location — both as explicit ``gold_chunk_ids`` and as
``gold_pages``. Page labels matter because the multi-scale chunker splits
a page into micro/meso/macro chunks; labeling pages lets the metrics give
credit when the harness retrieves a sibling/parent of the exact labeled
chunk (see :mod:`aws_rag.eval.metrics`).

The set is stored as JSONL (one item per line) so it diffs cleanly in git
and is trivially editable by a human reviewer after LLM generation.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# The five categories. Each maps to a retrieval concept it is designed to
# stress — see the plan's concept→hypothesis table.
#   identifier  – exact register / pin / part-number lookups (keyword branch)
#   conceptual  – "how does X work" (vector branch)
#   figure      – answerable via a diagram / curve (figure descriptions)
#   table_spec  – numeric spec lookups (table chunks)
#   synthesis   – spans multiple sections (macro summaries + navigation)
Category = Literal["identifier", "conceptual", "figure", "table_spec", "synthesis"]

CATEGORIES: tuple[Category, ...] = (
    "identifier",
    "conceptual",
    "figure",
    "table_spec",
    "synthesis",
)


class GoldenItem(BaseModel):
    """One labeled evaluation question with its ground-truth location."""

    question: str
    category: Category
    doc_id: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_pages: list[int] = Field(default_factory=list)
    answer_notes: str = ""
    # 'auto' = LLM-generated (needs review), 'human' = reviewed / hand-written.
    source: Literal["auto", "human"] = "auto"


class EvalSet(BaseModel):
    """An ordered collection of :class:`GoldenItem` with JSONL I/O."""

    items: list[GoldenItem] = Field(default_factory=list)

    def __iter__(self) -> Iterator[GoldenItem]:  # type: ignore[override]
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def by_category(self, category: Category) -> list[GoldenItem]:
        return [it for it in self.items if it.category == category]

    @classmethod
    def from_items(cls, items: Iterable[GoldenItem]) -> EvalSet:
        return cls(items=list(items))

    @classmethod
    def load(cls, path: Path | str) -> EvalSet:
        """Read a JSONL file. Blank lines are skipped."""
        p = Path(path)
        items: list[GoldenItem] = []
        with p.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(GoldenItem.model_validate_json(line))
                except Exception as e:  # noqa: BLE001 - want the line number
                    raise ValueError(
                        f"{p}:{line_no}: invalid GoldenItem JSONL line: {e}"
                    ) from e
        return cls(items=items)

    def save(self, path: Path | str) -> None:
        """Write JSONL, creating parent directories as needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for item in self.items:
                fh.write(item.model_dump_json())
                fh.write("\n")

    def append_jsonl(self, path: Path | str) -> None:
        """Append items to an existing JSONL file (used during generation)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            for item in self.items:
                fh.write(item.model_dump_json())
                fh.write("\n")
