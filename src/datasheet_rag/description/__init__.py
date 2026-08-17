"""Bedrock Claude vision wrapper for figure description generation.

The :class:`FigureDescriber` reads a cropped figure image plus its
caption + surrounding chunk text and asks a vision-capable Claude model
on Bedrock for a short, retrieval-tuned description that is then folded
into the chunk's ``context_text`` and persisted on the chunk row.

See :mod:`datasheet_rag.description.describer` for the implementation and
cost / prompt notes.
"""

from __future__ import annotations

from datasheet_rag.description.describer import (
    FigureDescriber,
    describe_figures_in_store,
)

__all__ = ["FigureDescriber", "describe_figures_in_store"]
