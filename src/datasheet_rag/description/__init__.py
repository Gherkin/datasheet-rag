"""Bedrock Claude vision wrapper for figure description generation.

The :class:`FigureDescriber` reads a cropped figure image plus its
caption + surrounding chunk text and asks a vision-capable Claude model
on Bedrock for a short, retrieval-tuned description that is then folded
into the chunk's ``context_text`` and persisted on the chunk row.

Where those inputs come from is pluggable (:class:`FigureSource`), so the
same describer serves a local sqlite store, a freshly parsed in-memory
graph, or a store that lives on another host — which is what lets the
vision model run on a GPU client while a GPU-less server holds the corpus
(``RAG_COMPUTE=client``, GH #43).

See :mod:`datasheet_rag.description.describer` for the implementation and
cost / prompt notes.
"""

from __future__ import annotations

from datasheet_rag.description.describer import (
    BackendFigureSource,
    DescriptionPlan,
    FigureDescriber,
    FigureInputs,
    FigureSource,
    GraphFigureSource,
    StoreFigureSource,
    apply_description_to_chunk,
    describe_figures_in_graph,
    describe_figures_in_store,
    describe_figures_via_backend,
    plan_figure_descriptions,
)

__all__ = [
    "BackendFigureSource",
    "DescriptionPlan",
    "FigureDescriber",
    "FigureInputs",
    "FigureSource",
    "GraphFigureSource",
    "StoreFigureSource",
    "apply_description_to_chunk",
    "describe_figures_in_graph",
    "describe_figures_in_store",
    "describe_figures_via_backend",
    "plan_figure_descriptions",
]
