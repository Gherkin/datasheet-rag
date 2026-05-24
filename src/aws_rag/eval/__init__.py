"""Retrieval-layer evaluation for the RAG store.

This package answers "which design concepts actually help retrieval?" by
turning each concept into a measurable hypothesis:

* :mod:`dataset`  – the golden Q&A set (``GoldenItem`` / ``EvalSet``), JSONL.
* :mod:`metrics`  – recall@k, MRR, nDCG@k with multi-scale hit matching.
* :mod:`harness`  – run a golden set through one search config; emit traces.
* :mod:`generate` – LLM-assisted golden-set generation from the corpus.
* :mod:`ablation` – toggle one concept at a time and measure the delta.

It is deliberately scoped to the *retrieval* layer (does the right chunk
come back?). The agent layer (run Claude against the MCP tools, judge the
answer + tool-use trajectory) is a separate fast-follow; the JSONL traces
written by :mod:`harness` are its on-ramp.
"""

from __future__ import annotations

from aws_rag.eval.dataset import CATEGORIES, Category, EvalSet, GoldenItem
from aws_rag.eval.harness import RunConfig, RunReport, run_eval
from aws_rag.eval.metrics import CategoryMetrics, QueryOutcome, is_hit

__all__ = [
    "CATEGORIES",
    "Category",
    "CategoryMetrics",
    "EvalSet",
    "GoldenItem",
    "QueryOutcome",
    "RunConfig",
    "RunReport",
    "is_hit",
    "run_eval",
]
