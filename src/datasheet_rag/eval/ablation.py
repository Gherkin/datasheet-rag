"""Ablation runners: toggle one concept at a time, measure the delta.

Two flavours:

* **Query-time ablations** are free — they re-query the *same* store with
  different :class:`RunConfig`s (search mode, level filter, RRF weights).
  ``default_matrix`` lays out the standard set; ``run_matrix`` executes it.

* **Index ablations** require rebuilding the vector index, because they
  change *what text gets embedded*. ``build_variant_store`` copies every
  chunk verbatim (so keyword/FTS behaviour is identical) but recomputes
  the embedding vector from a different payload — raw ``text`` instead of
  the enriched ``context_text``, or ``context_text`` with the figure
  description removed. Running the harness against the variant store and
  diffing against the baseline isolates that one embedding concept. These
  incur Bedrock cost, so callers should gate them behind an explicit flag.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from rich.console import Console

from datasheet_rag.chunking.linker import _chunk_sort_key
from datasheet_rag.chunking.summarizer import AbstractiveSummarizer
from datasheet_rag.eval.dataset import EvalSet
from datasheet_rag.eval.harness import Embedder, RunConfig, RunReport, run_eval
from datasheet_rag.models.chunk import Chunk, ChunkGraph, ChunkLevel, LayoutType
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import _row_to_chunk, insert_chunks

console = Console()

IndexVariant = Literal["raw_text", "no_figure_desc"]


class BatchEmbedder(Protocol):
    """Embedding interface the variant-store builder needs."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def default_matrix(base_k: int = 5) -> list[RunConfig]:
    """The standard query-time ablation set.

    Baseline is hybrid; the sweep covers the three search modes, the three
    chunk levels, and two RRF weight tilts. The hit-rate@k curve in every
    report covers the k sweep for free.
    """
    return [
        RunConfig(mode="hybrid", k=base_k, label="baseline hybrid"),
        RunConfig(mode="vector", k=base_k, label="vector only"),
        RunConfig(mode="keyword", k=base_k, label="keyword only"),
        RunConfig(mode="hybrid", k=base_k, level="macro", label="hybrid @macro"),
        RunConfig(mode="hybrid", k=base_k, level="meso", label="hybrid @meso"),
        RunConfig(mode="hybrid", k=base_k, level="micro", label="hybrid @micro"),
        RunConfig(
            mode="hybrid",
            k=base_k,
            vector_weight=2.0,
            keyword_weight=1.0,
            label="hybrid vec-tilt",
        ),
        RunConfig(
            mode="hybrid",
            k=base_k,
            vector_weight=1.0,
            keyword_weight=2.0,
            label="hybrid kw-tilt",
        ),
    ]


def run_matrix(
    conn: sqlite3.Connection,
    eval_set: EvalSet,
    configs: list[RunConfig],
    *,
    embedder: Embedder | None = None,
    trace_path: Path | str | None = None,
) -> list[RunReport]:
    """Run every config over the same store; return reports in order."""
    reports: list[RunReport] = []
    for config in configs:
        reports.append(
            run_eval(
                conn, eval_set, config, embedder=embedder, trace_path=trace_path
            )
        )
    return reports


# ---------------------------------------------------------------------------
# Index ablations (re-embedding required)
# ---------------------------------------------------------------------------


def _embedding_dim(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT embedding_dim FROM schema_version WHERE id = 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("source store has no schema_version row")
    return int(row["embedding_dim"])


def _variant_payload(chunk: Chunk, variant: IndexVariant) -> str:
    """The text to embed for this chunk under the given variant."""
    if variant == "raw_text":
        return chunk.text or chunk.context_text

    # no_figure_desc: keep context_text, but for figures drop the
    # description line we fold in during enrichment.
    if chunk.metadata.layout_type == LayoutType.FIGURE and chunk.figure_description:
        desc = chunk.figure_description
        kept = [
            ln
            for ln in chunk.context_text.splitlines()
            if desc not in ln and not ln.startswith("Description:")
        ]
        payload = "\n".join(kept).strip()
        return payload or chunk.figure_caption or chunk.text
    return chunk.context_text or chunk.text


def build_variant_store(
    src_conn: sqlite3.Connection,
    dst_path: Path | str,
    variant: IndexVariant,
    embedder: BatchEmbedder,
    *,
    doc_id: str | None = None,
    project_id: str | None = None,
    limit: int | None = None,
    verbose: bool = False,
) -> sqlite3.Connection:
    """Build a variant store: same chunks, vectors re-embedded from a
    different payload. Returns the open connection to the new store.

    Only the embedding vector differs from the source — chunk text /
    ``context_text`` columns (and therefore FTS/BM25 behaviour) are copied
    verbatim, isolating the embedding-input concept.
    """
    dim = _embedding_dim(src_conn)
    dst = connect(dst_path, embedding_dim=dim)

    clauses: list[str] = []
    params: list[object] = []
    if doc_id:
        clauses.append("doc_id = ?")
        params.append(doc_id)
    if project_id:
        clauses.append("project_id = ?")
        params.append(project_id)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM chunks{where} ORDER BY id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    rows = src_conn.execute(sql, params).fetchall()
    chunks = [_row_to_chunk(r) for r in rows]
    # Preserve project_id/group_name from the source row for parity.
    project_by_id = {r["id"]: r["project_id"] for r in rows}
    group_by_id = {r["id"]: r["group_name"] for r in rows}

    payloads: list[str] = []
    embeddable: list[Chunk] = []
    for chunk in chunks:
        payload = _variant_payload(chunk, variant)
        if not payload:
            if verbose:
                console.print(f"[yellow]skip[/] {chunk.id}: empty payload")
            continue
        payloads.append(payload)
        embeddable.append(chunk)

    if verbose:
        console.print(
            f"[cyan]variant={variant}[/]: embedding {len(embeddable)} chunks "
            f"(dim={dim}) into {dst_path}"
        )

    vectors_list = embedder.embed_texts(payloads) if payloads else []
    vectors = {c.id: v for c, v in zip(embeddable, vectors_list, strict=True)}

    # Insert chunks grouped by their original project/group so scoping
    # filters behave identically to the source.
    for chunk in chunks:
        insert_chunks(
            dst,
            [chunk],
            vectors=vectors,
            project_id=project_by_id.get(chunk.id),
            group_name=group_by_id.get(chunk.id),
        )
    return dst


# ---------------------------------------------------------------------------
# MACRO-summarizer ablation (extractive vs abstractive — content, not payload)
# ---------------------------------------------------------------------------
#
# Unlike the index ablations above (same chunk text, different embedding
# input), this one changes the *content* of MACRO chunks: the extractive
# summaries already stored get replaced with freshly-generated abstractive
# (Bedrock Claude) ones. See the README "Switch MACRO summaries from
# extractive to abstractive" TODO — this is the eval that gates the switch.


def _load_doc_graph_from_store(conn: sqlite3.Connection, doc_id: str) -> ChunkGraph:
    """Reconstruct a full :class:`ChunkGraph` (with ``children_ids``) from
    store rows.

    ``_row_to_chunk`` doesn't denormalize ``children_ids`` onto the row
    (see its docstring), so we rebuild it from ``parent_id`` using the same
    index-derived ordering :func:`datasheet_rag.chunking.linker._chunk_sort_key`
    uses for ``prev_id``/``next_id`` — that's the exact order the splitter
    originally appended children in, so grouping by ``parent_id`` over that
    order reproduces the original ``children_ids`` lists.
    """
    rows = conn.execute(
        "SELECT * FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchall()
    chunks = [_row_to_chunk(r) for r in rows]
    chunks.sort(key=_chunk_sort_key)

    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        if c.parent_id:
            children_by_parent[c.parent_id].append(c.id)
    for c in chunks:
        c.children_ids = children_by_parent.get(c.id, [])

    graph = ChunkGraph(doc_id=doc_id)
    for c in chunks:
        graph.add(c)
    return graph


def _fetch_vector(conn: sqlite3.Connection, chunk_id: str) -> list[float] | None:
    row = conn.execute(
        "SELECT embedding FROM chunk_vecs WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    if row is None or row["embedding"] is None:
        return None
    return np.frombuffer(row["embedding"], dtype=np.float32).tolist()


def build_macro_summarizer_variant_store(
    src_conn: sqlite3.Connection,
    dst_path: Path | str,
    doc_id: str,
    summarizer: AbstractiveSummarizer,
    embedder: BatchEmbedder,
    *,
    verbose: bool = False,
) -> sqlite3.Connection:
    """Build a variant store with MACRO chunks re-summarized abstractively.

    Loads ``doc_id``'s chunk graph from ``src_conn``, runs
    ``summarizer.summarize_graph`` over it (only MACRO ``text``/
    ``context_text``/``token_count`` change — MESO/MICRO chunks are
    untouched), re-embeds just the chapters whose ``context_text`` changed,
    and copies every other chunk's vector verbatim from the source store.

    This isolates the MACRO-summarizer concept the way
    :func:`build_variant_store` isolates an embedding-payload concept —
    the difference is that here the underlying chunk *content* changes, not
    just what gets embedded, so a generic payload-swap wouldn't do.

    Cost/latency for the run is left on ``summarizer.stats`` for the caller
    to report (see :class:`datasheet_rag.chunking.summarizer.SummarizerStats`).
    """
    dim = _embedding_dim(src_conn)
    dst = connect(dst_path, embedding_dim=dim)

    rows = src_conn.execute(
        "SELECT id, project_id, group_name FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchall()
    project_by_id = {r["id"]: r["project_id"] for r in rows}
    group_by_id = {r["id"]: r["group_name"] for r in rows}

    graph = _load_doc_graph_from_store(src_conn, doc_id)
    macro_chunks = graph.by_level(ChunkLevel.MACRO)
    original_context = {c.id: c.context_text for c in macro_chunks}

    if verbose:
        console.print(
            f"[cyan]macro-summarizer[/]: re-summarizing {doc_id} "
            f"({len(macro_chunks)} chapters) with {summarizer.model_id}…"
        )
    graph = summarizer.summarize_graph(graph)

    changed = [
        c for c in graph.by_level(ChunkLevel.MACRO)
        if c.context_text != original_context.get(c.id)
    ]
    payloads = [c.context_text or c.text for c in changed]
    vectors: dict[str, list[float]] = {}
    if payloads:
        new_vecs = embedder.embed_texts(payloads)
        vectors.update({c.id: v for c, v in zip(changed, new_vecs, strict=True)})

    if verbose:
        stats = summarizer.stats
        console.print(
            f"[cyan]macro-summarizer[/]: {len(changed)}/{len(macro_chunks)} chapters "
            f"changed · {stats.calls} Bedrock calls "
            f"({stats.avg_calls_per_chapter:.1f}/chapter, "
            f"{stats.avg_latency_ms_per_chapter:.0f} ms/chapter)"
        )

    # Every other chunk (and any MACRO chunk that didn't change — e.g. one
    # with no MESO children) carries its vector over verbatim.
    for chunk_id in graph.chunks:
        if chunk_id in vectors:
            continue
        vec = _fetch_vector(src_conn, chunk_id)
        if vec is not None:
            vectors[chunk_id] = vec

    for chunk in graph.chunks.values():
        insert_chunks(
            dst,
            [chunk],
            vectors=vectors,
            project_id=project_by_id.get(chunk.id),
            group_name=group_by_id.get(chunk.id),
        )
    return dst
