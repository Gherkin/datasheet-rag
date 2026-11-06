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
from datasheet_rag.store.documents import (
    add_revision,
    documents_of,
    format_document_id,
    revisions_of,
)
from datasheet_rag.store.fields import field_shapes
from datasheet_rag.store.metadata import get_metadata, set_metadata
from datasheet_rag.store.projects import project_scope_sql
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
        reports.append(run_eval(conn, eval_set, config, embedder=embedder, trace_path=trace_path))
    return reports


# ---------------------------------------------------------------------------
# Index ablations (re-embedding required)
# ---------------------------------------------------------------------------


def _embedding_dim(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT embedding_dim FROM schema_version WHERE id = 1").fetchone()
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
    rev_id: str | None = None,
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
    if rev_id:
        clauses.append("rev_id = ?")
        params.append(rev_id)
    if project_id:
        clause, clause_params = project_scope_sql(project_id, rev_id_column="rev_id")
        clauses.append(clause)
        params.extend(clause_params)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM chunks{where} ORDER BY id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    rows = src_conn.execute(sql, params).fetchall()
    chunks = [_row_to_chunk(r) for r in rows]

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

    # Keep each document's projects and per-project fields so scoping and
    # subsystem filters behave identically to the source.
    for chunk in chunks:
        insert_chunks(dst, [chunk], vectors=vectors)
    _copy_projects(src_conn, dst, sorted({c.rev_id for c in chunks}))
    return dst


def _copy_projects(src: sqlite3.Connection, dst: sqlite3.Connection, rev_ids: list[str]) -> None:
    """Give the revisions copied into ``dst`` the documents they have in ``src``.

    Revisions of one source document become revisions of one document in
    ``dst``, in the same order, so the newest-revision filter (GH #124)
    hides the same chunks. Each document gets its projects, tags and the
    values of the fields ``dst`` has registered (the built-ins; a fresh
    variant store has no user fields), and each revision its label and
    overrides — so scoping and field filters behave as in the source.
    """
    registered = field_shapes(dst)
    wanted = set(rev_ids)
    for document_id in sorted(set(documents_of(src, rev_ids).values())):
        revs = [d for d in reversed(revisions_of(src, document_id)) if d in wanted]
        if not revs:
            continue
        target = documents_of(dst, revs[:1])[revs[0]]
        for rev in revs[1:]:
            add_revision(dst, target, rev)
        ref = format_document_id(target)
        source = get_metadata(src, format_document_id(document_id))
        assert source is not None
        set_metadata(
            dst,
            ref,
            add_projects=source.projects,
            tags=source.tags,
            fields={k: v for k, v in source.fields.items() if k in registered},
        )
        for project, values in source.project_fields.items():
            kept = {name: value for name, value in values.items() if name in registered}
            if kept:
                set_metadata(dst, ref, fields=kept, project=project)
        for rev in revs:
            own = get_metadata(src, rev)
            assert own is not None
            overrides = {
                k: own.fields[k] for k in own.overrides if k in registered and k in own.fields
            }
            set_metadata(
                dst,
                rev,
                this_revision=True,
                revision=own.revision or "",
                fields=overrides or None,
                tags=own.tags if "tags" in own.overrides else None,
            )


# ---------------------------------------------------------------------------
# MACRO-summarizer ablation (extractive vs abstractive — content, not payload)
# ---------------------------------------------------------------------------
#
# Unlike the index ablations above (same chunk text, different embedding
# input), this one changes the *content* of MACRO chunks: the extractive
# summaries already stored get replaced with freshly-generated abstractive
# (Bedrock Claude) ones. See the README "Switch MACRO summaries from
# extractive to abstractive" TODO — this is the eval that gates the switch.


def _load_doc_graph_from_store(conn: sqlite3.Connection, rev_id: str) -> ChunkGraph:
    """Reconstruct a full :class:`ChunkGraph` (with ``children_ids``) from
    store rows.

    ``_row_to_chunk`` doesn't denormalize ``children_ids`` onto the row
    (see its docstring), so we rebuild it from ``parent_id`` using the same
    index-derived ordering :func:`datasheet_rag.chunking.linker._chunk_sort_key`
    uses for ``prev_id``/``next_id`` — that's the exact order the splitter
    originally appended children in, so grouping by ``parent_id`` over that
    order reproduces the original ``children_ids`` lists.
    """
    rows = conn.execute("SELECT * FROM chunks WHERE rev_id = ?", (rev_id,)).fetchall()
    chunks = [_row_to_chunk(r) for r in rows]
    chunks.sort(key=_chunk_sort_key)

    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        if c.parent_id:
            children_by_parent[c.parent_id].append(c.id)
    for c in chunks:
        c.children_ids = children_by_parent.get(c.id, [])

    graph = ChunkGraph(rev_id=rev_id)
    for c in chunks:
        graph.add(c)
    return graph


def _fetch_vector(conn: sqlite3.Connection, chunk_id: str) -> list[float] | None:
    row = conn.execute(
        "SELECT embedding FROM chunk_vecs WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    if row is None or row["embedding"] is None:
        return None
    vec: list[float] = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
    return vec


def build_macro_summarizer_variant_store(
    src_conn: sqlite3.Connection,
    dst_path: Path | str,
    rev_id: str,
    summarizer: AbstractiveSummarizer,
    embedder: BatchEmbedder,
    *,
    verbose: bool = False,
) -> sqlite3.Connection:
    """Build a variant store with MACRO chunks re-summarized abstractively.

    Loads ``rev_id``'s chunk graph from ``src_conn``, runs
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

    graph = _load_doc_graph_from_store(src_conn, rev_id)
    macro_chunks = graph.by_level(ChunkLevel.MACRO)
    original_context = {c.id: c.context_text for c in macro_chunks}

    if verbose:
        console.print(
            f"[cyan]macro-summarizer[/]: re-summarizing {rev_id} "
            f"({len(macro_chunks)} chapters) with {summarizer.model_id}…"
        )
    graph = summarizer.summarize_graph(graph)

    changed = [
        c for c in graph.by_level(ChunkLevel.MACRO) if c.context_text != original_context.get(c.id)
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
        insert_chunks(dst, [chunk], vectors=vectors)
    _copy_projects(src_conn, dst, [rev_id])
    return dst
