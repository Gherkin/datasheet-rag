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
from pathlib import Path
from typing import Literal, Protocol

from rich.console import Console

from aws_rag.eval.dataset import EvalSet
from aws_rag.eval.harness import Embedder, RunConfig, RunReport, run_eval
from aws_rag.models.chunk import Chunk, LayoutType
from aws_rag.store.schema import connect
from aws_rag.store.sqlite import _row_to_chunk, insert_chunks

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
