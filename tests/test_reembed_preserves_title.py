"""Regression test: a re-embed must not wipe a post-ingest inferred title."""
from __future__ import annotations
from pathlib import Path
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks, set_doc_title

DOC = "c" * 64

def _chunk(title: str) -> Chunk:
    md = ChunkMetadata(doc_id=DOC, doc_title=title, chapter_title="", section_title="",
                       page_numbers=[1], layout_type=LayoutType.TEXT, context_string="")
    return Chunk(id=f"{DOC}:L1:0", doc_id=DOC, level=ChunkLevel.MICRO, text="t",
                 context_text="t", token_count=1, metadata=md)

def test_reembed_keeps_inferred_title(tmp_path: Path) -> None:
    conn = connect(tmp_path / "s.sqlite", embedding_dim=get_settings().embedding_dimensions)
    # Ingest with no title, as Docling leaves a poorly-titled datasheet.
    insert_chunks(conn, [_chunk("")], project_id="p")
    # `rag repair titles` backfills one into the chunks table only.
    set_doc_title(conn, DOC, "Inferred Title")
    # Re-embedding replays the cached graph, which still carries the blank.
    insert_chunks(conn, [_chunk("")], project_id="p")
    got = conn.execute("SELECT doc_title FROM chunks WHERE doc_id = ?", (DOC,)).fetchone()[0]
    assert got == "Inferred Title", f"re-embed wiped the inferred title (got {got!r})"

def test_reembed_still_applies_a_real_new_title(tmp_path: Path) -> None:
    conn = connect(tmp_path / "s.sqlite", embedding_dim=get_settings().embedding_dimensions)
    insert_chunks(conn, [_chunk("Old")], project_id="p")
    insert_chunks(conn, [_chunk("New")], project_id="p")
    got = conn.execute("SELECT doc_title FROM chunks WHERE doc_id = ?", (DOC,)).fetchone()[0]
    assert got == "New"
