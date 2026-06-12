"""LLM-assisted golden-set generation from the corpus.

For each evaluation category we sample chunks that are *likely* to support
a question of that kind (figures for ``figure``, tables for ``table_spec``,
micro text/key-value chunks for ``identifier``, etc.), then ask a Bedrock
Claude model to write a realistic question answerable from that chunk plus
a short answer note. The source chunk seeds the ground truth
(``gold_chunk_ids`` + ``gold_pages``).

Everything is marked ``source="auto"``. A human reviews the emitted JSONL
— fixing categories, dropping unanswerable questions, broadening gold
labels — before it is trusted as ground truth.
"""

from __future__ import annotations

import json
import random
import sqlite3
from typing import Any

from rich.console import Console

from aws_rag.config import get_settings
from aws_rag.eval.dataset import CATEGORIES, Category, EvalSet, GoldenItem
from aws_rag.models.chunk import Chunk
from aws_rag.store.sqlite import _row_to_chunk

console = Console()

# Per-category SQL predicate selecting chunks that can plausibly support a
# question of that category. level: 0=MACRO, 1=MESO, 2=MICRO.
_CATEGORY_SQL: dict[Category, str] = {
    "identifier": "level = 2 AND layout_type IN ('text', 'key_value', 'list')",
    "conceptual": "level = 1 AND layout_type = 'text'",
    "figure": (
        "layout_type = 'figure' AND "
        "(figure_caption IS NOT NULL OR figure_description IS NOT NULL)"
    ),
    "table_spec": "layout_type = 'table'",
    "synthesis": "level = 0",
}

_CATEGORY_GUIDANCE: dict[Category, str] = {
    "identifier": (
        "Write a question that looks up a specific named identifier — a "
        "register name, bit field, pin name, signal name, or part number "
        "that appears verbatim in the content."
    ),
    "conceptual": (
        "Write a 'how does it work' / 'what is the purpose of' style "
        "question that requires understanding the explanation in the "
        "content, not just matching a keyword."
    ),
    "figure": (
        "Write a question that can only be answered by looking at the "
        "figure described — e.g. about a waveform, timing relationship, "
        "block-diagram connection, or a curve's shape. Refer to the figure "
        "by what it depicts (the signal/curve/block), never by its number."
    ),
    "table_spec": (
        "Write a question that looks up a specific numeric specification "
        "(a value, range, min/typ/max, or unit) found in the table."
    ),
    "synthesis": (
        "Write a broader question whose answer draws on the whole section "
        "summarised here, not a single sentence."
    ),
}

_SYSTEM_PROMPT = (
    "You generate evaluation questions for a retrieval system over "
    "electronics datasheets and reference manuals. Given one chunk of "
    "source content, you write a single realistic question a hardware "
    "engineer would ask that is answerable FROM THAT CHUNK, plus a short "
    "answer note.\n\n"
    "Write the question the way an engineer would actually type it into a "
    "search box: concise and keyword-forward, ideally under ~15 words. Do "
    "NOT write a verbose full-sentence essay.\n\n"
    "The question must stand on its own and test whether the system can FIND "
    "the content from the subject matter alone. Therefore:\n"
    "- NEVER reference a page number, figure number, table number, section "
    "title, or where the content sits in the document. The retriever's job "
    "is to locate it; telling it the location defeats the test.\n"
    "- Do NOT embed the answer (specific values, ranges) in the question.\n"
    "- Refer to figures/tables by WHAT THEY SHOW (the signal, curve, block, "
    "or quantity), not by their number.\n\n"
    "Never invent facts not present in the content. Respond with strict JSON "
    'only: {"question": "...", "answer_notes": "...", "answerable": true}. '
    "Set answerable=false if the content cannot support a good question of "
    "the requested kind."
)


def _chunk_context(chunk: Chunk) -> str:
    """Render the chunk's content for the prompt (text + figure fields)."""
    # NB: deliberately omit page numbers and section/chapter titles. Feeding
    # them to the model leaks the answer's location into the question text
    # ("on page 14", "in the Input Slew Rate section") — a real user does not
    # know where the answer lives. gold_pages is labeled from chunk metadata
    # downstream, so the model never needs to see the location to ground truth.
    parts: list[str] = []
    md = chunk.metadata
    parts.append(f"Content type: {md.layout_type.value}")
    if chunk.text:
        parts.append(f"Text:\n{chunk.text}")
    if chunk.figure_caption:
        parts.append(f"Figure caption: {chunk.figure_caption}")
    if chunk.figure_description:
        parts.append(f"Figure description: {chunk.figure_description}")
    return "\n".join(parts)


def _sample_chunks(
    conn: sqlite3.Connection,
    category: Category,
    *,
    n: int,
    doc_id: str | None,
    project_id: str | None,
    rng: random.Random,
) -> list[Chunk]:
    predicate = _CATEGORY_SQL[category]
    clauses = [predicate]
    params: list[object] = []
    if doc_id:
        clauses.append("doc_id = ?")
        params.append(doc_id)
    if project_id:
        clauses.append("project_id = ?")
        params.append(project_id)
    # Over-fetch a pool then sample in Python for a stable seed.
    sql = (
        f"SELECT * FROM chunks WHERE {' AND '.join(clauses)} "
        f"AND length(text) > 40 ORDER BY id LIMIT 500"
    )
    rows = conn.execute(sql, params).fetchall()
    chunks = [_row_to_chunk(r) for r in rows]
    rng.shuffle(chunks)
    return chunks[:n]


def _invoke_claude(client: Any, model_id: str, user_text: str, max_tokens: int) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
    }
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(response["body"].read())
    text_parts = [
        b.get("text", "")
        for b in payload.get("content", [])
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    return "\n".join(t for t in text_parts if t).strip()


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Extract the JSON object from the model response, tolerating fences."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def generate_golden_set(
    conn: sqlite3.Connection,
    *,
    per_category: int = 4,
    doc_id: str | None = None,
    project_id: str | None = None,
    model_id: str | None = None,
    max_tokens: int = 400,
    seed: int = 0,
    client: Any | None = None,
    verbose: bool = False,
) -> EvalSet:
    """Generate a reviewable golden set, ~``per_category`` items per category."""
    settings = get_settings()
    model_id = model_id or settings.description_model_id
    rng = random.Random(seed)

    if client is None:
        from aws_rag.local_models import get_chat_client

        client = get_chat_client(kind="text")

    items: list[GoldenItem] = []
    for category in CATEGORIES:
        sampled = _sample_chunks(
            conn,
            category,
            n=per_category,
            doc_id=doc_id,
            project_id=project_id,
            rng=rng,
        )
        if verbose:
            console.print(
                f"[cyan]{category}[/]: sampled {len(sampled)} candidate chunks"
            )
        for chunk in sampled:
            user_text = (
                f"{_CATEGORY_GUIDANCE[category]}\n\n"
                f"--- SOURCE CONTENT ---\n{_chunk_context(chunk)}"
            )
            try:
                raw = _invoke_claude(client, model_id, user_text, max_tokens)
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]generate failed[/] for {chunk.id}: {e}")
                continue
            parsed = _parse_response(raw)
            if not parsed or not parsed.get("answerable", False):
                if verbose:
                    console.print(f"[yellow]skip[/] {chunk.id}: not answerable")
                continue
            question = str(parsed.get("question", "")).strip()
            if not question:
                continue
            items.append(
                GoldenItem(
                    question=question,
                    category=category,
                    doc_id=chunk.doc_id,
                    gold_chunk_ids=[chunk.id],
                    gold_pages=list(chunk.metadata.page_numbers),
                    answer_notes=str(parsed.get("answer_notes", "")).strip(),
                    source="auto",
                )
            )

    return EvalSet(items=items)
