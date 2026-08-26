"""AWS Textract integration — layout-aware OCR for datasheets."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from datasheet_rag.aws import textract_client
from datasheet_rag.config import get_settings

console = Console()


# ---------------------------------------------------------------------------
# Synchronous (single-page, ≤ 10 MB) — good for quick testing
# ---------------------------------------------------------------------------


def analyze_document_sync(pdf_path: Path) -> dict[str, Any]:
    """Run synchronous AnalyzeDocument on a local PDF (single-page only).

    Returns the full Textract response dict.
    """
    settings = get_settings()
    client = textract_client()

    with open(pdf_path, "rb") as f:
        doc_bytes = f.read()

    console.print(f"[blue]Analyzing (sync)[/] {pdf_path.name} …")
    response: dict[str, Any] = client.analyze_document(
        Document={"Bytes": doc_bytes},
        FeatureTypes=settings.textract_features,  # type: ignore[arg-type]
    )
    console.print(f"[green]Done[/] — {len(response.get('Blocks', []))} blocks extracted")
    return response


# ---------------------------------------------------------------------------
# Asynchronous (multi-page) — required for real datasheets
# ---------------------------------------------------------------------------


def start_analysis(doc_id: str, s3_key: str) -> str:
    """Start an async Textract analysis job. Returns the job ID."""
    settings = get_settings()
    client = textract_client()

    # OutputConfig is intentionally omitted: it requires Textract to have
    # s3:PutObject on the output prefix, which is hard to configure and not
    # needed — GetDocumentAnalysis pagination returns all blocks regardless.
    params: dict[str, Any] = {
        "DocumentLocation": {
            "S3Object": {
                "Bucket": settings.s3_bucket,
                "Name": s3_key,
            }
        },
        "FeatureTypes": settings.textract_features,
    }

    console.print(f"[blue]Starting async analysis[/] for {s3_key} …")
    response = client.start_document_analysis(**params)
    job_id: str = response["JobId"]
    console.print(f"[green]Job started[/] — JobId={job_id}")
    return job_id


def wait_for_job(job_id: str, poll_interval: int = 5, timeout: int = 900) -> str:
    """Poll until the Textract job completes. Returns final status."""
    client = textract_client()
    deadline = time.monotonic() + timeout
    pages: int | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Waiting for Textract job {job_id[:8]}…", total=None)

        while time.monotonic() < deadline:
            resp = client.get_document_analysis(JobId=job_id, MaxResults=1)
            status: str = resp["JobStatus"]

            # Page count is available from the first response even while running.
            if pages is None:
                pages = resp.get("DocumentMetadata", {}).get("Pages")

            if status in ("SUCCEEDED", "FAILED", "PARTIAL_SUCCESS"):
                progress.update(task, description=f"Job {job_id[:8]}… {status}")
                if status == "FAILED":
                    msg = resp.get("StatusMessage") or "no details"
                    console.print(f"[red]Textract job failed:[/] {msg}")
                return status

            elapsed_s = int(time.monotonic() - (deadline - timeout))
            pages_info = f", {pages}pp" if pages else ""
            progress.update(
                task,
                description=(f"Textract {job_id[:8]}… {status}{pages_info} — {elapsed_s}s elapsed"),
            )
            remaining = deadline - time.monotonic()
            time.sleep(min(poll_interval, max(remaining, 0)))

    raise TimeoutError(f"Textract job {job_id} did not complete within {timeout}s")


def get_job_results(job_id: str) -> list[dict[str, Any]]:
    """Retrieve all pages of results for a completed async job."""
    client = textract_client()
    blocks: list[dict[str, Any]] = []
    next_token: str | None = None

    while True:
        kwargs: dict[str, Any] = {"JobId": job_id}
        if next_token:
            kwargs["NextToken"] = next_token

        resp = client.get_document_analysis(**kwargs)
        blocks.extend(resp.get("Blocks", []))
        next_token = resp.get("NextToken")

        if not next_token:
            break

    console.print(f"[green]Retrieved[/] {len(blocks)} blocks from job {job_id[:8]}…")
    return blocks


def save_blocks(blocks: list[dict[str, Any]], dest: Path) -> Path:
    """Persist Textract blocks to a local JSON file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(blocks, f, indent=2, default=str)
    console.print(f"[green]Saved[/] {len(blocks)} blocks → {dest}")
    return dest


def load_blocks(path: Path) -> list[dict[str, Any]]:
    """Load Textract blocks from a local JSON file saved by :func:`save_blocks`."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Layout parsing helpers
# ---------------------------------------------------------------------------


def layout_reading_order(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return all LAYOUT_* blocks in document reading order.

    Textract's LAYOUT feature already computes a column-aware reading order
    and exposes it as the ordered CHILD list of each PAGE block (it reads a
    full column top-to-bottom before moving to the next column). We trust
    that order. A plain geometric (page, top, left) sort would interleave
    columns on the multi-column datasheets that make up most of the corpus.

    Blocks that Textract did not sequence (no PAGE→CHILD entry) fall back to
    geometric order, placed after the sequenced blocks on their page.
    """
    id_map = {b["Id"]: b for b in blocks if "Id" in b}
    layout_blocks = [b for b in blocks if b.get("BlockType", "").startswith("LAYOUT_")]

    # Per-page rank from each PAGE block's ordered CHILD list (layout only).
    rank: dict[str, int] = {}
    for pb in (b for b in blocks if b.get("BlockType") == "PAGE"):
        position = 0
        for rel in pb.get("Relationships", []):
            if rel.get("Type") != "CHILD":
                continue
            for cid in rel["Ids"]:
                child = id_map.get(cid)
                if child and child.get("BlockType", "").startswith("LAYOUT_"):
                    rank[cid] = position
                    position += 1

    def _key(b: dict[str, Any]) -> tuple[int, int, int, float, float]:
        page = b.get("Page", 0)
        bb = b.get("Geometry", {}).get("BoundingBox", {})
        bid = b.get("Id", "")
        if bid in rank:
            return (page, 0, rank[bid], 0.0, 0.0)
        # Unsequenced: after ranked blocks on the page, ordered geometrically.
        return (page, 1, 0, bb.get("Top", 0.0), bb.get("Left", 0.0))

    return sorted(layout_blocks, key=_key)


def extract_layout_elements(blocks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Organise Textract blocks by layout type.

    Returns a dict keyed by BlockType (PAGE, LAYOUT_TITLE, LAYOUT_HEADER,
    LAYOUT_SECTION_HEADER, LAYOUT_TEXT, LAYOUT_TABLE, LAYOUT_FIGURE,
    TABLE, CELL, KEY_VALUE_SET, etc.).
    """
    by_type: dict[str, list[dict[str, Any]]] = {}
    for block in blocks:
        bt = block.get("BlockType", "UNKNOWN")
        by_type.setdefault(bt, []).append(block)
    return by_type


def _collect_text(block: dict[str, Any], id_map: dict[str, dict[str, Any]]) -> str:
    """Recursively collect text from a block and its children."""
    if "Text" in block:
        return block["Text"]

    child_ids = [rel["Ids"] for rel in block.get("Relationships", []) if rel["Type"] == "CHILD"]
    flat_ids = [cid for ids in child_ids for cid in ids]

    texts: list[str] = []
    for cid in flat_ids:
        child = id_map.get(cid)
        if child:
            texts.append(_collect_text(child, id_map))
    return " ".join(texts)
