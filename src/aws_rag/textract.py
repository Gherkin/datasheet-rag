"""AWS Textract integration — layout-aware OCR for datasheets."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from aws_rag.aws import s3_client, textract_client
from aws_rag.config import get_settings

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

    params: dict[str, Any] = {
        "DocumentLocation": {
            "S3Object": {
                "Bucket": settings.s3_bucket,
                "Name": s3_key,
            }
        },
        "FeatureTypes": settings.textract_features,
        "OutputConfig": {
            "S3Bucket": settings.s3_bucket,
            "S3Prefix": f"{settings.s3_textract_prefix}{doc_id}/",
        },
    }

    console.print(f"[blue]Starting async analysis[/] for {s3_key} …")
    response = client.start_document_analysis(**params)
    job_id: str = response["JobId"]
    console.print(f"[green]Job started[/] — JobId={job_id}")
    return job_id


def wait_for_job(job_id: str, poll_interval: int = 5, timeout: int = 600) -> str:
    """Poll until the Textract job completes. Returns final status."""
    client = textract_client()
    elapsed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Waiting for Textract job {job_id[:8]}…", total=None)

        while elapsed < timeout:
            resp = client.get_document_analysis(JobId=job_id, MaxResults=1)
            status: str = resp["JobStatus"]

            if status in ("SUCCEEDED", "FAILED", "PARTIAL_SUCCESS"):
                progress.update(task, description=f"Job {job_id[:8]}… {status}")
                return status

            time.sleep(poll_interval)
            elapsed += poll_interval

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


# ---------------------------------------------------------------------------
# Layout parsing helpers
# ---------------------------------------------------------------------------

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


def build_text_from_layout(blocks: list[dict[str, Any]]) -> str:
    """Reconstruct document text preserving layout ordering.

    Uses LAYOUT_* blocks for ordering when available, falls back to
    LINE blocks sorted by geometry.
    """
    id_map = {b["Id"]: b for b in blocks if "Id" in b}

    # Prefer layout blocks for ordering
    layout_blocks = [
        b for b in blocks
        if b.get("BlockType", "").startswith("LAYOUT_")
    ]

    if layout_blocks:
        # Sort by page, then top, then left
        layout_blocks.sort(key=lambda b: (
            b.get("Page", 0),
            b.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0),
            b.get("Geometry", {}).get("BoundingBox", {}).get("Left", 0),
        ))
        parts: list[str] = []
        for lb in layout_blocks:
            text = _collect_text(lb, id_map)
            if text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)

    # Fallback: LINE blocks sorted by geometry
    lines = [b for b in blocks if b.get("BlockType") == "LINE"]
    lines.sort(key=lambda b: (
        b.get("Page", 0),
        b.get("Geometry", {}).get("BoundingBox", {}).get("Top", 0),
        b.get("Geometry", {}).get("BoundingBox", {}).get("Left", 0),
    ))
    return "\n".join(b.get("Text", "") for b in lines)


def _collect_text(block: dict[str, Any], id_map: dict[str, dict[str, Any]]) -> str:
    """Recursively collect text from a block and its children."""
    if "Text" in block:
        return block["Text"]

    child_ids = [
        rel["Ids"]
        for rel in block.get("Relationships", [])
        if rel["Type"] == "CHILD"
    ]
    flat_ids = [cid for ids in child_ids for cid in ids]

    texts: list[str] = []
    for cid in flat_ids:
        child = id_map.get(cid)
        if child:
            texts.append(_collect_text(child, id_map))
    return " ".join(texts)
