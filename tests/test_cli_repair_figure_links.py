"""`rag repair figure-links` — reattach crops to figure chunks (GH #41).

A document ingested with `--skip-figures`, or restored from a store written
before figure links existed, leaves figure chunks that search can find and
`show_figure` cannot serve. When the crops are still on disk with their
manifest, they can be paired back up per page in reading order.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from datasheet_rag.cli import cli
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import get_chunk, insert_chunks

DOC = "f" * 64


def _figure_chunk(index: int, page: int, text: str = "[Figure]") -> Chunk:
    return Chunk(
        id=f"{DOC}:L2:{index}",
        doc_id=DOC,
        level=ChunkLevel.MICRO,
        text=text,
        context_text=text,
        token_count=2,
        metadata=ChunkMetadata(
            doc_id=DOC, doc_title="Widget Manual", page_numbers=[page],
            layout_type=LayoutType.FIGURE,
        ),
    )


@pytest.fixture()
def figures_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A doc's crops on disk, with the manifest the cropper wrote."""
    root = tmp_path / "figures"
    doc_dir = root / DOC
    doc_dir.mkdir(parents=True)
    monkeypatch.setattr(get_settings(), "figures_dir", root)

    entries = []
    for i, page in enumerate([3, 3, 5]):
        name = f"p{page:03d}_fig{i:03d}.png"
        (doc_dir / name).write_bytes(b"\x89PNGFAKE" + str(i).encode())
        entries.append({
            "block_id": f"docling_figure_{i + 1}",
            "page": page,
            "caption": f"Figure {i + 1}: view",
            # As written on the machine that cropped them — a path that does
            # not exist here.
            "image_path": f"/elsewhere/{DOC}/{name}",
        })
    entries.append({
        "block_id": "docling_formula_1",
        "page": 5,
        "caption": "",
        "image_path": f"/elsewhere/{DOC}/p005_formula003.png",
    })
    (doc_dir / "manifest.json").write_text(json.dumps({"figures": entries}))
    return root


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)
    insert_chunks(
        conn,
        [_figure_chunk(0, 3), _figure_chunk(1, 3), _figure_chunk(2, 5)],
        project_id="proj-a",
    )
    conn.commit()
    conn.close()
    return path


def _run(db_path: Path, *args: str):
    return CliRunner().invoke(cli, [*args, "--db", str(db_path)])


def test_dry_run_reports_without_writing(db_path: Path, figures_dir: Path) -> None:
    result = _run(db_path, "repair", "figure-links")
    assert result.exit_code == 0, result.output
    assert "3 figure chunk(s) can be relinked" in result.output
    assert "Dry run" in result.output

    conn = connect(db_path, embedding_dim=get_settings().embedding_dimensions)
    assert get_chunk(conn, f"{DOC}:L2:0").figure_image_path is None
    conn.close()


def test_apply_relinks_crops_and_captions(db_path: Path, figures_dir: Path) -> None:
    result = _run(db_path, "repair", "figure-links", "--apply")
    assert result.exit_code == 0, result.output
    assert "Relinked 3" in result.output

    conn = connect(db_path, embedding_dim=get_settings().embedding_dimensions)
    first = get_chunk(conn, f"{DOC}:L2:0")
    assert first.figure_available is True
    assert first.figure_image_path == f"{DOC}/p003_fig000.png"
    assert first.figure_caption == "Figure 1: view"
    # Page order is preserved: the second crop on page 3, then page 5's.
    assert get_chunk(conn, f"{DOC}:L2:1").figure_image_path.endswith("p003_fig001.png")
    assert get_chunk(conn, f"{DOC}:L2:2").figure_image_path.endswith("p005_fig002.png")
    conn.close()

    # Nothing left to do on a second pass.
    again = _run(db_path, "repair", "figure-links")
    assert "already has an image source" in again.output


def test_a_page_whose_counts_disagree_is_left_alone(
    db_path: Path, figures_dir: Path
) -> None:
    """Better an unrepaired page than a chunk pointing at the wrong picture."""
    conn = connect(db_path, embedding_dim=get_settings().embedding_dimensions)
    insert_chunks(conn, [_figure_chunk(3, 3)], project_id="proj-a")  # 3 chunks, 2 crops
    conn.commit()
    conn.close()

    result = _run(db_path, "repair", "figure-links", "--apply")
    assert result.exit_code == 0, result.output
    assert "page 3: 2 crop(s) vs 3 figure chunk(s)" in result.output

    conn = connect(db_path, embedding_dim=get_settings().embedding_dimensions)
    assert get_chunk(conn, f"{DOC}:L2:0").figure_image_path is None
    # Page 5 still paired off cleanly.
    assert get_chunk(conn, f"{DOC}:L2:2").figure_image_path is not None
    conn.close()


def test_a_document_without_crops_says_so(db_path: Path, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "figures_dir", tmp_path / "empty")
    result = _run(db_path, "repair", "figure-links")
    assert result.exit_code == 0, result.output
    assert "no manifest" in result.output
    assert "re-ingest" in result.output


def test_documents_needing_only_a_rechunk_are_summarised(
    db_path: Path, figures_dir: Path
) -> None:
    """A MESO figure chunk has no crop of its own — relinking cannot help."""
    conn = connect(db_path, embedding_dim=get_settings().embedding_dimensions)
    meso = _figure_chunk(0, 3)
    meso.id = f"{DOC}:L1:0"
    meso.level = ChunkLevel.MESO
    insert_chunks(conn, [meso], project_id="proj-a")
    conn.commit()
    conn.close()

    assert _run(db_path, "repair", "figure-links", "--apply").exit_code == 0

    # Second pass: the MICRO chunks are linked, only the MESO one is left.
    again = _run(db_path, "repair", "figure-links")
    assert again.exit_code == 0, again.output
    assert "coarser (MESO) figure chunks" in again.output
    assert "cannot pair" not in again.output
