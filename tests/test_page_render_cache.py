"""Page render cache integrity (GH #40).

An interrupted ingest used to leave half-written PNGs in
``<RAG_HOME>/page_render_cache/<doc_id>/``. The next run trusted them on
existence alone, Pillow blew up decoding the first damaged one, and --force
did not help because it never reached the render cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from datasheet_rag.figures import PageRenderError, render_pdf_pages


@pytest.fixture()
def pdf(tmp_path: Path) -> Path:
    """A minimal two-page PDF to render."""
    import fitz

    doc = fitz.open()
    for text in ("page one", "page two"):
        page = doc.new_page()
        page.insert_text((72, 72), text)
    path = tmp_path / "doc.pdf"
    doc.save(str(path))
    doc.close()
    return path


def _cache_file(cache_dir: Path, page: int, dpi: int = 72) -> Path:
    return cache_dir / f"p{page:04d}_{dpi}dpi.png"


def test_cache_round_trips_and_leaves_no_temp_files(pdf: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "render_cache"

    first = render_pdf_pages(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)
    assert set(first) == {1, 2}
    assert sorted(p.name for p in cache_dir.iterdir()) == [
        "p0001_72dpi.png",
        "p0002_72dpi.png",
    ]

    # Second call is served from disk and returns equivalent images.
    second = render_pdf_pages(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)
    assert second[1].size == first[1].size


def test_truncated_cached_page_is_discarded_and_re_rendered(pdf: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "render_cache"
    render_pdf_pages(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)

    # Simulate the interrupted render: a PNG header with the pixel data cut off.
    damaged = _cache_file(cache_dir, 1)
    intact_size = Image.open(_cache_file(cache_dir, 2)).size
    damaged.write_bytes(damaged.read_bytes()[:200])

    pages = render_pdf_pages(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)

    assert set(pages) == {1, 2}
    assert pages[1].size == intact_size  # re-rendered, not the truncated file
    Image.open(damaged).load()  # the replacement decodes cleanly


def test_force_clears_the_whole_cache(pdf: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "render_cache"
    render_pdf_pages(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)
    stray = cache_dir / "p0009_72dpi.png"
    stray.write_bytes(b"not a png at all")

    # Only page 1 is requested, but --force invalidates the entire cache.
    pages = render_pdf_pages(pdf, dpi=72, pages=[1], cache_dir=cache_dir, refresh_cache=True)

    assert set(pages) == {1}
    assert not stray.exists()
    assert not _cache_file(cache_dir, 2).exists()
    Image.open(_cache_file(cache_dir, 1)).load()


def test_render_failure_names_the_file_and_suggests_force(pdf: Path, tmp_path: Path) -> None:
    with pytest.raises(PageRenderError) as excinfo:
        render_pdf_pages(pdf, dpi=72, pages=[99], cache_dir=tmp_path / "render_cache")

    message = str(excinfo.value)
    assert "page 99" in message
    assert str(pdf) in message
    assert "--force" in message


def test_extract_figures_force_refreshes_the_documents_cache(
    pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--force reaches the render cache, not just the blocks/chunks caches."""
    from datasheet_rag.config import get_settings
    from datasheet_rag.figures import FigureRegion, extract_figures_from_regions

    settings = get_settings()
    monkeypatch.setattr(settings, "rag_home", tmp_path / "home")
    doc_id = "d" * 64
    cache_dir = settings.rag_home / "page_render_cache" / doc_id
    cache_dir.mkdir(parents=True)
    damaged = _cache_file(cache_dir, 1, dpi=100)
    damaged.write_bytes(b"\x89PNG\r\n\x1a\n truncated")
    # A page no region needs: only a cache-wide invalidation removes it.
    stray = _cache_file(cache_dir, 5, dpi=100)
    stray.write_bytes(b"\x89PNG\r\n\x1a\n truncated")

    regions = [FigureRegion(block_id="b1", page=1, left=0.1, top=0.1, width=0.5, height=0.2)]
    manifest = extract_figures_from_regions(
        pdf,
        regions,
        doc_id,
        output_dir=tmp_path / "figs",
        dpi=100,
        force=True,
    )

    assert len(manifest.figures) == 1
    assert not stray.exists()
    Image.open(damaged).load()  # the poisoned page was re-rendered
    Image.open(manifest.figures[0].image_path).load()
