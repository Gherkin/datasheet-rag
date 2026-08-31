"""Page render cache integrity (GH #40) and the memory bound on rendering (GH #59).

An interrupted ingest used to leave half-written PNGs in
``<RAG_HOME>/page_render_cache/<doc_id>/``. The next run trusted them on
existence alone, Pillow blew up decoding the first damaged one, and --force
did not help because it never reached the render cache.

The renderer also used to return every page at once, so a 900-page document at
300 DPI held ~23 GB of RGB and the kernel OOM-killed the server. It now streams
pages within a window sized from a memory budget; the tests at the bottom pin
that window down.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from datasheet_rag.figures import PageRenderError, iter_pdf_pages


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


def _render_all(pdf: Path, **kwargs: object) -> dict[int, Image.Image]:
    """Drain the page stream into a dict — fine for the 2-page fixtures here.

    Deliberately not a helper in ``figures``: materialising every page is the
    shape that OOM-killed the server (GH #59), and it should stay confined to
    tests that render a handful of pages.
    """
    return dict(iter_pdf_pages(pdf, **kwargs))  # type: ignore[arg-type]


def _cache_file(cache_dir: Path, page: int, dpi: int = 72) -> Path:
    return cache_dir / f"p{page:04d}_{dpi}dpi.png"


def test_cache_round_trips_and_leaves_no_temp_files(pdf: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "render_cache"

    first = _render_all(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)
    assert set(first) == {1, 2}
    assert sorted(p.name for p in cache_dir.iterdir()) == [
        "p0001_72dpi.png",
        "p0002_72dpi.png",
    ]

    # Second call is served from disk and returns equivalent images.
    second = _render_all(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)
    assert second[1].size == first[1].size


def test_truncated_cached_page_is_discarded_and_re_rendered(pdf: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "render_cache"
    _render_all(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)

    # Simulate the interrupted render: a PNG header with the pixel data cut off.
    damaged = _cache_file(cache_dir, 1)
    intact_size = Image.open(_cache_file(cache_dir, 2)).size
    damaged.write_bytes(damaged.read_bytes()[:200])

    pages = _render_all(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)

    assert set(pages) == {1, 2}
    assert pages[1].size == intact_size  # re-rendered, not the truncated file
    Image.open(damaged).load()  # the replacement decodes cleanly


def test_force_clears_the_whole_cache(pdf: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "render_cache"
    _render_all(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)
    stray = cache_dir / "p0009_72dpi.png"
    stray.write_bytes(b"not a png at all")

    # Only page 1 is requested, but --force invalidates the entire cache.
    pages = _render_all(pdf, dpi=72, pages=[1], cache_dir=cache_dir, refresh_cache=True)

    assert set(pages) == {1}
    assert not stray.exists()
    assert not _cache_file(cache_dir, 2).exists()
    Image.open(_cache_file(cache_dir, 1)).load()


def test_render_failure_names_the_file_and_suggests_force(pdf: Path, tmp_path: Path) -> None:
    with pytest.raises(PageRenderError) as excinfo:
        _render_all(pdf, dpi=72, pages=[99], cache_dir=tmp_path / "render_cache")

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


# ---------------------------------------------------------------------------
# The memory bound (GH #59)
# ---------------------------------------------------------------------------


@pytest.fixture()
def long_pdf(tmp_path: Path) -> Path:
    """A 40-page PDF — enough that eager rendering is visible in the cache dir."""
    import fitz

    doc = fitz.open()
    for n in range(40):
        page = doc.new_page()
        page.insert_text((72, 72), f"page {n + 1}")
    path = tmp_path / "long.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_pages_are_rendered_lazily_not_all_up_front(long_pdf: Path, tmp_path: Path) -> None:
    """Taking one page must not render the other 39.

    This is the OOM: the old renderer dispatched every page before returning
    anything, so peak memory scaled with the document. With streaming, only the
    in-flight window has been rendered by the time the first page arrives.
    """
    cache_dir = tmp_path / "render_cache"

    stream = iter_pdf_pages(long_pdf, dpi=72, pages=list(range(1, 41)), cache_dir=cache_dir)
    page_no, img = next(stream)

    assert page_no == 1
    assert img.size[0] > 0
    # The window is at most 8 workers plus the page just handed over; anything
    # near 40 means the whole document was rendered before the first yield.
    assert len(list(cache_dir.glob("*.png"))) <= 9
    stream.close()


def test_abandoning_the_stream_does_not_keep_rendering(long_pdf: Path, tmp_path: Path) -> None:
    """Closing the generator early shuts the pool down instead of draining it."""
    cache_dir = tmp_path / "render_cache"

    stream = iter_pdf_pages(long_pdf, dpi=72, pages=list(range(1, 41)), cache_dir=cache_dir)
    next(stream)
    stream.close()

    assert len(list(cache_dir.glob("*.png"))) <= 9
    assert not list(cache_dir.glob("*.tmp"))


def test_window_shrinks_as_pages_get_bigger() -> None:
    from datasheet_rag.figures import _render_window_size

    budget = 1024 * 1024 * 1024  # 1 GiB

    # A 300 DPI A4 page (~26 MB of RGB, ~78 MB per slot) fits many times over,
    # so the cap is the ordinary worker ceiling, not the budget.
    assert _render_window_size(78 * 1024 * 1024, budget, 900) >= 1
    assert _render_window_size(78 * 1024 * 1024, budget, 900) <= 8
    # A page that alone eats most of the budget renders on its own…
    assert _render_window_size(600 * 1024 * 1024, budget, 900) == 1
    # …and one bigger than the whole budget still renders, rather than dividing
    # the window down to zero and hanging.
    assert _render_window_size(4 * budget, budget, 900) == 1
    # Never more workers than there are pages to render.
    assert _render_window_size(1024, budget, 3) == 3


def test_slot_size_tracks_dpi_and_page_area(pdf: Path) -> None:
    from datasheet_rag.figures import _page_slot_bytes

    at_72 = _page_slot_bytes(pdf, [1], 72)
    at_300 = _page_slot_bytes(pdf, [1], 300)

    assert at_72 > 0
    # Area scales with the square of the resolution ratio (~17x for 72 → 300).
    assert at_300 > at_72 * 15


def test_extraction_crops_every_page_it_streams(pdf: Path, tmp_path: Path) -> None:
    """Regions spread over several pages survive the page-at-a-time rewrite.

    The manifest must stay in the caller's region order — figure filenames are
    numbered from it — even though cropping now happens grouped by page.
    """
    from datasheet_rag.figures import FigureRegion, extract_figures_from_regions

    regions = [
        FigureRegion(block_id="b2", page=2, left=0.1, top=0.1, width=0.4, height=0.2),
        FigureRegion(block_id="b1", page=1, left=0.1, top=0.4, width=0.4, height=0.2),
        FigureRegion(block_id="b3", page=2, left=0.1, top=0.5, width=0.4, height=0.2),
    ]

    manifest = extract_figures_from_regions(
        pdf,
        regions,
        "e" * 64,
        output_dir=tmp_path / "figs",
        dpi=72,
    )

    assert [f.region.block_id for f in manifest.figures] == ["b2", "b1", "b3"]
    assert [f.image_path.name for f in manifest.figures if f.image_path] == [
        "p002_fig000.png",
        "p001_fig001.png",
        "p002_fig002.png",
    ]
    for fig in manifest.figures:
        assert fig.image_path is not None
        Image.open(fig.image_path).load()
        assert fig.width_px > 0 and fig.height_px > 0
