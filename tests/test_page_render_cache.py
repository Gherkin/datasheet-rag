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

import subprocess
import sys
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

    # A 300 DPI A4 page (~25 MB of RGB, ~58 MB per slot) fits many times over,
    # so the cap is the ordinary worker ceiling, not the budget.
    assert _render_window_size(58 * 1024 * 1024, budget, 900) >= 1
    assert _render_window_size(58 * 1024 * 1024, budget, 900) <= 8
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


# ---------------------------------------------------------------------------
# Sizing the window must not change how failures surface (GH #40 × GH #59)
# ---------------------------------------------------------------------------


def test_unopenable_pdf_still_reports_a_page_render_error(tmp_path: Path) -> None:
    """Measuring pages up front must not downgrade the GH #40 error contract.

    The window is sized before the pool starts, which means the PDF is now
    opened on the main thread. If that open is what fails, the caller must
    still get the error that names the file, the page and the --force hint —
    not the bare ``FileNotFoundError`` PyMuPDF raises.
    """
    missing = tmp_path / "gone.pdf"

    with pytest.raises(PageRenderError) as excinfo:
        _render_all(missing, dpi=72, pages=[1])

    message = str(excinfo.value)
    assert str(missing) in message
    assert "page 1" in message
    assert "--force" in message


def test_unopenable_pdf_reports_a_page_render_error_without_a_page_list(
    tmp_path: Path,
) -> None:
    """Same contract when the page count itself is what could not be read."""
    missing = tmp_path / "gone.pdf"

    with pytest.raises(PageRenderError) as excinfo:
        _render_all(missing, dpi=72)

    assert str(missing) in str(excinfo.value)


def test_slot_sizing_falls_back_instead_of_raising(tmp_path: Path) -> None:
    from datasheet_rag.figures import _page_slot_bytes

    # An unreadable document still yields a usable slot estimate: sizing the
    # window is not the step that gets to reject a PDF.
    assert _page_slot_bytes(tmp_path / "gone.pdf", [1], 72) > 0


def test_warm_cache_survives_a_moved_source_pdf(pdf: Path, tmp_path: Path) -> None:
    """A fully cached re-render must not need the original file.

    Sizing the window opens the PDF, which would otherwise turn a cache hit
    into a hard failure whenever the source moved between ingests.
    """
    cache_dir = tmp_path / "render_cache"
    _render_all(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)

    pdf.unlink()
    pages = _render_all(pdf, dpi=72, pages=[1, 2], cache_dir=cache_dir)

    assert set(pages) == {1, 2}


def test_zero_memory_budget_is_honoured_not_read_as_unset(
    pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``memory_budget_mb=0`` means the smallest window, not the default one."""
    from datasheet_rag import figures

    real = figures._render_window_size
    seen: list[int] = []

    def spy(slot_bytes: int, budget_bytes: int, n_pages: int) -> int:
        seen.append(budget_bytes)
        return real(slot_bytes, budget_bytes, n_pages)

    monkeypatch.setattr(figures, "_render_window_size", spy)

    stream = figures.iter_pdf_pages(pdf, dpi=72, pages=[1, 2], memory_budget_mb=0)
    next(stream)
    stream.close()

    assert seen == [0]


def test_budget_fallback_tracks_the_settings_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """The no-config fallback is the settings default, not a second literal.

    Two copies of the number would drift, and which one you got would depend
    only on whether a config happened to load. Both sides are asserted against
    the literal rather than against each other: comparing the fallback to the
    expression it is implemented as cannot fail, so it would pin nothing.
    """
    from datasheet_rag import figures
    from datasheet_rag.config import Settings

    def no_settings() -> object:
        raise RuntimeError("no RAG_HOME")

    monkeypatch.setattr("datasheet_rag.config.get_settings", no_settings)

    assert Settings.model_fields["render_memory_budget_mb"].default == 1024
    assert figures._render_budget_mb() == 1024


def test_slot_sizing_keeps_what_it_measured_before_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A document that dies mid-measurement must not fall back to A4.

    ``fitz.open`` is tolerant enough to open a file whose later pages are
    damaged. Throwing away the pages already measured would substitute an A4
    slot for a fold-out that was sized correctly moments earlier — undersizing
    the slot, widening the window, and overshooting the very budget this
    machinery exists to hold (GH #59).
    """
    import fitz

    from datasheet_rag.figures import (
        _A4_HEIGHT_PT,
        _A4_WIDTH_PT,
        _SLOT_BYTES_PER_PX,
        _page_slot_bytes,
    )

    class _Rect:
        def __init__(self, width: float, height: float) -> None:
            self.width, self.height = width, height

    class _Page:
        def __init__(self, rect: _Rect) -> None:
            self.rect = rect

    class _Doc:
        """Page 1 is an A2 fold-out; page 2 is damaged and raises."""

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> _Page:
            if index == 0:
                return _Page(_Rect(1191, 1684))
            raise RuntimeError("cannot find page 2 in the file")

        def __enter__(self) -> _Doc:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    monkeypatch.setattr(fitz, "open", lambda *a, **k: _Doc())

    measured = _page_slot_bytes(tmp_path / "damaged.pdf", [1, 2], 72)

    # At 72 DPI the scale is 1, so the slot is the page area times the
    # in-flight bytes per pixel — the A2 page, not an A4 one.
    a2_slot = 1191 * 1684 * _SLOT_BYTES_PER_PX
    a4_slot = _A4_WIDTH_PT * _A4_HEIGHT_PT * _SLOT_BYTES_PER_PX
    assert measured == a2_slot > a4_slot


# The child reports, on stderr, how far one A2 page at 300 DPI pushed peak RSS above what
# the process had already touched, next to the slot the window was sized with.
# A warm-up render first pulls in the pool, the progress bar and PyMuPDF's
# fonts, so the baseline holds everything except the page itself.
#
# The peak is VmHWM, not ru_maxrss: ru_maxrss survives exec, so a child of a
# pytest process that has grown past one page would report the parent's peak.
_PEAK_PROBE = """
import sys
from pathlib import Path
import fitz
from datasheet_rag.figures import _page_slot_bytes, iter_pdf_pages

def peak_bytes():
    with open("/proc/self/status") as f:
        return next(int(l.split()[1]) for l in f if l.startswith("VmHWM:")) * 1024

tmp = Path(sys.argv[1])
for name, (w, h) in {"small": (100, 100), "a2": (1191, 1684)}.items():
    doc = fitz.open()
    doc.new_page(width=w, height=h).insert_text((50, 50), name)
    doc.save(str(tmp / f"{name}.pdf"))
    doc.close()

def render(name):
    for _, img in iter_pdf_pages(tmp / f"{name}.pdf", dpi=300, cache_dir=tmp / name):
        img.load()

render("small")
base = peak_bytes()
render("a2")
peak = peak_bytes()
print(peak - base, _page_slot_bytes(tmp / "a2.pdf", [1], 300), file=sys.stderr)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="reads VmHWM from /proc")
def test_one_page_stays_within_its_slot(tmp_path: Path) -> None:
    """Rendering and caching one page fits the slot the window budgets for it.

    The slot is what turns RAG_RENDER_MEMORY_BUDGET_MB into a worker count, so
    a slot smaller than the real footprint lets the pool overshoot the budget
    (GH #59). Reading the pixmap through ``.samples`` instead of
    ``.samples_mv`` adds a third full copy and fails this (GH #63). It runs in
    a fresh interpreter because the peak is a high-water mark for the whole
    process, and earlier tests would already have set it.
    """
    out = subprocess.run(
        [sys.executable, "-c", _PEAK_PROBE, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stderr.splitlines()  # stdout carries the progress bar
    # The report is the last line; a warning on stderr may come before it.
    grew, slot = (int(v) for v in out[-1].split())

    # The page must actually have been measured, or the check below is empty.
    assert grew > slot // 2
    # A few percent of slack for allocator rounding; the .samples copy would
    # add ~40%.
    assert grew <= slot * 1.05


# One solid-colour 2000 x 2000 image per page: 12 MB each once decoded, but
# only a few KB each in the PDF. Every page gets its own colour, so a page drawn
# with another page's image, or with a freed one, shows up as a wrong pixel.
# Each image fills a 2000 pt page, so at 72 DPI MuPDF decodes it at full size;
# drawn smaller, it decodes a subsampled copy and the store barely grows.
_IMAGE_PAGES = 12
_IMAGE_PX = 2000


def _page_colour(page_no: int) -> tuple[int, int, int]:
    return (page_no * 20 % 256, 255 - page_no * 20 % 256, page_no * 7 % 256)


def _write_image_pdf(path: Path, images_per_page: int = 1) -> None:
    """With more than one image, each is squeezed into its own horizontal band."""
    import fitz

    doc = fitz.open()
    for page_no in range(1, _IMAGE_PAGES + 1):
        page = doc.new_page(width=_IMAGE_PX, height=_IMAGE_PX)
        band = _IMAGE_PX / images_per_page
        for k in range(images_per_page):
            colour = _page_colour(page_no + k * _IMAGE_PAGES)
            rgb = bytes(colour) * (_IMAGE_PX * _IMAGE_PX)
            pix = fitz.Pixmap(fitz.csRGB, _IMAGE_PX, _IMAGE_PX, rgb, False)
            rect = fitz.Rect(0, k * band, _IMAGE_PX, (k + 1) * band)
            page.insert_image(rect, pixmap=pix, keep_proportion=False)
    doc.save(str(path), deflate=True)
    doc.close()


def test_emptying_the_store_is_safe_with_several_workers(tmp_path: Path) -> None:
    """Every worker empties MuPDF's store while the others are still drawing.

    The store is shared by the threads (GH #79). Emptying it drops only the
    store's own reference, so no page may come out with an image that another
    worker freed or swapped. Several rounds, so the timing varies.
    """
    import os

    from datasheet_rag.figures import _page_slot_bytes, _render_window_size

    path = tmp_path / "images.pdf"
    _write_image_pdf(path)
    pages = list(range(1, _IMAGE_PAGES + 1))
    budget = 4096 * 1024 * 1024
    workers = _render_window_size(_page_slot_bytes(path, pages, 72), budget, len(pages))
    if workers < 2:
        pytest.skip(f"only {os.cpu_count()} CPU(s): the pool renders serially")

    for _ in range(5):
        for page_no, img in iter_pdf_pages(path, dpi=72, memory_budget_mb=4096):
            assert img.getpixel((_IMAGE_PX // 2, _IMAGE_PX // 2)) == _page_colour(page_no)


_STORE_PROBE = """
import sys
from pathlib import Path
import fitz
from datasheet_rag.figures import iter_pdf_pages

def peak_bytes():
    with open("/proc/self/status") as f:
        return next(int(l.split()[1]) for l in f if l.startswith("VmHWM:")) * 1024

tmp = Path(sys.argv[1])
warm = fitz.open()
warm.new_page().insert_text((50, 50), "warm")
warm.save(str(tmp / "warm.pdf"))
warm.close()

def render(name):
    # A budget below two slots forces one worker, so one page is in flight.
    for _, img in iter_pdf_pages(tmp / name, dpi=72, memory_budget_mb=4):
        img.load()

render("warm.pdf")
base = peak_bytes()
render("images.pdf")
print(peak_bytes() - base, file=sys.stderr)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="reads VmHWM from /proc")
def test_decoded_images_do_not_pile_up_across_pages(tmp_path: Path) -> None:
    """A page's decoded images are gone before the next page is rendered.

    MuPDF's store used to keep them all, up to its own cap and outside the
    render budget (GH #79). Twelve pages of 12 MB images then grew the process
    by about 200 MB; with the store emptied after each page, only the page in
    flight holds its image. It runs in a fresh interpreter because the peak is
    a high-water mark for the whole process.
    """
    _write_image_pdf(tmp_path / "images.pdf")
    out = subprocess.run(
        [sys.executable, "-c", _STORE_PROBE, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stderr.splitlines()  # stdout carries the progress bar
    grew = int(out[-1])

    one_image = _IMAGE_PX * _IMAGE_PX * 3
    # Measured on PyMuPDF 1.27.2: about 6 images' worth with the store emptied
    # (one page's pixmap, its PIL copy, its own decoded image — GH #81 — and
    # allocator slack), about 17 without. The line sits between the two.
    assert grew < 10 * one_image


_RETAINED_PROBE = """
import sys
from pathlib import Path
import fitz
from datasheet_rag.figures import iter_pdf_pages

def rss_bytes():
    with open("/proc/self/status") as f:
        return next(int(l.split()[1]) for l in f if l.startswith("VmRSS:")) * 1024

tmp = Path(sys.argv[1])
warm = fitz.open()
warm.new_page().insert_text((50, 50), "warm")
warm.save(str(tmp / "warm.pdf"))
warm.close()

def render(name):
    for _, img in iter_pdf_pages(tmp / name, dpi=72, memory_budget_mb=4096):
        img.load()
        del img

render("warm.pdf")
base = rss_bytes()
render("images.pdf")
print(rss_bytes() - base, file=sys.stderr)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="reads VmRSS from /proc")
def test_freed_page_memory_goes_back_to_the_os(tmp_path: Path) -> None:
    """Once the pages are freed, the process shrinks back to its size before.

    glibc kept the freed page buffers in its worker-thread arenas, and
    malloc_trim did not return them (GH #80). The MCP server is long-lived,
    so every large ingest left it holding them. It runs in a fresh interpreter
    so that no earlier test has already grown the arenas.
    """
    import os

    from datasheet_rag.figures import _page_slot_bytes, _render_window_size

    path = tmp_path / "images.pdf"
    # Two images a page, each drawn at half its height: with one full-page
    # image a page, glibc kept only about two images' worth even before.
    _write_image_pdf(path, images_per_page=2)
    pages = list(range(1, _IMAGE_PAGES + 1))
    budget = 4096 * 1024 * 1024
    workers = _render_window_size(_page_slot_bytes(path, pages, 72), budget, len(pages))
    if workers < 2:
        pytest.skip(f"only {os.cpu_count()} CPU(s): one worker retained little anyway")

    out = subprocess.run(
        [sys.executable, "-c", _RETAINED_PROBE, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stderr.splitlines()  # stdout carries the progress bar
    retained = int(out[-1])

    one_image = _IMAGE_PX * _IMAGE_PX * 3
    # Measured with 8 workers: 7 to 13 images' worth without the fixed mmap
    # threshold.
    assert retained < 3 * one_image
