# TODO — docling-native-pdf branch

Issues found during local testing (2026-06-06) on TMC2202/TMC2208/TMC2224
datasheet (81 pages, single-column) and MP6004 datasheet (23 pages,
two-column).

---

## Fixed in this session

- **`[docling]` optional dependency missing from `pyproject.toml`** — the
  parser code referenced `pip install 'aws-rag[docling]'` in its error
  messages but the extra didn't exist. Added `docling>=2.0.0` and
  `pymupdf>=1.24.0`.

- **`_step()` called with wrong signature in `ingest` command** — lines 947
  and 997 called `_step(step_n, total_steps, label)` but the helper only
  takes `label`; `total_steps` was also undefined. Fixed to `_step(label)`.

- **`TableItem.export_to_markdown()` deprecation** — docling 2.x requires the
  `doc` argument. Was emitting one warning per table per run (~53 warnings on
  the TMC2208 doc). Fixed: `item.export_to_markdown(doc=doc)`.

- **`figure_block_id` mismatch — images never linked to chunks** — `_build_outline`
  used a single global counter (`docling_N`) for all items, while
  `_build_figure_regions` used per-kind counters (`docling_figure_N`,
  `docling_formula_N`). The splitter's figure lookup keyed on the manifest
  block_id, so every lookup missed and `figure_image_path` was `None` on all
  figure chunks. Fixed: added separate `figure_counter` / `formula_counter` in
  `_build_outline` that produce IDs matching `_build_figure_regions`.
  Result: 47/47 figure MICRO chunks now have `figure_image_path` set.

- **Captions always empty** — `_item_caption()` called `item.caption` (singular)
  but docling 2.x uses `item.captions` (list of `RefItem`); even that is only
  populated for ~40 % of figures. Fixed: stateful approach — track a
  `pending_caption_element` / `pending_region` through the iteration loop and
  assign the text of the first following `CAPTION`-labelled item to it.
  `CAPTION` items are now consumed and never turned into standalone text chunks.
  Result: 40/47 figure chunks have captions; chunk `text` is the caption string
  (used as the embedding content).

- **Double-spaces from PDF justified-text positioning** — 182/620 non-table
  MICRO chunks had extra intra-word spaces (e.g. `"The  TMC22xx  family  scores
  with  power  density"`). Added `_clean_text()` helper using
  `re.sub(r" {2,}", " ", ...)` applied to all text fields in `_build_outline`.
  Result: 0 affected chunks.

- **Page-header logo extracted as a figure on every page** — Docling classifies
  the MPS logo (repeated on every page of MP6004.pdf) as `PICTURE` rather than
  `PAGE_HEADER`, producing 21 spurious logo chunks. Added `_dedup_repeating_figures`
  (size bucket → pixel hash → frequency+position gate): drops any PICTURE that
  appears with identical pixel content on ≥ 3 pages AND sits in the header/footer
  zone. Validated: 21 copies dropped, 58 real figures and 27 formulas kept.

- **Adjacent-column figure crop bleed** — `crop_figure()` added 2 % padding on
  all edges; the column gutter on two-column datasheets is only ~1 %, so the
  padding bridged the gap and captured a sliver of the adjacent figure. Added
  `_compute_adjacent_crop_caps()` and extended `crop_figure()` with per-edge cap
  params. Only padding is constrained — bbox content is never removed. Validated
  on MP6004: 25 edges capped, zero content loss.

- **Table of Contents section creates ~110 useless MICRO chunks** — ToC entries
  were parsed as TEXT elements producing a MESO with 110 MICRO children. Added
  `_filter_toc_sections()` (title-match + trailing-number heuristic), applied
  after `_build_outline`. Validated on TMC2208: 110-element ToC section dropped.

---

## Open issues

### 1. Page-header logo extracted as a figure on every page — FIXED

**Symptom:** On multi-page datasheets with a repeated logo in the page header
(e.g. the MPS logo on every page of MP6004.pdf), docling classifies the logo
as `PICTURE` rather than `PAGE_HEADER`. Our `_SKIP` filter only drops
`PAGE_HEADER`-labelled items, so the logo passes through as a figure.

**Fix:** Three-stage dedup in `_dedup_repeating_figures` (called from
`convert_pdf` between `_build_figure_regions` and `_build_outline`):
1. Size bucket — group regions by normalised (width, height) rounded to 2 dp.
2. Pixel hash — crop each region via PyMuPDF at 72 dpi, take MD5 of samples.
3. Frequency + position — if ≥ 3 hash-identical copies span different pages
   AND each sits in the header (top < 0.10) or footer (top+height > 0.90),
   all copies are dropped from both `figure_regions` and the outline.

**Validated on MP6004.pdf:** 21/23 logo copies dropped (21 out of 106 total
regions); 58 real figures and 27 formulas remain. Chunk count: 250 (was 271).

### 2. Adjacent-column bleed on two-column performance-characteristic charts — FIXED

**Symptom:** In two-column datasheets (tested: MP6004), figure crop_figure()
was adding 2 % padding on all edges. The column gutter between adjacent figures
is typically only ~1 %, so the padding bridged the gap and captured a sliver of
the neighbouring column.

**Root cause:** The bboxes themselves did NOT overlap — the gap between adjacent-
column figures is real but smaller than padding_pct. The bleed was padding-only.

**Fix:** `_compute_adjacent_crop_caps()` in `figures.py` detects adjacent-column
pairs (same page, ≥ 30 % vertical co-occurrence, gap < padding_pct). For each
pair it returns per-region crop-edge caps:
- Left figure: `max_right = b.left` (no padding past neighbour's left bbox edge)
- Right figure: `min_left = a.right` (no padding past neighbour's right bbox edge)

`crop_figure()` was extended with `max_right / min_left / max_bottom / min_top`
params; caps are passed in by `extract_figures_from_regions`. Bboxes are never
modified — only padding is constrained. Actual bbox overlap (gap < 0) is left
alone with a log message.

**Validated on MP6004.pdf:** 25 region edges capped across pages 6/8/9/10.
Example p8: fig_24 right=0.3596, fig_27 left=0.3706, gap=0.011; right crop now
capped at 0.3706 (was 0.3796). Gap between crops = 0.011 (original gap preserved,
no content removed).

### 3. Table of Contents page creates an oversized MESO with ~110 MICRO children — FIXED

**Symptom:** The TMC2208 ToC page was parsed as a section with 110 TEXT children
(one per "title … page_number" entry). The splitter produced a MESO with 110
MICRO children that are useless for retrieval.

**Fix:** `_is_toc_section()` + `_filter_toc_sections()` in `docling_parser.py`,
applied after `_flush` in `_build_outline`. Detects by title match
(`_TOC_TITLES`: "table of contents", "contents", "index", "toc") or by content
heuristic (≥ 10 TEXT elements, ≥ 60 % match `r"^.+[.\s]{2,}\d+\s*$"`).

**Validated on TMC2208:** "Table of Contents" section with 110 elements dropped.
581 chunks remain (MESO 117, MICRO 463).

### 4. Formula image description / text extraction (future)

**Symptom:** When docling cannot extract LaTeX/MathML from a formula, the chunk
`text` is `"[Formula]"` and vector search cannot find it before describe-figures
runs.

**Planned approach:** A vision model (describe-figures) that also attempts to
output the formula as structured plaintext (LaTeX or readable notation), not just
a prose description. This is a larger task involving the description prompt and
possibly a post-processing normaliser. Not tackled yet.
