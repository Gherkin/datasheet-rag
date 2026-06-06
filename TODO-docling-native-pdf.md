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

### 2. Adjacent-column bleed on two-column performance-characteristic charts (cosmetic)

**Symptom:** In two-column datasheets (tested: MP6004), bounding boxes for
left-column charts extend slightly past the column gutter, capturing a sliver
of the right-column chart's Y-axis label. Visually the crop is still clearly
the left-column chart, but the right edge is slightly contaminated.

**Root cause:** Docling's layout detector does not model the column gutter
explicitly; its bounding box for a left-column element may extend a few percent
past the true column boundary.

**Proposed fix:** After docling analysis, detect two-column pages (check whether
any two figure bboxes on the same page are side-by-side with overlapping
vertical ranges) and clip figure bboxes to the detected column midpoint before
cropping. No fix needed in `_build_figure_regions` itself — this would be a
post-processing step in `figures.extract_figures_from_regions`.

### 3. Table of Contents page creates an oversized MESO with ~110 MICRO children (low)

**Symptom:** The TMC2208 ToC page (page 4) is parsed as a single `SECTION_HEADER`
block containing all ToC entries as individual TEXT elements. The splitter
creates one MESO chunk for the entire ToC page, which ends up with 110 MICRO
children (one per ToC entry: "Section title … page number"). These are not
useful for retrieval — queries about document content should not land on ToC
entries.

**Proposed fix:** Detect ToC sections (heuristic: section title is "Table of
Contents" / "Contents" / "Index", or >50 % of text elements on the page match
a "text … number" pattern) and skip them entirely in `_build_outline`, or
assign them to a dedicated `LayoutType.TOC` so the search layer can filter
them out.

### 4. Formulas stored as image-only with no text fallback (low)

**Symptom:** When docling cannot extract LaTeX/MathML source for a formula
(common in older PDFs), the `ContentElement.text` is `"[Formula]"`. The crop
image is correct and will be handled by describe-figures, but until that step
runs the chunk text is uninformative. Vector search cannot find it by content
before description.

**Proposed fix:** Run a lightweight OCR pass on formula crops using PyMuPDF's
own text extraction (it can often recover inline formula text that docling
misses) and populate `text` with the result. Only applies when docling returns
no formula text. Could be gated behind a `--formula-ocr` flag.
