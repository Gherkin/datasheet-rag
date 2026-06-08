# AWS RAG Pipeline — Electronics Datasheet Intelligence

## Overview

A multi-scale, context-aware RAG pipeline that performs layout-aware OCR on
electronics datasheets, embeds them using hierarchical chunking with concept
linking, and provides a ReAct agent for intelligent retrieval.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your AWS settings

# Upload a datasheet
rag upload path/to/datasheet.pdf

# Run Textract analysis (async, multi-page)
rag analyze <doc_id>

# Or quick single-page sync analysis
rag analyze path/to/single-page.pdf --mode sync

# Extract text preserving layout
rag extract-text output/<doc_id>_blocks.json

# Inspect Textract block structure
rag inspect-layout output/<doc_id>_blocks.json

# Chunk into multi-scale graph
rag chunk output/<doc_id>_blocks.json

# Embed and store (Bedrock Titan v2 → SQLite + sqlite-vec + FTS5)
rag embed output/<doc_id>_chunks.json --project-id my-board

# Generate vision-LLM descriptions for figure chunks (Bedrock Claude 3 Haiku)
rag describe-figures --project-id my-board --missing-only

# Tag the document (sidecar — no re-ingest needed)
rag metadata set <doc_id> --mpn STM32H743VIT6 --manufacturer ST --subsystem mcu

# Backfill AI-inferred titles for documents with blank/generic titles
rag fix-titles

# Save a document's source PDF to disk
rag download <doc_id> [-o path/or/dir/]

# Print clickable URLs to read a document's PDF in the browser
rag open <doc_id>

# Search the store (hybrid by default)
rag search "I2C clock stretching" --project-id my-board -k 5
rag search "ESD HBM rating" --mode keyword
rag search "thermal resistance junction-to-ambient" --mode vector

# List uploaded documents
rag list
```

## Global config options

Settings live in `aws_rag.config.Settings`, read as `RAG_*` environment
variables. The single shared RAG store — db, PDFs, figures, caches, and
config — lives at `~/.rag` (or `$RAG_HOME` if overridden), so the global
config file is `~/.rag/config.env`. Set `RAG_TABLE_STRUCTURE_MODE` and
friends there once and every project pointed at this RAG instance picks
them up (a cwd-local `.env` can still layer per-checkout overrides on top,
but for one shared instance `~/.rag/config.env` is the place to reach for).
Most settings are self-explanatory (AWS region/profile, model IDs, batch
sizes); this section covers the ones with real tradeoffs worth
understanding before you change them.

### Table parsing mode (`RAG_TABLE_STRUCTURE_MODE`)

Docling recognises table structure with its TableFormer model, which has two
modes:

- **`fast`** (default) — roughly **2.4x faster**. Good enough for the vast
  majority of tables. Known weakness: it can misparse complex multi-level
  headers, producing a single header cell's text duplicated across many
  columns (a "garbled header"). When that happens, `rag ingest` prints a
  loud `Table parsing warning` and the chunking pipeline automatically drops
  the garbled header from the embedded text — see
  `aws_rag.docling_parser._detect_garbled_header`.
- **`accurate`** — slower, more precise cell-boundary detection. **It is not
  a guaranteed fix**: on a real-world complex nested header (a 64-pin
  TQFP/VQFN pin-mux table), re-running it in ACCURATE mode produced a
  *smaller* table but a *differently* garbled header (`'MUXEN=1 PMUX
  Values'` repeated 15x instead of the FAST-mode garble). The garbled-header
  safety net therefore applies — and matters — in both modes.

Set the global default with `RAG_TABLE_STRUCTURE_MODE=fast|accurate` in your
env file, or override it for one ingest with `rag ingest --accurate-tables`
/ `--fast-tables`.

If you only need to fix a handful of tables flagged by a parsing warning,
don't pay for a full re-ingest of a multi-thousand-page document — use:

```bash
# Re-run just pages 36-38 with ACCURATE tables, geometrically match the
# results onto the cached layout outline, and patch only those tables in.
rag reconvert-tables path/to/datasheet.pdf --pages 36-38

# Preview the size/garbled-header delta without writing anything
rag reconvert-tables path/to/datasheet.pdf --pages 36 --dry-run
```

This converts only the given page range (Docling's native `page_range`
support keeps true PDF page numbers in the result), matches each fresh table
to its cached counterpart by page number + bounding-box overlap (tables are
geometrically self-contained leaf elements, unlike sections/headings which
can't be safely tree-merged across two independent conversions), patches
just the matched tables' cells/text, and invalidates the cached chunk graph
so the next `rag ingest` re-derives chunks and embeddings from the patched
outline — without re-running Docling layout analysis on the rest of the
document.

## Wiring Claude Code to a project

Copy `.mcp.json.example` to `.mcp.json` in your electronics project's repo
and edit the env block:

```json
{
  "mcpServers": {
    "aws-rag": {
      "command": "rag-mcp",
      "env": {
        "RAG_SQLITE_DB_PATH": "/path/to/aws-rag/output/rag.sqlite",
        "RAG_DEFAULT_PROJECT_ID": "pcb-rev-a"
      }
    }
  }
}
```

Claude Code will launch one `rag-mcp` process per project; tools come
back scoped to that project unless the agent explicitly overrides
`project_id` in a tool call.

## Shell completion

`rag` is a Click CLI, which ships built-in tab completion for bash, zsh,
and fish — no extra packages needed. Add one of the following to your
shell's startup file:

```bash
# ~/.bashrc
eval "$(_RAG_COMPLETE=bash_source rag)"

# ~/.zshrc
eval "$(_RAG_COMPLETE=zsh_source rag)"

# ~/.config/fish/completions/rag.fish
_RAG_COMPLETE=fish_source rag | source
```

`eval` re-invokes Python on every shell startup. For a snappier shell,
generate the script once and source the file instead:

```bash
_RAG_COMPLETE=bash_source rag > ~/.rag-complete.bash
echo 'source ~/.rag-complete.bash' >> ~/.bashrc
```

Re-run the generation step whenever subcommands or options change.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Phase 1: Ingestion                   │
├─────────────────────────────────────────────────────────┤
│  PDF ──► S3 ──► Textract (LAYOUT+TABLES+FORMS)         │
│                    │                                    │
│                    ▼                                    │
│            Layout-Aware Blocks JSON                     │
│            (preserves tables, headers, sections)        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│             Phase 2: Hierarchical Chunking              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Level 0 (Macro)  ── Full chapter summaries (~2k tok)   │
│       │                                                 │
│       ├── Level 1 (Meso) ── Subsections (~512 tok)      │
│       │       │                                         │
│       │       ├── Level 2 (Micro) ── Paragraphs/tables  │
│       │       │                        (~128 tok)       │
│       │       └── ...                                   │
│       └── ...                                           │
│                                                         │
│  Each chunk stores:                                     │
│   • content_embedding (the text itself)                 │
│   • context_embedding (chapter + section + doc meta)    │
│   • parent_id, prev_id, next_id, chapter_root_id       │
│   • level (0/1/2)                                       │
│   • layout_type (text, table, figure, key-value)        │
│   • page_numbers                                        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Phase 3: Concept Graph Layer                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  For each chunk, an LLM extracts concepts:              │
│   • "thermal resistance calculation"                    │
│   • "ESD protection rating"                             │
│   • "I2C clock stretching"                              │
│   • "dropout voltage vs. load current"                  │
│                                                         │
│  Concepts are:                                          │
│   • Embedded in the same vector space                   │
│   • Linked to chunks (many-to-many)                     │
│   • Linked to each other (similarity + co-occurrence)   │
│                                                         │
│  Query modes:                                           │
│   1. Standard: query → vector search → chunks           │
│   2. Concept-augmented:                                 │
│      query → chunks → extract concept → find other      │
│      chunks sharing that concept (within/across docs)   │
│   3. Concept navigation:                                │
│      concept → related concepts → their chunks          │
│                                                         │
│  Storage: DynamoDB (graph edges) + Vector DB (embeds)   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│             Phase 2.5: Local Embedding Store            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Implemented locally first to keep costs minimal:       │
│   • Bedrock Titan Embed Text v2 (1024-dim, $0.02/1M tok)│
│   • SQLite + sqlite-vec for vector KNN                  │
│   • FTS5 with porter stemming for keyword/BM25          │
│   • Hybrid retrieval via Reciprocal Rank Fusion         │
│   • doc_metadata sidecar table — tag / re-tag without   │
│     re-embedding (project_id, mpn, manufacturer, …)     │
│                                                         │
│  Migration path: same schema → Postgres + pgvector +    │
│  tsvector when multi-user / hosted is needed.           │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         Phase 3: MCP Server (Claude Code etc.)          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  `rag-mcp` runs an MCP stdio server with these tools:   │
│   • search(query, mode, k, project_id?, doc_id?, …)     │
│   • get_chunk(chunk_id, include_neighbors?)             │
│   • navigate(chunk_id, direction)                       │
│        directions: parent, children, prev, next,        │
│                    chapter_root                         │
│   • zoom_in / zoom_out — sugar over navigate            │
│   • list_documents(project_id?, group?, mpn?,           │
│                    manufacturer?)                       │
│   • get_document_metadata(doc_id)                       │
│   • stats(project_id?, doc_id?)                         │
│                                                         │
│  One MCP server per project — scope it via              │
│  RAG_DEFAULT_PROJECT_ID in the .mcp.json env block.     │
│  See .mcp.json.example for the exact config.            │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Phase 4: ReAct Agent                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Tools:                                                 │
│   • vector_search(query, level, top_k)                  │
│   • navigate(chunk_id, direction)                       │
│        directions: parent, child, prev, next, chapter   │
│   • concept_search(query, concept_id, scope)            │
│        scope: same_document, all_documents              │
│   • get_context(chunk_id)                               │
│        returns: chapter_title, section_summary,         │
│                 chunk_text, neighboring chunks           │
│   • zoom_in(chunk_id)                                   │
│        returns finer-grained children                   │
│   • zoom_out(chunk_id)                                  │
│        returns parent summary                           │
│                                                         │
│  The agent reasons about:                               │
│   1. Which zoom level matches the query abstraction     │
│   2. Whether to navigate to related chunks              │
│   3. Whether to use concept search for lateral moves    │
│   4. When it has enough context to answer               │
└─────────────────────────────────────────────────────────┘
```

## Roadmap / Ideas

### Formula extraction / description (not implemented)

When Docling cannot extract LaTeX/MathML from a formula region, the chunk
`text` falls back to a placeholder (`"[Formula]"`) and is effectively
unsearchable until `describe-figures` runs on it.

- **TODO:** teach the vision-LLM description step to output formulas as
  structured plaintext (LaTeX or readable notation) rather than a prose
  description, so formula chunks become searchable on their mathematical
  content, not just a caption-level summary.

### Relevance-feedback navigation — "more like this" / "less like these" (not implemented)

A future MCP tool that lets the agent refine a result set *by example* instead
of by rephrasing the query. The agent passes chunk IDs it judges relevant
(`keep`) and/or irrelevant (`drop`); the tool returns a fresh ranking pulled
toward the kept chunks and away from the dropped ones.

- **Internally** this is classic Rocchio relevance feedback: build a modified
  query vector `q' = α·q + β·centroid(keep) − γ·centroid(drop)`, renormalize
  (Titan v2 vectors are L2-normalized), and re-run KNN. `vector_search()`
  already accepts a raw `query_embedding`, so no new retrieval plumbing is
  needed — only the vector-assembly step.
- **Why a verb, not a knob:** an LLM agent has no way to choose a continuous α,
  can't introspect the embedding geometry, and re-querying in natural language
  is its native strength. So we deliberately do NOT expose vector arithmetic.
  We expose a discrete semantic action ("more like these"), which is a judgment
  the agent is genuinely good at making, and hide the math.
- **Framework fit:** this is a third navigation axis. We already have
  forward/back (`prev`/`next`) and up/down (`zoom_in`/`zoom_out`); "more like
  this" is a *lateral / associative* jump to related material elsewhere in the
  corpus by semantic similarity.
- **Open questions before building:**
  - Does it beat the baseline of the agent simply rephrasing and re-querying?
    It should win for associative discovery ("more like these specific
    results") but not for "make the query more specific" (just re-query).
  - Tool-count discipline — every MCP tool competes for the agent's attention
    and can degrade tool selection across the whole set. Only add if associative
    cross-corpus jumps are a gap the agent actually hits.
  - Titan v2 is contrastively trained, not linearly compositional, so treat the
    centroid math as *steering*, not exact concept algebra. Mean-centering the
    corpus may help if results look off.
  - In `hybrid_search`, a steered vector only affects the KNN branch; the BM25
    branch still needs text.

### Switch MACRO summaries from extractive to abstractive (evaluated — not switching yet)

The chunking pipeline defaults to the **extractive** summarizer for MACRO
(chapter) chunks, and the main ingest path hardcodes it — the **abstractive**
(Bedrock Claude) summarizer is opt-in only via `aws-rag chunk --summarizer
abstractive`.

Extractive is fast and free but crude: it concatenates the *leading* sentences
of the most heavily content-weighted blocks and hard-truncates to a char
budget. There's no notion of which sentence is most informative (position +
layout-type weight only), so a section's key fact can be lost to truncation,
and the output reads as fragments rather than a real description. Abstractive
produces purpose-written descriptions tuned for semantic search.

**Eval result (2026-06-07):** ran `rag eval ablate --index-ablation
macro-summarizer` (Claude Haiku 4.5, re-embedded, queried at `hybrid @macro`)
on two docs with very different structures:

| doc | extractive MRR/nDCG | abstractive MRR/nDCG | `synthesis` ranks (extractive → abstractive) |
|---|---|---|---|
| MAX40025 datasheet (13 chapters, all describing the same 2-3 near-identical comparator variants) | 0.650 / 0.635 | 0.806 / 0.720 | 1,1,1 → 1,2,1 |
| CC Linux Software Guide (9 chapters, each on a distinct subsystem) | 0.812 / 0.744 | 1.000 / 0.851 | 1,1,4,1 → 1,1,1,1 |

Abstractive is a clean win on the heterogeneous guide (one query goes from
rank 4 to rank 1, zero regressions) but causes a small `synthesis`-category
regression on the datasheet (one query rank 1 → 2). Root cause: when every
chapter genuinely describes the same handful of devices, the LLM tends to
open each summary with similar device-level framing, which raises
inter-chapter similarity and slightly hurts discriminability — a structural
property of single-product-family datasheets that prompt tuning only
partially escapes (tried two opposing tunings; both plateaued around the same
MRR). One line in `_macro_summary_prompt` ("open with what's distinct about
this chapter") recovered about half the regression at no cost and is now in
place.

Cost/latency scales with how much each chapter has to say: ~8-10 s/chapter on
the sparse datasheet vs. ~33-36 s/chapter on the content-rich guide — a real
addition to ingest latency worth weighing once the quality question is
settled.

- **Verdict: don't flip the default yet.** The quality delta is
  document-structure-dependent — abstractive clearly wins on documents with
  long, topically-distinct sections, but can mildly regress single-product
  datasheets where every chapter overlaps. The golden set (3 + 4 `synthesis`
  items across 2 docs) is too small to set policy on; growing it — especially
  with more single-product datasheets — would clarify whether that regression
  generalizes or is a one-off.
- Wiring abstractive through the main ingest path (currently hardcoded to
  `summarizer_mode="extractive"` at the two `ingest` call sites in `cli.py`)
  is a 10-minute change whenever we do flip the default — not worth doing
  speculatively.

## AWS Services Used

| Service              | Purpose                                 |
|---------------------|-----------------------------------------|
| S3                  | PDF storage, Textract output            |
| Textract            | Layout-aware OCR (LAYOUT, TABLES, FORMS)|
| Bedrock (Titan/Cohere) | Embeddings                           |
| OpenSearch Serverless| Vector database with metadata filtering |
| DynamoDB            | Chunk graph, concept associations       |
| Bedrock (Claude)    | Concept extraction, ReAct agent         |

## Project Structure

```
src/aws_rag/
├── __init__.py
├── config.py           # Pydantic settings
├── aws.py              # AWS client factory
├── storage.py          # S3 operations
├── textract.py         # Textract analysis + layout parsing
├── cli.py              # Click CLI
├── models/             # (Phase 2) Data models
│   ├── chunk.py        #   Hierarchical chunk model
│   └── concept.py      #   Concept graph model
├── chunking/           # (Phase 2) Chunking pipeline
│   ├── splitter.py     #   Multi-scale text splitting
│   └── context.py      #   Context enrichment
├── embedding/          # (Phase 2.5) Bedrock Titan v2 wrapper
│   └── embedder.py     #   Concurrent batched embedding + retries
├── store/              # (Phase 2.5) Local SQLite store
│   ├── schema.py       #   Connect + DDL (chunks, vecs, fts, sidecar)
│   ├── sqlite.py       #   CRUD helpers
│   ├── search.py       #   vector / keyword / hybrid (RRF)
│   └── metadata.py     #   Doc-level metadata sidecar
├── mcp/                # (Phase 3) MCP server for Claude Code
│   └── server.py       #   FastMCP tools: search, navigate, zoom, …
├── concepts/           # (Phase 3) Concept extraction (planned)
│   ├── extractor.py    #   LLM-based concept extraction
│   └── graph.py        #   Concept graph operations
└── agent/              # (Phase 4) ReAct agent (planned)
    ├── tools.py        #   Agent tool definitions
    └── agent.py        #   ReAct loop
```
