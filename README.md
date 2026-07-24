# Datasheet-RAG - PDF RAG stack for electronics docs

## Overview

A multi-scale, context-aware RAG pipeline that performs layout-aware OCR on
electronics datasheets, embeds them using hierarchical chunking with concept
linking, and provides a ReAct agent for intelligent retrieval.

## Installation

You can either run without a dedicated backend, with the files stored on
your local disk, or with a remote backend. Then for actually doing 
ingestation, you need to chose between Textract and Docling for PDF handling,
and then if you want to run the models used for image-handling, cleanup and 
embedding locally or through Bedrock.  

These choices are handled via extras on the python package. The available
options are:
* aws - required for running any part of the pipeline with AWS
* docling - required if not using Textract for PDF-handling
* local-hf - required for running models locally using huggingface
* server - required for serving the backend over HTTP
* token - used for more accurate token usage counting towards AWS
* dev - test requirements

The recommended setup for running the backend:
* Docling for PDFhandling since Textract becomes expensive fast. Note: Docling requires the PDFs to contain text, and not be scanned.
* Use AWS Bedrock for some models, primarily the image description unless
you have a lot of memory for running models.
* Embedding model locally using huggingface, less latency and its quite small.  

To install just the base for running with an external backend:

```bash
pip install "git+https://github.com/Gherkin/aws-rag.git"
```
To run the recommended setup locally (no server)
```bash
pip install "aws-rag[[aws,docling,local-hf] @ git+https://github.com/Gherkin/aws-rag.git"
```

You can also run the backend as a dedicated server. This is recommended even if you run fully local, since there is quite a bit of latency overhead in restarting the stack every time you run a query.  

TODO: document how to install the docker container (with/without the HTTPS proxy)

TODO: explain integration into harnesses, MCP or the straight runnable

## Configuration

## Usage

### Ingestation


-- ai crap beyond this -- 

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

### Model backends (`RAG_EMBEDDING_BACKEND` / `RAG_TEXT_BACKEND` / `RAG_VISION_BACKEND`)

Every model call picks a backend **per capability** — each `local` (default; run
on this machine, no per-call AWS cost) or `bedrock` (AWS). Three independent
switches so they can be mixed:

| Setting | Covers | Local runtime knob |
|---|---|---|
| `RAG_EMBEDDING_BACKEND` | text embeddings | `RAG_LOCAL_EMBEDDING_RUNTIME` |
| `RAG_TEXT_BACKEND` | titling, summaries, eval | `RAG_LOCAL_TEXT_RUNTIME` |
| `RAG_VISION_BACKEND` | figure descriptions, table repair | `RAG_LOCAL_VISION_RUNTIME` |

Each local runtime knob is **`huggingface`** (in-process via PyTorch/CUDA — full
precision, robust, precise VRAM control, any HF model id) or **`ollama`** (the
[Ollama](https://ollama.com) server — lighter deps, GGUF/quantized).

**Recommended on a consumer GPU: embeddings-only-local.** Embeddings are the
search hot path (query embedding is ~71 ms locally vs ~325 ms Bedrock RTT),
run offline with no credentials, are at quality parity, and cost ~nothing
either way — so local is a pure win there. Text and vision stay on Bedrock:
text (titles/summaries) is *slightly* better on Haiku at negligible cost, and
vision is well above any 12 GB-local VLM. Total Bedrock spend for a whole
corpus is ~$1-2 one-time (mostly figure descriptions).

```bash
pip install 'aws-rag[local-hf]'   # sentence-transformers (embeddings); torch/CUDA
# embeddings download from the HF Hub on first use
```

```env
# Recommended hybrid (embeddings local, the rest on Bedrock):
RAG_EMBEDDING_BACKEND=local
RAG_TEXT_BACKEND=bedrock
RAG_VISION_BACKEND=bedrock

RAG_LOCAL_EMBEDDING_RUNTIME=huggingface
RAG_LOCAL_EMBEDDING_MODEL=BAAI/bge-m3      # HF repo id; output dim must match ↓
RAG_EMBEDDING_DIMENSIONS=1024
```

For more-local setups, flip `RAG_TEXT_BACKEND` / `RAG_VISION_BACKEND` to `local`
and set the runtime/model (needs `aws-rag[local]` for Ollama and `ollama serve`,
or `aws-rag[local-hf]` for in-process). Vision example:

```env
RAG_VISION_BACKEND=local
RAG_LOCAL_VISION_RUNTIME=huggingface
RAG_LOCAL_VISION_MODEL=Qwen/Qwen2.5-VL-3B-Instruct  # 3B fits 12GB; 7B/32B/72B need more
# RAG_LOCAL_HF_LOAD_4BIT=true              # 4-bit (bitsandbytes) to fit 7B+ on 12-24GB
RAG_TEXT_BACKEND=local
RAG_LOCAL_TEXT_RUNTIME=ollama              # ollama pull qwen2.5:7b
RAG_LOCAL_TEXT_MODEL=qwen2.5:7b
```
```

Vision model vs VRAM (4-bit): ~8B → ~6-8 GB (12 GB GPU, *below* Haiku on diagrams);
~32B → ~24 GB (*≈ Haiku*); ~72B → ~48 GB (*≈ Sonnet*). The `huggingface` runtime
is hardware-portable: on a bigger box just point `RAG_LOCAL_VISION_MODEL` at a
larger repo id.

> **bge-m3 only works via the `huggingface` runtime.** On Ollama its llama.cpp
> F16 path emits `NaN` (HTTP 500) on some inputs (e.g. the `---` separator we put
> in every chunk), aborting ingestion. For `RAG_LOCAL_EMBEDDING_RUNTIME=ollama`
> use `mxbai-embed-large` instead.

> **Re-embed when switching `RAG_EMBEDDING_BACKEND`** (or the embedding
> model/dimension). Vectors from different models aren't comparable and the
> dimension is baked into the sqlite-vec table, so switching needs a re-embed on
> a fresh DB. The text/vision backends flip freely.

The `--model` overrides on `rag describe-figures` / `repair-tables` /
`infer-title` are Bedrock model IDs and are ignored when that capability runs
locally (local model names come from the settings above).

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

If `table_structure_untrustworthy` flags a table's header band (garbled
repeated text, or a data row fused into the header), `rag repair-tables
<doc_id>` re-transcribes just that header band from a cropped page image via
a vision-capable Bedrock model — the table's data-row column count (`C`) and
header-row count (`H`) come from Docling's grid geometry, which stays
reliable even when the header text doesn't. The proposed H×C grid must
exactly tile the header band and pass anti-degenerate / anti-fusion checks
(`table_repair.validate_header_grid`); rejected or skipped tables (header
band too large to crop reliably, or data rows don't agree on a column count)
keep their existing structure-free reading-order rendering — repair is
additive, never a regression. Use `rag table-structure-sweep` +
`rag repair-tables --dry-run` to preview what would be touched.

**TODO — wider cross-document validation:** the detector heuristics and
`validate_header_grid`'s invariants were derived from one document
(PIC32CK1025GC01100, doc_id `d44efe...`), with a small tracked fixture corpus
(`tests/fixtures/table_repair_corpus/`, exercised by
`tests/test_table_repair_corpus.py`) adding a second, structurally different
document (doc_id `928d4097...`) as a false-positive control. Before treating
the repair logic as generally reliable across vendors, run
`rag table-structure-sweep` + `rag repair-tables --dry-run` against more
datasheets from other silicon vendors (ST, TI, NXP, Renesas, …) with varying
table styles (register-bit-field, pin-mux, electrical-characteristics,
multi-page), spot-check flagged tables manually, and grow the fixture corpus
with any new defect shapes found.

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

To point the same MCP server at a **shared remote server** instead of a
local SQLite file, drop the local-store env vars and set `RAG_SERVER_URL`
(plus `RAG_SERVER_TOKEN` if the server requires auth) — exactly like the
CLI. The server then owns the database, embedder, figures, and PDFs:

```json
{
  "mcpServers": {
    "aws-rag": {
      "command": "rag-mcp",
      "env": {
        "RAG_SERVER_URL": "https://rag.internal:8080",
        "RAG_SERVER_TOKEN": "your-read-or-ingest-token",
        "RAG_DEFAULT_PROJECT_ID": "pcb-rev-a"
      }
    }
  }
}
```

Whenever `RAG_SERVER_URL` is set it takes precedence (remote mode);
otherwise the server runs locally against `RAG_SQLITE_DB_PATH`. The
Claude Desktop `.mcpb` bundle exposes the same choice as **Server URL** /
**Access Token** vs **SQLite Database Path** fields in its config UI.

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

## Remote server & security

A team can share one corpus by running the FastAPI server (`rag-server`, or the
Docker image) and pointing clients at it with `RAG_SERVER_URL`. The server owns
the sqlite DB + embedder; clients talk to it over HTTP.

### Thin-client ingest

In remote mode `rag ingest <pdf>` **uploads the raw PDF** and the server runs
the whole pipeline — detect PDF type → Docling/Textract parse → figure cropping
→ chunk → embed → describe → store — streaming step-by-step progress back to the
client as it goes. The client needs only `httpx` from the base install; the
heavy Docling/torch and Textract stack lives on the server. Scanned-PDF OCR uses
the server's AWS role, so clients never need Textract credentials.

Pass `--local-parse` to instead parse on the client and ship the finished chunk
graph to the server (the older behaviour) — useful for advanced/offline-parse
workflows, but it needs the full parse stack locally. `--dry-run` and
`--show-cost` are client-side estimation and always parse locally. The raw-PDF
route is `POST /ingest-pdf`; the pre-parsed `POST /ingest` (chunk graph) path is
unchanged.

### Auth tiers

Reads are cheap; ingest induces real cost (Bedrock/LLM embed + figure
description). Auth is therefore tiered:

- **Shared read token** — `RAG_SERVER_READ_TOKEN` (or mount a secret file via
  `RAG_SERVER_TOKEN_FILE`). Everyone allowed to search uses it. Sent by clients
  as `RAG_SERVER_TOKEN`. Until a read token *or* any API key exists, reads are
  open (trusted-LAN default).
- **Per-client ingest keys** — each ingesting client gets its own key, so
  cost-inducing actions are individually attributable and revocable. Scopes:
  `read` / `ingest` / `admin` (each implies the lower).

Bootstrap the first admin key (writes straight to the DB), then manage keys at
runtime via the admin API — no restart, revocation is immediate:

```bash
# one-time bootstrap (inside the container or wherever the DB lives)
rag-server create-key --label bootstrap --scope admin

# mint a per-client ingest key
curl -H "Authorization: Bearer <admin-token>" \
     -H 'Content-Type: application/json' \
     -d '{"label":"alice-laptop","scopes":["ingest"]}' \
     https://rag.example.com/admin/keys
# → returns the plaintext token ONCE; paste it into that client's `rag init`.

# revoke a client
curl -X DELETE -H "Authorization: Bearer <admin-token>" \
     https://rag.example.com/admin/keys/<id>
```

### Traceability

Every ingest / figure-description / title-inference / delete is recorded in the
`audit_log` table (who, what, when, outcome) and mirrored as a structured log
line. Read it via `GET /audit` (admin scope).

### TLS (LAN-only, real Let's Encrypt cert)

Terminate TLS at the bundled **Caddy** reverse proxy; the app stays plain HTTP
on the internal docker network. The server has no inbound internet, so prove
domain ownership with the **manual DNS-01** challenge — `deploy/get-cert.sh`
runs certbot in a container, prints a TXT record to add to your own DNS zone,
then **polls public DNS and continues automatically once the record is live**
(no keypress, no guessing about propagation). The cert lands in `./certs/`. No
DNS-provider API or plugin needed; works with any public domain whose
nameservers answer public queries.

```bash
# 1. Get a cert (prints _acme-challenge.<domain> TXT, waits until it resolves):
./deploy/get-cert.sh rag.example.com you@example.com

# 2. Point an A record for rag.example.com at this host's LAN IP (192.168.x.x).

# 3. Start the server behind Caddy (only :443 is published; 8080 stays internal):
RAG_DOMAIN=rag.example.com \
  docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d
```

Clients then use `export RAG_SERVER_URL=https://rag.example.com`. Certs last
~90 days; re-run `get-cert.sh` to renew (fresh TXT each time, auto-detected),
then `docker compose ... restart caddy`. (For a non-Docker host, `rag-server
tls-setup` does the same certbot flow locally.)

Also lock CORS via `RAG_SERVER_CORS_ORIGINS` (empty = no cross-origin access;
server-to-server CLI/MCP traffic is unaffected).
