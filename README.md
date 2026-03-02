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

# List uploaded documents
rag list
```

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
├── embedding/          # (Phase 2) Embedding pipeline
│   └── embedder.py     #   Bedrock embedding calls
├── concepts/           # (Phase 3) Concept extraction
│   ├── extractor.py    #   LLM-based concept extraction
│   └── graph.py        #   Concept graph operations
└── agent/              # (Phase 4) ReAct agent
    ├── tools.py        #   Agent tool definitions
    └── agent.py        #   ReAct loop
```
