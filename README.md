# Datasheet-RAG - PDF RAG stack for electronics docs

[![CI](https://github.com/Gherkin/datasheet-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Gherkin/datasheet-rag/actions/workflows/ci.yml)

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)

## Overview

A multi-scale, context-aware RAG pipeline that performs layout-aware analysis on
electronics datasheets and embeds them using hierarchical chunking.

Some Features:
* Embeds navigational breadcrumbs with text (chapter/section/page/etc)
* Does visual analysis of figures to embed them as text
* Transcribes formulas into LaTeX from their cropped image
* Different chunk sizes to provide both detail and overview
* Provides rendered pages/figures inline in chat using an MCP App
* Weighted vector and keyword searching

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
* tokens - used for more accurate token usage counting towards AWS
* dev - test requirements

The recommended setup for running the backend:
* Docling for PDFhandling since Textract becomes expensive fast. Note: Docling requires the PDFs to contain text, and not be scanned.
* Use AWS Bedrock for some models, primarily the image description unless
you have a lot of memory for running models.
* Embedding model locally using huggingface, less latency and its quite small.  

To install just the base for running with an external backend:

```bash
pip install "git+https://github.com/Gherkin/datasheet-rag.git"
```
To run the recommended setup locally (no server)
```bash
pip install "datasheet-rag[aws,docling,local-hf] @ git+https://github.com/Gherkin/datasheet-rag.git"
```

You can also run the backend as a dedicated server. This is recommended even if you run fully local, since there is quite a bit of latency overhead in restarting the stack every time you run a query.  

### Running the server in Docker

Prebuilt images are published to GHCR:

| Tag | Use it when |
| --- | --- |
| `ghcr.io/gherkin/datasheet-rag:cpu` (= `:latest`) | Any host without an NVIDIA GPU. Built against PyTorch's CPU wheels, so it skips ~2.5GB of CUDA libraries. |
| `ghcr.io/gherkin/datasheet-rag:cuda` | An NVIDIA host, to GPU-accelerate local bge-m3 embeddings and local VLMs. Needs the NVIDIA Container Toolkit and `--gpus all`. |

There is no separate image per backend. Both tags carry every extra
(`server,aws,docling,local-hf`), and the choice between Bedrock and local
models is made with environment variables at run time — the same ones
documented in `~/.rag/config.env`.

Everything stateful (sqlite db, PDFs, cropped figures, and the downloaded
HuggingFace models) lives under `/data`, so a single volume is all you need.

**Bedrock for everything** — smallest setup, nothing runs locally:

```bash
docker run -d --name rag-server -p 8080:8080 \
  -v rag-data:/data \
  -e RAG_EMBEDDING_BACKEND=bedrock \
  -e RAG_TEXT_BACKEND=bedrock \
  -e RAG_VISION_BACKEND=bedrock \
  -e AWS_REGION=eu-west-1 \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... \
  ghcr.io/gherkin/datasheet-rag:cpu
```

**The recommended hybrid** — bge-m3 embeddings in-process, Bedrock for text and
vision. Here credentials come from a mounted profile instead of static keys:

```bash
docker run -d --name rag-server -p 8080:8080 \
  -v rag-data:/data \
  -v ~/.aws:/home/rag/.aws \
  -e AWS_PROFILE=rag -e AWS_REGION=eu-west-1 \
  -e RAG_EMBEDDING_BACKEND=local \
  -e RAG_LOCAL_EMBEDDING_RUNTIME=huggingface \
  -e RAG_LOCAL_EMBEDDING_MODEL=BAAI/bge-m3 \
  -e RAG_EMBEDDING_DIMENSIONS=1024 \
  -e RAG_TEXT_BACKEND=bedrock \
  -e RAG_VISION_BACKEND=bedrock \
  ghcr.io/gherkin/datasheet-rag:cpu
```

Mount `~/.aws` read-write, not `:ro` — an SSO profile makes botocore refresh and
cache its bearer token underneath it, and a read-only mount fails that write.

**On a GPU**, swap the tag and add `--gpus all`:

```bash
docker run -d --name rag-server -p 8080:8080 --gpus all \
  -v rag-data:/data \
  ghcr.io/gherkin/datasheet-rag:cuda
```

**Fully local**, with a text model on the host's Ollama and no AWS at all:

```bash
docker run -d --name rag-server -p 8080:8080 --gpus all \
  -v rag-data:/data \
  --add-host=host.docker.internal:host-gateway \
  -e RAG_EMBEDDING_BACKEND=local \
  -e RAG_TEXT_BACKEND=local \
  -e RAG_VISION_BACKEND=local \
  -e RAG_LOCAL_TEXT_RUNTIME=ollama \
  -e RAG_OLLAMA_HOST=http://host.docker.internal:11434 \
  ghcr.io/gherkin/datasheet-rag:cuda
```

The first start downloads bge-m3 (~2GB) into the volume. It is cached there, so
only the first run pays for it.

#### Where the models run: `RAG_COMPUTE`

By default the server does all the work — parsing, embeddings, figure
descriptions, title inference — and a client needs nothing but the base
install. That is the right split when the server is the machine with the GPU.

If it isn't, invert it. Set `RAG_COMPUTE=client` **on the client** and the
server becomes purely a vector store: your workstation parses the PDF, embeds
the chunks, describes the figures and infers the title, then uploads the
finished graph plus its vectors. The server never loads a model, so it runs
happily on a NAS or a small VPS — the `:cpu` image with no Bedrock credentials
is enough.

```bash
# On the client (the machine with the GPU):
export RAG_SERVER_URL=http://rag.internal:8080
export RAG_COMPUTE=client
export RAG_EMBEDDING_BACKEND=local     # this machine now needs the model config
export RAG_LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
export RAG_EMBEDDING_DIMENSIONS=1024
```

The client needs the extras for whatever it now runs — `docling` to parse,
`local-hf` for local models, `aws` for Bedrock:

```bash
pip install "datasheet-rag[docling,local-hf] @ git+https://github.com/Gherkin/datasheet-rag.git"
```

It applies to every command that would otherwise reach a model, not just
ingest: `rag search` embeds the query locally, `rag repair figures` runs the
vision model locally against figures it pulls from the server, and `rag repair
titles` does the same for the text model. The MCP server inherits it too.
Override it for one command with `rag --compute client|server ...`.

**Both ends must agree on the embedding model.** Vectors from two different
models are not comparable, so a client that embeds with the wrong one writes
rows that never match. The client checks the server's `/health` before its
first embedding and refuses outright on a dimension mismatch (warning, but
continuing, if only the model *name* differs). `RAG_COMPUTE` has no effect in
local mode, where everything already runs locally.

| | `RAG_COMPUTE=server` (default) | `RAG_COMPUTE=client` |
| --- | --- | --- |
| PDF parse (Docling/Textract) | server | client |
| Chunk + query embeddings | server | client |
| Figure descriptions (vision) | server | client |
| Title inference (text) | server | client |
| Database, PDFs, figure crops | server | server |
| Client needs models installed | no | yes |

#### Auth

Reads are open until you set a read token or create an API key. To lock the
server down, set `RAG_SERVER_READ_TOKEN` and mint per-client ingest keys. The
image's entrypoint is `rag-server`, so its subcommands are a one-shot container
away — no running server, no `exec`:

```bash
docker run --rm -v rag-data:/data ghcr.io/gherkin/datasheet-rag:cpu \
  create-key --label bootstrap --scope admin
```

Clients then point at it with `export RAG_SERVER_URL=http://<host>:8080` — and, if the server has no GPU, `export RAG_COMPUTE=client` (see above).

#### Bind mounts

The server runs as uid 1000. A named volume (`-v rag-data:/data`) inherits that
ownership and needs no thought. If you'd rather bind-mount a host directory,
its ownership wins instead, so run as yourself and make sure the directory is
yours:

```bash
docker run -d --name rag-server -p 8080:8080 \
  --user "$(id -u):$(id -g)" -v "$HOME/.rag/rag-data:/data" \
  ghcr.io/gherkin/datasheet-rag:cpu
```

#### Compose, and HTTPS

`docker-compose.yml` is an optional convenience wrapper around the same image —
worth it for a pinned env file and a restart policy, but never required.

HTTPS is the one case that still wants Compose, because TLS terminates in a
separate Caddy container. Get a certificate via the DNS-01 challenge (works for
a LAN-only host with no inbound internet), then bring up both services:

```bash
./deploy/get-cert.sh rag.example.com you@example.com
RAG_DOMAIN=rag.example.com RAG_DATA_DIR=$HOME/.rag/rag-data \
  docker compose -f docker-compose.yml -f docker-compose.proxy.yml up -d
```

The overlay stops publishing port 8080 to the host, so only Caddy's :443 is
reachable, and clients use `https://rag.example.com`.

### Connecting an agent (MCP)

The tools an agent uses — `search`, `navigate`, `show_figure`, `show_page` and
the rest — are exposed over MCP. There are two ways to reach them.

**Against a server: point the client at `/mcp`.** The server hosts the MCP
endpoint itself, so there is nothing to install on the client machine and no
local process to keep alive. Add this to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "datasheet-rag": {
      "type": "http",
      "url": "http://rag.internal:8080/mcp/pcb-rev-a",
      "headers": { "Authorization": "Bearer <read-token>" }
    }
  }
}
```

The last path segment is the project id the tools scope themselves to — the
remote equivalent of the `.rag.toml` a local server would find in your
checkout. Drop it (`/mcp`) to search the whole store, or pass an
`X-RAG-Project` header instead if your client pins the URL. Individual tool
calls can still override it with a `project_id` argument.

The endpoint takes the same credentials as the REST API: a read token or any
API key with the `read` scope (`ingest` and `admin` imply it). While the server
is in open mode, `/mcp` is open too. Set `RAG_SERVER_MCP_ENABLED=false` to
serve the REST API only.

`show_pdf` is absent here. It works by serving the PDF from a loopback port on
the machine the MCP server runs on, which is the *server* in this setup — so
the URL would be useless to you. `show_page` renders a page inline instead and
works either way; see [#45](https://github.com/Gherkin/datasheet-rag/issues/45)
for serving PDFs from the server properly.

<details>
<summary>DNS-rebinding protection</summary>

The MCP SDK can reject requests whose `Host` header it does not recognise,
which guards a *desktop-local* MCP server against a web page in the user's
browser reaching it. This server is normally reached by several names and
through a proxy, and is guarded by a bearer token, so the check is off unless
you enumerate the hosts you serve:

```bash
-e RAG_MCP_ALLOWED_HOSTS="rag.internal:8080,rag.example.com"
```

Entries are exact or port-wildcarded (`rag.internal:*`). There is no `*`
catch-all — an allowlist you set is an allowlist that is enforced.
</details>

**Without a server: run `rag-mcp` over stdio.** This is the right shape when
the store is a sqlite file on your own disk. The client launches the process
and scopes it via the environment (or a `.rag.toml` in the checkout it starts
in). See `.mcp.json.example` for both variants, and `scripts/install-mcp.sh`
to wire it into Claude Code.


### Shell completion

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

## Configuration
Configuration is mainly done in `~/.rag/config.env`. A default and heavily commented config file can be generated by running `rag config init`. See that file for details on the configurable parameters, but the main things are the address to the remote server if not running fully local, pipeline options (such as docling or textract for ingestion, and which AI models to use).  

In addition to the global configuration file, the parameters can be overridden by an `.env` file in the local directory, by environment variables or directly to the cli.  

You can move the global config by setting the `RAG_HOME` environment variable.  

### Static Metadata Configuration

The ingestion command will automatically pick up metadata configured in `rag.toml` files present on the path between the working directory you are in when executing the command and the directory where the actual pdf file is stored. `rag.toml`-files closer to a pdf will override those further away if a metadata is specified more than once. example:
```
project dir/
  datasheets/
    .rag.toml <- specifies project-id
    subsystem-A/
      .rag.toml <- specifies subsystem, eventual arbitrary tags
      Manufacturer-A/
        .rag.toml <- specifies manufacturer
        MPN-A/
           .rag.toml <- mpn/tags
           datasheet.pdf
           reference-manual.pdf
           ...
        ...
      ...
    ...
```

This way you dont have to specify all metadata at runtime, and it is transferrable/reusable if you arent using one central server for a project.  

The format for `rag.toml` is as follows:
```
project_id   = "stm32-h7-devboard"
tags         = ["pinout", "power-tree", "reference-manual"]

[fields]
mpn          = ["STM32H743VIT6", "STM32H743ZIT6"]   # or a single string
manufacturer = "STMicroelectronics"
doc_type     = "reference-manual"
subsystem    = "power"           # per project: set in project_id
acme-id      = "D-0172"          # a field registered with `rag fields add`
```

`subsystem` is set under `[fields]`. The old top-level `group` and `subsystem` keys were removed; a file that still has them fails with an error that says what to write instead.

A document can belong to several projects. `project_id` adds the document to that project when it is ingested; the projects it was already in are kept. The same datasheet ingested from two project folders ends up in both projects. To leave a project, run `rag metadata <doc> --remove-project <project>` (`--add-project` adds one by hand).

`tags` are combined across every `.rag.toml` on the path. `[fields]` are merged name by name, and the file closest to the pdf wins for a given name. `--field name=value` (and the `--mpn`, `--manufacturer` and `--doc-type` shortcuts) on the command line win over both.

The schema is closed. Any of the following makes the command fail with an error that names the file:
- a key that is not listed above (any other value goes under `[fields]`)
- a value of the wrong type: `project_id` is a string, `tags` is a list of strings, and a `[fields]` value is a string or, for a list field such as `mpn`, a list of strings
- a file that is not valid TOML

A bad file inside a directory given to `rag ingest ./datasheets` fails only the pdfs below it; the rest are still ingested.

Whether a `[fields]` name is registered is checked when the values are written, not when the file is read. An unregistered name fails the write.

### Metadata fields

A field is a named, typed piece of document metadata. A field must be registered before any document can hold a value for it. You can filter on it with `rag list --field`, and agents can filter on it in the MCP `search` and `list_documents` tools. Four fields are built in: `mpn` (a list), `manufacturer`, `doc_type` and `subsystem` (one value per project, see below). Register your own for things the built-ins don't cover, such as an internal document number, a vendor's document number or an internal part number:

```
rag fields add acme-id --description "ACME document number"
rag fields add intel-id --description "Intel document number"
rag fields add acme-pn --shape list --role part
rag fields list
```

Set the values with `rag metadata <doc> --field acme-id=D-0172`. Repeat `--field` for a list field; `--mpn` is repeatable the same way. Remove a value with `--unset-field acme-id`. Filtering matches whole values and ignores case, and a list field matches if any of its values is equal. `rag list --columns mpn,acme-id` shows a column per field. The MCP tool `list_metadata_fields` tells an agent which fields exist.

`rag fields remove <name>` refuses while any document has a value for the field. `rag fields remove <name> --clear` deletes those values with the field. Built-in fields cannot be removed.

#### Fields with one value per project

Most fields hold one value per document. Some describe the role a document plays on one board. The built-in `subsystem` is one of these: one regulator datasheet can be `power` on one board and `aux` on another. These fields have the scope `project` and hold one value per (document, project):

```
rag metadata <doc> --project netdaq  --subsystem power
rag metadata <doc> --project board-b --subsystem aux
rag fields add board-ref --scope project     # your own per-project field
```

A write goes to the project named with `--project`. Without it, the `.rag.toml` `project_id` is used. If neither is given, the write fails. The project is never guessed from the projects the document is in. The document must be in that project. When a document leaves a project, its per-project values there are deleted.

`rag ingest --subsystem` writes to the ingest project. `rag list --subsystem` and `rag search --subsystem` match the value in the listed or searched project, or in any project with `--global`. The MCP `fields` filter does the same.

### Projects from a BOM

A project's BOM already says which parts the board uses. A BOM sync puts the documents for those parts into the project. Another system sends the project's whole BOM to the server, each time it changes:

```
POST /bom/sync          (an ingest key)
{"project": "netdaq",
 "lines": [{"ref": "U8,U9", "parts": {"acme-pn": "100-2345", "mpn": "TLA2518IRTER"}}],
 "dry_run": false}
```

The same from a CSV file, such as a KiCad BOM export:

```
rag bom sync bom.csv --project netdaq --column MPN=mpn --column "Internal PN=acme-pn" --ref-column Reference --dry-run
```

- A BOM line names a part by one or more part numbers. Each must be a field with the `part` role and one value per document: `mpn`, or your own, such as `rag fields add acme-pn --role part`.
- A line matches a document when one of its part numbers equals the document's value of that field, ignoring case. Matched documents join the project.
- Documents an earlier sync added that the BOM no longer matches leave the project, with their per-project values. Documents added by hand (`--add-project`, or at ingest) are never removed, so app notes and standards stay. To keep a document the BOM added, add it by hand: `rag metadata D12 --add-project netdaq`. The `.rag.toml` project does not do this: it only adds documents that are not in the project yet.
- A line that matches nothing is listed in the reply. When a field has the `mfr-part` role (`mpn` has), the line also gets *suggestions*: documents whose value is the start of the line's part number, such as `TLA2518` for `TLA2518IRTER`. A suggestion does not join the project. To accept it, add the BOM's part number to the document's list: `rag metadata D12 --mpn-alias TLA2518IRTER`. For another field, `--field` replaces the whole list, so name every value.
- Sending the same BOM twice changes nothing the second time. `dry_run` (`--dry-run`) reports what would change and writes nothing.

### Documents and revisions

A datasheet often comes in several revisions: Rev A, Rev B, Rev C. Each PDF is a *revision* with its own `rev_id` (the SHA-256 of the file). Revisions of one datasheet are grouped in a *document*, named `D12`. Every command that takes a document accepts either the `D12` or any of its revisions' `rev_id`s.

The `rev_id` used to be called `doc_id`. Opening an older store renames it in the database. Cached pipeline files on disk (chunk graphs, layouts, figure manifests) are not renamed automatically: a command that reads one stops and asks you to run `rag repair caches --apply`. Run it on each machine that ran the pipeline.

Metadata belongs to the document: projects, tags and field values are set once and every revision has them. A revision has its own label (`--revision "Rev C"`, free text), title and chunks.

Search, `rag list` and the MCP tools show only the newest revision of each document. Pass `--all-revisions` (MCP: `all_revisions=true`) to include the older ones. Commands that work on one PDF (`rag get page`, `show_pdf`, …) use the newest revision when given a `D12`.

```
rag ingest rev-c.pdf --new-revision-of D12 --revision "Rev C"   # newest revision of D12
rag revisions D12                          # list, newest first
rag revisions add D12 <rev_id>             # a stored PDF becomes D12's newest revision
rag revisions move <rev_id> --below <rev_id>
rag revisions unlink <rev_id>              # back into a document of its own
```

The order is the one you set; labels are never parsed. A new revision goes on top. `rag revisions add` merges the PDF's own document into `D12`: projects and tags are joined, and if both hold different values for a field it stops and names the field. `unlink` gives the PDF a new document with a copy of the old one's metadata. `rag delete <rev_id>` deletes one revision; `rag delete D12` deletes all of them. A document's metadata goes with its last revision.

To give one revision a different value, write it on that revision only. Listings mark such values with `*`:

```
rag metadata <rev_id> --this-revision --field doc_type=errata
rag metadata <rev_id> --this-revision --unset-field doc_type   # inherit again
```

Projects and per-project fields are always set on the whole document.

To change the whole document, name it by its `D12`. `rag metadata <rev_id>` without `--this-revision` stops with an error when it would change fields, tags or projects of a document with more than one revision, because a `rev_id` may mean only that PDF. `--title` and `--revision` are always the revision's own; on such a `rev_id` they do not pick up the `.rag.toml` project, tags or fields. For a document with one revision, the `rev_id` and the `D12` mean the same thing.

#### The identity rule

Without help nothing links on its own: an ingested PDF becomes a new document unless `--new-revision-of` names one. A store can hold one *identity rule* that does the linking at ingest:

```
rag revisions rule set "intel-id OR ti-id OR (acme-pn AND doc_type)"
rag revisions rule show
rag revisions rule clear
```

The rule is an expression over registered, document-scoped fields, with `AND`, `OR` and parentheses (`AND` binds tighter). When a PDF is ingested without `--new-revision-of`, the field values the ingest sets are compared with every document's values. A field matches only when both have it and the values are equal (case-insensitive); a list field must hold the same values, not just share one. One matching document: the PDF becomes its newest revision. None: a new document. Two or more: the ingest stops before parsing and names them.

There is no default rule. `manufacturer + mpn + doc_type` would fold together documents that are not revisions of each other (one Intel CPU has hundreds of datasheets, errata and guides with the same values). The ingest output always says what happened (`linked to D12 as newest revision (rule: intel-id)`, `new document D14`), so a wrong link is seen at once and fixed with `rag revisions unlink`. Setting or changing the rule does not relink documents already stored on its own.

To relink stored documents by the rule, run:

```
rag revisions apply-rule --dry-run   # print what would be linked
rag revisions apply-rule
```

Matching works as at ingest. A group of matching documents is linked into one only when every pair in the group matches (a chain where A matches B and B matches C, but A does not match C, is skipped), at most one of them already has several revisions, and no field holds different values in two of them. Skipped groups are listed with the reason. A linked document holds the fields of all its members, so it can match a document none of them matched alone; matching repeats until nothing more links.

**The order of a linked document is a guess.** It comes from ingest time, which is when the PDF was first ingested into this store, not when it was published. A document that already had several revisions keeps its order, and its newest stays on top. Each other PDF goes below that top, just above the highest revision ingested before it (at the bottom if there is none). A revision with no ingest time is passed over. Otherwise all PDFs are ordered by ingest time, the latest ingested as newest. The top revision is the one search shows. Every linked document is listed so you can check its order and fix it with `rag revisions move`.

## Usage

### Ingestation
To ingest a new PDF into the system, you use 
```
rag ingest </path/to/datasheet.pdf>
```
There are several options to this command that are documented in the help-text. the options for configuring the ingestion parameters are generally sane and shouldnt need to be touched generally, in addition you can set metadata directly when ingesting instead of doing it later, example:  
```
rag ingest tlv6722.pdf --project-id 'test-project' --subsystem 'ethernet phy' --manufacturer 'Texas Instruments' --mpn 'TLV6722' 
```
any metadata specified in any `rag.toml` files found will also be applied, although those specified as cli parameters will override the file. 

You can also recursively ingest all pdfs found by specifying a directory to the command such as 
```
rag ingest ./datasheets
```   
