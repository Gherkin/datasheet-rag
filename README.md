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
manufacturer = "STMicroelectronics"
group        = "psu"
subsystem    = "power"
mpn          = "STM32H743VIT6"
tags         = ["pinout", "power-tree", "reference-manual"]
```

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
