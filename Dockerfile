# datasheet-rag HTTP server image.
#
# Runs `rag-server` (FastAPI/uvicorn) wrapping the sqlite store + embedder.
# The database, source PDFs, cropped figures AND the HuggingFace model cache
# all live under /data, so a single volume is all the persistence you need:
#
#   docker run -d --name rag -p 8080:8080 -v rag-data:/data \
#     ghcr.io/gherkin/datasheet-rag:cpu
#
# Published variants (see .github/workflows/docker-publish.yml):
#   :cpu    TORCH_VARIANT=cpu   — PyTorch CPU wheels. No nvidia-* payload.
#   :cuda   TORCH_VARIANT=cuda  — PyTorch CUDA wheels, for `docker run --gpus all`.
# Both carry the same extras, so every backend in the README is reachable by
# setting environment variables at run time — there is no separate "bedrock"
# or "local" image to choose between.

FROM python:3.11-slim

# poppler-utils: pdf2image (show_page / page rendering).
# gcc + libc6-dev: torch's inductor/Triton backend JIT-compiles a small C stub
# the first time the docling layout models run on the GPU. Without a compiler
# on PATH the server-side parse dies with "Failed to find C compiler"; without
# libc6-dev it gets one step further and dies on "stdlib.h: No such file or
# directory" — Debian's gcc only *Recommends* libc6-dev, and we build with
# --no-install-recommends, so the headers must be named explicitly.
# libsqlite3-0: the sqlite-vec loadable extension (the wheel ships a prebuilt
# .so, but having the runtime lib avoids a load failure on some bases).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        gcc \
        libc6-dev \
        libsqlite3-0 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Which torch wheels to install. "cpu" pulls them from PyTorch's CPU-only index
# BEFORE the package install, so the later resolve sees torch already satisfied
# and never reaches for PyPI's default build — the one that drags in ~2.5GB of
# nvidia-* CUDA libraries a CPU-only host will never execute. "cuda" skips this
# step and lets the normal PyPI wheels (CUDA-enabled) come in as dependencies.
# torch is NOT optional here: the `docling` extra depends on it transitively
# (docling -> docling-slim[standard] -> torch), so it lands in the image
# whether or not `local-hf` is selected.
ARG TORCH_VARIANT=cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TORCH_VARIANT" = "cpu" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
    fi

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Which optional-dependency extras to install. The default installs every
# extra the server can use, because the choice between them is a RUNTIME one
# (RAG_EMBEDDING_BACKEND / RAG_TEXT_BACKEND / RAG_VISION_BACKEND):
#   server   – FastAPI/uvicorn, the HTTP layer itself
#   aws      – boto3 clients for the Bedrock text/vision/embedding backends
#   docling  – server-side PDF parse behind POST /ingest-pdf; without it a
#              remote `rag ingest` fails with "No module named 'docling'"
#   local-hf – in-process sentence-transformers/bge-m3 embeddings + local VLMs
# Trim it for a smaller private build, e.g. RAG_EXTRAS=server,aws for a
# query-only node that never ingests.
ARG RAG_EXTRAS=server,aws,docling,local-hf
# A BuildKit cache mount keeps pip's download cache (torch is ~800MB) across
# builds. `COPY src` above invalidates this layer on any source change, but
# with the cache the reinstall reuses already-downloaded wheels instead of
# re-fetching them — turning a one-line code change from a multi-minute
# re-download into a quick reinstall. (Dropped --no-cache-dir so the mount is
# actually populated; the cache lives in the builder, not the final image.)
RUN --mount=type=cache,target=/root/.cache/pip pip install ".[${RAG_EXTRAS}]"

# HF_HOME under /data is what makes a single volume sufficient: the bge-m3
# download (~2GB) persists next to the database instead of needing its own
# mount at /root/.cache/huggingface. Everything stateful is now under /data.
ENV RAG_HOME=/data \
    HF_HOME=/data/hf-cache \
    RAG_SERVER_HOST=0.0.0.0 \
    RAG_SERVER_PORT=8080

# Run as a non-root user. A *named volume* inherits this ownership from the
# image, so the default `-v rag-data:/data` just works. A *bind mount* keeps
# the host directory's owner instead — run those with `--user $(id -u):$(id -g)`
# so the server can write to them.
RUN useradd --create-home --uid 1000 rag \
    && mkdir -p /data \
    && chown -R rag:rag /data
USER rag

VOLUME ["/data"]

EXPOSE 8080

# Healthcheck hits the unauthenticated /health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status==200 else 1)"

# ENTRYPOINT (not CMD) so the image forwards arguments to the `rag-server`
# console script. That puts its subcommands one `docker run` away, with no
# compose file and no long-running container to `exec` into:
#   docker run --rm -v rag-data:/data ghcr.io/gherkin/datasheet-rag:cpu \
#       create-key --label bootstrap --scope admin
# For a shell, override it: `docker run --rm -it --entrypoint bash <image>`.
ENTRYPOINT ["rag-server"]
CMD []
