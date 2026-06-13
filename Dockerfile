# RAG HTTP server image.
#
# Runs `rag-server` (FastAPI/uvicorn) wrapping the sqlite store + embedder.
# The database, source PDFs and cropped figures all live under /data, which
# is expected to be a bind-mounted / named volume so they survive container
# recreation (see docker-compose.yml).
#
# Build a Bedrock-backed server (no GPU, embeddings via AWS):
#   docker build -t aws-rag-server .
# For fully local embeddings/vision instead, add the local-hf extra below
# and provide a GPU (heavy — pulls torch). Bedrock is the simplest default.

FROM python:3.11-slim

# poppler-utils: pdf2image (show_page / page rendering).
# build-essential + libsqlite3-dev: belt-and-suspenders for the sqlite-vec
# loadable extension under slim (the wheel ships a prebuilt .so, but having
# the dev libs avoids a load failure on some bases).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        poppler-utils \
        libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

# Default: server + Bedrock embeddings + docling (for any server-side parsing).
# Swap to '.[server,local,local-hf]' (and a GPU runtime) for local embeddings.
RUN pip install --no-cache-dir '.[server,docling]'

ENV RAG_HOME=/data \
    RAG_SERVER_HOST=0.0.0.0 \
    RAG_SERVER_PORT=8080

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8080

# Healthcheck hits the unauthenticated /health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status==200 else 1)"

CMD ["rag-server"]
