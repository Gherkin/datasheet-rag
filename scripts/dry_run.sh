#!/usr/bin/env bash
# End-to-end dry-run for the aws-rag pipeline.
#
# What this does:
#   1. Installs the project (editable, with dev extras).
#   2. Runs the test suite (in-memory SQLite + mocked Bedrock — no AWS hit).
#   3. Embeds an existing chunks.json against Bedrock Titan v2.
#   4. Sets some document metadata.
#   5. Runs hybrid / vector / keyword searches.
#   6. Launches the MCP server smoke test (scripts/mcp_smoke.py).
#
# Requires: AWS credentials with Bedrock access (Titan Embed Text v2 must
# be enabled in your account & region — eu-west-1 by default).
#
# Run from the repo root:
#   bash scripts/dry_run.sh
#
# Override the chunks file or project ID via env:
#   CHUNKS_JSON=output/foo_chunks.json PROJECT_ID=test bash scripts/dry_run.sh

set -euo pipefail

# Pick the first chunks file under output/ unless one was passed in.
CHUNKS_JSON="${CHUNKS_JSON:-$(ls output/*_chunks.json 2>/dev/null | head -n1 || true)}"
PROJECT_ID="${PROJECT_ID:-dryrun}"

if [[ -z "${CHUNKS_JSON}" ]]; then
    echo "ERROR: no output/*_chunks.json found. Run 'rag chunk' first."
    exit 1
fi

DOC_ID="$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d['doc_id'])
" "${CHUNKS_JSON}")"

banner() {
    printf "\n\033[1;34m== %s ==\033[0m\n" "$*"
}

banner "1. Install (editable + dev extras)"
pip install -e ".[dev]" --quiet
echo "OK"

banner "2. Run tests"
pytest tests/ -q

banner "3. Embed (Bedrock Titan v2)"
echo "doc_id     = ${DOC_ID}"
echo "project_id = ${PROJECT_ID}"
echo "source     = ${CHUNKS_JSON}"
rag embed "${CHUNKS_JSON}" --project-id "${PROJECT_ID}"

banner "4. Set document metadata"
rag metadata set "${DOC_ID}" \
    --project-id "${PROJECT_ID}" \
    --doc-type datasheet \
    --tag dryrun

banner "5a. Hybrid search"
rag search "specification" --project-id "${PROJECT_ID}" -k 3

banner "5b. Keyword-only search (good for exact identifiers)"
rag search "register" --project-id "${PROJECT_ID}" --mode keyword -k 3

banner "5c. Vector-only search (good for conceptual queries)"
rag search "thermal characteristics" --project-id "${PROJECT_ID}" --mode vector -k 3

banner "6. MCP server smoke test"
export RAG_DEFAULT_PROJECT_ID="${PROJECT_ID}"
python3 scripts/mcp_smoke.py

banner "Done"
echo "Dry-run completed without errors."
echo "DB: $(python3 -c 'from aws_rag.config import get_settings; print(get_settings().sqlite_db_path)')"
