#!/usr/bin/env bash
# Build aws-rag.mcpb — a self-contained MCPB bundle.
#
# Usage:
#   cd mcp-bundle && ./pack.sh
#   # or from the project root:
#   ./mcp-bundle/pack.sh
#
# Output: aws-rag.mcpb in the project root.
#
# Steps:
#   1. Copy the aws_rag source tree into server/src/ so the bundle is
#      self-contained and does not require a local editable install.
#   2. Zip manifest.json + server/ into a .mcpb archive.
#   3. Clean up the temporary server/src/ copy.
#
# Requirements: uv, zip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC="${PROJECT_ROOT}/src/aws_rag"
BUNDLE_SRC="${SCRIPT_DIR}/server/src/aws_rag"
OUTPUT="${PROJECT_ROOT}/aws-rag.mcpb"

# --------------------------------------------------------------------------
# Validate
# --------------------------------------------------------------------------
if [[ ! -d "${SRC}" ]]; then
    echo "ERROR: source directory not found: ${SRC}" >&2
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv is not installed. Install it from https://github.com/astral-sh/uv" >&2
    exit 1
fi

if ! command -v zip &>/dev/null; then
    echo "ERROR: zip is not installed." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Copy source into bundle
# --------------------------------------------------------------------------
echo "Copying source → server/src/aws_rag ..."
rm -rf "${SCRIPT_DIR}/server/src"
mkdir -p "${SCRIPT_DIR}/server/src"
cp -r "${SRC}" "${BUNDLE_SRC}"

# Strip __pycache__ and .pyc files to keep the archive lean.
find "${BUNDLE_SRC}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${BUNDLE_SRC}" -name "*.pyc" -delete 2>/dev/null || true

# --------------------------------------------------------------------------
# Validate the uv project can be resolved (deps check)
# --------------------------------------------------------------------------
echo "Checking uv dependency resolution ..."
uv lock --project "${SCRIPT_DIR}/server" --check 2>/dev/null \
    || uv lock --project "${SCRIPT_DIR}/server"

# --------------------------------------------------------------------------
# Pack the bundle
# --------------------------------------------------------------------------
echo "Creating ${OUTPUT} ..."
rm -f "${OUTPUT}"

cd "${SCRIPT_DIR}"
zip -r "${OUTPUT}" \
    manifest.json \
    server/main.py \
    server/pyproject.toml \
    server/uv.lock \
    server/src/ \
    ${ICON_ARG:-}

# --------------------------------------------------------------------------
# Clean up
# --------------------------------------------------------------------------
rm -rf "${SCRIPT_DIR}/server/src"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
SIZE=$(du -sh "${OUTPUT}" | cut -f1)
echo ""
echo "Bundle created: ${OUTPUT} (${SIZE})"
echo ""
echo "Install in Claude for macOS by double-clicking the .mcpb file."
echo "Or add to claude_desktop_config.json manually:"
echo ""
echo "  mcpb install ${OUTPUT}"
