#!/usr/bin/env bash
# Registers the aws-rag MCP server with Claude Code (user scope, available in
# every project), mirroring the configuration options of the aws-rag.mcpb
# Claude Desktop extension. Run once: ./scripts/install-mcp.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_DIR="$REPO_DIR/mcp-bundle/server"

if ! command -v claude >/dev/null 2>&1; then
    echo "error: 'claude' CLI not found on PATH" >&2
    exit 1
fi

prompt() {
    local var_name="$1" message="$2" default="${3-}"
    local input
    if [ -n "$default" ]; then
        read -r -p "$message [$default]: " input
        printf -v "$var_name" '%s' "${input:-$default}"
    else
        read -r -p "$message: " input
        printf -v "$var_name" '%s' "$input"
    fi
}

prompt_secret() {
    local var_name="$1" message="$2"
    local input
    read -r -s -p "$message: " input
    echo
    printf -v "$var_name" '%s' "$input"
}

echo "aws-rag MCP setup"
echo "================="
echo "Choose a mode:"
echo "  1) Remote — talk to a shared aws-rag server over HTTP"
echo "  2) Local  — open a SQLite database on this machine directly"
read -r -p "Mode [1]: " mode
mode="${mode:-1}"

env_args=()

if [ "$mode" = "1" ]; then
    prompt SERVER_URL "Server URL (e.g. https://rag.internal:8080)"
    if [ -z "$SERVER_URL" ]; then
        echo "error: Server URL is required for remote mode" >&2
        exit 1
    fi
    env_args+=(-e "RAG_SERVER_URL=$SERVER_URL")

    prompt_secret SERVER_TOKEN "Access Token (leave blank if not required)"
    if [ -n "$SERVER_TOKEN" ]; then
        env_args+=(-e "RAG_SERVER_TOKEN=$SERVER_TOKEN")
    fi
else
    prompt DB_PATH "SQLite Database Path (e.g. /path/to/output/rag.sqlite)"
    if [ -z "$DB_PATH" ]; then
        echo "error: SQLite Database Path is required for local mode" >&2
        exit 1
    fi
    env_args+=(-e "RAG_SQLITE_DB_PATH=$DB_PATH")

    prompt S3_BUCKET "S3 Bucket (for PDF storage / Textract, optional)"
    if [ -n "$S3_BUCKET" ]; then
        env_args+=(-e "RAG_S3_BUCKET=$S3_BUCKET")
    fi

    prompt AWS_REGION "AWS Region" "eu-west-1"
    env_args+=(-e "AWS_REGION=$AWS_REGION")

    prompt FIGURES_DIR "Figures Directory (optional, defaults to <db_path parent>/figures)"
    if [ -n "$FIGURES_DIR" ]; then
        env_args+=(-e "RAG_FIGURES_DIR=$FIGURES_DIR")
    fi
fi

prompt DEFAULT_PROJECT_ID "Default Project ID (optional, leave blank to search globally)"
if [ -n "$DEFAULT_PROJECT_ID" ]; then
    env_args+=(-e "RAG_DEFAULT_PROJECT_ID=$DEFAULT_PROJECT_ID")
fi

claude mcp remove aws-rag -s user >/dev/null 2>&1 || true

claude mcp add aws-rag -s user "${env_args[@]}" -- \
    uv run --project "$SERVER_DIR" python "$SERVER_DIR/main.py"

echo
echo "Registered 'aws-rag' as a user-scope MCP server."
echo "Restart Claude Code (or run /mcp) to connect."
