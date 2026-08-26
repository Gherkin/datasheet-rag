#!/usr/bin/env bash
# Registers the datasheet-rag MCP server with Claude Code (user scope, available in
# every project), mirroring the configuration options of the datasheet-rag.mcpb
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

echo "datasheet-rag MCP setup"
echo "================="
echo "Choose a mode:"
echo "  1) Server  — connect straight to a datasheet-rag server's /mcp endpoint"
echo "  2) Local   — open a SQLite database on this machine directly"
echo "  3) Bridge  — run a local stdio server that forwards to a remote server"
echo "               (only for clients that cannot speak HTTP MCP)"
read -r -p "Mode [1]: " mode
mode="${mode:-1}"

env_args=()

# Mode 1 needs no local process at all: the server hosts the MCP endpoint, so
# Claude Code just holds a URL and a token. Registered and done, well before
# the stdio plumbing the other two modes share below.
if [ "$mode" = "1" ]; then
    prompt SERVER_URL "Server URL (e.g. https://rag.internal:8080)"
    if [ -z "$SERVER_URL" ]; then
        echo "error: Server URL is required" >&2
        exit 1
    fi
    prompt DEFAULT_PROJECT_ID "Default Project ID (optional, leave blank to search globally)"

    # The project id is a path segment on the endpoint — the remote stand-in
    # for the .rag.toml a local server would find in your checkout.
    url="${SERVER_URL%/}/mcp"
    if [ -n "$DEFAULT_PROJECT_ID" ]; then
        url="$url/$DEFAULT_PROJECT_ID"
    fi

    header_args=()
    prompt_secret SERVER_TOKEN "Access Token (leave blank if the server is open)"
    if [ -n "$SERVER_TOKEN" ]; then
        header_args+=(--header "Authorization: Bearer $SERVER_TOKEN")
    fi

    claude mcp remove datasheet-rag -s user >/dev/null 2>&1 || true
    claude mcp add --transport http datasheet-rag -s user "${header_args[@]}" "$url"

    echo
    echo "Registered 'datasheet-rag' at $url (user scope)."
    echo "Restart Claude Code (or run /mcp) to connect."
    exit 0
fi

if [ "$mode" = "3" ]; then
    prompt SERVER_URL "Server URL (e.g. https://rag.internal:8080)"
    if [ -z "$SERVER_URL" ]; then
        echo "error: Server URL is required for bridge mode" >&2
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

claude mcp remove datasheet-rag -s user >/dev/null 2>&1 || true

claude mcp add datasheet-rag -s user "${env_args[@]}" -- \
    uv run --project "$SERVER_DIR" python "$SERVER_DIR/main.py"

echo
echo "Registered 'datasheet-rag' as a user-scope MCP server."
echo "Restart Claude Code (or run /mcp) to connect."
