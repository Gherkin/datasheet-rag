"""MCP Bundle entry point for aws-rag.

Adds the bundle's src/ directory to sys.path so the aws_rag package is found
whether this is run from the extracted bundle (src/ populated by pack.sh) or
directly from the development tree.
"""

import sys
from pathlib import Path

# When running from an extracted .mcpb, the aws_rag source lives in src/
# next to this file (populated by pack.sh). When running from the dev tree
# (mcp-bundle/server/), the project src is two levels up.
_here = Path(__file__).parent.resolve()
for candidate in [_here / "src", _here.parent.parent / "src"]:
    if (candidate / "aws_rag").is_dir():
        sys.path.insert(0, str(candidate))
        break

from aws_rag.mcp.server import main  # noqa: E402

if __name__ == "__main__":
    main()
