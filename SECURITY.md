# Security policy

## Reporting a vulnerability

Please report security issues privately, not as a public issue.

Use GitHub's [private vulnerability
reporting](https://github.com/Gherkin/datasheet-rag/security/advisories/new) —
it opens a draft advisory that only the maintainers can see.

Include enough to reproduce: version or commit, how the server was configured
(auth mode, whether it sits behind a proxy), and the request or input that
triggers the problem.

Expect an acknowledgement within a week. This is a small project maintained by
one person in their own time, so please treat that as a best effort rather than
a commitment. You will be credited in the advisory unless you would rather not
be.

## Supported versions

The project has not cut a release yet. Fixes land on `main`, and the
`ghcr.io/gherkin/datasheet-rag` images are rebuilt from it. If you are running
an older image, updating is the fix.

## What is in scope

The parts of this project that make security decisions:

- **The HTTP server's authentication** (`src/datasheet_rag/server/auth.py`) —
  the shared read token, per-client API keys, and the `read` ⊂ `ingest` ⊂
  `admin` scope ladder. Anything that lets a caller act beyond the scope its
  credential carries, or that leaks a token or key.
- **Key storage** — API keys are stored hashed; the plaintext is shown once at
  creation. Anything that recovers a key from the database is in scope.
- **The audit log** (`src/datasheet_rag/server/audit.py`) — a way to perform an
  audited action without leaving a row, or to forge one.
- **Path handling on ingest and retrieval** — reaching a file outside the
  configured store via a document id, chunk id, or figure path.
- **The `/mcp` endpoint** — it takes the same credentials as the REST API, so
  anything that reaches a tool without them.

## What is not

- **Open mode.** With no read token and no API keys configured, the server
  allows every request, by design — it is the trusted-LAN default and it says
  so on startup. That it is open is not a vulnerability; a way *past* configured
  credentials is.
- **DNS-rebinding protection being off by default.** The MCP `Host` allowlist
  is opt-in (`RAG_MCP_ALLOWED_HOSTS`) because this server is normally reached
  by several names and through a proxy. See the README for the reasoning.
- **The loopback PDF viewer** (`show_pdf`). It serves the PDF from an
  unauthenticated port on the machine running the MCP server, for a local
  single-user process. It is deliberately not exposed by the HTTP server.
- **Costs.** A credential with the `ingest` scope can run up a Bedrock or
  Textract bill. That is what the scope means — hand it out accordingly.
- **Vulnerabilities in dependencies** with no exploitable path here. Report
  those upstream; if there *is* a path through this project, that path is in
  scope and we want to hear about it.
