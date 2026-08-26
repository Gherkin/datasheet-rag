# Contributing

Thanks for looking. This is a small project, so the process is short: open an
issue if you want to talk it through first, otherwise send a pull request.

## Setting up

Development uses [uv](https://docs.astral.sh/uv/). `uv.lock` is committed, so
one command gets you the exact environment CI uses:

```bash
git clone https://github.com/Gherkin/datasheet-rag
cd datasheet-rag
uv sync --locked --extra dev --extra server --extra aws --extra docling
```

That is CI's set. `dev` brings the tooling (pytest, ruff, mypy, boto3-stubs);
`server` brings FastAPI, which the server tests exercise through its
TestClient; `aws` brings botocore, which two tests import directly; `docling`
is there for pymupdf, which two test files import as `fitz` at module scope.

Add `--extra local-hf` if you are working on the in-process HuggingFace
backends. It is several gigabytes of torch wheels, and nothing in the test
suite needs it — every import of it in `src/` is lazy and guarded — so CI
leaves it out.

For what the extras mean in a *user* install rather than a dev one, see the
[README](README.md#installation).

The tree also carries a `.git-blame-ignore-revs` for whole-tree mechanical
rewrites. Enable it once per clone so `git blame` skips them:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Running the checks

The four gates CI enforces, in the order it runs them:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest
```

All four are expected to pass on `main`. `ruff format .` (without `--check`)
applies the formatting, and `ruff check --fix .` applies the mechanical lint
fixes.

Ruff's rule selection, the 100-column limit, and mypy's strict mode all live in
`pyproject.toml` — there is no separate config to keep in sync.

## Tests

`pytest` runs the whole suite in a few seconds. It is fully mocked: no AWS
credentials, no network, no PDFs fetched. Please keep it that way — a test that
needs either should mock at the boundary the existing tests use, or be marked
so it can be skipped.

The MCP tools are split into a thin `@mcp.tool()` wrapper over an `_impl`
function precisely so tests can drive the `_impl` layer without a transport.
Follow that shape when adding one.

`tests/conftest.py` points `RAG_HOME` at a throwaway directory at import time,
so the suite never reads your real `~/.rag/config.env` or touches your store.

## Type annotations

`mypy --strict` covers `src/`. Two things in it are deliberate rather than
accidental, and both carry a comment where they sit:

- Optional third-party modules that arrive untyped, or that live behind the
  `local-hf` extra, are listed in `pyproject.toml`'s mypy overrides.
- Five MCP tools that return content blocks carry no return annotation.
  Annotating one makes FastMCP publish an `outputSchema` and start returning
  `structuredContent`, which changes what the tool puts on the wire.

## Pull requests

- Keep unrelated reformatting out of the diff — `ruff format` on a file you did
  not otherwise touch buries the change you actually made.
- Commit messages: a short imperative subject, and a body explaining *why* when
  the reason is not obvious from the diff. `git log` has the house style.
- If you fix a bug, add the test that would have caught it.

## Security

Please do not open a public issue for a security problem. See
[SECURITY.md](SECURITY.md).
