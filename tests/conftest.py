"""Test isolation from the developer's machine configuration.

Settings layer a global ``<RAG_HOME>/config.env`` (default ``~/.rag/config.env``)
on top of process env. A real machine may set ``RAG_EMBEDDING_BACKEND=local``,
``RAG_SQLITE_DB_PATH=...`` etc. there — none of which should bleed into the test
suite. Pointing ``RAG_HOME`` at a throwaway temp dir *before* ``datasheet_rag.config``
is imported makes the layered env_file resolve to a nonexistent file, so tests
run against field defaults (and never touch the real store).
"""

from __future__ import annotations

import os
import tempfile

# Must run at import time, before datasheet_rag.config computes its RAG_HOME default.
_TEST_RAG_HOME = tempfile.mkdtemp(prefix="datasheet-rag-tests-")
os.environ["RAG_HOME"] = _TEST_RAG_HOME
