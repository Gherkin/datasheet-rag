"""Authentication & authorization for the RAG server.

Two tiers:

* a **shared read token** (``RAG_SERVER_READ_TOKEN`` / legacy ``RAG_SERVER_TOKEN``)
  that grants the ``read`` scope to everyone who holds it; and
* **per-client API keys** (stored hashed in the ``api_keys`` table) that carry
  ``ingest`` and/or ``admin`` scopes for cost-inducing and management calls.

When neither a read token nor any API key exists the server runs in *open
mode* (trusted-LAN dev) and every request is allowed — matching the historical
default. The first thing this module does on a guarded request is decide, in
constant time, which scope a presented bearer token carries.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from datasheet_rag.store import lookup_api_key

if TYPE_CHECKING:
    import sqlite3

    from datasheet_rag.config import Settings


class Scope(IntEnum):
    """Ordered scopes — a higher scope implies every lower one."""

    READ = 1
    INGEST = 2
    ADMIN = 3


_SCOPE_BY_NAME = {s.name.lower(): s for s in Scope}


def parse_scopes(names: list[str]) -> set[Scope]:
    """Map scope name strings to the implied set (admin ⊇ ingest ⊇ read)."""
    highest = Scope.READ
    seen = False
    for n in names:
        s = _SCOPE_BY_NAME.get(n.lower().strip())
        if s is not None:
            highest = max(highest, s)
            seen = True
    if not seen:
        return set()
    return {s for s in Scope if s <= highest}


@dataclass(frozen=True)
class KeyContext:
    """The authenticated identity attached to a request."""

    label: str
    scopes: frozenset[Scope]

    def allows(self, scope: Scope) -> bool:
        return scope in self.scopes

    @classmethod
    def anonymous(cls) -> KeyContext:
        return cls(label="anonymous", scopes=frozenset(Scope))


def _bearer(authorization: str | None) -> str | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[len("Bearer ") :].strip() or None


def _read_token_matches(token: str, expected: str) -> bool:
    return hmac.compare_digest(token.encode(), expected.encode())


def auth_enabled(conn: sqlite3.Connection, settings: Settings) -> bool:
    """True when any credential is configured (read token or any API key)."""
    if settings.effective_read_token():
        return True
    from datasheet_rag.store import count_api_keys

    return count_api_keys(conn) > 0


def resolve_context(
    conn: sqlite3.Connection, settings: Settings, token: str | None
) -> KeyContext | None:
    """Resolve a presented bearer token to a :class:`KeyContext`, else ``None``."""
    if token is None:
        return None
    read_token = settings.effective_read_token()
    if read_token and _read_token_matches(token, read_token):
        return KeyContext(label="shared-read", scopes=frozenset({Scope.READ}))
    rec = lookup_api_key(conn, token)
    if rec is not None:
        scopes = parse_scopes(rec.scopes)
        if scopes:
            return KeyContext(label=rec.label, scopes=frozenset(scopes))
    return None
