"""Control-plane store helpers: API keys and the audit log.

These operate on the same sqlite connection as the chunk store (tables are
created by :func:`datasheet_rag.store.schema._ensure_control_tables`). Functions are
intentionally small module-level helpers, mirroring the rest of ``store``.

Tokens are never stored in plaintext — only their SHA-256 hash. The plaintext
is generated and returned once by :func:`create_api_key` and never again.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime


def hash_token(token: str) -> str:
    """Return the hex SHA-256 of a bearer token (the at-rest representation)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApiKeyRecord:
    id: str
    label: str
    scopes: list[str]
    created_at: str | None
    revoked_at: str | None

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None


def _row_to_record(row: sqlite3.Row) -> ApiKeyRecord:
    return ApiKeyRecord(
        id=row["id"],
        label=row["label"],
        scopes=json.loads(row["scopes"]),
        created_at=row["created_at"],
        revoked_at=row["revoked_at"],
    )


def create_api_key(
    conn: sqlite3.Connection, *, label: str, scopes: list[str]
) -> tuple[ApiKeyRecord, str]:
    """Mint a new API key.

    Returns the stored record and the **plaintext token** (shown once — the
    caller must surface it immediately; only its hash is persisted).
    """
    token = secrets.token_urlsafe(32)
    key_id = secrets.token_hex(6)
    conn.execute(
        "INSERT INTO api_keys(id, label, token_sha256, scopes) VALUES (?, ?, ?, ?)",
        (key_id, label, hash_token(token), json.dumps(scopes)),
    )
    conn.commit()
    rec = ApiKeyRecord(
        id=key_id, label=label, scopes=scopes, created_at=None, revoked_at=None
    )
    return rec, token


def lookup_api_key(conn: sqlite3.Connection, token: str) -> ApiKeyRecord | None:
    """Resolve a presented token to a non-revoked key record, else ``None``."""
    row = conn.execute(
        "SELECT * FROM api_keys WHERE token_sha256 = ?", (hash_token(token),)
    ).fetchone()
    if row is None:
        return None
    rec = _row_to_record(row)
    return None if rec.revoked else rec


def list_api_keys(
    conn: sqlite3.Connection, *, include_revoked: bool = True
) -> list[ApiKeyRecord]:
    sql = "SELECT * FROM api_keys"
    if not include_revoked:
        sql += " WHERE revoked_at IS NULL"
    sql += " ORDER BY created_at"
    return [_row_to_record(r) for r in conn.execute(sql).fetchall()]


def revoke_api_key(conn: sqlite3.Connection, key_id: str) -> bool:
    """Revoke a key by id. Returns True if a live key was revoked."""
    cur = conn.execute(
        "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
        (datetime.now(UTC).isoformat(), key_id),
    )
    conn.commit()
    return cur.rowcount > 0


def count_api_keys(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0])


def record_audit(
    conn: sqlite3.Connection,
    *,
    action: str,
    status: str,
    key_label: str | None = None,
    client_ip: str | None = None,
    doc_id: str | None = None,
    project_id: str | None = None,
    detail: dict | None = None,
    error: str | None = None,
) -> None:
    """Append one row to the audit log."""
    conn.execute(
        """
        INSERT INTO audit_log(
            ts, key_label, client_ip, action, doc_id, project_id,
            detail_json, status, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(UTC).isoformat(),
            key_label,
            client_ip,
            action,
            doc_id,
            project_id,
            json.dumps(detail) if detail is not None else None,
            status,
            error,
        ),
    )
    conn.commit()


def list_audit(
    conn: sqlite3.Connection,
    *,
    doc_id: str | None = None,
    since: str | None = None,
    limit: int = 200,
) -> list[dict]:
    sql = "SELECT * FROM audit_log"
    clauses, params = [], []
    if doc_id:
        clauses.append("doc_id = ?")
        params.append(doc_id)
    if since:
        clauses.append("ts >= ?")
        params.append(since)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]
