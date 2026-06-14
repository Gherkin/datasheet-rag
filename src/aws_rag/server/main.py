"""Console-script entry point for the RAG HTTP server (``rag-server``).

Subcommands:

* (no args)      – run the uvicorn server.
* ``create-key`` – mint an API key directly in the DB (bootstrap the first
                   admin/ingest key before any admin key exists).
* ``tls-setup``  – obtain a Let's Encrypt cert via the DNS-01 challenge
                   (works for a LAN-only server with no inbound internet).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from aws_rag.server.app import build_app

logger = logging.getLogger("aws_rag.server")

# Module-level app so `uvicorn aws_rag.server.main:app` works directly.
app = build_app()


def _startup_posture() -> str:
    """One-line description of the server's auth + CORS posture, for the log."""
    from aws_rag.config import get_settings
    from aws_rag.server.deps import get_backend
    from aws_rag.store import count_api_keys

    s = get_settings()
    be = get_backend()
    n_keys = count_api_keys(be.conn)
    has_read = bool(s.effective_read_token())
    if not has_read and n_keys == 0:
        mode = "OPEN (no read token, no API keys — all requests allowed)"
    else:
        parts = ["shared read token set" if has_read else "no read token"]
        parts.append(f"{n_keys} API key(s)")
        mode = ", ".join(parts)
    cors = s.cors_origins_list()
    cors_desc = f"CORS allowlist={cors}" if cors else "CORS locked (no origins)"
    return f"auth: {mode}; {cors_desc}"


def _run_server() -> None:
    import uvicorn

    host = os.environ.get("RAG_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("RAG_SERVER_PORT", "8080"))
    logging.basicConfig(level=logging.INFO)
    logger.info("rag-server starting on %s:%s — %s", host, port, _startup_posture())
    uvicorn.run(app, host=host, port=port, log_level="info")


def _cmd_create_key(args: argparse.Namespace) -> int:
    from aws_rag.server.deps import get_backend
    from aws_rag.store import create_api_key

    be = get_backend()
    with be.write_lock:
        rec, token = create_api_key(be.conn, label=args.label, scopes=args.scope)
    print(f"Created API key '{rec.label}' (id={rec.id}, scopes={rec.scopes})")
    print("\nToken (shown once — store it now, it is not recoverable):\n")
    print(f"  {token}\n")
    return 0


def _cmd_tls_setup(args: argparse.Namespace) -> int:
    """Obtain a cert via DNS-01. Wraps certbot's manual (copy-paste) flow, or
    the Cloudflare-automated flow when a token is supplied."""
    import shutil
    import subprocess

    certbot = shutil.which("certbot")
    if certbot is None:
        print(
            "certbot not found. Install it (e.g. `pip install certbot "
            "certbot-dns-cloudflare` or your distro package) and re-run.",
            file=sys.stderr,
        )
        return 1

    cert_dir = args.cert_dir
    cmd = [
        certbot, "certonly",
        "-d", args.domain,
        "--config-dir", cert_dir,
        "--work-dir", cert_dir,
        "--logs-dir", cert_dir,
        "--agree-tos",
        "-m", args.email,
    ]
    if args.cf_token:
        # Automated DNS-01 via the Cloudflare plugin — no copy-paste.
        creds = os.path.join(cert_dir, "cloudflare.ini")
        os.makedirs(cert_dir, exist_ok=True)
        with open(creds, "w") as fh:
            fh.write(f"dns_cloudflare_api_token = {args.cf_token}\n")
        os.chmod(creds, 0o600)
        cmd += [
            "--non-interactive",
            "--dns-cloudflare",
            "--dns-cloudflare-credentials", creds,
        ]
    else:
        # Manual DNS-01: certbot prints the exact _acme-challenge TXT record
        # to paste into your DNS panel, then waits for you to continue.
        cmd += ["--manual", "--preferred-challenges", "dns"]
        print(
            "\nManual DNS-01: certbot will print a TXT record "
            f"(_acme-challenge.{args.domain}). Paste it into your DNS panel, "
            "wait for propagation, then press Enter to continue.\n"
        )

    print("Running:", " ".join(cmd), "\n")
    if args.print_only:
        return 0
    rc = subprocess.call(cmd)
    if rc == 0:
        print(
            f"\nCert issued under {cert_dir}. Point an A record for "
            f"{args.domain} at the server's LAN IP and start the Caddy proxy "
            "(docker-compose.proxy.yml)."
        )
    return rc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-server")
    sub = p.add_subparsers(dest="cmd")

    ck = sub.add_parser("create-key", help="mint an API key directly in the DB")
    ck.add_argument("--label", required=True, help="client identity for the key")
    ck.add_argument(
        "--scope",
        action="append",
        default=None,
        choices=["read", "ingest", "admin"],
        help="scope(s) for the key (repeatable); default: ingest",
    )

    ts = sub.add_parser("tls-setup", help="obtain a Let's Encrypt cert via DNS-01")
    ts.add_argument("--domain", required=True, help="e.g. rag.example.com")
    ts.add_argument("--email", required=True, help="ACME account email")
    ts.add_argument(
        "--cert-dir", default="./certs", help="output dir for cert/key (default ./certs)"
    )
    ts.add_argument(
        "--cf-token",
        default=None,
        help="Cloudflare API token for fully-automated DNS-01 (skips copy-paste)",
    )
    ts.add_argument(
        "--print-only", action="store_true", help="print the command without running"
    )
    return p


def run() -> None:
    """Entry point: dispatch a subcommand, or run the server when none given."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "create-key":
        if not args.scope:
            args.scope = ["ingest"]
        sys.exit(_cmd_create_key(args))
    if args.cmd == "tls-setup":
        sys.exit(_cmd_tls_setup(args))
    _run_server()


if __name__ == "__main__":  # pragma: no cover
    run()
