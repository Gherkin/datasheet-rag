#!/bin/sh
# certbot --manual-auth-hook: print the TXT record to add, then POLL public DNS
# until it's actually live before letting certbot validate. certbot passes the
# challenge in $CERTBOT_DOMAIN / $CERTBOT_VALIDATION. Exit 0 = record confirmed
# live (certbot proceeds); non-zero = give up (certbot aborts cleanly).
#
# NOTE: certbot CAPTURES a hook's stdout and only shows it after the hook
# returns — which would hide our instructions during the (blocking) poll. So we
# write everything to the controlling terminal (/dev/tty) when there is one,
# falling back to stdout for non-interactive runs.
set -e

# Test-OPEN /dev/tty (a mere `-w` check can pass while open() fails with ENXIO
# when there's no controlling terminal).
if { : > /dev/tty; } 2>/dev/null; then OUT=/dev/tty; else OUT=/dev/stdout; fi

RECORD="_acme-challenge.${CERTBOT_DOMAIN}"

{
  echo ""
  echo "=================================================================="
  echo " Add this DNS TXT record in your zone. This tool then watches"
  echo " public DNS and continues automatically once it's live —"
  echo " no keypress, no guessing about propagation:"
  echo ""
  echo "   Name:  ${RECORD}"
  echo "   Type:  TXT"
  echo "   Value: ${CERTBOT_VALIDATION}"
  echo "=================================================================="
  echo ""
} > "$OUT"

# Poll via python+dnspython (installed on demand — the certbot image has pip
# and outbound network). Query public resolvers so we see real propagation,
# not a stale local cache. All progress goes to the terminal, not certbot's
# captured stdout.
python3 - > "$OUT" 2>&1 <<'PY'
import os, sys, time, subprocess
try:
    import dns.resolver, dns.exception
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--root-user-action=ignore", "dnspython"],
        check=True,
    )
    import dns.resolver, dns.exception

name = "_acme-challenge." + os.environ["CERTBOT_DOMAIN"]
want = os.environ["CERTBOT_VALIDATION"]

# Use the container's SYSTEM resolver by default — the same path a working
# `dig` on the host uses. Many networks block outbound :53 to public resolvers
# like 1.1.1.1, so don't force those. Override with DNS_SERVERS="1.1.1.1 8.8.8.8"
# if your system resolver is the one that's stale/broken.
override = os.environ.get("DNS_SERVERS", "").split()

def query():
    r = dns.resolver.Resolver(configure=True)
    if override:
        r.nameservers = override
    r.timeout = 5
    r.lifetime = 10
    ans = r.resolve(name, "TXT", raise_on_no_answer=True)
    return [
        "".join(p.decode() if isinstance(p, bytes) else p for p in rr.strings)
        for rr in ans
    ]

deadline = time.time() + 1200  # 20 minutes
attempt = 0
while time.time() < deadline:
    attempt += 1
    try:
        values = query()
        if want in values:
            print(f"[ok] {name} TXT is live — continuing with issuance.", flush=True)
            sys.exit(0)
        shown = ", ".join(v[:14] + "…" for v in values)
        print(
            f"[..] attempt {attempt}: found {len(values)} TXT record(s) [{shown}] "
            f"but NOT the value this run needs — UPDATE the record to the value "
            f"shown above. retry in 10s",
            flush=True,
        )
    except dns.resolver.NXDOMAIN:
        print(f"[..] attempt {attempt}: {name} does not exist yet (NXDOMAIN); retry in 10s", flush=True)
    except dns.resolver.NoAnswer:
        print(f"[..] attempt {attempt}: no TXT records at {name} yet; retry in 10s", flush=True)
    except dns.exception.DNSException as exc:
        print(
            f"[..] attempt {attempt}: DNS query failed "
            f"({exc.__class__.__name__}: {exc}). If this repeats, your resolver "
            f"may block this lookup — try DNS_SERVERS=... ; retry in 10s",
            flush=True,
        )
    time.sleep(10)

print(f"[!!] Timed out after 20 min waiting for {name}. Aborting.", flush=True)
sys.exit(1)
PY
