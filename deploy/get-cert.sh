#!/usr/bin/env bash
# Obtain a Let's Encrypt cert for a LAN-only server via the MANUAL DNS-01
# challenge — Docker-native, no DNS-provider API, no plugin.
#
# certbot prints a TXT record; you add it to your own DNS zone. The auth hook
# then POLLS public DNS and continues automatically once the record is live —
# so you never guess about propagation and a too-early check can't fail the run.
# The cert lands in ./certs/ (certbot's letsencrypt layout), ready for the Caddy
# proxy (docker-compose.proxy.yml).
#
#   ./deploy/get-cert.sh rag.example.com you@example.com
#
# Renewal: Let's Encrypt certs last ~90 days. Re-run this exact command to
# renew — certbot prints a fresh TXT record, the hook waits for it, then:
#   docker compose -f docker-compose.yml -f docker-compose.proxy.yml restart caddy
set -euo pipefail

DOMAIN="${1:?usage: get-cert.sh <domain> <email>}"
EMAIL="${2:?usage: get-cert.sh <domain> <email>}"
CERT_DIR="${CERT_DIR:-$PWD/certs}"
HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"

# STAGING=1 uses Let's Encrypt's staging CA: the full flow runs (great for a
# first test) but the cert is UNTRUSTED and there are no rate limits. Drop it
# for the real, browser-trusted cert.
STAGING_FLAG=""
if [ "${STAGING:-0}" = "1" ]; then
  STAGING_FLAG="--test-cert"
  echo "[staging] using Let's Encrypt STAGING CA — cert will be UNTRUSTED."
fi

mkdir -p "$CERT_DIR"

echo "Requesting a cert for $DOMAIN via manual DNS-01 (auto-detecting propagation)."

# -it so you can Ctrl-C; the auth hook handles the wait, no Enter needed.
docker run -it --rm \
  -v "$CERT_DIR:/etc/letsencrypt" \
  -v "$HOOK_DIR:/hooks:ro" \
  certbot/certbot certonly \
  --manual \
  --preferred-challenges dns \
  --manual-auth-hook /hooks/dns-auth-hook.sh \
  --manual-cleanup-hook /hooks/dns-cleanup-hook.sh \
  --agree-tos \
  $STAGING_FLAG \
  -m "$EMAIL" \
  -d "$DOMAIN"

echo
echo "Cert written under $CERT_DIR/live/$DOMAIN/"
echo "Next:"
echo "  1. Point an A record for $DOMAIN at this host's LAN IP."
echo "  2. RAG_DOMAIN=$DOMAIN docker compose \\"
echo "       -f docker-compose.yml -f docker-compose.proxy.yml up -d"
