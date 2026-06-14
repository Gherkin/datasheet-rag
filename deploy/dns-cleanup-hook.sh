#!/bin/sh
# certbot --manual-cleanup-hook: runs after validation. We can't edit your DNS
# for you (that's the whole point of the manual flow), so just remind you the
# challenge record is no longer needed.
echo ""
echo "[done] Validation finished. You may now remove the TXT record:"
echo "       _acme-challenge.${CERTBOT_DOMAIN}"
