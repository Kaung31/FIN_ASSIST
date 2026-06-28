#!/usr/bin/env bash
# Spin up (or reuse) the Azure AI Search service and build the index — ready for live testing.
# The service name is read from AZURE_SEARCH_ENDPOINT in .env (https://<name>.search.windows.net),
# so a recreated service keeps the same endpoint; only the admin key changes (rewritten into .env).
#
# Usage:
#   az login
#   AZ_RG=<your-resource-group> [AZ_LOCATION=eastus] [AZ_SKU=basic] ./scripts/search_up.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${AZ_RG:?set AZ_RG to your Azure resource group, e.g. AZ_RG=my-rg ./scripts/search_up.sh}"
AZ_LOCATION="${AZ_LOCATION:-eastus}"
AZ_SKU="${AZ_SKU:-basic}"

ENDPOINT=$(grep -E '^AZURE_SEARCH_ENDPOINT=' .env | cut -d= -f2- | tr -d ' ')
SEARCH=$(echo "$ENDPOINT" | sed -E 's#https?://([^.]+)\..*#\1#')
: "${SEARCH:?could not read a service name from AZURE_SEARCH_ENDPOINT in .env}"
echo "AI Search service: $SEARCH  (rg=$AZ_RG, sku=$AZ_SKU, location=$AZ_LOCATION)"

if az search service show -n "$SEARCH" -g "$AZ_RG" >/dev/null 2>&1; then
  echo "Service already exists — reusing it."
else
  echo "Creating service (takes a few minutes)…"
  az search service create -n "$SEARCH" -g "$AZ_RG" \
    --sku "$AZ_SKU" --semantic-search free -l "$AZ_LOCATION"
fi

echo "Refreshing AZURE_SEARCH_KEY in .env…"
KEY=$(az search admin-key show --service-name "$SEARCH" -g "$AZ_RG" --query primaryKey -o tsv)
.venv/bin/python - "$KEY" <<'PY'
import sys, pathlib
key = sys.argv[1]
p = pathlib.Path(".env")
out = [("AZURE_SEARCH_KEY=" + key) if l.startswith("AZURE_SEARCH_KEY=") else l
       for l in p.read_text().splitlines()]
p.write_text("\n".join(out) + "\n")
print("  updated AZURE_SEARCH_KEY")
PY

echo "Building the index…"
.venv/bin/python scripts/create_index.py

echo
echo "Ready. Now set MOCK_MODE=false in .env, then ingest + query, e.g.:"
echo "  python scripts/ingest_local.py data/samples/your_earnings.pdf"
