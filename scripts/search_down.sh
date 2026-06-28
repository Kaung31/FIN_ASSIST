#!/usr/bin/env bash
# Delete the Azure AI Search service to STOP billing (it bills per hour it exists, not per query).
# Recreate it later with scripts/search_up.sh. Document Intelligence + Azure OpenAI are pay-per-call,
# so there's nothing to tear down for those.
#
# Usage:
#   AZ_RG=<your-resource-group> ./scripts/search_down.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${AZ_RG:?set AZ_RG to your Azure resource group, e.g. AZ_RG=my-rg ./scripts/search_down.sh}"

ENDPOINT=$(grep -E '^AZURE_SEARCH_ENDPOINT=' .env | cut -d= -f2- | tr -d ' ')
SEARCH=$(echo "$ENDPOINT" | sed -E 's#https?://([^.]+)\..*#\1#')
: "${SEARCH:?could not read a service name from AZURE_SEARCH_ENDPOINT in .env}"

echo "Deleting AI Search service '$SEARCH' (rg=$AZ_RG) — billing stops after this."
az search service delete -n "$SEARCH" -g "$AZ_RG" --yes
echo "Done. Run scripts/search_up.sh to recreate it before your next test."
