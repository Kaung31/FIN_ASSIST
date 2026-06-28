# Infrastructure

Three Azure resources are required. Provision via Portal, `az` CLI, or IaC (Bicep/Terraform).

1. **Azure AI Document Intelligence** — copy endpoint + key into `.env`.
2. **Azure AI Search** — **Standard tier or above** (semantic ranker is not available on free).
   Enable the semantic ranker on the service. Copy endpoint + admin key.
3. **Azure OpenAI** — create two deployments:
   - a chat/vision-capable model (e.g. `gpt-4.1`), and
   - an embedding model (`text-embedding-3-large`, 3072 dims).
   Copy endpoint + key; put the *deployment names* in `.env`.

Auth: for production prefer Microsoft Entra / Managed Identity (`USE_AAD_AUTH=true`) and grant the
identity the relevant data-plane roles, rather than shipping keys.

> Add a `main.bicep` here if you want one-command provisioning. Keep secrets out of source control.
