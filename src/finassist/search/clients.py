"""Azure AI Search client construction (admin index client + per-index query client).

All Azure AI Search client creation lives here. Auth: admin key for local dev, or
DefaultAzureCredential (Microsoft Entra) when settings.use_aad_auth is true.
"""

from __future__ import annotations

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

from config.settings import get_settings


def _credential():
    settings = get_settings()
    if settings.use_aad_auth:
        from azure.identity import DefaultAzureCredential

        return DefaultAzureCredential()
    if not settings.azure_search_key:
        raise RuntimeError(
            "AZURE_SEARCH_KEY is empty and USE_AAD_AUTH is false. Set the admin key in .env."
        )
    return AzureKeyCredential(settings.azure_search_key)


def get_index_client() -> SearchIndexClient:
    """Admin client for creating/updating the index (used by scripts/create_index.py)."""
    settings = get_settings()
    return SearchIndexClient(endpoint=settings.azure_search_endpoint, credential=_credential())


def get_search_client() -> SearchClient:
    """Query/document client bound to the configured index (used by indexer + retriever)."""
    settings = get_settings()
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=_credential(),
    )
