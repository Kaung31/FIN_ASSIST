"""Create (or update) the Azure AI Search index. Run once before ingesting.

Usage: python scripts/create_index.py

Requires a configured Azure AI Search service (Standard tier or above for the semantic ranker)
with credentials in .env. The vector field dimension is taken from settings.embedding_dimensions
and must match the embedding deployment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo-root `config` package importable when run as `python scripts/create_index.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from finassist.search.clients import get_index_client
from finassist.search.index_schema import build_index


def main() -> None:
    settings = get_settings()
    index = build_index()
    client = get_index_client()
    result = client.create_or_update_index(index)

    semantic = "on" if settings.azure_search_use_semantic_ranker else "off"
    print(
        f"Index '{result.name}' ready ({len(result.fields)} fields, "
        f"vector dim={settings.embedding_dimensions}, semantic ranker={semantic})."
    )


if __name__ == "__main__":
    main()
