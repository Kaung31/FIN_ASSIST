"""Azure AI Search index definition: filterable metadata + HNSW vector field + semantic config.

Retrieval is hybrid (BM25 keyword + vector via RRF) re-ranked by the semantic ranker (L2), which
replaces a separate reranker. The semantic ranker needs Standard tier or above. Build the index
once with scripts/create_index.py. The vector dimension must equal settings.embedding_dimensions.
"""

from __future__ import annotations

from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from config.settings import get_settings

SEMANTIC_CONFIG_NAME = "earnings-semantic"
VECTOR_PROFILE_NAME = "earnings-hnsw-profile"
_HNSW_CONFIG_NAME = "earnings-hnsw"


def build_index() -> SearchIndex:
    """Return the SearchIndex definition for earnings chunks (mirrors analysis.models.Chunk)."""
    settings = get_settings()

    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchableField(name="section", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True),
        SearchableField(
            name="company", type=SearchFieldDataType.String, filterable=True, facetable=True
        ),
        SearchableField(
            name="fiscal_period", type=SearchFieldDataType.String, filterable=True, facetable=True
        ),
        SimpleField(
            name="filing_date", type=SearchFieldDataType.String, filterable=True, sortable=True
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=settings.embedding_dimensions,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=_HNSW_CONFIG_NAME)],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME, algorithm_configuration_name=_HNSW_CONFIG_NAME
            )
        ],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="section")],
                ),
            )
        ]
    )

    return SearchIndex(
        name=settings.azure_search_index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )
