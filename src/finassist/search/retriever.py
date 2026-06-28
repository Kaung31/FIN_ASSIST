"""Read side: Azure AI Search hybrid (BM25 + vector) retrieval + semantic ranker + filters.

Per Microsoft's benchmarking, hybrid retrieval re-ranked by the semantic ranker (L2) is the
recommended default — it replaces a separate reranker. We embed the query, run a vector query
alongside the keyword query (RRF fuses them), let the semantic ranker reorder, and scope results
with an OData filter (company / period / doc). Returns canned results in MOCK_MODE.
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from finassist.analysis.models import Chunk, RetrievedChunk
from finassist.embeddings import embedder
from finassist.search import clients, index_schema

logger = logging.getLogger(__name__)

# Pull enough vector candidates for the semantic ranker to work with, then trim to `top`.
_VECTOR_K = 50


def search(
    query: str,
    *,
    top: int = 8,
    doc_id: str | None = None,
    company: str | None = None,
    fiscal_period: str | None = None,
    filing_date: str | None = None,
) -> list[RetrievedChunk]:
    """Hybrid + semantic retrieval, optionally filtered by company / period / doc / filing date."""
    settings = get_settings()
    if settings.mock_mode:
        from finassist.mocks import fixtures

        return fixtures.mock_search_results(
            query, top=top, doc_id=doc_id, company=company, fiscal_period=fiscal_period
        )

    from azure.search.documents.models import VectorizedQuery

    vector = embedder.embed_query(query)
    client = clients.get_search_client()

    kwargs: dict = {}
    if settings.azure_search_use_semantic_ranker:
        kwargs["query_type"] = "semantic"
        kwargs["semantic_configuration_name"] = index_schema.SEMANTIC_CONFIG_NAME

    results = client.search(
        search_text=query,
        vector_queries=[
            VectorizedQuery(
                vector=vector, k_nearest_neighbors=_VECTOR_K, fields="content_vector"
            )
        ],
        filter=build_filter(
            doc_id=doc_id, company=company, fiscal_period=fiscal_period, filing_date=filing_date
        ),
        top=top,
        **kwargs,
    )
    return [_to_retrieved(result) for result in results]


def _to_retrieved(result) -> RetrievedChunk:  # noqa: ANN001 - AI Search result dict
    chunk = Chunk(
        chunk_id=result["chunk_id"],
        doc_id=result["doc_id"],
        content=result["content"],
        page=result["page"],
        section=result.get("section"),
        content_type=result.get("content_type", "text"),
        company=result.get("company"),
        fiscal_period=result.get("fiscal_period"),
        filing_date=result.get("filing_date"),
    )
    return RetrievedChunk(
        chunk=chunk,
        score=result.get("@search.score", 0.0),
        reranker_score=result.get("@search.reranker_score"),
    )


def passes_relevance_gate(hits: list[RetrievedChunk]) -> bool:
    """True if retrieval is usable: the best semantic-ranker score clears relevance_min_score.

    Self-RAG-lite: lets callers refuse on low-quality (not just empty) retrieval. Only judges when
    semantic scores exist — if the ranker is off (scores are None), it never gates (returns True).
    """
    threshold = get_settings().relevance_min_score
    if threshold <= 0 or not hits:
        return bool(hits)
    scored = [h.reranker_score for h in hits if h.reranker_score is not None]
    if not scored:
        return True  # no semantic scores to judge → don't gate
    return max(scored) >= threshold


def build_filter(
    *,
    doc_id: str | None = None,
    company: str | None = None,
    fiscal_period: str | None = None,
    filing_date: str | None = None,
) -> str | None:
    """Build an OData $filter string (escaping single quotes), or None."""
    clauses = []
    if doc_id:
        clauses.append(f"doc_id eq '{_escape(doc_id)}'")
    if company:
        clauses.append(f"company eq '{_escape(company)}'")
    if fiscal_period:
        clauses.append(f"fiscal_period eq '{_escape(fiscal_period)}'")
    if filing_date:
        clauses.append(f"filing_date eq '{_escape(filing_date)}'")
    return " and ".join(clauses) or None


def _escape(value: str) -> str:
    return value.replace("'", "''")
