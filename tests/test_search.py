"""Azure AI Search index schema (no network) + mock-mode retrieval + OData filter."""

from __future__ import annotations

from config.settings import get_settings
from finassist.search import retriever
from finassist.search.index_schema import (
    SEMANTIC_CONFIG_NAME,
    VECTOR_PROFILE_NAME,
    build_index,
)

EXPECTED_FIELDS = {
    "chunk_id",
    "doc_id",
    "content",
    "page",
    "section",
    "content_type",
    "company",
    "fiscal_period",
    "filing_date",
    "content_vector",
}


def test_index_fields_and_vector_dim():
    index = build_index()
    assert {f.name for f in index.fields} == EXPECTED_FIELDS
    vector_field = next(f for f in index.fields if f.name == "content_vector")
    assert vector_field.vector_search_dimensions == get_settings().embedding_dimensions  # 1536
    assert vector_field.vector_search_profile_name == VECTOR_PROFILE_NAME


def test_chunk_id_is_the_only_key():
    index = build_index()
    assert [f.name for f in index.fields if getattr(f, "key", False)] == ["chunk_id"]


def test_semantic_configuration_present():
    index = build_index()
    names = {c.name for c in index.semantic_search.configurations}
    assert SEMANTIC_CONFIG_NAME in names


def test_mock_retrieval_returns_table_context():
    results = retriever.search(
        "total revenue", top=12, company="Acme Corp", fiscal_period="FY2025 Q2"
    )
    assert results
    assert any(
        r.chunk.content_type == "table" and "Total revenue" in r.chunk.content for r in results
    )


def test_filter_scopes_to_company():
    assert retriever.search("revenue", company="Nonexistent Inc") == []


def test_build_filter_escapes_quotes():
    odata = retriever.build_filter(company="Acme's Corp", fiscal_period="FY2025 Q2")
    assert "company eq 'Acme''s Corp'" in odata
    assert "fiscal_period eq 'FY2025 Q2'" in odata
    assert retriever.build_filter() is None


def test_relevance_gate():
    from finassist.analysis.models import Chunk, RetrievedChunk

    def hit(reranker):
        chunk = Chunk(chunk_id="1", doc_id="d", content="x", page=1)
        return RetrievedChunk(chunk=chunk, score=0.0, reranker_score=reranker)

    assert retriever.passes_relevance_gate([hit(3.0)]) is True  # clears threshold (1.0)
    assert retriever.passes_relevance_gate([hit(0.1)]) is False  # below threshold
    assert retriever.passes_relevance_gate([]) is False  # empty
    assert retriever.passes_relevance_gate([hit(None)]) is True  # no semantic score → don't gate
