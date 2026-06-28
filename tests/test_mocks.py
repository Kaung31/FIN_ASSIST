"""Mock-mode infrastructure: canned fixtures + the LLM client's mock short-circuit."""

from __future__ import annotations

import json

from config.settings import get_settings
from finassist.llm.client import LLMClient
from finassist.mocks import fixtures


def test_mock_parsed_document_shape():
    doc = fixtures.mock_parsed_document()
    assert doc.company == "Acme Corp"
    assert doc.fiscal_period == "FY2025 Q2"
    assert len(doc.tables) == 2
    assert len(doc.figures) == 1
    # current + prior columns present so the math guardrail has real numbers to diff.
    income = doc.tables[0]
    assert income.rows[0] == ["($ in millions, except per share)", "Q2 FY2025", "Q2 FY2024"]
    assert any(row[0] == "Total revenue" for row in income.rows)
    assert income.markdown.startswith("| ")


def test_chat_response_routes_by_intent():
    summary_system = "You produce STRUCTURED SUMMARIES conforming to the EarningsSummary schema."
    anomaly_system = "You EXPLAIN pre-computed anomalies in earnings data."
    qa_system = "You are a financial document Q&A assistant."

    summary = fixtures.mock_chat_response(summary_system, "ctx", json_mode=True)
    assert json.loads(summary)["company"] == "Acme Corp"

    explanation = fixtures.mock_chat_response(anomaly_system, "ctx")
    assert "expenses" in explanation.lower()

    answer = fixtures.mock_chat_response(qa_system, "What was revenue?")
    assert "1,250" in answer


def test_mock_search_results_return_tables_and_respect_company_filter():
    results = fixtures.mock_search_results("total revenue", top=12, company="Acme Corp")
    assert any(r.chunk.content_type == "table" for r in results)
    # descending scores
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # a non-matching company returns nothing (so refusal is exercised)
    assert fixtures.mock_search_results("x", company="Nope Inc") == []


def test_llm_client_mock_mode_makes_no_network_calls():
    llm = LLMClient()
    assert llm.mock_mode is True

    vectors = llm.embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == get_settings().embedding_dimensions

    assert isinstance(llm.chat("Q&A assistant", "hi"), str)
    assert isinstance(llm.describe_image(b"\x89PNG", "describe this"), str)
