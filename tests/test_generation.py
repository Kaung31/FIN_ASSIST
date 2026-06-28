"""Summarizer + grounded Q&A in mock mode (mock retriever returns the canned result set)."""

from __future__ import annotations

import pytest

from finassist.analysis.summarizer import summarize
from finassist.llm.client import LLMClient
from finassist.qa.rag_chat import answer


def test_summary_numbers_come_from_python_with_citations():
    summary = summarize("Acme Corp", "FY2025 Q2", LLMClient())
    assert summary.headline  # prose from the LLM
    metrics = {m.name.lower(): m for m in summary.key_metrics}
    assert metrics["total revenue"].value == 1250.0
    assert metrics["total revenue"].prior_value == 1000.0
    assert metrics["total revenue"].yoy_change_pct == pytest.approx(25.0)
    assert metrics["total revenue"].citation is not None
    assert metrics["total revenue"].citation.filing_date == "2025-03-30"  # version-aware (Task 5)
    assert summary.citations
    assert all(c.filing_date == "2025-03-30" for c in summary.citations)
    assert {"Cloud", "Devices"} <= {s.segment for s in summary.segments}


def test_qa_grounded_answer_has_citations():
    response = answer("What was total revenue in Q2 FY2025?", LLMClient())
    assert response.grounded is True
    assert response.citations
    assert all(c.filing_date == "2025-03-30" for c in response.citations)  # version-aware (Task 5)
    assert "1,250" in response.answer


def test_qa_refuses_when_no_context():
    response = answer("What is the capital of France?", LLMClient(), company="Nonexistent Inc")
    assert response.grounded is False
    assert response.citations == []


def test_qa_refuses_on_low_relevance(monkeypatch):
    # Chunks come back, but all below relevance_min_score → refuse, no fabricated citations.
    from finassist.analysis.models import Chunk, RetrievedChunk
    from finassist.search import retriever

    low = [
        RetrievedChunk(
            chunk=Chunk(chunk_id="x", doc_id="d", content="unrelated", page=1, company="Acme Corp"),
            score=0.01,
            reranker_score=0.1,  # below the default threshold of 1.0
        )
    ]
    monkeypatch.setattr(retriever, "search", lambda *a, **k: low)
    response = answer("What was total revenue?", LLMClient())
    assert response.grounded is False
    assert response.citations == []
