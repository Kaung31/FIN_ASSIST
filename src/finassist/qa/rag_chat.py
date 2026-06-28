"""Grounded Q&A: retrieve, then answer ONLY from retrieved context, with citations.

If retrieval returns nothing relevant, we decline rather than answer from general knowledge.
Citations are attached from the retrieved chunks the answer is grounded in.
"""

from __future__ import annotations

from pathlib import Path

from finassist.analysis.models import Citation, QAResponse, RetrievedChunk
from finassist.llm.client import LLMClient
from finassist.search import retriever

_SYSTEM = (Path(__file__).parents[3] / "config/prompts/qa_system.txt").read_text()

_REFUSAL = "I don't have that information in the provided documents."


def answer(
    question: str,
    llm: LLMClient,
    *,
    company: str | None = None,
    fiscal_period: str | None = None,
    top: int = 6,
) -> QAResponse:
    """Answer a question grounded in the indexed documents, with citations."""
    retrieved = retriever.search(
        question,
        top=top,
        company=company,
        fiscal_period=fiscal_period,
    )
    # Refuse on empty OR low-relevance retrieval (Self-RAG-lite), same grounded-refusal path.
    if not retrieved or not retriever.passes_relevance_gate(retrieved):
        return QAResponse(question=question, answer=_REFUSAL, citations=[], grounded=False)

    user = (
        f"Question: {question}\n\n"
        "Answer using ONLY the context passages below; cite the passage ids you used. If the "
        "answer is not in the context, say you don't have that information.\n\n"
        f"Context passages:\n{_format_context(retrieved)}"
    )
    answer_text = llm.chat(_SYSTEM, user).strip()

    citations = [
        Citation(
            chunk_id=hit.chunk.chunk_id,
            doc_id=hit.chunk.doc_id,
            page=hit.chunk.page,
            section=hit.chunk.section,
            filing_date=hit.chunk.filing_date,
        )
        for hit in retrieved
    ]
    return QAResponse(question=question, answer=answer_text, citations=citations, grounded=True)


def _format_context(retrieved: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{hit.chunk.chunk_id} | p{hit.chunk.page} | {hit.chunk.section}] {hit.chunk.content}"
        for hit in retrieved
    )
