"""POST /chat — grounded Q&A over the indexed documents."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from finassist.analysis.models import QAResponse
from finassist.api.deps import get_llm
from finassist.llm.client import LLMClient
from finassist.qa import rag_chat

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    fiscal_period: str | None = None


@router.post("", response_model=QAResponse)
async def chat(
    req: ChatRequest,
    llm: LLMClient = Depends(get_llm),
) -> QAResponse:
    return rag_chat.answer(
        req.question,
        llm,
        company=req.company,
        fiscal_period=req.fiscal_period,
    )
