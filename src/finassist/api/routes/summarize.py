"""GET /summarize — structured EarningsSummary for a company/period."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from finassist.analysis import summarizer
from finassist.analysis.models import EarningsSummary
from finassist.api.deps import get_llm
from finassist.llm.client import LLMClient

router = APIRouter()


@router.get("", response_model=EarningsSummary)
async def get_summary(
    company: str | None = None,
    fiscal_period: str | None = None,
    llm: LLMClient = Depends(get_llm),
) -> EarningsSummary:
    return summarizer.summarize(company, fiscal_period, llm)
