"""GET /anomalies — period-over-period anomaly report."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from finassist.analysis import anomaly_detector
from finassist.analysis.models import AnomalyReport
from finassist.api.deps import get_llm
from finassist.llm.client import LLMClient
from finassist.search import retriever

router = APIRouter()

_QUERY = (
    "income statement statements of operations revenue cost expenses operating income "
    "net income EPS segment results"
)


@router.get("", response_model=AnomalyReport)
async def get_anomalies(
    company: str | None = None,
    fiscal_period: str | None = None,
    current_period: str | None = None,
    prior_period: str | None = None,
    llm: LLMClient = Depends(get_llm),
) -> AnomalyReport:
    retrieved = retriever.search(_QUERY, top=12, company=company, fiscal_period=fiscal_period)
    if not retriever.passes_relevance_gate(retrieved):
        retrieved = []  # low-relevance → no anomalies rather than from noise
    return anomaly_detector.report_from_retrieved(
        retrieved,
        llm,
        company=company,
        current_period=current_period,
        prior_period=prior_period,
    )
