"""Anomaly math is deterministic and must be exact — these tests pin that behavior."""

from __future__ import annotations

import pytest

from finassist.analysis import math_guardrail
from finassist.analysis.anomaly_detector import detect, explain
from finassist.analysis.models import FinancialTable
from finassist.llm.client import LLMClient


def _income_table() -> FinancialTable:
    return FinancialTable(
        page=2,
        caption="Income statement",
        rows=[
            ["Metric", "Q2 FY2025", "Q2 FY2024"],
            ["Total revenue", "1,250", "1,000"],
            ["Operating income", "350", "250"],
            ["Cost of revenue", "500", "450"],
        ],
        markdown="(unused)",
    )


def test_detect_pct_change_is_exact_and_computed_in_python():
    anomalies = {a.metric: a for a in detect([_income_table()], "FY2025 Q2", "FY2024 Q2")}
    assert anomalies["Total revenue"].pct_change == pytest.approx(25.0)
    assert anomalies["Total revenue"].abs_delta == pytest.approx(250.0)
    assert anomalies["Operating income"].pct_change == pytest.approx(40.0)
    # Cost of revenue is +11.1% < 20% threshold → not flagged.
    assert "Cost of revenue" not in anomalies


def test_detect_on_sample_document_fixture(sample_document):
    anomalies = detect(sample_document.tables, "FY2025 Q2", "FY2024 Q2")
    revenue = next(a for a in anomalies if a.metric == "Revenue")
    assert revenue.pct_change == pytest.approx(20.0)
    assert revenue.severity.value == "medium"


def test_severity_thresholds():
    assert math_guardrail.severity(60).value == "high"
    assert math_guardrail.severity(30).value == "medium"
    assert math_guardrail.severity(5).value == "low"


def test_yoy_pct_guards_zero_and_none():
    assert math_guardrail.yoy_pct(120, 100) == pytest.approx(20.0)
    assert math_guardrail.yoy_pct(100, 0) is None
    assert math_guardrail.yoy_pct(None, 100) is None


def test_explain_fills_text_without_changing_numbers():
    anomalies = detect([_income_table()], "FY2025 Q2", "FY2024 Q2")
    before = {a.metric: (a.current_value, a.prior_value, a.pct_change) for a in anomalies}

    report = explain(anomalies, LLMClient())

    assert report.anomalies and all(a.explanation for a in report.anomalies)
    after = {a.metric: (a.current_value, a.prior_value, a.pct_change) for a in report.anomalies}
    assert before == after  # the LLM must not touch the numbers
