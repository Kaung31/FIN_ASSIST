"""API smoke test in mock mode: ingest, then chat / summarize / anomalies."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from finassist.api import deps
from finassist.api.main import app


@pytest.fixture(scope="module")
def client():
    deps.get_llm.cache_clear()
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_full_flow_ingest_then_query(client):
    # 1. ingest (mock DI parse + canned index)
    resp = client.post(
        "/ingest", files={"file": ("acme.pdf", b"%PDF-1.7 dummy", "application/pdf")}
    )
    assert resp.status_code == 200, resp.text
    ingested = resp.json()
    assert ingested["chunks"] > 0
    company, period = ingested["company"], ingested["fiscal_period"]

    # 2. chat — grounded answer with citations
    resp = client.post(
        "/chat",
        json={
            "question": "What was total revenue in Q2 FY2025?",
            "company": company,
            "fiscal_period": period,
        },
    )
    assert resp.status_code == 200
    chat = resp.json()
    assert chat["grounded"] is True
    assert chat["citations"]

    # 3. summarize — numbers from Python
    resp = client.get("/summarize", params={"company": company, "fiscal_period": period})
    assert resp.status_code == 200
    summary = resp.json()
    assert any(
        m["name"].lower() == "total revenue" and m["value"] == 1250.0
        for m in summary["key_metrics"]
    )

    # 4. anomalies — deterministic deltas + explanations
    resp = client.get("/anomalies", params={"company": company, "fiscal_period": period})
    assert resp.status_code == 200
    report = resp.json()
    assert report["anomalies"]
    assert all(a["explanation"] for a in report["anomalies"])
