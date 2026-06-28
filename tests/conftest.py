"""Shared fixtures. Unit tests must NOT touch the network — mock the LLM and search clients."""

from __future__ import annotations

import os

# Tests are hermetic: always mock the keyed cloud APIs and use an in-memory Qdrant, regardless of
# the local .env (which may have MOCK_MODE=false for a live run). These env vars override .env and
# must be set before config.settings is first imported/cached.
os.environ["MOCK_MODE"] = "true"
os.environ["QDRANT_URL"] = ""

import pytest

from config.settings import get_settings
from finassist.analysis.models import FinancialTable, ParsedDocument

get_settings.cache_clear()


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.integration (hits real keyed cloud APIs).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration-marked tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="integration test; pass --run-integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fake_llm():
    """A stub LLMClient. Override .chat/.embed/.describe_image per test."""

    class _FakeLLM:
        def chat(self, system, user, *, json_mode=False):
            return "{}"

        def embed(self, texts):
            return [[0.0] * 8 for _ in texts]

        def describe_image(self, image_bytes, instruction):
            return "A bar chart of revenue by quarter."

    return _FakeLLM()


@pytest.fixture
def sample_document() -> ParsedDocument:
    table = FinancialTable(
        page=2,
        caption="Income Statement",
        rows=[["Metric", "FY25 Q2", "FY24 Q2"], ["Revenue", "1,200", "1,000"]],
        markdown="| Metric | FY25 Q2 | FY24 Q2 |\n|---|---|---|\n| Revenue | 1,200 | 1,000 |",
    )
    return ParsedDocument(
        doc_id="doc1",
        source_name="sample.pdf",
        company="Acme Corp",
        fiscal_period="FY2025 Q2",
        page_count=10,
        raw_text="...",
        tables=[table],
    )
