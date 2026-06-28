"""Shared FastAPI dependencies: one LLMClient per process.

Azure AI Search clients are constructed inside search/ (mocked in MOCK_MODE), so there is no
vector-store client to share here anymore.
"""

from __future__ import annotations

from functools import lru_cache

from finassist.llm.client import LLMClient


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()
