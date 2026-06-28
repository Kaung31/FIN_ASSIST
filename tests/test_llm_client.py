"""Smoke tests for the Azure OpenAI client wrapper.

These hit the real Azure OpenAI resource configured in .env, so they are marked
``integration`` and are skipped unless ``pytest --run-integration`` is passed.
"""

from __future__ import annotations

import base64

import pytest

from finassist.llm.client import LLMClient

# A 1x1 transparent PNG — enough to exercise the vision request path.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture(scope="module")
def llm() -> LLMClient:
    return LLMClient()


@pytest.mark.integration
def test_chat_returns_text(llm: LLMClient):
    out = llm.chat("You are a terse test assistant.", "Reply with exactly: pong")
    assert isinstance(out, str)
    assert out.strip() != ""


@pytest.mark.integration
def test_embed_returns_consistent_vectors(llm: LLMClient):
    # llm.embed() is the Azure OpenAI embedding path (e.g. 3-small=1536, 3-large=3072), which is
    # independent of the bge pipeline dim used for Qdrant. Just assert non-trivial, consistent dims.
    vectors = llm.embed(["Total revenue increased.", "Net income rose."])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 256


@pytest.mark.integration
def test_describe_image_returns_text(llm: LLMClient):
    description = llm.describe_image(_TINY_PNG, "Describe this image in one short sentence.")
    assert isinstance(description, str)
    assert description.strip() != ""


def test_non_retryable_error_fails_fast(monkeypatch):
    """A 404 (e.g. DeploymentNotFound) must NOT be retried — it should raise immediately."""
    from types import SimpleNamespace

    import httpx
    from openai import NotFoundError

    calls = {"n": 0}

    def boom(**kwargs):
        calls["n"] += 1
        raise NotFoundError(
            "deployment not found",
            response=httpx.Response(404, request=httpx.Request("POST", "http://x")),
            body={"code": "DeploymentNotFound"},
        )

    llm = LLMClient()  # constructed in mock mode, so no real client is built
    # Force the real code path with a fake client that always 404s.
    llm._settings = SimpleNamespace(mock_mode=False, azure_openai_chat_deployment="d")
    llm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=boom))
    )

    with pytest.raises(NotFoundError):
        llm.chat("system", "user")
    assert calls["n"] == 1  # called once, no retries
