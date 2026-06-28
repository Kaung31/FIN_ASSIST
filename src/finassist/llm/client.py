"""Thin wrapper around the Azure OpenAI client (chat, vision, embeddings).

In MOCK_MODE the keyed LLM/VLM is never contacted: ``chat`` / ``describe_image`` / ``embed``
return canned, schema-valid content from ``finassist.mocks.fixtures``. With real keys
(``MOCK_MODE=false``) it uses the ``openai`` AzureOpenAI class — API key for local dev, or
DefaultAzureCredential (Microsoft Entra) when ``settings.use_aad_auth`` is true.

All Azure OpenAI access in the app goes through this module so retries, auth, the mock path,
and the deployment-name indirection live in one place.
"""

from __future__ import annotations

import base64
import hashlib
import logging

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Only retry transient failures. 4xx (e.g. 404 DeploymentNotFound, 401 auth) are permanent —
# retrying them just wastes time, so they propagate immediately (reraise=True, original error).
_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def _retry(func):
    return retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=20),
        reraise=True,
    )(func)


def _build_client():
    """Construct a real AzureOpenAI client. Only called when MOCK_MODE is false."""
    from openai import AzureOpenAI

    settings = get_settings()
    if settings.use_aad_auth:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            azure_ad_token_provider=token_provider,
        )
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        api_key=settings.azure_openai_key,
    )


class LLMClient:
    """Chat / vision / embedding access to Azure OpenAI, with a MOCK_MODE short-circuit."""

    def __init__(self) -> None:
        self._settings = get_settings()
        # In mock mode we never construct the real client (no endpoint/key required).
        self._client = None if self._settings.mock_mode else _build_client()

    @property
    def mock_mode(self) -> bool:
        return self._settings.mock_mode

    @_retry
    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Single-turn chat completion. Set json_mode=True for structured-output prompts."""
        if self.mock_mode:
            from finassist.mocks import fixtures

            return fixtures.mock_chat_response(system, user, json_mode=json_mode)

        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self._settings.azure_openai_chat_deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,  # deterministic; we're extracting facts, not being creative
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    @_retry
    def describe_image(self, image_bytes: bytes, instruction: str) -> str:
        """Vision call: describe a cropped chart/figure. Used by multimodal.chart_describer."""
        if self.mock_mode:
            from finassist.mocks import fixtures

            return fixtures.mock_chart_description()

        b64 = base64.b64encode(image_bytes).decode()
        resp = self._client.chat.completions.create(
            model=self._settings.azure_openai_chat_deployment,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    @_retry
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed texts via Azure OpenAI (the embedding path used by embeddings.embedder)."""
        if self.mock_mode:
            return [self._mock_vector(t) for t in texts]

        resp = self._client.embeddings.create(
            model=self._settings.azure_openai_embedding_deployment,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def _mock_vector(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding of the configured dimension (no network)."""
        dim = self._settings.embedding_dimensions
        digest = hashlib.sha256(text.encode()).digest()
        return [(digest[i % len(digest)] / 255.0) - 0.5 for i in range(dim)]
