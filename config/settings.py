"""Central configuration. Everything reads settings from here — never os.environ directly.

All external calls are Azure: Document Intelligence (parsing), Azure OpenAI (chat + embeddings),
and Azure AI Search (hybrid + semantic retrieval). Mock mode (the default) mocks all three so the
full pipeline and tests run at zero cost — unlike a local vector store, AI Search and Document
Intelligence have no free local equivalent, so real retrieval can only be tested against the cloud.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Mode ───────────────────────────────────────────────────────────────────
    # When true, all three Azure services are mocked. Flip to false once keys are in .env.
    mock_mode: bool = True

    # ── Azure Document Intelligence (parsing) ──────────────────────────────────
    azure_docintel_endpoint: str = ""
    azure_docintel_key: str = ""

    # ── Azure OpenAI (chat + embeddings) ───────────────────────────────────────
    # IMPORTANT: azure_openai_endpoint must be the BARE resource URL ending in
    # ".openai.azure.com/" — never append /openai, /openai/v1, or /deployments (the SDK adds the
    # path). The *_deployment values are the deployment NAMES, passed to the SDK as `model`.
    azure_openai_endpoint: str = ""
    azure_openai_key: str = ""
    azure_openai_api_version: str = "2024-10-21"  # a current GA version
    azure_openai_chat_deployment: str = "gpt-4o-mini"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536  # text-embedding-3-small = 1536 (3-large = 3072)

    # ── Azure AI Search (hybrid + semantic ranker) ─────────────────────────────
    azure_search_endpoint: str = ""
    azure_search_key: str = ""
    azure_search_index_name: str = "earnings-index"
    # Semantic ranker requires Standard tier or above (it replaces a separate reranker).
    azure_search_use_semantic_ranker: bool = True
    # Relevance gate: refuse generation when the best semantic-ranker score is below this
    # (Azure semantic score is 0-4; 0 disables the gate). Tune against the golden set — too high
    # over-refuses and hurts context recall.
    relevance_min_score: float = 1.0

    # ── Eval / tracing ─────────────────────────────────────────────────────────
    enable_phoenix: bool = False

    # ── App ────────────────────────────────────────────────────────────────────
    # Prefer DefaultAzureCredential (Managed Identity / az login) over keys where possible.
    use_aad_auth: bool = False
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so we parse the environment only once."""
    return Settings()  # type: ignore[call-arg]
