"""FastAPI application entry point. Run: uvicorn finassist.api.main:app --reload"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APIStatusError

from config.settings import get_settings
from finassist.api.routes import anomalies, chat, ingest, summarize

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinAssist",
    description="Multimodal financial-report assistant. Document analysis, not investment advice.",
    version="0.1.0",
)

app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(summarize.router, prefix="/summarize", tags=["summarize"])
app.include_router(anomalies.router, prefix="/anomalies", tags=["anomalies"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


@app.exception_handler(APIStatusError)
async def _azure_status_error(request: Request, exc: APIStatusError) -> JSONResponse:
    """Turn Azure OpenAI HTTP errors into a clean, actionable JSON response."""
    body = exc.body if isinstance(exc.body, dict) else {}
    code = body.get("code")
    detail = body.get("message") or str(exc)
    if code == "DeploymentNotFound" or exc.status_code == 404:
        detail = (
            "Azure OpenAI deployment not found. Check AZURE_OPENAI_CHAT_DEPLOYMENT and "
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env match deployments that exist in your "
            "Azure OpenAI resource (Azure AI Foundry → Deployments)."
        )
    logger.warning("Azure OpenAI error %s: %s", exc.status_code, code or detail)
    return JSONResponse(
        status_code=502,
        content={
            "error": "azure_openai",
            "status": exc.status_code,
            "code": code,
            "detail": detail,
        },
    )


@app.exception_handler(APIConnectionError)
async def _azure_conn_error(request: Request, exc: APIConnectionError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"error": "azure_openai_connection", "detail": str(exc)[:300]},
    )


@app.exception_handler(Exception)
async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Return a JSON body for any unhandled error (never plain-text 'Internal Server Error')."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal", "type": type(exc).__name__, "detail": str(exc)[:500]},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
