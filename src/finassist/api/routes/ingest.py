"""POST /ingest — upload a PDF, run the full pipeline, index it."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from finassist import pipeline
from finassist.api.deps import get_llm
from finassist.llm.client import LLMClient

router = APIRouter()


@router.post("")
async def ingest_pdf(
    file: UploadFile = File(...),
    llm: LLMClient = Depends(get_llm),
) -> dict:
    """Parse → crop figures → describe → chunk → embed → index. Returns doc_id + counts."""
    pdf_bytes = await file.read()
    return pipeline.ingest(pdf_bytes, file.filename or "upload.pdf", llm)
