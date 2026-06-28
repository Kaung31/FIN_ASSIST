"""Ingest a local PDF end-to-end (parse → describe → chunk → embed → index) and run a demo query.

Usage: python scripts/ingest_local.py path/to/earnings.pdf

Runs entirely in MOCK_MODE with no keys (canned Azure calls; real chunking/validation). With real
keys + MOCK_MODE=false it hits Document Intelligence, Azure OpenAI, and Azure AI Search.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo-root `config` package importable when run as `python scripts/ingest_local.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finassist import pipeline
from finassist.llm.client import LLMClient
from finassist.qa import rag_chat


def main(pdf_path: str) -> None:
    data = Path(pdf_path).read_bytes()
    llm = LLMClient()

    result = pipeline.ingest(data, Path(pdf_path).name, llm)
    print("Ingested:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    question = "What was total revenue this quarter and how did it change year over year?"
    response = rag_chat.answer(
        question, llm, company=result["company"], fiscal_period=result["fiscal_period"]
    )
    print(f"\nDemo question: {question}")
    print(f"  grounded: {response.grounded}")
    print(f"  answer:   {response.answer}")
    print(f"  citations: {[(c.section, f'p{c.page}') for c in response.citations]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/ingest_local.py <pdf_path>")
    main(sys.argv[1])
