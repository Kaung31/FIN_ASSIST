"""Embedder prepends a [company | period | statement] context header to the embedded text."""

from __future__ import annotations

from finassist.analysis.models import Chunk
from finassist.embeddings import embedder


def _chunk(**kw) -> Chunk:
    base = {"chunk_id": "1", "doc_id": "d", "content": "Total revenue 100", "page": 1}
    base.update(kw)
    return Chunk(**base)


def test_embed_text_prepends_context_header():
    text = embedder._embed_text(
        _chunk(company="Apple Inc.", fiscal_period="Q2 2024", section="Income statement")
    )
    assert text.startswith("[Apple Inc. | Q2 2024 | Income statement]")
    assert "Total revenue 100" in text  # original content preserved after the header


def test_embed_text_falls_back_to_content_type_label():
    text = embedder._embed_text(_chunk(company="Apple Inc.", content_type="table"))
    assert text.startswith("[Apple Inc. | table]")


def test_embed_chunks_embeds_the_header(monkeypatch):
    captured = {}

    def fake_embed_texts(texts):
        captured["texts"] = texts
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(embedder, "embed_texts", fake_embed_texts)
    chunks = [_chunk(company="Apple Inc.", fiscal_period="Q2 2024", section="Income statement")]
    embedder.embed_chunks(chunks)

    assert captured["texts"][0].startswith("[Apple Inc.")  # header anchored the embedded text
    assert chunks[0].vector is not None  # vector still filled
