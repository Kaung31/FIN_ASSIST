"""Flat section/table/figure chunking with metadata."""

from __future__ import annotations

from finassist.chunking.chunker import chunk
from finassist.mocks.fixtures import mock_parsed_document


def test_tables_become_whole_chunks():
    chunks = chunk(mock_parsed_document())
    table_chunks = [c for c in chunks if c.content_type == "table"]
    assert len(table_chunks) == 2  # income statement + segment table
    assert any("Total revenue" in c.content for c in table_chunks)


def test_narrative_becomes_text_chunks_and_undescribed_figure_is_skipped():
    chunks = chunk(mock_parsed_document())
    assert any(c.content_type == "text" for c in chunks)
    assert not any(c.content_type == "figure" for c in chunks)  # no description yet


def test_described_figure_becomes_chunk():
    doc = mock_parsed_document()
    doc.figures[0].description = "Bar chart of quarterly revenue."
    chunks = chunk(doc)
    assert any(c.content_type == "figure" for c in chunks)


def test_all_chunks_carry_metadata():
    chunks = chunk(mock_parsed_document())
    assert chunks
    assert all(c.doc_id and c.company == "Acme Corp" for c in chunks)
    assert all(c.fiscal_period == "FY2025 Q2" for c in chunks)
    # source_type (content_type) + a section label on every chunk — used for the context header.
    assert all(c.content_type in {"table", "figure", "text"} for c in chunks)
    assert all(c.section for c in chunks)
    assert all(c.filing_date == "2025-03-30" for c in chunks)  # filing-date awareness (Task 5)
