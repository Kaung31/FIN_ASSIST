"""Parser (mock mode) + markdown table extraction + financial number parsing."""

from __future__ import annotations

from finassist.ingestion import parser
from finassist.ingestion.table_extractor import parse_number, tables_from_markdown


def test_parse_mock_mode_returns_tables_text_and_figure():
    doc = parser.parse("acme_fy2025_q2.pdf")
    assert doc.company == "Acme Corp"
    assert doc.fiscal_period == "FY2025 Q2"
    assert len(doc.tables) == 2
    assert doc.figures and doc.figures[0].page == 4
    assert "revenue" in doc.raw_text.lower()


def test_parse_accepts_bytes_with_source_name():
    # In mock mode the bytes are ignored but the source_name flows through to the document.
    doc = parser.parse(b"%PDF-1.7 ...", source_name="report.pdf")
    assert doc.source_name == "report.pdf"
    assert doc.page_count == 4


def test_parse_number_handles_financial_formats():
    assert parse_number("1,250") == 1250.0
    assert parse_number("$1,250.5") == 1250.5
    assert parse_number("(50)") == -50.0
    assert parse_number("25%") == 25.0
    assert parse_number("1.20") == 1.2
    assert parse_number("—") is None
    assert parse_number("") is None
    assert parse_number(None) is None


def test_tables_from_markdown_parses_grid():
    md = (
        "Some narrative line.\n"
        "| Metric | Q2 FY2025 | Q2 FY2024 |\n"
        "|---|---|---|\n"
        "| Total revenue | 1,250 | 1,000 |\n"
        "| Diluted EPS | 1.20 | 0.88 |\n"
        "\nTrailing prose.\n"
    )
    tables = tables_from_markdown(md, page=2)
    assert len(tables) == 1
    table = tables[0]
    assert table.page == 2
    assert table.rows[0] == ["Metric", "Q2 FY2025", "Q2 FY2024"]
    assert ["Total revenue", "1,250", "1,000"] in table.rows


def test_infer_company_from_registrant_line():
    # The 10-Q/10-K cover anchors the issuer name to the "(Exact name of Registrant ...)" line.
    plain = (
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n\n"
        "Apple Inc.\n"
        "(Exact name of the Registrant as specified in its charter)\n"
        "California"
    )
    assert parser._infer_company(plain) == "Apple Inc."
    # Document Intelligence wraps the name in a <figure> tag in real markdown.
    di_markdown = (
        "Commission File Number: 001-36743\n\n"
        "<figure>\n\nApple Inc.\n\n</figure>\n\n"
        "(Exact name of Registrant as specified in its charter)\n\nCalifornia"
    )
    assert parser._infer_company(di_markdown) == "Apple Inc."
    assert parser._infer_company("no cover page here") is None


def test_filing_date_inference_and_mock_doc():
    cover = "For the quarterly period ended June 28, 2025"
    assert parser._infer_filing_date(cover) == "June 28, 2025"
    assert parser._infer_filing_date("no date in here") is None
    assert parser.parse("acme.pdf").filing_date == "2025-03-30"  # carried on the mock document
