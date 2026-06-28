"""Markdown table handling + financial number parsing.

LlamaParse/Docling return financial statements as markdown tables. We preserve them losslessly
(raw cell strings AND a markdown rendering): markdown is what we feed the LLM; the raw cells are
what the math guardrail parses into numbers. The LLM never reads numbers out of prose when a
table exists.
"""

from __future__ import annotations

import re

from finassist.analysis.models import FinancialTable

# Matches the first numeric token in a messy cell, e.g. "$1,250.5 *" -> "1,250.5".
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_BLANK_CELLS = {"", "-", "—", "–", "n/a", "na", "nm", "—%", "*"}


def parse_number(cell: str | None) -> float | None:
    """Parse a financial table cell into a float, or None when it isn't a number.

    Handles thousands separators, currency/percent symbols, parenthesised negatives, and
    common blank markers ("—", "N/A", ...).
    """
    if cell is None:
        return None
    s = str(cell).strip()
    if s.lower() in _BLANK_CELLS:
        return None

    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]
    s = s.replace("$", "").replace("%", "").replace(",", "").strip().rstrip("*").strip()

    try:
        value = float(s)
    except ValueError:
        match = _NUM_RE.search(s)
        if not match:
            return None
        try:
            value = float(match.group().replace(",", ""))
        except ValueError:
            return None
    return -value if negative else value


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render a 2D grid as a GitHub-flavoured markdown table (first row is the header)."""
    if not rows:
        return ""
    header, *body = rows
    width = len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        # pad/truncate ragged rows to the header width so the markdown stays well-formed.
        cells = (row + [""] * width)[:width]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _is_separator_row(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(c and set(c) <= set("-: ") and "-" in c for c in cells)


def markdown_to_rows(block: str) -> list[list[str]]:
    """Parse a markdown table block into rows, dropping the `---` separator row."""
    rows = []
    for line in block.splitlines():
        if not _is_table_row(line) or _is_separator_row(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def tables_from_markdown(
    markdown: str, *, page: int = 1, caption: str | None = None
) -> list[FinancialTable]:
    """Extract every markdown table block in ``markdown`` as a FinancialTable."""
    tables: list[FinancialTable] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            rows = markdown_to_rows("\n".join(buffer))
            if len(rows) >= 2:  # header + at least one data row
                tables.append(
                    FinancialTable(
                        page=page, caption=caption, rows=rows, markdown=rows_to_markdown(rows)
                    )
                )
        buffer.clear()

    for line in markdown.splitlines():
        if _is_table_row(line):
            buffer.append(line)
        else:
            flush()
    flush()
    return tables


_HTML_TABLE_RE = re.compile(r"<table.*?</table>", re.IGNORECASE | re.DOTALL)


def strip_tables(markdown: str) -> str:
    """Return ``markdown`` with tables removed (both pipe tables and DI's HTML <table> blocks)."""
    without_html = _HTML_TABLE_RE.sub("", markdown)
    return "\n".join(line for line in without_html.splitlines() if not _is_table_row(line))


def to_financial_tables(di_tables) -> list[FinancialTable]:  # noqa: ANN001 - DI DocumentTable list
    """Map Azure Document Intelligence structured tables into FinancialTable objects.

    DI's markdown renders tables as HTML, so we reconstruct the grid from the structured cells
    (row_index/column_index) instead — that's what the math guardrail parses for numbers.
    """
    tables: list[FinancialTable] = []
    for table in di_tables or []:
        n_rows, n_cols = table.row_count, table.column_count
        grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        for cell in table.cells:
            row, col = cell.row_index, cell.column_index
            if 0 <= row < n_rows and 0 <= col < n_cols:
                grid[row][col] = (cell.content or "").strip()

        page = 1
        if table.bounding_regions:
            page = table.bounding_regions[0].page_number
        caption = None
        if getattr(table, "caption", None) and table.caption:
            caption = table.caption.content

        tables.append(
            FinancialTable(page=page, caption=caption, rows=grid, markdown=rows_to_markdown(grid))
        )
    return tables
