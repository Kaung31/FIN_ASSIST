"""Describe cropped charts/figures with the vision model so they become searchable + citable.

A chart is useless to a text retriever until it's described. Ask the model for a factual
description: chart type, axes/units, series, and the concrete values/trends it can read. The
description is then chunked and embedded like any other text (content_type="figure").

In MOCK_MODE the LLM client returns a canned description, so figures flow through the pipeline
without a real vision call (and without a real cropped image).
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import get_settings
from finassist.analysis.models import Figure
from finassist.llm.client import LLMClient

logger = logging.getLogger(__name__)

_INSTRUCTION = (
    "Describe this financial chart factually for search indexing. State the chart type, what the "
    "axes/units are, each data series, and the specific values or trends shown. Report only what "
    "is visible. Do not give analysis, opinions, or investment advice."
)


def describe(
    figures: list[Figure], llm: LLMClient, *, instruction: str = _INSTRUCTION
) -> list[Figure]:
    """Fill figure.description for each figure (in place); returns the same list."""
    for figure in figures:
        image_bytes = _read_image(figure)
        if not image_bytes and not get_settings().mock_mode:
            logger.warning("Figure on page %s has no image to describe; skipping.", figure.page)
            continue
        figure.description = llm.describe_image(image_bytes, instruction)
    return figures


def _read_image(figure: Figure) -> bytes:
    if figure.image_path and Path(figure.image_path).exists():
        return Path(figure.image_path).read_bytes()
    return b""  # mock mode: the canned VLM ignores the bytes
