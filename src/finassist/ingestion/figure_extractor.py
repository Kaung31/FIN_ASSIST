"""Crop chart/figure regions out of the PDF so the vision model can describe them.

Uses PyMuPDF (fitz) to render each figure's bounding box to a PNG and set Figure.image_path.
In MOCK_MODE there's no real PDF, so this is a no-op (descriptions come from the mock VLM).
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import get_settings
from finassist.analysis.models import Figure

logger = logging.getLogger(__name__)


def crop_figures(
    pdf_bytes: bytes, figures: list[Figure], out_dir: str, *, zoom: float = 2.0
) -> list[Figure]:
    """Render each figure's bbox to a PNG; return figures with image_path populated."""
    if get_settings().mock_mode or not pdf_bytes:
        return figures  # no real PDF to crop in mock mode

    import fitz  # PyMuPDF

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for index, figure in enumerate(figures):
            if figure.bbox is None:
                continue
            page = document[figure.page - 1]
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom), clip=fitz.Rect(*figure.bbox)
            )
            path = out / f"figure_p{figure.page}_{index}.png"
            pixmap.save(str(path))
            figure.image_path = str(path)
            logger.info("Cropped figure on page %s -> %s", figure.page, path)
    finally:
        document.close()
    return figures
