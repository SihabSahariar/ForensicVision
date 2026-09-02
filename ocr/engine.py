"""Optional OCR support.

OCR is optional (S25): the application works fully without it, and reports
exactly which engine produced a reading. Every result carries the disclaimer
that a character sequence read from an enhanced image reflects the *derivative*
and is not proof of the content of the degraded original.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.constants import OCR_DISCLAIMER
from core.image_utils import ensure_uint8_rgb

logger = logging.getLogger(__name__)

__all__ = ["OcrResult", "OcrEngine", "available_engines", "run_ocr"]


@dataclass
class OcrResult:
    """The outcome of one OCR pass."""

    text: str = ""
    engine: str = ""
    lines: List[Tuple[str, float]] = field(default_factory=list)
    mean_confidence: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether the engine produced a reading."""
        return not self.error

    def summary(self) -> str:
        """Return a display string including the confidence."""
        if self.error:
            return f"({self.error})"
        if not self.text.strip():
            return "(no text detected)"
        return self.text.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "engine": self.engine,
            "text": self.text,
            "lines": [{"text": t, "confidence": c} for t, c in self.lines],
            "mean_confidence": round(self.mean_confidence, 4),
            "error": self.error,
            "disclaimer": OCR_DISCLAIMER,
        }


def _tesseract_available() -> bool:
    """Whether pytesseract and the tesseract binary are both present."""
    try:
        import pytesseract  # noqa: PLC0415, F401
    except ImportError:
        return False
    from app.config import get_config

    configured = get_config().tesseract_cmd
    if configured:
        return bool(shutil.which(configured) or configured)
    return shutil.which("tesseract") is not None


def _paddle_available() -> bool:
    """Whether PaddleOCR can be imported."""
    try:
        import paddleocr  # noqa: PLC0415, F401

        return True
    except Exception:
        return False


def available_engines() -> List[str]:
    """Return the OCR engines usable on this installation."""
    engines = []
    if _tesseract_available():
        engines.append("tesseract")
    if _paddle_available():
        engines.append("paddleocr")
    return engines


def _preprocess(image: np.ndarray, upscale: int = 2) -> np.ndarray:
    """Prepare a region for OCR.

    Small text benefits substantially from a modest interpolated enlargement
    and local contrast normalisation. This is a *reading aid* applied only to
    the OCR input; it never touches the image shown to the investigator or any
    exported derivative.
    """
    rgb = ensure_uint8_rgb(image)[..., :3]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if upscale > 1:
        gray = cv2.resize(
            gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
        )
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


class OcrEngine:
    """Thin wrapper over whichever OCR backend is installed."""

    def __init__(self, engine: str = "auto") -> None:
        """Create the wrapper.

        Args:
            engine: ``"auto"``, ``"tesseract"``, ``"paddleocr"`` or ``"none"``.
        """
        self._requested = engine
        self._paddle = None

    def resolve(self) -> Optional[str]:
        """Return the engine that will actually be used."""
        if self._requested == "none":
            return None
        engines = available_engines()
        if not engines:
            return None
        if self._requested in engines:
            return self._requested
        return engines[0]

    def read(
        self, image: np.ndarray, preprocess: bool = True, upscale: int = 2
    ) -> OcrResult:
        """Run OCR over ``image``.

        Args:
            image: Region to read.
            preprocess: Apply the OCR-only reading aid.
            upscale: Enlargement factor used by the reading aid.

        Returns:
            An :class:`OcrResult`; failures are reported in ``error`` rather
            than raised, because a failed reading is a normal outcome.
        """
        engine = self.resolve()
        if engine is None:
            return OcrResult(
                error=(
                    "No OCR engine is installed. Install pytesseract with the "
                    "Tesseract binary, or paddleocr."
                )
            )

        prepared = _preprocess(image, upscale) if preprocess else (
            cv2.cvtColor(ensure_uint8_rgb(image)[..., :3], cv2.COLOR_RGB2GRAY)
        )

        try:
            if engine == "tesseract":
                return self._read_tesseract(prepared)
            return self._read_paddle(prepared)
        except Exception as exc:
            logger.exception("OCR failed")
            return OcrResult(engine=engine, error=str(exc))

    # ---------------------------------------------------------------- engines
    def _read_tesseract(self, gray: np.ndarray) -> OcrResult:
        """Run Tesseract and collect per-word confidences."""
        import pytesseract  # noqa: PLC0415
        from app.config import get_config

        configured = get_config().tesseract_cmd
        if configured:
            pytesseract.pytesseract.tesseract_cmd = configured

        data = pytesseract.image_to_data(
            gray, output_type=pytesseract.Output.DICT
        )
        words: List[Tuple[str, float]] = []
        for text, confidence in zip(data.get("text", []), data.get("conf", [])):
            cleaned = str(text).strip()
            try:
                score = float(confidence)
            except (TypeError, ValueError):
                score = -1.0
            if cleaned and score >= 0:
                words.append((cleaned, score / 100.0))

        text = pytesseract.image_to_string(gray).strip()
        mean = float(np.mean([c for _, c in words])) if words else 0.0
        return OcrResult(
            text=text, engine="tesseract", lines=words, mean_confidence=mean
        )

    def _read_paddle(self, gray: np.ndarray) -> OcrResult:
        """Run PaddleOCR and collect per-line confidences."""
        from paddleocr import PaddleOCR  # noqa: PLC0415

        if self._paddle is None:
            self._paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        raw = self._paddle.ocr(rgb, cls=True)

        lines: List[Tuple[str, float]] = []
        for page in raw or []:
            for entry in page or []:
                try:
                    text, confidence = entry[1]
                    lines.append((str(text), float(confidence)))
                except (IndexError, TypeError, ValueError):
                    continue

        mean = float(np.mean([c for _, c in lines])) if lines else 0.0
        return OcrResult(
            text="\n".join(t for t, _ in lines),
            engine="paddleocr",
            lines=lines,
            mean_confidence=mean,
        )


def run_ocr(image: np.ndarray, engine: str = "auto") -> OcrResult:
    """Convenience wrapper around :class:`OcrEngine`."""
    return OcrEngine(engine).read(image)
