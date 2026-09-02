"""Optional OCR integration (Tesseract / PaddleOCR)."""

from ocr.engine import OcrEngine, OcrResult, available_engines, run_ocr

__all__ = ["OcrEngine", "OcrResult", "available_engines", "run_ocr"]
