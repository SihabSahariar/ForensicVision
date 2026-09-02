"""Stylesheet loading and shared palette constants for the GUI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

from app.paths import styles_dir

logger = logging.getLogger(__name__)

__all__ = ["Palette", "load_stylesheet", "apply_theme", "severity_for"]


class Palette:
    """Colour tokens mirrored from ``dark_theme.qss``.

    Widgets that paint themselves (the viewer, histogram plots, ROI overlays)
    read these instead of hard-coding hex strings, so the theme stays coherent.
    """

    BG_0 = "#101216"
    BG_1 = "#161920"
    BG_2 = "#1c2029"
    BG_3 = "#232833"
    VIEWPORT = "#08090c"
    LINE = "#2c323d"
    FG_0 = "#d7dce4"
    FG_1 = "#9aa3b2"
    FG_2 = "#6c7382"
    ACCENT = "#3d8bcd"
    ACCENT_DIM = "#23364a"
    WARN = "#d99a2b"
    ERROR = "#d0555a"
    OK = "#4c9e6b"
    LOCK = "#c8863c"
    ROI = "#4fd0c8"
    CROSSHAIR = "#e0555a"

    #: Colour-blind-safe channel colours for histograms.
    CH_R = "#e06c75"
    CH_G = "#7fbf6a"
    CH_B = "#5f9fe0"
    CH_K = "#c3cad6"

    @staticmethod
    def qcolor(value: str, alpha: int = 255) -> QColor:
        """Return ``value`` as a :class:`QColor` with optional alpha."""
        colour = QColor(value)
        colour.setAlpha(alpha)
        return colour


def severity_for(score: float) -> str:
    """Map a 0..1 degradation score to a QSS ``severity`` property value.

    Args:
        score: Normalised severity in ``[0, 1]``.

    Returns:
        ``"low"``, ``"medium"`` or ``"high"``.
    """
    if score >= 0.66:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def load_stylesheet(name: str = "dark_theme") -> str:
    """Read a ``.qss`` file from :func:`app.paths.styles_dir`.

    Args:
        name: Stylesheet basename without extension.

    Returns:
        The stylesheet text, or an empty string when the file is missing.
    """
    path: Path = styles_dir() / f"{name}.qss"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.error("Stylesheet not found: %s", path)
        return ""


def apply_theme(app: QApplication, name: str = "dark_theme") -> bool:
    """Apply the named stylesheet and a matching Qt palette to ``app``.

    A palette is set in addition to the stylesheet because several native
    widgets (tooltips, disabled text, standard dialogs) ignore QSS.

    Returns:
        ``True`` when a stylesheet was loaded.
    """
    sheet = load_stylesheet(name)
    if sheet:
        app.setStyleSheet(sheet)

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(Palette.BG_0))
    palette.setColor(QPalette.WindowText, QColor(Palette.FG_0))
    palette.setColor(QPalette.Base, QColor("#0c0e12"))
    palette.setColor(QPalette.AlternateBase, QColor("#12151b"))
    palette.setColor(QPalette.ToolTipBase, QColor(Palette.BG_2))
    palette.setColor(QPalette.ToolTipText, QColor(Palette.FG_0))
    palette.setColor(QPalette.Text, QColor(Palette.FG_0))
    palette.setColor(QPalette.Button, QColor(Palette.BG_2))
    palette.setColor(QPalette.ButtonText, QColor(Palette.FG_0))
    palette.setColor(QPalette.BrightText, QColor(Palette.ERROR))
    palette.setColor(QPalette.Link, QColor(Palette.ACCENT))
    palette.setColor(QPalette.Highlight, QColor(Palette.ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(Palette.FG_2))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(Palette.FG_2))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(Palette.FG_2))
    app.setPalette(palette)
    return bool(sheet)


def refresh_style(widget) -> None:
    """Re-apply the stylesheet to ``widget`` after a dynamic property change.

    Qt only re-evaluates property selectors when the style is re-polished.
    """
    style: Optional[object] = widget.style()
    if style is None:  # pragma: no cover - defensive
        return
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
