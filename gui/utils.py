"""Small GUI helpers shared between panels and dialogs."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QMessageBox, QWidget

from app.constants import SYNTHESIS_WARNING

logger = logging.getLogger(__name__)

__all__ = [
    "reveal_in_file_manager",
    "open_with_default_application",
    "confirm_synthesis",
    "show_error",
    "elide",
]


def reveal_in_file_manager(path: os.PathLike | str) -> bool:
    """Show ``path`` in the platform's file manager.

    Returns:
        ``True`` when a file manager was launched.
    """
    target = Path(path)
    try:
        if sys.platform.startswith("win"):
            if target.is_file():
                subprocess.Popen(["explorer", "/select,", str(target)])
            else:
                os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":  # pragma: no cover - not a target platform
            subprocess.Popen(["open", "-R", str(target)])
        else:
            directory = target if target.is_dir() else target.parent
            subprocess.Popen(["xdg-open", str(directory)])
        return True
    except Exception as exc:
        logger.warning("Could not reveal %s: %s", target, exc)
        return False


def open_with_default_application(path: os.PathLike | str) -> bool:
    """Open ``path`` with the OS default handler.

    Returns:
        ``True`` when the handler was launched.
    """
    target = Path(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":  # pragma: no cover
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return True
    except Exception as exc:
        logger.warning("Could not open %s: %s", target, exc)
        return False


def confirm_synthesis(
    parent: Optional[QWidget],
    operation: str,
    extra: str = "",
) -> bool:
    """Ask the investigator to confirm a potentially generative operation.

    Args:
        parent: Dialog parent.
        operation: Short description, e.g. ``"Real-ESRGAN x4plus"``.
        extra: Additional model-specific caution text.

    Returns:
        ``True`` when the user chose to proceed.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Generative operation")
    box.setText(f"<b>{operation}</b> may synthesise image content.")
    body = SYNTHESIS_WARNING
    if extra:
        body = f"{body}\n\n{extra}"
    box.setInformativeText(body)
    box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
    box.setDefaultButton(QMessageBox.Cancel)
    box.button(QMessageBox.Ok).setText("Proceed")
    return box.exec_() == QMessageBox.Ok


def show_error(
    parent: Optional[QWidget], title: str, message: str, detail: str = ""
) -> None:
    """Display an error dialog with optional expandable detail."""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    if detail:
        box.setDetailedText(detail)
    box.setStandardButtons(QMessageBox.Ok)
    box.exec_()


def elide(text: str, limit: int = 60) -> str:
    """Truncate ``text`` in the middle so both ends stay readable."""
    if len(text) <= limit:
        return text
    keep = (limit - 3) // 2
    return f"{text[:keep]}...{text[-keep:]}"
