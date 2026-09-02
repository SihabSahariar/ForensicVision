"""Metadata and hash dock.

Shows file properties, all three digests and the extracted EXIF. Metadata is
only ever read from evidence - the panel has no editing affordances, because
the application must never be the reason a stored original changed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.image_io import ImageData
from forensic.hashing import MD5_ADVISORY, HashSet
from forensic.metadata import FileMetadata, format_gps, human_size
from gui.widgets.common import KeyValueTable

logger = logging.getLogger(__name__)

__all__ = ["MetadataPanel"]


class MetadataPanel(QWidget):
    """Dock widget presenting file, hash and EXIF metadata.

    Signals:
        verifyRequested: The user asked to re-verify the file's integrity.
    """

    verifyRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._metadata: Optional[FileMetadata] = None
        self._hashes: Optional[HashSet] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter fields...")
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        self._table = KeyValueTable()
        layout.addWidget(self._table, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self._copy_button = QPushButton("Copy SHA-256")
        self._copy_button.setEnabled(False)
        self._copy_button.clicked.connect(self._copy_sha256)
        buttons.addWidget(self._copy_button)

        self._verify_button = QPushButton("Verify integrity")
        self._verify_button.setEnabled(False)
        self._verify_button.clicked.connect(self.verifyRequested.emit)
        buttons.addWidget(self._verify_button)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------ public
    def set_metadata(
        self,
        metadata: Optional[FileMetadata],
        hashes: Optional[HashSet] = None,
        image: Optional[ImageData] = None,
        title: str = "",
    ) -> None:
        """Populate the panel.

        Args:
            metadata: Extracted file metadata.
            hashes: Digests for the file.
            image: Decoded image, used for pixel-level properties.
            title: Optional heading, e.g. the derivative label.
        """
        self._metadata = metadata
        self._hashes = hashes
        self._table.setRowCount(0)

        if metadata is None and image is None:
            self._copy_button.setEnabled(False)
            self._verify_button.setEnabled(False)
            return

        if title:
            self._table.add_section(title)

        self._table.add_section("File")
        if metadata is not None:
            self._table.append_rows(
                [
                    ("Filename", metadata.filename),
                    ("Path", metadata.path),
                    ("Size", f"{human_size(metadata.size_bytes)} "
                             f"({metadata.size_bytes:,} bytes)"),
                    ("Format", metadata.container or "unknown"),
                    ("Dimensions", metadata.dimensions),
                    ("Megapixels", str(metadata.megapixels or "unknown")),
                    ("Channels", str(metadata.channels or "unknown")),
                    ("Bit depth", f"{metadata.bit_depth or 'unknown'} bits/channel"),
                    ("ICC profile", "present" if metadata.icc_profile_present else "absent"),
                    ("Modified (UTC)",
                     metadata.mtime.strftime("%Y-%m-%d %H:%M:%S")
                     if metadata.mtime else "unknown"),
                ]
            )
        if image is not None:
            self._table.append_rows(
                [
                    ("Decoded dtype", str(image.dtype)),
                    ("Alpha channel", "yes" if image.has_alpha else "no"),
                ]
            )

        if hashes is not None:
            self._table.add_section("Hashes")
            self._table.append_rows(
                [
                    ("SHA-256 (primary)", hashes.sha256),
                    ("SHA-512", hashes.sha512),
                    ("MD5 (legacy)", hashes.md5),
                    ("MD5 note", MD5_ADVISORY),
                ]
            )
            self._copy_button.setEnabled(True)
            self._verify_button.setEnabled(True)

        if metadata is not None and metadata.gps:
            self._table.add_section("GPS")
            coordinates = format_gps(metadata.gps)
            if coordinates:
                self._table.append_rows([("Coordinates", coordinates)])
            self._table.append_rows(
                [(key.replace("_", " ").title(), value)
                 for key, value in metadata.gps.items()]
            )

        if metadata is not None:
            highlights = metadata.highlights()
            if highlights:
                self._table.add_section("EXIF highlights")
                self._table.append_rows(highlights)

            if metadata.exif:
                self._table.add_section(f"EXIF (all {len(metadata.exif)} tags)")
                self._table.append_rows(sorted(metadata.exif.items()))

            if metadata.warnings:
                self._table.add_section("Extraction warnings")
                self._table.append_rows(
                    [(f"Warning {i + 1}", w) for i, w in enumerate(metadata.warnings)]
                )

        self._apply_filter(self._filter.text())

    def clear(self) -> None:
        """Empty the panel."""
        self.set_metadata(None, None, None)

    # --------------------------------------------------------------- handlers
    def _apply_filter(self, text: str) -> None:
        """Hide rows that do not match ``text``."""
        needle = text.strip().lower()
        for row in range(self._table.rowCount()):
            if not needle:
                self._table.setRowHidden(row, False)
                continue
            key_item = self._table.item(row, 0)
            value_item = self._table.item(row, 1)
            haystack = " ".join(
                item.text().lower() for item in (key_item, value_item) if item
            )
            self._table.setRowHidden(row, needle not in haystack)

    def _copy_sha256(self) -> None:
        """Copy the primary digest to the clipboard."""
        if self._hashes is None:
            return
        QApplication.clipboard().setText(self._hashes.sha256)
        logger.info("SHA-256 copied to clipboard")
