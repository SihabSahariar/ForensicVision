"""Small reusable widgets shared by the dock panels."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.theme import refresh_style, severity_for

__all__ = [
    "SectionLabel",
    "HLine",
    "ScoreBar",
    "KeyValueTable",
    "CollapsibleSection",
    "BannerLabel",
    "set_property",
]


def set_property(widget: QWidget, name: str, value) -> None:
    """Set a dynamic QSS property and re-polish ``widget``."""
    widget.setProperty(name, value)
    refresh_style(widget)


class SectionLabel(QLabel):
    """An uppercase section heading used inside dock panels."""

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setProperty("role", "section")


class BannerLabel(QLabel):
    """A highlighted advisory strip (used for forensic warnings)."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setProperty("role", "banner")
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)


class HLine(QFrame):
    """A one-pixel horizontal separator."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setProperty("role", "hline")
        self.setFixedHeight(1)
        self.setFrameShape(QFrame.NoFrame)


class ScoreBar(QWidget):
    """A labelled 0-100 severity bar used by the analysis panel.

    Emits :attr:`clicked` so the panel can open a detail view for the metric.
    """

    clicked = pyqtSignal(str)

    def __init__(
        self,
        key: str,
        label: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._key = key
        self._score = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(3)

        header = QHBoxLayout()
        header.setSpacing(6)
        self._name = QLabel(label)
        self._value = QLabel("--")
        self._value.setProperty("role", "mono")
        self._value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._value.setMinimumWidth(34)
        header.addWidget(self._name, 1)
        header.addWidget(self._value, 0)
        layout.addLayout(header)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(9)
        self._bar.setProperty("severity", "low")
        layout.addWidget(self._bar)

        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{label} - click for the technical breakdown")

    @property
    def key(self) -> str:
        """The degradation key this bar represents."""
        return self._key

    @property
    def score(self) -> float:
        """Current normalised score in ``[0, 1]``."""
        return self._score

    def set_score(self, score: Optional[float]) -> None:
        """Update the bar; ``None`` renders an unknown state."""
        if score is None:
            self._score = 0.0
            self._bar.setValue(0)
            self._value.setText("--")
            set_property(self._bar, "severity", "low")
            return
        self._score = max(0.0, min(1.0, float(score)))
        percent = int(round(self._score * 100))
        self._bar.setValue(percent)
        self._value.setText(str(percent))
        set_property(self._bar, "severity", severity_for(self._score))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Emit :attr:`clicked` for left-button releases."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._key)
        super().mouseReleaseEvent(event)


class KeyValueTable(QTableWidget):
    """A compact read-only two-column table for metadata display."""

    def __init__(
        self,
        headers: Sequence[str] = ("Field", "Value"),
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(list(headers))
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setWordWrap(False)
        self.setShowGrid(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setHighlightSections(False)
        self.verticalHeader().setDefaultSectionSize(21)

    def set_rows(self, rows: Iterable[Tuple[str, str]]) -> None:
        """Replace the table contents with ``rows``."""
        items = list(rows)
        self.setRowCount(len(items))
        for index, (key, value) in enumerate(items):
            key_item = QTableWidgetItem(str(key))
            value_item = QTableWidgetItem(str(value))
            value_item.setToolTip(str(value))
            self.setItem(index, 0, key_item)
            self.setItem(index, 1, value_item)

    def add_section(self, title: str) -> None:
        """Append a visually distinct section header row."""
        row = self.rowCount()
        self.insertRow(row)
        item = QTableWidgetItem(title.upper())
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self.setItem(row, 0, item)
        self.setItem(row, 1, QTableWidgetItem(""))
        self.setSpan(row, 0, 1, 2)

    def append_rows(self, rows: Iterable[Tuple[str, str]]) -> None:
        """Append ``rows`` without clearing existing content."""
        for key, value in rows:
            row = self.rowCount()
            self.insertRow(row)
            self.setItem(row, 0, QTableWidgetItem(str(key)))
            value_item = QTableWidgetItem(str(value))
            value_item.setToolTip(str(value))
            self.setItem(row, 1, value_item)


class CollapsibleSection(QWidget):
    """A titled container that can be folded away to save vertical space."""

    toggled = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        expanded: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._expanded = expanded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self._button = QToolButton()
        self._button.setText(title)
        self._button.setCheckable(True)
        self._button.setChecked(expanded)
        self._button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._button.setStyleSheet(
            "QToolButton { font-size: 11px; font-weight: 600; letter-spacing: 0.7px; "
            "text-transform: uppercase; padding: 4px 2px; }"
        )
        self._button.clicked.connect(self._on_toggle)
        outer.addWidget(self._button)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 2, 0, 6)
        self._body_layout.setSpacing(5)
        self._body.setVisible(expanded)
        outer.addWidget(self._body)

    @property
    def body(self) -> QWidget:
        """The container widget clients add their content to."""
        return self._body

    def add_widget(self, widget: QWidget) -> None:
        """Append ``widget`` to the section body."""
        self._body_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        """Append a nested layout to the section body."""
        self._body_layout.addLayout(layout)

    def set_expanded(self, expanded: bool) -> None:
        """Expand or collapse the section."""
        self._expanded = bool(expanded)
        self._button.setChecked(self._expanded)
        self._button.setArrowType(Qt.DownArrow if self._expanded else Qt.RightArrow)
        self._body.setVisible(self._expanded)

    def _on_toggle(self, checked: bool) -> None:
        self.set_expanded(checked)
        self.toggled.emit(checked)
