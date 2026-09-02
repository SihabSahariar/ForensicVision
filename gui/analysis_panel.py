"""Degradation analysis dock.

Renders the indicator bars, and opens a technical breakdown when one is
clicked. The heuristic qualifier is displayed permanently rather than buried in
a tooltip - a number on a bar invites over-reading, and the caveat has to be as
visible as the number.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from analysis.analyzer import AnalysisReport
from analysis.base import MetricResult
from app.constants import (
    DEGRADATION_LABELS,
    DEGRADATION_ORDER,
    HEURISTIC_DISCLAIMER,
)
from gui.widgets.common import HLine, ScoreBar, SectionLabel

logger = logging.getLogger(__name__)

__all__ = ["AnalysisPanel", "MetricDetailDialog"]


class MetricDetailDialog(QDialog):
    """Shows the raw measurements behind one indicator."""

    def __init__(self, metric: MetricResult, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Analysis detail - {metric.label}")
        self.setMinimumSize(620, 480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        heading = QLabel(f"{metric.label}: {metric.percent} / 100")
        heading.setProperty("role", "heading")
        layout.addWidget(heading)

        caveat = QLabel(HEURISTIC_DISCLAIMER)
        caveat.setProperty("role", "banner")
        caveat.setWordWrap(True)
        layout.addWidget(caveat)

        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(metric.detail_text())
        text.setStyleSheet(
            "font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 11px;"
        )
        layout.addWidget(text, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class AnalysisPanel(QWidget):
    """Dock widget presenting every degradation indicator.

    Signals:
        analyseRequested: The user asked for a (re-)analysis.
        detailRequested: ``(key)`` - a metric's detail view was opened.
    """

    analyseRequested = pyqtSignal()
    detailRequested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._report: Optional[AnalysisReport] = None
        self._bars: Dict[str, ScoreBar] = {}
        self._build_ui()

    # ------------------------------------------------------------------ build
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll, 1)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        scroll.setWidget(container)

        layout.addWidget(SectionLabel("Degradation indicators"))

        for key in DEGRADATION_ORDER:
            bar = ScoreBar(key, DEGRADATION_LABELS[key])
            bar.clicked.connect(self._on_bar_clicked)
            bar.setContextMenuPolicy(Qt.CustomContextMenu)
            bar.customContextMenuRequested.connect(
                lambda position, name=key: self._show_bar_menu(name, position)
            )
            self._bars[key] = bar
            layout.addWidget(bar)

        layout.addWidget(HLine())

        self._summary = QLabel("No analysis has been run.")
        self._summary.setWordWrap(True)
        self._summary.setProperty("role", "hint")
        layout.addWidget(self._summary)

        self._caveat = QLabel(HEURISTIC_DISCLAIMER)
        self._caveat.setWordWrap(True)
        self._caveat.setProperty("role", "banner")
        self._caveat.setVisible(False)
        layout.addWidget(self._caveat)

        layout.addStretch(1)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(10, 6, 10, 10)
        buttons.setSpacing(6)

        self._analyse_button = QPushButton("Analyse  (A)")
        self._analyse_button.setProperty("accent", True)
        self._analyse_button.clicked.connect(self.analyseRequested.emit)
        buttons.addWidget(self._analyse_button)

        self._detail_button = QPushButton("Detailed analysis")
        self._detail_button.setEnabled(False)
        self._detail_button.clicked.connect(self._show_dominant_detail)
        buttons.addWidget(self._detail_button)

        outer.addLayout(buttons)

    # ------------------------------------------------------------------ state
    @property
    def report(self) -> Optional[AnalysisReport]:
        """The report currently displayed."""
        return self._report

    def set_report(self, report: Optional[AnalysisReport]) -> None:
        """Display ``report``, or clear the panel when ``None``."""
        self._report = report
        if report is None:
            self.clear()
            return

        for key, bar in self._bars.items():
            metric = report.get(key)
            bar.set_score(metric.score if metric else None)
            if metric is not None:
                bar.setToolTip(
                    f"{metric.label}: {metric.percent}/100 ({metric.severity})\n"
                    f"{metric.method}\n\nClick for the full breakdown."
                )

        actionable = report.actionable()
        if actionable:
            names = ", ".join(f"{m.label} ({m.percent})" for m in actionable[:4])
            self._summary.setText(f"Above action threshold: {names}")
        else:
            self._summary.setText(
                "No indicator exceeded the action threshold. Restoration is not "
                "indicated by the measurements."
            )
        self._caveat.setVisible(True)
        self._detail_button.setEnabled(True)

    def clear(self) -> None:
        """Reset every bar to the unknown state."""
        self._report = None
        for bar in self._bars.values():
            bar.set_score(None)
            bar.setToolTip("")
        self._summary.setText("No analysis has been run.")
        self._caveat.setVisible(False)
        self._detail_button.setEnabled(False)

    def set_busy(self, busy: bool) -> None:
        """Disable the analyse button while an analysis is running."""
        self._analyse_button.setEnabled(not busy)
        self._analyse_button.setText("Analysing..." if busy else "Analyse  (A)")

    # ---------------------------------------------------------- context menu
    def _show_bar_menu(self, key: str, position) -> None:
        """Offer detail and copy actions for one indicator."""
        if self._report is None:
            return
        metric = self._report.get(key)
        if metric is None:
            return

        menu = QMenu(self)
        menu.addAction(
            f"Technical detail for {metric.label}...",
            lambda: MetricDetailDialog(metric, self).exec_(),
        )
        menu.addSeparator()
        menu.addAction(
            f"Copy score ({metric.percent}/100)",
            lambda: self._copy(str(metric.percent)),
        )
        menu.addAction(
            "Copy raw measurements",
            lambda: self._copy(metric.detail_text()),
        )
        menu.addAction(
            "Copy every indicator",
            lambda: self._copy(self._all_scores_text()),
        )
        menu.addSeparator()
        menu.addAction("Re-analyse", self.analyseRequested.emit)
        menu.exec_(self._bars[key].mapToGlobal(position))

    def _all_scores_text(self) -> str:
        """Render every indicator as text, for pasting into notes."""
        if self._report is None:
            return ""
        lines = [f"{m.label}: {m.percent}/100 ({m.severity})"
                 for m in self._report.ordered()]
        lines.append("")
        lines.append(HEURISTIC_DISCLAIMER)
        return "\n".join(lines)

    def _copy(self, text: str) -> None:
        """Put ``text`` on the clipboard."""
        QApplication.clipboard().setText(text)

    # --------------------------------------------------------------- handlers
    def _on_bar_clicked(self, key: str) -> None:
        """Open the detail dialog for the clicked indicator."""
        self.detailRequested.emit(key)
        if self._report is None:
            return
        metric = self._report.get(key)
        if metric is None:
            return
        MetricDetailDialog(metric, self).exec_()

    def _show_dominant_detail(self) -> None:
        """Open the detail dialog for the most severe indicator."""
        if self._report is None:
            return
        dominant = self._report.dominant
        if dominant is not None:
            MetricDetailDialog(dominant, self).exec_()
