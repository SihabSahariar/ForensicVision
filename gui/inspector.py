"""Tabbed inspector: one dock hosting every side panel.

The original layout gave each panel its own dock - two on the left, two on the
right, two along the bottom. Between them they consumed roughly 600 px of
width and 220 px of height, squeezing the image into a small central box, and
the taller panels were clipped: the analysis dock showed five of its nine
indicators.

Collapsing them into a single tabbed dock returns that space to the image while
making every panel fully visible when selected. The image is the thing an
examiner is actually looking at, so it gets the room.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.analysis_panel import AnalysisPanel
from gui.case_explorer import CaseExplorer
from gui.history_panel import HistoryPanel
from gui.log_panel import LogPanel
from gui.metadata_panel import MetadataPanel
from gui.restoration_panel import RestorationPanel
from gui.widgets.common import SectionLabel

logger = logging.getLogger(__name__)

__all__ = ["InspectorPanel", "InspectorTab"]


class InspectorTab:
    """Stable identifiers for the inspector's tabs."""

    CASE = "case"
    ANALYSIS = "analysis"
    RESTORE = "restore"
    HISTORY = "history"
    LOG = "log"

    #: Display order and short labels. Labels are kept short deliberately -
    #: five tabs have to fit across a dock narrow enough to leave the image
    #: room, which is the entire point of the consolidation.
    ORDER = (
        (CASE, "Case", "Evidence tree and file metadata"),
        (ANALYSIS, "Analysis", "Degradation indicators"),
        (RESTORE, "Restore", "Restoration models and the staged pipeline"),
        (HISTORY, "History", "Processing history and per-step provenance"),
        (LOG, "Log", "Application log"),
    )


class InspectorPanel(QWidget):
    """Hosts the case, analysis, restoration, history and log panels.

    Signals:
        tabChanged: ``(tab_id)`` whenever the visible panel changes.
    """

    tabChanged = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.case_explorer = CaseExplorer()
        self.metadata_panel = MetadataPanel()
        self.analysis_panel = AnalysisPanel()
        self.restoration_panel = RestorationPanel()
        self.history_panel = HistoryPanel()
        self.log_panel = LogPanel()

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setUsesScrollButtons(True)
        self._tabs.setElideMode(Qt.ElideNone)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._index_for: Dict[str, int] = {}
        self._build()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._tabs)

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        """Populate the tab bar."""
        pages = {
            InspectorTab.CASE: self._build_case_page(),
            InspectorTab.ANALYSIS: self.analysis_panel,
            InspectorTab.RESTORE: self.restoration_panel,
            InspectorTab.HISTORY: self.history_panel,
            InspectorTab.LOG: self.log_panel,
        }
        for tab_id, label, tooltip in InspectorTab.ORDER:
            index = self._tabs.addTab(pages[tab_id], label)
            self._tabs.setTabToolTip(index, tooltip)
            self._index_for[tab_id] = index

    def _build_case_page(self) -> QWidget:
        """Combine the evidence tree and the metadata table in one page.

        They are two views of the same selection - the tree chooses an item,
        the table describes it - so they belong together rather than competing
        for separate docks.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Vertical)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(6, 6, 6, 2)
        top_layout.setSpacing(4)
        top_layout.addWidget(SectionLabel("Case contents"))
        top_layout.addWidget(self.case_explorer, 1)
        splitter.addWidget(top)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(6, 2, 6, 6)
        bottom_layout.setSpacing(4)
        bottom_layout.addWidget(SectionLabel("Metadata and hashes"))
        bottom_layout.addWidget(self.metadata_panel, 1)
        splitter.addWidget(bottom)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)
        return page

    # ----------------------------------------------------------------- public
    @property
    def current_tab(self) -> str:
        """Identifier of the visible tab."""
        index = self._tabs.currentIndex()
        for tab_id, position in self._index_for.items():
            if position == index:
                return tab_id
        return InspectorTab.CASE

    def show_tab(self, tab_id: str) -> None:
        """Bring ``tab_id`` to the front."""
        index = self._index_for.get(tab_id)
        if index is not None:
            self._tabs.setCurrentIndex(index)

    def tab_widget(self) -> QTabWidget:
        """The underlying tab widget, for wiring shortcuts."""
        return self._tabs

    def next_tab(self, delta: int = 1) -> None:
        """Cycle the visible tab by ``delta`` positions, wrapping around."""
        count = self._tabs.count()
        if count:
            self._tabs.setCurrentIndex((self._tabs.currentIndex() + delta) % count)

    def set_tab_badge(self, tab_id: str, badge: str = "") -> None:
        """Append a short marker to a tab's label, e.g. a pending count."""
        index = self._index_for.get(tab_id)
        if index is None:
            return
        label = next(
            (text for key, text, _ in InspectorTab.ORDER if key == tab_id), tab_id
        )
        self._tabs.setTabText(index, f"{label} {badge}".strip())

    # --------------------------------------------------------------- handlers
    def _on_tab_changed(self, index: int) -> None:
        """Re-emit the change with the stable tab identifier."""
        for tab_id, position in self._index_for.items():
            if position == index:
                self.tabChanged.emit(tab_id)
                return
