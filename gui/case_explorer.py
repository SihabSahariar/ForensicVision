"""Case explorer dock.

Presents the case as a tree of Evidence / Derivatives / Analysis / Reports and
emits selection signals. Derivatives are nested under the evidence they descend
from, so the chain of custody is visible in the tree rather than only in the
provenance records.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLineEdit,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.case_manager import CaseManager
from database.models import Derivative, Evidence
from gui.theme import Palette

logger = logging.getLogger(__name__)

__all__ = ["CaseExplorer", "NodeKind"]


class NodeKind:
    """Role values stored on tree items."""

    ROOT = "root"
    GROUP = "group"
    EVIDENCE = "evidence"
    DERIVATIVE = "derivative"
    REPORT = "report"
    ANALYSIS = "analysis"


_ROLE_KIND = Qt.UserRole + 1
_ROLE_ID = Qt.UserRole + 2
_ROLE_PATH = Qt.UserRole + 3


class CaseExplorer(QWidget):
    """Tree view of the open case.

    Signals:
        evidenceSelected: ``(evidence_id)``
        derivativeSelected: ``(derivative_id)``
        reportSelected: ``(path)``
        importRequested: The user asked to add evidence.
        compareRequested: ``(derivative_id)`` - compare against its original.
        revealRequested: ``(path)`` - show the file in the OS file manager.
    """

    evidenceSelected = pyqtSignal(int)
    derivativeSelected = pyqtSignal(int)
    reportSelected = pyqtSignal(str)
    importRequested = pyqtSignal()
    compareRequested = pyqtSignal(int)
    revealRequested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._case: Optional[CaseManager] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter items...")
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Item", "Detail"])
        self._tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tree.setUniformRowHeights(True)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(self._on_double_clicked)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(self._tree, 1)

    # ------------------------------------------------------------------ public
    def set_case(self, case: Optional[CaseManager]) -> None:
        """Bind the explorer to ``case`` and refresh."""
        self._case = case
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the tree from the database."""
        expanded = self._expanded_keys()
        selected = self._selected_key()
        self._tree.clear()

        if self._case is None:
            placeholder = QTreeWidgetItem(["No case is open", ""])
            placeholder.setForeground(0, QBrush(QColor(Palette.FG_2)))
            self._tree.addTopLevelItem(placeholder)
            return

        case = self._case
        counts = case.counts()
        root = QTreeWidgetItem([case.case_id, case.case.title or ""])
        root.setData(0, _ROLE_KIND, NodeKind.ROOT)
        font = root.font(0)
        font.setBold(True)
        root.setFont(0, font)
        self._tree.addTopLevelItem(root)

        evidence_group = self._group(root, "Evidence", counts.get("evidence", 0))
        derivative_group = self._group(root, "Derivatives", counts.get("derivatives", 0))
        analysis_group = self._group(root, "Analysis", counts.get("analyses", 0))
        report_group = self._group(root, "Reports", counts.get("reports", 0))

        derivatives_by_evidence: Dict[int, List[Derivative]] = {}
        for derivative in case.list_derivatives():
            derivatives_by_evidence.setdefault(derivative.evidence_id, []).append(
                derivative
            )

        for evidence in case.list_evidence():
            item = QTreeWidgetItem(
                [evidence.original_filename, f"{evidence.dimensions}"]
            )
            item.setData(0, _ROLE_KIND, NodeKind.EVIDENCE)
            item.setData(0, _ROLE_ID, int(evidence.id))
            item.setData(0, _ROLE_PATH, evidence.stored_path)
            item.setToolTip(
                0,
                f"{evidence.original_filename}\n"
                f"SHA-256: {evidence.sha256}\n"
                f"{evidence.dimensions}, {evidence.bit_depth}-bit, "
                f"{evidence.size_bytes:,} bytes",
            )
            evidence_group.addChild(item)

            for derivative in derivatives_by_evidence.get(evidence.id, []):
                child = QTreeWidgetItem(
                    [Path(derivative.path).name,
                     f"{derivative.width} x {derivative.height}"]
                )
                child.setData(0, _ROLE_KIND, NodeKind.DERIVATIVE)
                child.setData(0, _ROLE_ID, int(derivative.id))
                child.setData(0, _ROLE_PATH, derivative.path)
                if derivative.model_kind == "neural":
                    child.setForeground(0, QBrush(QColor(Palette.WARN)))
                    child.setToolTip(
                        0,
                        f"{Path(derivative.path).name}\n"
                        f"Pipeline: {derivative.model_name}\n"
                        f"SHA-256: {derivative.sha256}\n\n"
                        "This derivative was produced with a learned model.\n"
                        + (
                            "Detail in it may be synthesised rather than "
                            "recovered."
                            if (derivative.provenance or {}).get("may_synthesise")
                            else "No step in it is capable of synthesising "
                            "structure."
                        ),
                    )
                else:
                    child.setToolTip(
                        0,
                        f"{Path(derivative.path).name}\n"
                        f"Pipeline: {derivative.model_name}\n"
                        f"SHA-256: {derivative.sha256}",
                    )
                item.addChild(child)

        # The flat Derivatives group mirrors the nested view for quick access.
        for derivatives in derivatives_by_evidence.values():
            for derivative in derivatives:
                flat = QTreeWidgetItem(
                    [Path(derivative.path).name, derivative.operation]
                )
                flat.setData(0, _ROLE_KIND, NodeKind.DERIVATIVE)
                flat.setData(0, _ROLE_ID, int(derivative.id))
                flat.setData(0, _ROLE_PATH, derivative.path)
                derivative_group.addChild(flat)

        for record in case.repository.list_reports(case.case_pk):
            item = QTreeWidgetItem(
                [Path(record.path).name, record.created_at.strftime("%Y-%m-%d %H:%M")]
            )
            item.setData(0, _ROLE_KIND, NodeKind.REPORT)
            item.setData(0, _ROLE_PATH, record.path)
            report_group.addChild(item)

        analysis_count = counts.get("analyses", 0)
        if analysis_count:
            note = QTreeWidgetItem(
                [f"{analysis_count} analysis record(s)", "see Analysis dock"]
            )
            note.setForeground(0, QBrush(QColor(Palette.FG_1)))
            analysis_group.addChild(note)

        self._tree.expandToDepth(1)
        self._restore_expanded(expanded)
        self._restore_selection(selected)
        self._apply_filter(self._filter.text())

    # ---------------------------------------------------------------- helpers
    def _group(self, parent: QTreeWidgetItem, label: str, count: int) -> QTreeWidgetItem:
        """Create a category node under ``parent``."""
        item = QTreeWidgetItem([label, str(count)])
        item.setData(0, _ROLE_KIND, NodeKind.GROUP)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setForeground(0, QBrush(QColor(Palette.FG_1)))
        parent.addChild(item)
        return item

    def _iter_items(self):
        """Yield every item in the tree."""
        stack = [self._tree.topLevelItem(i) for i in range(self._tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item is None:
                continue
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _expanded_keys(self) -> set:
        """Return keys of expanded nodes so state survives a refresh."""
        return {
            item.text(0) for item in self._iter_items() if item.isExpanded()
        }

    def _restore_expanded(self, keys: set) -> None:
        """Re-expand nodes whose labels were expanded before the refresh."""
        for item in self._iter_items():
            if item.text(0) in keys:
                item.setExpanded(True)

    def _selected_key(self):
        """Return ``(kind, id)`` of the current selection."""
        items = self._tree.selectedItems()
        if not items:
            return None
        item = items[0]
        return item.data(0, _ROLE_KIND), item.data(0, _ROLE_ID)

    def _restore_selection(self, key) -> None:
        """Reselect the previously selected node, if it still exists."""
        if key is None:
            return
        kind, identifier = key
        for item in self._iter_items():
            if item.data(0, _ROLE_KIND) == kind and item.data(0, _ROLE_ID) == identifier:
                self._tree.setCurrentItem(item)
                return

    def _apply_filter(self, text: str) -> None:
        """Hide leaf nodes that do not match ``text``."""
        needle = text.strip().lower()
        for item in self._iter_items():
            kind = item.data(0, _ROLE_KIND)
            if kind in (NodeKind.ROOT, NodeKind.GROUP):
                continue
            match = (
                not needle
                or needle in item.text(0).lower()
                or needle in item.text(1).lower()
            )
            item.setHidden(not match)

    # --------------------------------------------------------------- handlers
    def _on_selection_changed(self) -> None:
        """Emit the appropriate selection signal."""
        items = self._tree.selectedItems()
        if not items:
            return
        item = items[0]
        kind = item.data(0, _ROLE_KIND)
        identifier = item.data(0, _ROLE_ID)
        if kind == NodeKind.EVIDENCE and identifier is not None:
            self.evidenceSelected.emit(int(identifier))
        elif kind == NodeKind.DERIVATIVE and identifier is not None:
            self.derivativeSelected.emit(int(identifier))
        elif kind == NodeKind.REPORT:
            path = item.data(0, _ROLE_PATH)
            if path:
                self.reportSelected.emit(str(path))

    def _on_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        """Compare a derivative against its original on double click."""
        if item.data(0, _ROLE_KIND) == NodeKind.DERIVATIVE:
            identifier = item.data(0, _ROLE_ID)
            if identifier is not None:
                self.compareRequested.emit(int(identifier))

    def _show_context_menu(self, position) -> None:
        """Build the context menu for the item under the cursor."""
        item = self._tree.itemAt(position)
        menu = QMenu(self)

        menu.addAction("Import evidence...", self.importRequested.emit)

        if item is not None:
            kind = item.data(0, _ROLE_KIND)
            path = item.data(0, _ROLE_PATH)
            identifier = item.data(0, _ROLE_ID)

            if kind == NodeKind.DERIVATIVE and identifier is not None:
                menu.addSeparator()
                menu.addAction(
                    "Compare with original",
                    lambda: self.compareRequested.emit(int(identifier)),
                )
            if path:
                menu.addSeparator()
                menu.addAction(
                    "Show in file manager",
                    lambda: self.revealRequested.emit(str(path)),
                )

        menu.addSeparator()
        menu.addAction("Refresh", self.refresh)
        menu.exec_(self._tree.viewport().mapToGlobal(position))
