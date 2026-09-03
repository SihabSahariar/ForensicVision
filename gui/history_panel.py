"""Processing history dock.

Renders the derivative tree for the selected evidence, with each step's model,
parameters, timing and input/output digests. The history is append-only: there
is no UI affordance for editing or deleting an entry, and Forensic Safe Mode
additionally refuses such an operation at the guard level.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QHeaderView,
    QMenu,
    QPlainTextEdit,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.case_manager import CaseManager
from database.models import Derivative, Evidence, ProcessingStep
from gui.theme import Palette
from gui.widgets.common import SectionLabel

logger = logging.getLogger(__name__)

__all__ = ["HistoryPanel"]

_ROLE_KIND = Qt.UserRole + 1
_ROLE_ID = Qt.UserRole + 2


class HistoryPanel(QWidget):
    """Shows the derivation tree and per-step technical detail.

    Signals:
        derivativeSelected: ``(derivative_id)`` when a node is chosen.
        compareRequested: ``(derivative_id)`` to open the comparison view.
        revealRequested: ``(path)`` to show a file in the OS file manager.
    """

    derivativeSelected = pyqtSignal(int)
    compareRequested = pyqtSignal(int)
    revealRequested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._case: Optional[CaseManager] = None
        self._evidence: Optional[Evidence] = None
        self._steps: Dict[int, List[ProcessingStep]] = {}
        self._derivatives: Dict[int, Derivative] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter, 1)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        top_layout.addWidget(SectionLabel("Processing history"))

        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["Step", "Device", "Time"])
        self._tree.setUniformRowHeights(True)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        top_layout.addWidget(self._tree, 1)
        splitter.addWidget(top)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)
        bottom_layout.addWidget(SectionLabel("Step detail"))
        self._detail = QPlainTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setStyleSheet(
            "font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 11px;"
        )
        bottom_layout.addWidget(self._detail, 1)
        splitter.addWidget(bottom)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    # ------------------------------------------------------------------ public
    def set_case(self, case: Optional[CaseManager]) -> None:
        """Bind the panel to ``case``."""
        self._case = case
        self._evidence = None
        self.refresh()

    def set_evidence(self, evidence: Optional[Evidence]) -> None:
        """Show the history for ``evidence``."""
        self._evidence = evidence
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the tree from the database."""
        self._tree.clear()
        self._detail.clear()
        self._steps.clear()
        self._derivatives.clear()

        if self._case is None or self._evidence is None:
            root = QTreeWidgetItem(["No evidence selected", "", ""])
            root.setForeground(0, QBrush(QColor(Palette.FG_2)))
            self._tree.addTopLevelItem(root)
            return

        case = self._case
        evidence = self._evidence

        root = QTreeWidgetItem([f"Original: {evidence.original_filename}",
                                "", ""])
        root.setData(0, _ROLE_KIND, "evidence")
        root.setData(0, _ROLE_ID, int(evidence.id))
        root.setToolTip(0, f"SHA-256: {evidence.sha256}")
        font = root.font(0)
        font.setBold(True)
        root.setFont(0, font)
        self._tree.addTopLevelItem(root)

        for step in case.repository.list_steps(case.case_pk, evidence.id):
            key = int(step.derivative_id or 0)
            self._steps.setdefault(key, []).append(step)

        derivatives = case.list_derivatives(evidence.id)
        for derivative in derivatives:
            self._derivatives[int(derivative.id)] = derivative

        children: Dict[Optional[int], List[Derivative]] = {}
        for derivative in derivatives:
            children.setdefault(derivative.parent_derivative_id, []).append(derivative)

        self._add_derivatives(root, children, None)
        self._tree.expandAll()

    # ---------------------------------------------------------------- helpers
    def _add_derivatives(
        self,
        parent_item: QTreeWidgetItem,
        children: Dict[Optional[int], List[Derivative]],
        parent_id: Optional[int],
    ) -> None:
        """Recursively add derivative nodes and their steps."""
        for derivative in children.get(parent_id, []):
            label = Path(derivative.path).name
            item = QTreeWidgetItem([label, "", ""])
            item.setData(0, _ROLE_KIND, "derivative")
            item.setData(0, _ROLE_ID, int(derivative.id))
            if derivative.model_kind == "neural":
                item.setForeground(0, QBrush(QColor(Palette.WARN)))
            parent_item.addChild(item)

            for step in self._steps.get(int(derivative.id), []):
                step_item = QTreeWidgetItem(
                    [
                        f"{step.sequence + 1}. {step.model_name}",
                        step.device,
                        f"{step.duration_s:.2f}s",
                    ]
                )
                step_item.setData(0, _ROLE_KIND, "step")
                step_item.setData(0, _ROLE_ID, int(step.id))
                if step.model_kind == "neural":
                    step_item.setForeground(0, QBrush(QColor(Palette.WARN)))
                item.addChild(step_item)

            self._add_derivatives(item, children, int(derivative.id))

    def _find_step(self, step_id: int) -> Optional[ProcessingStep]:
        """Return the step with the given primary key."""
        for steps in self._steps.values():
            for step in steps:
                if int(step.id) == step_id:
                    return step
        return None

    # ---------------------------------------------------------- context menu
    def _show_context_menu(self, position) -> None:
        """Offer per-node actions: view, compare, copy digests, reveal."""
        item = self._tree.itemAt(position)
        if item is None:
            return

        kind = item.data(0, _ROLE_KIND)
        identifier = item.data(0, _ROLE_ID)
        menu = QMenu(self)

        if kind == "derivative" and identifier is not None:
            derivative = self._derivatives.get(int(identifier))
            if derivative is not None:
                menu.addAction(
                    "View this derivative",
                    lambda: self.derivativeSelected.emit(int(identifier)),
                )
                menu.addAction(
                    "Compare with the original",
                    lambda: self.compareRequested.emit(int(identifier)),
                )
                menu.addSeparator()
                menu.addAction(
                    "Copy SHA-256",
                    lambda: self._copy(derivative.sha256),
                )
                menu.addAction(
                    "Copy file path", lambda: self._copy(derivative.path)
                )
                menu.addSeparator()
                menu.addAction(
                    "Show in file manager",
                    lambda: self.revealRequested.emit(derivative.path),
                )

        elif kind == "evidence" and self._evidence is not None:
            evidence = self._evidence
            menu.addAction("Copy SHA-256", lambda: self._copy(evidence.sha256))
            menu.addAction("Copy SHA-512", lambda: self._copy(evidence.sha512))
            menu.addAction(
                "Copy stored path", lambda: self._copy(evidence.stored_path)
            )
            menu.addSeparator()
            menu.addAction(
                "Show in file manager",
                lambda: self.revealRequested.emit(evidence.stored_path),
            )

        elif kind == "step" and identifier is not None:
            step = self._find_step(int(identifier))
            if step is not None:
                menu.addAction(
                    "Copy input SHA-256", lambda: self._copy(step.input_sha256)
                )
                menu.addAction(
                    "Copy output SHA-256", lambda: self._copy(step.output_sha256)
                )
                menu.addSeparator()
                menu.addAction(
                    "Copy step detail",
                    lambda: self._copy(self._describe_step(step)),
                )

        if menu.isEmpty():
            return
        menu.addSeparator()
        menu.addAction("Refresh", self.refresh)
        menu.exec_(self._tree.viewport().mapToGlobal(position))

    def _copy(self, text: str) -> None:
        """Put ``text`` on the clipboard."""
        QApplication.clipboard().setText(str(text))
        logger.info("Copied %d characters to the clipboard", len(str(text)))

    # --------------------------------------------------------------- handlers
    def _on_selection_changed(self) -> None:
        """Render detail for the selected node."""
        items = self._tree.selectedItems()
        if not items:
            return
        item = items[0]
        kind = item.data(0, _ROLE_KIND)
        identifier = item.data(0, _ROLE_ID)

        if kind == "evidence" and self._evidence is not None:
            evidence = self._evidence
            self._detail.setPlainText(
                "\n".join(
                    [
                        "ORIGINAL EVIDENCE",
                        "=" * 17,
                        f"Filename   : {evidence.original_filename}",
                        f"Stored at  : {evidence.stored_path}",
                        f"Dimensions : {evidence.dimensions}",
                        f"Bit depth  : {evidence.bit_depth}",
                        f"Size       : {evidence.size_bytes:,} bytes",
                        f"Imported   : {evidence.imported_at}",
                        "",
                        f"SHA-256    : {evidence.sha256}",
                        f"SHA-512    : {evidence.sha512}",
                        f"MD5        : {evidence.md5}  (legacy reference only)",
                    ]
                )
            )
        elif kind == "derivative" and identifier is not None:
            derivative = self._derivatives.get(int(identifier))
            if derivative is not None:
                self.derivativeSelected.emit(int(identifier))
                self._detail.setPlainText(self._describe_derivative(derivative))
        elif kind == "step" and identifier is not None:
            step = self._find_step(int(identifier))
            if step is not None:
                self._detail.setPlainText(self._describe_step(step))

    @staticmethod
    def _describe_derivative(derivative: Derivative) -> str:
        """Render a derivative's record as text."""
        lines = [
            "DERIVATIVE",
            "=" * 10,
            f"File       : {Path(derivative.path).name}",
            f"Path       : {derivative.path}",
            f"Dimensions : {derivative.width} x {derivative.height}",
            f"Operation  : {derivative.operation}",
            f"Pipeline   : {derivative.model_name}",
            f"Kind       : {derivative.model_kind}",
            f"Created    : {derivative.created_at}",
            "",
            f"SHA-256    : {derivative.sha256}",
            f"SHA-512    : {derivative.sha512}",
            f"MD5        : {derivative.md5}  (legacy reference only)",
        ]
        provenance = derivative.provenance or {}
        if provenance.get("may_synthesise"):
            lines += [
                "",
                "WARNING: a step in this pipeline can synthesise. Detail in "
                "this image may be invented rather than recovered.",
            ]
        elif derivative.model_kind == "neural":
            lines += [
                "",
                "Produced with a learned model. No step in the pipeline is "
                "capable of synthesising structure.",
            ]
        provenance = derivative.provenance
        if provenance:
            lines += ["", "PROVENANCE", "-" * 10,
                      json.dumps(provenance, indent=2, default=str)]
        return "\n".join(lines)

    @staticmethod
    def _describe_step(step: ProcessingStep) -> str:
        """Render a processing step's record as text."""
        lines = [
            "PROCESSING STEP",
            "=" * 15,
            f"Sequence   : {step.sequence + 1}",
            f"Operation  : {step.operation}",
            f"Model      : {step.model_name} {step.model_version}".rstrip(),
            f"Kind       : {step.model_kind}",
            f"Device     : {step.device}",
            f"Started    : {step.started_at}",
            f"Duration   : {step.duration_s:.3f} s",
            f"Status     : {step.status}",
            "",
            f"Input      : {step.input_size}",
            f"  sha256   : {step.input_sha256}",
            f"Output     : {step.output_size}",
            f"  sha256   : {step.output_sha256}",
            "",
            "PARAMETERS",
            "-" * 10,
            json.dumps(step.parameters, indent=2, default=str),
        ]
        if step.message:
            lines += ["", f"Message    : {step.message}"]
        return "\n".join(lines)
