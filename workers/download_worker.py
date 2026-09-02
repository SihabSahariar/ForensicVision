"""Model-weight download worker.

Downloads are always user-initiated from the Model Manager (S15/S43); this
worker only performs a transfer that has already been approved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from restoration.base import WeightSpec
from restoration.weights import DownloadResult, download_weight, install_from_file
from workers.base import BaseWorker

logger = logging.getLogger(__name__)

__all__ = ["DownloadWorker", "InstallFileWorker"]


class DownloadWorker(BaseWorker):
    """Fetches one weight file, verifying its digest where one is published."""

    description = "Downloading model weights"

    def __init__(self, spec: WeightSpec, destination: Optional[Path] = None) -> None:
        super().__init__()
        self._spec = spec
        self._destination = destination

    def execute(self) -> DownloadResult:
        """Download and verify the weight file."""
        self.report_status(f"Downloading {self._spec.filename}...")

        def progress(done: int, total: int) -> None:
            if total > 0:
                percent = int(done * 100 / total)
                self.report(
                    percent,
                    f"{self._spec.filename}  "
                    f"{done / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MiB",
                )
            else:
                self.report(0, f"{done / (1024 * 1024):.1f} MiB downloaded")

        return download_weight(
            self._spec,
            destination_dir=self._destination,
            progress=progress,
            cancelled=self.is_cancelled,
        )


class InstallFileWorker(BaseWorker):
    """Installs a locally-obtained weight file, verifying its digest."""

    description = "Installing model weights"

    def __init__(
        self, spec: WeightSpec, source: Path, destination: Optional[Path] = None
    ) -> None:
        super().__init__()
        self._spec = spec
        self._source = Path(source)
        self._destination = destination

    def execute(self) -> DownloadResult:
        """Copy and verify the weight file."""
        self.report_status(f"Installing {self._spec.filename}...")

        def progress(done: int, total: int) -> None:
            if total > 0:
                self.report(int(done * 100 / total), "Verifying digest")

        return install_from_file(
            self._spec,
            self._source,
            destination_dir=self._destination,
            progress=progress,
        )
