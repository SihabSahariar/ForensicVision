"""Compute device discovery and VRAM reporting.

Torch is imported lazily so the GUI still launches on a machine with no ML
stack installed - a hard requirement from the specification (S42).
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import List, Optional

from app.constants import DeviceKind

logger = logging.getLogger(__name__)

__all__ = [
    "GpuInfo",
    "DeviceReport",
    "probe_devices",
    "get_device_report",
    "refresh_device_report",
    "resolve_torch_device",
    "empty_cache",
]


@dataclass
class GpuInfo:
    """A single CUDA device."""

    index: int
    name: str
    total_memory_mb: int
    capability: str = ""
    used_memory_mb: int = 0

    @property
    def free_memory_mb(self) -> int:
        """Approximate free VRAM in MiB."""
        return max(0, self.total_memory_mb - self.used_memory_mb)

    def describe(self) -> str:
        """Return a one-line human readable summary."""
        total_gb = self.total_memory_mb / 1024.0
        used_gb = self.used_memory_mb / 1024.0
        return f"GPU {self.index}: {self.name} ({used_gb:.1f} / {total_gb:.1f} GB)"


@dataclass
class DeviceReport:
    """Snapshot of the machine's compute capabilities."""

    torch_available: bool = False
    torch_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    cudnn_version: str = ""
    gpus: List[GpuInfo] = field(default_factory=list)
    cpu_name: str = ""
    cpu_threads: int = 0
    error: str = ""

    @property
    def gpu_count(self) -> int:
        """Number of visible CUDA devices."""
        return len(self.gpus)

    @property
    def has_gpu(self) -> bool:
        """Whether at least one usable CUDA device is present."""
        return self.cuda_available and bool(self.gpus)

    def primary_gpu(self) -> Optional[GpuInfo]:
        """Return the first CUDA device, if any."""
        return self.gpus[0] if self.gpus else None

    def device_label(self, index: int = 0) -> str:
        """Return a display label for the selected device."""
        if self.has_gpu and 0 <= index < len(self.gpus):
            return self.gpus[index].name
        return self.cpu_name or "CPU"

    def summary_line(self, index: int = 0) -> str:
        """Return the compact string used in the status bar."""
        if not self.torch_available:
            return "Device: CPU (PyTorch not installed)"
        if self.has_gpu and 0 <= index < len(self.gpus):
            gpu = self.gpus[index]
            total_gb = gpu.total_memory_mb / 1024.0
            used_gb = gpu.used_memory_mb / 1024.0
            return (
                f"Device: {gpu.name}  |  CUDA {self.cuda_version}  |  "
                f"VRAM {used_gb:.1f} / {total_gb:.1f} GB"
            )
        return f"Device: CPU ({self.cpu_threads} threads)  |  CUDA unavailable"


def _cpu_name() -> str:
    """Best-effort CPU model string across platforms."""
    name = platform.processor() or platform.machine()
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return name or "Unknown CPU"


def probe_devices() -> DeviceReport:
    """Interrogate the machine and return a fresh :class:`DeviceReport`."""
    import os

    report = DeviceReport(
        cpu_name=_cpu_name(),
        cpu_threads=os.cpu_count() or 1,
    )

    try:
        import torch  # noqa: PLC0415 - optional heavy import
    except Exception as exc:  # pragma: no cover - torch not installed
        report.error = f"PyTorch unavailable: {exc}"
        logger.info("PyTorch not available; neural models will be disabled")
        return report

    report.torch_available = True
    report.torch_version = torch.__version__
    try:
        report.cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - broken driver
        report.error = f"CUDA probe failed: {exc}"
        report.cuda_available = False

    if not report.cuda_available:
        return report

    report.cuda_version = getattr(torch.version, "cuda", "") or ""
    try:
        cudnn = torch.backends.cudnn.version()
        report.cudnn_version = str(cudnn) if cudnn else ""
    except Exception:  # pragma: no cover
        report.cudnn_version = ""

    try:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            total_mb = int(total_bytes // (1024 * 1024))
            used_mb = int((total_bytes - free_bytes) // (1024 * 1024))
            report.gpus.append(
                GpuInfo(
                    index=index,
                    name=props.name,
                    total_memory_mb=total_mb or int(props.total_memory // (1024 * 1024)),
                    capability=f"{props.major}.{props.minor}",
                    used_memory_mb=used_mb,
                )
            )
    except Exception as exc:  # pragma: no cover - driver dependent
        report.error = f"Could not enumerate CUDA devices: {exc}"
        logger.warning("CUDA enumeration failed", exc_info=True)

    return report


_cached_report: Optional[DeviceReport] = None


def get_device_report(refresh: bool = False) -> DeviceReport:
    """Return the cached device report, probing on first use.

    Args:
        refresh: Force a re-probe (used to update live VRAM figures).
    """
    global _cached_report
    if _cached_report is None or refresh:
        _cached_report = probe_devices()
    return _cached_report


def refresh_device_report() -> DeviceReport:
    """Re-probe device state, refreshing VRAM usage."""
    return get_device_report(refresh=True)


def resolve_torch_device(preference: str = "auto", cuda_index: int = 0):
    """Return a ``torch.device`` honouring ``preference`` with CPU fallback.

    Args:
        preference: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        cuda_index: Index of the CUDA device to select.

    Returns:
        A ``torch.device`` instance.

    Raises:
        RuntimeError: PyTorch is not installed.
    """
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for neural inference") from exc

    if preference == DeviceKind.CPU.value:
        return torch.device("cpu")

    report = get_device_report()
    if report.has_gpu:
        index = cuda_index if 0 <= cuda_index < report.gpu_count else 0
        return torch.device(f"cuda:{index}")

    if preference == DeviceKind.CUDA.value:
        logger.warning("CUDA requested but unavailable; falling back to CPU")
    return torch.device("cpu")


def empty_cache() -> None:
    """Release cached CUDA blocks; safe to call when torch is absent."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover
        pass
