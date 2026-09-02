"""Application configuration.

Settings are stored as JSON under :func:`app.paths.config_dir` and exposed as a
frozen-ish dataclass. A single process-wide instance is available through
:func:`get_config`, but every consumer accepts an explicit ``AppConfig`` so the
engine can be driven with an arbitrary configuration from CLI or tests
(dependency injection).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

from app.constants import (
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
    DeviceKind,
)
from app.paths import config_dir, default_cases_dir, ensure_dir, weights_dir

logger = logging.getLogger(__name__)

__all__ = ["AppConfig", "get_config", "set_config", "load_config", "save_config"]

_CONFIG_FILENAME = "settings.json"


@dataclass
class AppConfig:
    """Runtime configuration for the whole application.

    Attributes:
        cases_root: Parent directory in which case folders are created.
        weights_root: Directory holding downloaded model weights.
        device: ``"cuda"``, ``"cpu"`` or ``"auto"``.
        cuda_index: Which CUDA device to use when several are present.
        use_fp16: Enable half precision on CUDA where the model supports it.
        tile_size: Tile edge length for tiled inference, 0 disables tiling.
        tile_overlap: Overlap between neighbouring tiles, in pixels.
        auto_reduce_tile: Halve the tile size and retry after a CUDA OOM.
        safe_mode: Forensic Safe Mode default state.
        max_worker_threads: Upper bound for the shared thread pool.
        allow_model_download: Master switch for any outbound weight download.
        theme: Stylesheet name (currently only ``"dark"``).
        recent_cases: Most-recently-opened case directories, newest first.
        confirm_synthesis: Require confirmation before generative operations.
        ocr_engine: ``"auto"``, ``"tesseract"``, ``"paddleocr"`` or ``"none"``.
        tesseract_cmd: Explicit path to the ``tesseract`` binary, if needed.
    """

    cases_root: str = field(default_factory=lambda: str(default_cases_dir()))
    weights_root: str = field(default_factory=lambda: str(weights_dir()))
    device: str = "auto"
    cuda_index: int = 0
    use_fp16: bool = True
    tile_size: int = DEFAULT_TILE_SIZE
    tile_overlap: int = DEFAULT_TILE_OVERLAP
    auto_reduce_tile: bool = True
    safe_mode: bool = True
    max_worker_threads: int = 4
    allow_model_download: bool = True
    theme: str = "dark"
    recent_cases: list = field(default_factory=list)
    confirm_synthesis: bool = True
    ocr_engine: str = "auto"
    tesseract_cmd: str = ""

    # -- derived helpers ---------------------------------------------------- #
    @property
    def cases_path(self) -> Path:
        """Return :attr:`cases_root` as a :class:`~pathlib.Path`."""
        return Path(self.cases_root)

    @property
    def weights_path(self) -> Path:
        """Return :attr:`weights_root` as a :class:`~pathlib.Path`."""
        return Path(self.weights_root)

    def resolved_device(self) -> str:
        """Resolve ``"auto"`` against the actual hardware.

        Returns:
            ``"cuda"`` when CUDA is usable and permitted, otherwise ``"cpu"``.
        """
        if self.device == DeviceKind.CPU.value:
            return DeviceKind.CPU.value
        try:
            import torch  # noqa: PLC0415 - optional heavy import

            if torch.cuda.is_available():
                return DeviceKind.CUDA.value
        except Exception:  # pragma: no cover - torch missing or broken
            logger.debug("CUDA probe failed; falling back to CPU", exc_info=True)
        return DeviceKind.CPU.value

    def push_recent_case(self, path: str, limit: int = 10) -> None:
        """Record ``path`` as the most recently opened case."""
        entries = [p for p in self.recent_cases if p != path]
        entries.insert(0, path)
        self.recent_cases = entries[:limit]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Build a config from ``data``, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


_config: Optional[AppConfig] = None


def _config_file() -> Path:
    return config_dir() / _CONFIG_FILENAME


def load_config() -> AppConfig:
    """Load configuration from disk, returning defaults on any failure."""
    path = _config_file()
    if not path.exists():
        return AppConfig()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return AppConfig.from_dict(data)
    except Exception:
        logger.warning("Could not read %s; using defaults", path, exc_info=True)
        return AppConfig()


def save_config(config: Optional[AppConfig] = None) -> Path:
    """Persist ``config`` (or the process-wide instance) to disk."""
    cfg = config if config is not None else get_config()
    ensure_dir(config_dir())
    path = _config_file()
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)
    tmp.replace(path)
    logger.debug("Configuration written to %s", path)
    return path


def get_config() -> AppConfig:
    """Return the process-wide configuration, loading it on first use."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: AppConfig) -> None:
    """Replace the process-wide configuration instance."""
    global _config
    _config = config
