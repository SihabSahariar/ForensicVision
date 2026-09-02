"""Central logging configuration.

The application installs three handlers:

* a rotating file handler under :func:`app.paths.log_dir`;
* a console handler (stderr) for development;
* an in-process ring-buffer handler that the GUI log dock renders.

The GUI handler is deliberately Qt-free so this module can be imported by
headless components.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, List, Optional

from app.paths import ensure_dir, log_dir

__all__ = [
    "LogRecordEntry",
    "MemoryLogHandler",
    "configure_logging",
    "get_memory_handler",
    "get_log_file_path",
]

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BUFFER = 20000

_memory_handler: "Optional[MemoryLogHandler]" = None
_log_file_path: Optional[Path] = None
_configured = False


@dataclass(frozen=True)
class LogRecordEntry:
    """An immutable snapshot of a single emitted log record."""

    timestamp: datetime
    level: str
    levelno: int
    logger: str
    message: str

    def formatted(self) -> str:
        """Return the record rendered as a single display line."""
        stamp = self.timestamp.strftime(_DATE_FORMAT)
        return f"{stamp} [{self.level}] {self.logger}: {self.message}"


class MemoryLogHandler(logging.Handler):
    """Thread-safe ring buffer of log records with change notification.

    The GUI subscribes with :meth:`add_listener`. Listeners are invoked on the
    thread that emitted the record, so a Qt listener must marshal to the GUI
    thread itself (the log dock does this with a queued signal).
    """

    def __init__(self, capacity: int = _MAX_BUFFER) -> None:
        super().__init__()
        self._buffer: Deque[LogRecordEntry] = deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._listeners: List[Callable[[LogRecordEntry], None]] = []

    # -- logging.Handler API ------------------------------------------------ #
    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = LogRecordEntry(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                levelno=record.levelno,
                logger=record.name,
                message=record.getMessage(),
            )
            if record.exc_info:
                entry = LogRecordEntry(
                    timestamp=entry.timestamp,
                    level=entry.level,
                    levelno=entry.levelno,
                    logger=entry.logger,
                    message=entry.message + "\n" + self.format(record),
                )
        except Exception:  # pragma: no cover - defensive
            self.handleError(record)
            return

        with self._lock:
            self._buffer.append(entry)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(entry)
            except Exception:  # pragma: no cover - never break logging
                pass

    # -- public API --------------------------------------------------------- #
    def add_listener(self, listener: Callable[[LogRecordEntry], None]) -> None:
        """Register ``listener`` to be called for every subsequent record."""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[LogRecordEntry], None]) -> None:
        """Unregister a previously added listener; missing entries are ignored."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def snapshot(self) -> List[LogRecordEntry]:
        """Return a copy of the buffered records, oldest first."""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Discard buffered records (does not affect the log file)."""
        with self._lock:
            self._buffer.clear()


def configure_logging(level: int = logging.INFO, console: bool = True) -> Path:
    """Install application-wide logging handlers.

    Repeated calls are no-ops so plugins and tests can call it defensively.

    Args:
        level: Root logger level.
        console: When ``True`` also log to stderr.

    Returns:
        The path of the active log file.
    """
    global _memory_handler, _log_file_path, _configured

    if _configured and _log_file_path is not None:
        return _log_file_path

    directory = ensure_dir(log_dir())
    _log_file_path = directory / "forensicvision.log"

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        _log_file_path, maxBytes=8 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(stream=sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    _memory_handler = MemoryLogHandler()
    _memory_handler.setFormatter(formatter)
    root.addHandler(_memory_handler)

    # Third-party libraries are far too chatty at INFO.
    for noisy in ("PIL", "matplotlib", "urllib3", "torch", "ultralytics"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    return _log_file_path


def get_memory_handler() -> MemoryLogHandler:
    """Return the shared in-memory handler, configuring logging if needed."""
    if _memory_handler is None:
        configure_logging()
    assert _memory_handler is not None  # for type checkers
    return _memory_handler


def get_log_file_path() -> Optional[Path]:
    """Return the active log file path, or ``None`` before configuration."""
    return _log_file_path
