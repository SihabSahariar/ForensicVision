"""Forensic Safe Mode.

When enabled (the default) the application enforces:

* imported originals are marked read-only on the filesystem;
* the original is never a valid write target for any operation;
* destructive operations (delete evidence, purge history) are refused;
* every state-changing operation emits an audit event;
* every derivative gets a fresh hash recorded before it can be used further.

The guard is a process-wide singleton because it is a policy, not a per-window
setting; the GUI reflects its state in the status bar.
"""

from __future__ import annotations

import logging
import os
import stat
import sys
from pathlib import Path
from typing import Callable, List, Optional

from core.exceptions import SafeModeViolation

logger = logging.getLogger(__name__)

__all__ = ["SafeModeGuard", "get_guard", "safe_mode_enabled"]

#: Text shown in the status bar while safe mode is active.
SAFE_MODE_BANNER: str = "\U0001F512  FORENSIC SAFE MODE ENABLED"

#: Explanation shown when the user toggles safe mode off.
SAFE_MODE_OFF_WARNING: str = (
    "Forensic Safe Mode is being disabled.\n\n"
    "With safe mode off, original evidence files lose their read-only "
    "protection and destructive operations become available. Actions are still "
    "logged, but the application can no longer guarantee that originals are "
    "unmodified.\n\n"
    "Only disable safe mode when you have an explicit, documented reason."
)


class SafeModeGuard:
    """Policy object consulted before any potentially destructive operation."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self._protected: List[Path] = []
        self._listeners: List[Callable[[bool], None]] = []

    # ------------------------------------------------------------------ state
    @property
    def enabled(self) -> bool:
        """Whether safe mode is currently active."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable safe mode, notifying listeners."""
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        logger.warning("Forensic Safe Mode %s", "ENABLED" if enabled else "DISABLED")
        if enabled:
            self._reapply_protection()
        for listener in list(self._listeners):
            try:
                listener(enabled)
            except Exception:  # pragma: no cover - listener robustness
                logger.exception("Safe-mode listener failed")

    def add_listener(self, listener: Callable[[bool], None]) -> None:
        """Register a callback invoked whenever the mode changes."""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[bool], None]) -> None:
        """Unregister a previously added callback."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    # ------------------------------------------------------------ protections
    def protect_file(self, path: os.PathLike | str) -> bool:
        """Mark ``path`` read-only and remember it for re-protection.

        Returns:
            ``True`` when the permission change succeeded. A failure is logged
            and returned as ``False``; the logical protections (refusing to
            write to originals) remain in force regardless.
        """
        file_path = Path(path)
        if file_path not in self._protected:
            self._protected.append(file_path)
        return self._set_readonly(file_path, True)

    def unprotect_file(self, path: os.PathLike | str) -> bool:
        """Restore write permission on ``path``.

        Raises:
            SafeModeViolation: Safe mode is enabled.
        """
        if self._enabled:
            raise SafeModeViolation(
                "Cannot remove write protection while Forensic Safe Mode is enabled."
            )
        file_path = Path(path)
        if file_path in self._protected:
            self._protected.remove(file_path)
        return self._set_readonly(file_path, False)

    def is_protected(self, path: os.PathLike | str) -> bool:
        """Whether ``path`` is a registered original."""
        return Path(path) in self._protected

    def _reapply_protection(self) -> None:
        for path in list(self._protected):
            if path.exists():
                self._set_readonly(path, True)

    @staticmethod
    def _set_readonly(path: Path, readonly: bool) -> bool:
        """Toggle the read-only attribute in a cross-platform way."""
        try:
            if not path.exists():
                return False
            mode = path.stat().st_mode
            if readonly:
                new_mode = mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            else:
                new_mode = mode | stat.S_IWUSR
            os.chmod(path, new_mode)
            return True
        except OSError as exc:
            logger.warning("Could not change permissions on %s: %s", path, exc)
            return False

    # ------------------------------------------------------------- assertions
    def assert_can_write(self, path: os.PathLike | str) -> None:
        """Raise if ``path`` is a protected original.

        Args:
            path: Intended write target.

        Raises:
            SafeModeViolation: The target is registered evidence and safe mode
                is enabled.
        """
        target = Path(path)
        if self._enabled and target in self._protected:
            raise SafeModeViolation(
                f"Forensic Safe Mode: refusing to write to original evidence "
                f"'{target.name}'. Export a derivative instead."
            )

    def assert_can_delete(self, description: str = "this item") -> None:
        """Raise when a delete is attempted under safe mode."""
        if self._enabled:
            raise SafeModeViolation(
                f"Forensic Safe Mode: deleting {description} is disabled. "
                "Disable safe mode explicitly if removal is authorised."
            )

    def assert_can_modify_history(self) -> None:
        """Raise when history alteration is attempted under safe mode."""
        if self._enabled:
            raise SafeModeViolation(
                "Forensic Safe Mode: the processing history is append-only and "
                "cannot be edited or cleared."
            )

    def status_text(self) -> str:
        """Return the status-bar string for the current mode."""
        return SAFE_MODE_BANNER if self._enabled else "Safe mode OFF"

    def describe(self) -> str:
        """Return a longer description used in reports."""
        if self._enabled:
            return (
                "Forensic Safe Mode was ENABLED for this session. Original "
                "evidence was write-protected, destructive operations were "
                "disabled, and all processing steps were recorded."
            )
        return (
            "Forensic Safe Mode was DISABLED for this session. Original "
            "evidence write-protection was not enforced by the application."
        )


_guard: Optional[SafeModeGuard] = None


def get_guard(default_enabled: bool = True) -> SafeModeGuard:
    """Return the process-wide guard, creating it on first use."""
    global _guard
    if _guard is None:
        _guard = SafeModeGuard(default_enabled)
    return _guard


def safe_mode_enabled() -> bool:
    """Convenience predicate for the current safe-mode state."""
    return get_guard().enabled


def platform_supports_readonly() -> bool:
    """Whether the platform honours the read-only attribute meaningfully."""
    return sys.platform.startswith("win") or os.name == "posix"
