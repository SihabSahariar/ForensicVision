"""Filesystem locations used by ForensicVision.

All paths are resolved lazily so the module stays importable in frozen
(PyInstaller) builds where ``__file__`` points inside a bundle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "is_frozen",
    "app_root",
    "resource_root",
    "user_data_dir",
    "config_dir",
    "log_dir",
    "weights_dir",
    "default_cases_dir",
    "styles_dir",
    "ensure_dir",
]


def is_frozen() -> bool:
    """Return ``True`` when running from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Return the directory containing the application package tree."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_root() -> Path:
    """Return the directory containing bundled read-only resources.

    Under PyInstaller this is ``sys._MEIPASS``; in a source checkout it is the
    repository root.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return app_root()


def user_data_dir() -> Path:
    """Return the per-user writable data directory.

    Windows uses ``%LOCALAPPDATA%``; Linux honours ``XDG_DATA_HOME``.
    """
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(base) / "ForensicVision"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "forensicvision"
    return Path.home() / ".local" / "share" / "forensicvision"


def config_dir() -> Path:
    """Return the writable configuration directory."""
    if sys.platform.startswith("win"):
        return user_data_dir() / "config"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "forensicvision"
    return Path.home() / ".config" / "forensicvision"


def log_dir() -> Path:
    """Return the directory that holds rotating application logs."""
    return user_data_dir() / "logs"


def weights_dir() -> Path:
    """Return the directory that stores downloaded model weights.

    A source checkout keeps weights inside the repository (``models/weights``)
    so a developer can inspect them; a frozen build stores them next to the
    user's data because the bundle directory may be read-only.
    """
    env_override = os.environ.get("FORENSICVISION_WEIGHTS_DIR")
    if env_override:
        return Path(env_override)
    if is_frozen():
        return user_data_dir() / "weights"
    return app_root() / "models" / "weights"


def default_cases_dir() -> Path:
    """Return the default parent directory for new cases."""
    env_override = os.environ.get("FORENSICVISION_CASES_DIR")
    if env_override:
        return Path(env_override)
    if is_frozen():
        return user_data_dir() / "cases"
    return app_root() / "cases"


def styles_dir() -> Path:
    """Return the directory containing Qt stylesheets."""
    return resource_root() / "gui" / "styles"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (including parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
