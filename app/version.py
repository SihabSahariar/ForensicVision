"""Application version metadata for ForensicVision."""

from __future__ import annotations

import platform

__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "APP_VERSION_INFO",
    "APP_ORG",
    "APP_ORG_DOMAIN",
    "APP_DESCRIPTION",
    "build_string",
]

APP_NAME: str = "ForensicVision"
APP_VERSION_INFO: tuple = (1, 0, 0)
APP_VERSION: str = ".".join(str(part) for part in APP_VERSION_INFO)
APP_ORG: str = "ForensicVision"
APP_ORG_DOMAIN: str = "forensicvision.local"
APP_DESCRIPTION: str = "Forensic image analysis and enhancement workstation"


def build_string() -> str:
    """Return a human readable build identifier.

    Returns:
        A string such as ``ForensicVision 1.0.0 (CPython 3.11.4 / Windows)``.
    """
    return (
        f"{APP_NAME} {APP_VERSION} "
        f"(CPython {platform.python_version()} / {platform.system()})"
    )
