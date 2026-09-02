"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging_setup import configure_logging  # noqa: E402
from core.image_io import ImageData, save_image  # noqa: E402

configure_logging(console=False)


@pytest.fixture(autouse=True, scope="session")
def _isolate_user_config(tmp_path_factory):
    """Keep the test run out of the developer's real configuration.

    :func:`app.config.save_config` writes to the per-user config directory, and
    several code paths call it - ``MainWindow.closeEvent`` among them. Without
    this redirect a GUI test would persist its throw-away ``cases_root`` and
    ``weights_root`` into the real ``settings.json``, pointing a genuine
    install at a temporary directory that pytest then deletes.
    """
    import app.config as config_module

    sandbox = tmp_path_factory.mktemp("config")
    original_dir = config_module.config_dir
    original_config = config_module._config

    config_module.config_dir = lambda: sandbox
    config_module._config = None
    try:
        yield sandbox
    finally:
        config_module.config_dir = original_dir
        config_module._config = original_config


@pytest.fixture(scope="session")
def scene() -> np.ndarray:
    """A deterministic synthetic test scene as an ``uint8`` RGB array."""
    from scripts.make_sample import build_scene

    return build_scene(480, 320, seed=7)


@pytest.fixture()
def sample_png(tmp_path: Path, scene: np.ndarray) -> Path:
    """A lossless PNG copy of the test scene."""
    path = tmp_path / "sample.png"
    save_image(ImageData(pixels=scene), path)
    return path


@pytest.fixture()
def sample_jpeg(tmp_path: Path, scene: np.ndarray) -> Path:
    """A heavily compressed, downscaled JPEG - the CCTV-like case."""
    import cv2

    from scripts.make_sample import degrade

    degraded = degrade(scene, "cctv")
    path = tmp_path / "sample_cctv.jpg"
    cv2.imwrite(str(path), degraded[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 28])
    return path


@pytest.fixture()
def case(tmp_path: Path):
    """An open case backed by a temporary directory."""
    from core.case_manager import CaseManager
    from forensic.safe_mode import SafeModeGuard

    guard = SafeModeGuard(enabled=True)
    manager = CaseManager.create(
        parent=tmp_path / "cases",
        title="Automated test case",
        investigator="Test Runner",
        organisation="ForensicVision CI",
        guard=guard,
    )
    yield manager
    manager.close()


@pytest.fixture(scope="session")
def registry():
    """The model registry with every family registered."""
    from restoration import register_all_models
    from restoration.registry import ModelRegistry

    register_all_models()
    return ModelRegistry
