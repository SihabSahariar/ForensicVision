"""ForensicVision application entry point.

Usage::

    python main.py                     # launch the GUI
    python main.py --case cases/CASE-0001
    python main.py --image evidence.jpg --no-case
    python main.py --check             # environment self-test, no GUI
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

# Make the repository importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import get_config, save_config  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402
from app.paths import ensure_dir, weights_dir  # noqa: E402
from app.version import APP_NAME, APP_ORG, APP_ORG_DOMAIN, APP_VERSION, build_string  # noqa: E402

logger = logging.getLogger("forensicvision")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="forensicvision",
        description=f"{APP_NAME} - forensic image analysis and enhancement workstation",
    )
    parser.add_argument("--case", metavar="DIR", help="Open this case folder on start")
    parser.add_argument("--image", metavar="FILE", help="Open this image on start")
    parser.add_argument(
        "--no-case", action="store_true",
        help="With --image, inspect the file without creating a case",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"),
        help="Override the configured compute device for this session",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Report the environment and model status, then exit",
    )
    parser.add_argument(
        "--self-test", action="store_true", dest="self_test",
        help=(
            "Run a functional end-to-end self-test (analyse, restore, report) "
            "in a temporary directory, then exit. Use this to verify a build."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug-level logging"
    )
    parser.add_argument(
        "--version", action="version", version=build_string()
    )
    return parser.parse_args(argv)


def run_self_test() -> int:
    """Print an environment report without starting Qt.

    Returns:
        Process exit code; 0 when the application can start.
    """
    from core.device import get_device_report
    from restoration import REGISTRATION_REPORT, register_all_models
    from restoration.registry import ModelRegistry

    print(build_string())
    print("=" * 72)

    print("\nCore dependencies")
    print("-" * 17)
    required = ("PyQt5", "numpy", "cv2", "PIL", "sqlalchemy", "reportlab")
    optional = ("torch", "torchvision", "exifread", "pytesseract", "paddleocr",
                "ultralytics", "skimage")
    failures = []
    for name in required:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "present")
            print(f"  OK    {name:16s} {version}")
        except Exception as exc:
            print(f"  FAIL  {name:16s} {exc}")
            failures.append(name)

    print("\nOptional dependencies")
    print("-" * 21)
    for name in optional:
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "present")
            print(f"  OK    {name:16s} {version}")
        except Exception:
            print(f"  --    {name:16s} not installed")

    print("\nCompute device")
    print("-" * 14)
    report = get_device_report()
    print(f"  {report.summary_line()}")
    for gpu in report.gpus:
        print(f"  {gpu.describe()}  capability {gpu.capability}")
    if report.error:
        print(f"  note: {report.error}")

    print("\nRestoration models")
    print("-" * 18)
    register_all_models()
    rows = ModelRegistry.status_table()
    ready = [r for r in rows if r["status"] == "installed"]
    print(f"  {len(ready)} of {len(rows)} models ready")
    for row in rows:
        marker = "OK  " if row["status"] == "installed" else "--  "
        print(f"  {marker}{row['display_name']:34s} {row['status_label']}")
    for family, state in REGISTRATION_REPORT.items():
        if state.get("status") != "ok":
            print(f"  family '{family}' failed: {state.get('error')}")

    print("\nPaths")
    print("-" * 5)
    config = get_config()
    print(f"  cases   : {config.cases_path}")
    print(f"  weights : {weights_dir()}")

    if failures:
        print(f"\nFAILED: missing required package(s): {', '.join(failures)}")
        return 1
    print("\nEnvironment OK - the application can start.")
    return 0


def run_functional_self_test() -> int:
    """Exercise the application end to end and report the outcome.

    Returns:
        Process exit code; 0 when every check that ran passed.
    """
    from app.selftest import run_self_test as run_functional

    print(build_string())
    print("=" * 72)
    print("\nFunctional self-test")
    print("-" * 20)

    report = run_functional(verbose=True)

    ran = len(report.results) - len(report.skipped)
    print()
    if report.failures:
        print(f"FAILED: {len(report.failures)} of {ran} checks did not pass.")
        for result in report.failures:
            print(f"  - {result.name}: {result.detail}")
        return 1

    message = f"All {ran} functional checks passed."
    if report.skipped:
        message += f" {len(report.skipped)} skipped:"
    print(message)
    for result in report.skipped:
        print(f"  - {result.name}: {result.detail}")
    return 0


def _install_excepthook() -> None:
    """Log unhandled exceptions instead of dying silently."""
    def handler(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical(
            "Unhandled exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox

            if QApplication.instance() is not None:
                QMessageBox.critical(
                    None,
                    "Unexpected error",
                    f"{exc_type.__name__}: {exc_value}\n\n"
                    "The error has been written to the application log "
                    "(View > Application Log).",
                )
        except Exception:  # pragma: no cover - last resort
            pass

    sys.excepthook = handler


def main(argv: Optional[List[str]] = None) -> int:
    """Application entry point.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    log_path = configure_logging(
        level=logging.DEBUG if args.debug else logging.INFO
    )
    logger.info("Starting %s", build_string())
    logger.info("Log file: %s", log_path)

    if args.check:
        return run_self_test()

    if args.self_test:
        return run_functional_self_test()

    # Qt setup must happen before any widget is constructed.
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    application = QApplication(sys.argv[:1] + (argv or []))
    application.setApplicationName(APP_NAME)
    application.setApplicationVersion(APP_VERSION)
    application.setOrganizationName(APP_ORG)
    application.setOrganizationDomain(APP_ORG_DOMAIN)

    from gui.theme import apply_theme

    if not apply_theme(application):
        logger.warning("Stylesheet could not be loaded; using the default palette")

    config = get_config()
    if args.device:
        config.device = args.device
        logger.info("Compute device overridden on the command line: %s", args.device)

    ensure_dir(config.cases_path)
    ensure_dir(weights_dir())

    # Registering models imports torch lazily per family; failures are recorded
    # rather than raised so the GUI always starts (S42).
    from restoration import register_all_models

    register_all_models()

    _install_excepthook()

    from gui.main_window import MainWindow

    window = MainWindow(config)
    window.show()

    if args.case:
        window._open_case_path(Path(args.case))  # noqa: SLF001 - startup wiring
    if args.image:
        image_path = Path(args.image)
        if args.no_case or not args.case:
            window._load_standalone_path(image_path)  # noqa: SLF001

    save_config(config)
    return application.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
