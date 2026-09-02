"""Build ForensicVision.exe with PyInstaller.

Model weights are deliberately **not** bundled. They are large, several carry
licences that restrict redistribution, and a frozen copy would drift from the
upstream release. The executable resolves them at run time from the user's
weights directory, and the Model Manager installs them on request.

Usage::

    python scripts/build_windows.py                 # one-folder build
    python scripts/build_windows.py --onefile       # single .exe (slower start)
    python scripts/build_windows.py --console       # keep a console for logs
    python scripts/build_windows.py --clean
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.version import APP_NAME, APP_VERSION  # noqa: E402

DIST = ROOT / "dist"
BUILD = ROOT / "build"
SPEC = ROOT / f"{APP_NAME}.spec"

#: Modules PyInstaller's static analysis cannot see, because they are imported
#: lazily by name in :func:`restoration.register_all_models`.
HIDDEN_IMPORTS: List[str] = [
    "restoration.classical.models",
    "restoration.realesrgan.model",
    "restoration.realesrgan.arch",
    "restoration.restormer.model",
    "restoration.restormer.arch",
    "restoration.nafnet.model",
    "restoration.nafnet.arch",
    "restoration.dncnn.model",
    "restoration.dncnn.arch",
    "restoration.fbcnn.model",
    "restoration.fbcnn.arch",
    "restoration.swinir.model",
    "restoration.swinir.arch",
    "restoration.codeformer",
    "restoration.lama",
    "sqlalchemy.dialects.sqlite",
    "PIL._tkinter_finder",
]

#: Packages excluded from the bundle.
#:
#: PyInstaller's bundled ``hook-torch`` collects torch's submodules, which
#: reaches ``torch.utils.tensorboard``. On a machine that also has TensorFlow
#: installed - a very common shared-environment situation - that import chain
#: drags in TensorFlow, Keras and protobuf, which both balloons the build by
#: gigabytes and crashes PyInstaller's isolated hook subprocesses with protobuf
#: access violations. Excluding the whole family at the top level is what stops
#: it; excluding ``torch.utils.tensorboard`` alone is not enough, because the
#: hook probes the package before graph exclusions apply.
EXCLUDES: List[str] = [
    # --- Qt bindings we do not use --------------------------------------- #
    "PyQt6",
    "PySide2",
    "PySide6",
    "tkinter",
    # --- Deep-learning frameworks we do not use --------------------------- #
    "tensorflow",
    "tensorboard",
    "tensorboard_data_server",
    "keras",
    "jax",
    "jaxlib",
    "flax",
    "paddle",
    "paddleocr",
    # The standalone ONNX packages, not torch.onnx. PyInstaller's isolated
    # hook subprocess dies with an access violation while probing
    # ``onnx.reference``. Measured: torch and torch.onnx both import fine with
    # the standalone onnx package unavailable, because torch.onnx defers it.
    "onnx",
    "onnxscript",
    "onnxruntime",
    # Lazy under torch, so safe to drop.
    "torch.utils.tensorboard",
    #
    # Do NOT add torch.testing, torch.distributions, torch.fx or torch.nn here.
    # Measured: ``import torch`` pulls all four in eagerly, so excluding any of
    # them makes the frozen torch fail with "cannot import name 'nn' from
    # partially initialized module 'torch'". That failure is silent - the app
    # reports "PyTorch not installed", falls back to CPU and offers only the
    # classical operators, which looks like a legitimate configuration.
    # --- Data-science and cloud tooling pulled in transitively ------------ #
    "pandas",
    "openpyxl",
    "botocore",
    "boto3",
    "s3transfer",
    "lxml",
    "matplotlib",
    "seaborn",
    "sklearn",
    "statsmodels",
    # --- Notebooks and test tooling --------------------------------------- #
    "IPython",
    "ipykernel",
    "notebook",
    "jupyter",
    "jupyter_core",
    "pytest",
    "_pytest",
    "setuptools._distutils",
    # --- Optional at run time; see the note below ------------------------- #
    "ultralytics",
]

#: Explanation surfaced in the build output and the shipped README, so the
#: absence of a feature in the binary is never a surprise.
EXCLUSION_NOTES: List[str] = [
    "TensorFlow / Keras / JAX: not used. Excluded to stop PyInstaller's torch "
    "hook from pulling them in on machines where they happen to be installed.",
    "Ultralytics YOLO: object detection is unavailable in this build. It is "
    "AGPL-3.0, which would impose obligations on anyone redistributing the "
    "binary, and it drags in pandas and its dependency tree. Face detection "
    "(OpenCV YuNet) is unaffected and still works. Run from source with "
    "'pip install ultralytics' if object detection is required.",
    "PaddleOCR: not bundled. Tesseract remains usable if installed on the "
    "target machine.",
]

#: Data files copied next to the executable.
DATA: List[tuple] = [
    ("gui/styles/dark_theme.qss", "gui/styles"),
    ("THIRD_PARTY_LICENSES.md", "."),
    ("LICENSE", "."),
    ("docs", "docs"),
]


def check_pyinstaller() -> bool:
    """Report whether PyInstaller is importable."""
    try:
        import PyInstaller  # noqa: PLC0415, F401

        return True
    except ImportError:
        return False


def clean() -> None:
    """Remove previous build artefacts."""
    for target in (DIST, BUILD):
        if target.exists():
            shutil.rmtree(target)
            print(f"removed {target}")
    if SPEC.exists():
        SPEC.unlink()
        print(f"removed {SPEC}")


def build_command(onefile: bool, console: bool, icon: Optional[Path]) -> List[str]:
    """Assemble the PyInstaller command line."""
    separator = ";" if sys.platform.startswith("win") else ":"

    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        f"--name={APP_NAME}",
        "--onefile" if onefile else "--onedir",
        "--console" if console else "--windowed",
        f"--distpath={DIST}",
        f"--workpath={BUILD}",
        f"--specpath={ROOT}",
    ]

    for module in HIDDEN_IMPORTS:
        command.append(f"--hidden-import={module}")
    for module in EXCLUDES:
        command.append(f"--exclude-module={module}")

    for source, destination in DATA:
        path = ROOT / source
        if path.exists():
            command.append(f"--add-data={path}{separator}{destination}")
        else:
            print(f"  note: skipping missing data file {source}")

    if icon and icon.exists():
        command.append(f"--icon={icon}")

    command.append(str(ROOT / "main.py"))
    return command


def write_readme(target: Path, onefile: bool) -> None:
    """Write the end-user note that ships beside the executable."""
    location = "ForensicVision.exe" if onefile else "ForensicVision/ForensicVision.exe"
    target.write_text(
        f"""{APP_NAME} {APP_VERSION}
{'=' * (len(APP_NAME) + len(APP_VERSION) + 1)}

Run {location} to start the application.

MODEL WEIGHTS ARE NOT INCLUDED
------------------------------
This build ships the application only. Neural model weights are large and
several carry licences that restrict redistribution, so they are installed on
request instead of being bundled.

Every classical restoration operator - deconvolution, denoising, deblocking,
dehazing, tone and contrast correction, Lanczos enlargement - works
immediately with no download. Those operators are deterministic and cannot
introduce detail that is not derivable from the evidence.

To add the neural models:

  1. Start {APP_NAME}.
  2. Open Tools > Model Manager.
  3. Review each model's licence and source, then press Install.

Weights are stored under:
  %LOCALAPPDATA%\\ForensicVision\\weights

OBJECT DETECTION IS NOT INCLUDED
--------------------------------
Ultralytics YOLO is not bundled: it is AGPL-3.0, which would impose licensing
obligations on anyone redistributing this binary. Analysis > Detect Objects
will report it as unavailable.

Face detection is unaffected - it uses OpenCV's own YuNet detector, which is
MIT-licensed and installed from the Model Manager like any other model.

To use object detection, run ForensicVision from source with:
  pip install ultralytics

GPU ACCELERATION
----------------
CUDA is used automatically when an NVIDIA GPU and a suitable driver are
present. The status bar reports the active device and VRAM. Without a GPU the
application falls back to the CPU; neural models will be considerably slower.

FORENSIC NOTICE
---------------
Algorithmic image enhancement modifies image data. AI-based restoration may
infer or synthesize structures that are not directly represented in the source
image. Enhanced imagery is a derivative representation and should not
automatically be interpreted as an exact recovery of information absent from
the original evidence.

See docs/ for the full documentation and THIRD_PARTY_LICENSES.md for the
licences of every integrated component.
""",
        encoding="utf-8",
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onefile", action="store_true",
                        help="Produce a single executable (slower to start)")
    parser.add_argument("--console", action="store_true",
                        help="Keep a console window for log output")
    parser.add_argument("--clean", action="store_true",
                        help="Remove build artefacts and exit")
    parser.add_argument("--icon", type=Path, default=ROOT / "assets" / "icons" / "app.ico")
    args = parser.parse_args(argv)

    if args.clean:
        clean()
        return 0

    if not check_pyinstaller():
        print(
            "PyInstaller is not installed.\n\n"
            "  python -m pip install pyinstaller\n"
        )
        return 1

    print(f"Building {APP_NAME} {APP_VERSION}")
    print(f"  mode   : {'one-file' if args.onefile else 'one-folder'}")
    print(f"  window : {'console' if args.console else 'windowed'}")
    print(f"  output : {DIST}")
    print()

    command = build_command(args.onefile, args.console, args.icon)
    print("  " + " \\\n    ".join(command))
    print()

    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\nBuild failed with exit code {result.returncode}")
        return result.returncode

    write_readme(DIST / "README.txt", args.onefile)

    executable = (
        DIST / f"{APP_NAME}.exe" if args.onefile
        else DIST / APP_NAME / f"{APP_NAME}.exe"
    )
    print("\nBuild complete.")
    if executable.exists():
        print(f"  executable : {executable}")
        print(f"  size       : {executable.stat().st_size / (1024 * 1024):.1f} MiB")
    print(f"  notes      : {DIST / 'README.txt'}")
    print(
        "\nModel weights were not bundled. Install them from the running "
        "application via Tools > Model Manager."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
