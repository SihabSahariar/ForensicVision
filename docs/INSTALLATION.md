# Installation

Python **3.10 - 3.12**. 3.11 is the reference version.

---

## Windows

```powershell
git clone <repository> ForensicVision
cd ForensicVision

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python main.py --check
python main.py
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Linux (Ubuntu 22.04 / 24.04)

PyQt5 needs a few system libraries that are not pulled in by pip:

```bash
sudo apt update
sudo apt install -y \
    python3.11 python3.11-venv python3-pip \
    libgl1 libglib2.0-0 \
    libxcb-xinerama0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
    libxcb-xkb1 libxkbcommon-x11-0 libdbus-1-3
```

Then:

```bash
git clone <repository> ForensicVision
cd ForensicVision

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python main.py --check
python main.py
```

### Wayland

Qt 5 defaults to the X11 backend under XWayland, which works. For native
Wayland:

```bash
sudo apt install -y qtwayland5
QT_QPA_PLATFORM=wayland python main.py
```

### Headless servers

The engine is GUI-free and usable without a display:

```bash
python scripts/download_models.py --list
python -m pytest tests/ -q          # runs headless via Qt's offscreen platform
```

To run the GUI over SSH, forward X11 (`ssh -X`) or use `xvfb-run`.

---

## Verifying the installation

```bash
python main.py --check
```

Reports Python and dependency versions, the compute device, the status of every
registered model, and the resolved case and weights directories. Exit code 0
means the application can start.

Expected output on a working GPU install:

```
Compute device
--------------
  Device: NVIDIA GeForce RTX 3070 Laptop GPU  |  CUDA 12.6  |  VRAM 1.0 / 8.0 GB

Restoration models
------------------
  10 of 30 models ready
```

Ten ready with no downloads is correct: those are the classical operators.

---

## Optional features

Each is genuinely optional; the corresponding UI reports the missing dependency
rather than failing.

```bash
python -m pip install -r requirements-optional.txt
```

### OCR

Tesseract needs its native binary as well as the Python binding:

```bash
# Linux
sudo apt install -y tesseract-ocr
python -m pip install pytesseract

# Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
# then either add it to PATH or set the path in Tools > Preferences > OCR.
```

PaddleOCR is an alternative:

```bash
python -m pip install paddleocr paddlepaddle
```

### Object detection

```bash
python -m pip install ultralytics
```

The YOLO checkpoint downloads on first use and is copied into the weights
folder so it appears alongside the restoration models.

---

## Directory layout

| | Windows | Linux |
|---|---|---|
| Cases | `<repo>/cases` | `<repo>/cases` |
| Weights | `<repo>/models/weights` | `<repo>/models/weights` |
| Config | `%LOCALAPPDATA%\ForensicVision\config` | `~/.config/forensicvision` |
| Logs | `%LOCALAPPDATA%\ForensicVision\logs` | `~/.local/share/forensicvision/logs` |

A frozen (PyInstaller) build stores cases and weights under the user data
directory instead, because the bundle directory may be read-only.

Override with environment variables:

```bash
export FORENSICVISION_CASES_DIR=/mnt/evidence/cases
export FORENSICVISION_WEIGHTS_DIR=/opt/forensicvision/weights
```

Or in **Tools > Preferences > Folders**.

---

## Command line

```
python main.py                          launch
python main.py --check                  environment self-test, no GUI
python main.py --case cases/CASE-0001   open a case on start
python main.py --image frame.jpg        inspect one image without a case
python main.py --device cpu             force CPU for this session
python main.py --debug                  debug-level logging
python main.py --version
```

---

## Troubleshooting

**`qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`**
Install the xcb libraries listed above. `QT_DEBUG_PLUGINS=1 python main.py`
names the missing library.

**`ImportError: libGL.so.1`**
`sudo apt install libgl1`

**The application starts but reports 0 models ready**
Run `python main.py --check`. If PyTorch is missing the classical operators
should still be ready; if they are not, the OpenCV or numpy install is broken.

**CUDA is not detected**
See [GPU_SETUP.md](GPU_SETUP.md).

**`PermissionError` when deleting a case folder**
Stored originals are read-only by design. Clear the attribute first:
`chmod -R u+w cases/CASE-0001` (Linux) or `attrib -R /S /D cases\CASE-0001`
(Windows).
