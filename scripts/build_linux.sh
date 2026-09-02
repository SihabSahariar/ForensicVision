#!/usr/bin/env bash
#
# Build ForensicVision for Linux.
#
#   ./scripts/build_linux.sh              # PyInstaller one-folder build
#   ./scripts/build_linux.sh --appimage   # additionally package an AppImage
#
# Model weights are deliberately not bundled - see docs/MODELS.md.
#
# Tested on Ubuntu 22.04 and 24.04.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APP_NAME="ForensicVision"
DIST="$ROOT/dist"
BUILD="$ROOT/build"
MAKE_APPIMAGE=0

for arg in "$@"; do
    case "$arg" in
        --appimage) MAKE_APPIMAGE=1 ;;
        --clean) rm -rf "$DIST" "$BUILD" "$ROOT/$APP_NAME.spec"; echo "cleaned"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

echo "== checking build prerequisites =="
python3 -c "import PyInstaller" 2>/dev/null || {
    echo "PyInstaller is not installed:  python3 -m pip install pyinstaller" >&2
    exit 1
}
python3 -c "import PyQt5" 2>/dev/null || {
    echo "PyQt5 is not installed:  python3 -m pip install -r requirements.txt" >&2
    exit 1
}

echo "== building with PyInstaller =="
python3 -m PyInstaller \
    --noconfirm --clean \
    --name "$APP_NAME" \
    --onedir --windowed \
    --distpath "$DIST" --workpath "$BUILD" --specpath "$ROOT" \
    --hidden-import restoration.classical.models \
    --hidden-import restoration.realesrgan.model \
    --hidden-import restoration.realesrgan.arch \
    --hidden-import restoration.restormer.model \
    --hidden-import restoration.restormer.arch \
    --hidden-import restoration.nafnet.model \
    --hidden-import restoration.nafnet.arch \
    --hidden-import restoration.dncnn.model \
    --hidden-import restoration.dncnn.arch \
    --hidden-import restoration.fbcnn.model \
    --hidden-import restoration.fbcnn.arch \
    --hidden-import restoration.swinir.model \
    --hidden-import restoration.swinir.arch \
    --hidden-import restoration.codeformer \
    --hidden-import restoration.lama \
    --hidden-import sqlalchemy.dialects.sqlite \
    --exclude-module matplotlib \
    --exclude-module tkinter \
    --exclude-module pytest \
    --exclude-module PyQt6 \
    --exclude-module PySide6 \
    --add-data "gui/styles/dark_theme.qss:gui/styles" \
    --add-data "THIRD_PARTY_LICENSES.md:." \
    --add-data "LICENSE:." \
    --add-data "docs:docs" \
    main.py

echo "== build complete: $DIST/$APP_NAME =="

if [ "$MAKE_APPIMAGE" -eq 1 ]; then
    echo "== packaging AppImage =="
    command -v appimagetool >/dev/null 2>&1 || {
        echo "appimagetool not found. Download it from:" >&2
        echo "  https://github.com/AppImage/AppImageKit/releases" >&2
        echo "and place it on PATH as 'appimagetool'." >&2
        exit 1
    }

    APPDIR="$BUILD/${APP_NAME}.AppDir"
    rm -rf "$APPDIR"
    mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
             "$APPDIR/usr/share/icons/hicolor/256x256/apps"

    cp -r "$DIST/$APP_NAME/." "$APPDIR/usr/bin/"

    cat > "$APPDIR/$APP_NAME.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=ForensicVision
Comment=Forensic image analysis and enhancement workstation
Exec=ForensicVision
Icon=forensicvision
Categories=Graphics;Science;
Terminal=false
DESKTOP
    cp "$APPDIR/$APP_NAME.desktop" "$APPDIR/usr/share/applications/"

    if [ -f "$ROOT/assets/icons/app.png" ]; then
        cp "$ROOT/assets/icons/app.png" "$APPDIR/forensicvision.png"
        cp "$ROOT/assets/icons/app.png" \
           "$APPDIR/usr/share/icons/hicolor/256x256/apps/forensicvision.png"
    else
        # appimagetool requires an icon; generate a placeholder.
        python3 - <<'PY'
import os
from pathlib import Path
try:
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (256, 256), (16, 18, 22, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 216, 216], outline=(61, 139, 205, 255), width=8)
    draw.ellipse([88, 88, 168, 168], outline=(215, 220, 228, 255), width=8)
    target = Path(os.environ["APPDIR"]) / "forensicvision.png"
    img.save(target)
    (Path(os.environ["APPDIR"]) / "usr/share/icons/hicolor/256x256/apps/forensicvision.png").write_bytes(target.read_bytes())
except Exception as exc:
    raise SystemExit(f"could not generate a placeholder icon: {exc}")
PY
    fi

    cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="$HERE/usr/bin:${LD_LIBRARY_PATH:-}"
exec "$HERE/usr/bin/ForensicVision" "$@"
APPRUN
    chmod +x "$APPDIR/AppRun"

    APPIMAGE_EXTRACT_AND_RUN=1 appimagetool "$APPDIR" "$DIST/${APP_NAME}-x86_64.AppImage"
    echo "== AppImage: $DIST/${APP_NAME}-x86_64.AppImage =="
fi

echo
echo "Model weights were not bundled. Install them from the running"
echo "application via Tools > Model Manager, or with:"
echo "  python3 scripts/download_models.py --install realesrgan_x4plus"
