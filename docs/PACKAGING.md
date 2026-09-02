# Packaging

Model weights are **not** bundled into either build. They are large, several
carry licences that restrict redistribution, and a frozen copy would drift from
the upstream release. The executable resolves them at run time from the user's
weights directory and the Model Manager installs them on request.

Every classical operator works in a fresh build with no download at all.

---

## Windows

```powershell
python -m pip install pyinstaller
python scripts\build_windows.py
```

Output: `dist\ForensicVision\ForensicVision.exe` plus `dist\README.txt`.

| Option | Effect |
|---|---|
| `--onefile` | Single `.exe`. Simpler to hand over, noticeably slower to start (it unpacks to a temporary directory on every launch). |
| `--console` | Keeps a console window so log output is visible. Useful for diagnosing a build. |
| `--clean` | Remove `build/`, `dist/` and the spec file. |
| `--icon PATH` | Use a specific `.ico`. |

### Size

A one-folder build is roughly 1.2-2.5 GB, dominated by PyTorch's CUDA runtime
libraries. To ship a much smaller CPU-only build:

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python scripts\build_windows.py
```

That lands around 400-600 MB. Neural models still work, just slowly.

### Code signing

Unsigned executables trigger SmartScreen. With a code-signing certificate:

```powershell
signtool sign /fd SHA256 /t http://timestamp.digicert.com `
    /f cert.pfx /p PASSWORD dist\ForensicVision\ForensicVision.exe
```

### Installer

For an MSI or a Start-menu entry, wrap the one-folder output with
[Inno Setup](https://jrsoftware.org/isinfo.php) or WiX. Install to
`%PROGRAMFILES%` and leave cases and weights under `%LOCALAPPDATA%`, which is
where a frozen build already resolves them.

---

## Linux

```bash
python3 -m pip install pyinstaller
./scripts/build_linux.sh
```

Output: `dist/ForensicVision/ForensicVision`.

### AppImage

```bash
# appimagetool must be on PATH:
#   https://github.com/AppImage/AppImageKit/releases
./scripts/build_linux.sh --appimage
```

Output: `dist/ForensicVision-x86_64.AppImage`.

```bash
chmod +x ForensicVision-x86_64.AppImage
./ForensicVision-x86_64.AppImage
```

**Build on the oldest distribution you intend to support.** glibc is forward
compatible but not backward compatible, so a binary built on Ubuntu 24.04 will
not run on 22.04, while the reverse works. Ubuntu 22.04 is a good baseline.

### Running from source instead

For a controlled deployment, a virtualenv plus a `.desktop` file is often
simpler than freezing:

```ini
[Desktop Entry]
Type=Application
Name=ForensicVision
Exec=/opt/forensicvision/.venv/bin/python /opt/forensicvision/main.py
Icon=/opt/forensicvision/assets/icons/app.png
Categories=Graphics;Science;
Terminal=false
```

Place it in `~/.local/share/applications/` or `/usr/share/applications/`.

---

## Verifying a build

```bash
dist/ForensicVision/ForensicVision --check
```

**Check the "Compute device" line first.** This is the trap:

```
Compute device
--------------
  Device: CPU (PyTorch not installed)
  note: PyTorch unavailable: cannot import name 'nn' from partially
        initialized module 'torch' ... circular import
```

A frozen build with a broken torch does not crash. It reports *PyTorch not
installed*, falls back to CPU, and offers only the ten classical operators -
which looks exactly like a legitimate CPU-only configuration. On a working
build the line names your GPU (or a plain CPU with no error note), and the
model count is 10 of 30 rather than 0 of 30 neural models being loadable.

The usual cause is over-eager exclusion. `torch/__init__.py` imports several of
its own submodules eagerly, so excluding `torch.onnx`, `torch.testing`,
`torch.distributions` or `torch.utils.tensorboard` breaks the package. Exclude
TensorFlow at the *top* level instead - that is what actually stops the bloat
chain, without touching torch.

Then confirm by hand:

1. It starts and shows the dark theme.
2. **Tools > Model Manager** lists all 30 models with correct statuses.
3. Install one small model (DnCNN, 2.6 MiB) - proves the download worker and
   the writable weights directory both work in the frozen environment.
4. Creating a case and importing an image works.
5. Analysis populates every indicator.
6. A classical operator runs (proves the engine is wired without any download).
7. A neural operator runs (proves torch actually imported).
8. A PDF report generates.

Step 2 catches missing hidden imports: the model families are imported lazily
by name, so a missing one shows up there as a family reporting an error rather
than as a crash.

---

## Common problems

**`ModuleNotFoundError` for a `restoration.*` module at run time**
The families are imported by name in `register_all_models`, so PyInstaller's
static analysis cannot see them. They are listed in `HIDDEN_IMPORTS` in
`scripts/build_windows.py`; add any new family there.

**The build takes 20+ minutes and the log fills with protobuf access violations**
PyInstaller's `hook-torch` collects torch's submodules, which reaches
`torch.utils.tensorboard`. If TensorFlow is installed in the same environment -
common on a shared machine - that import chain drags in TensorFlow, Keras and
protobuf, crashing PyInstaller's isolated hook subprocesses and adding
gigabytes. The `EXCLUDES` list already excludes `tensorflow`, `tensorboard`,
`keras` and `jax` at the top level, which is the fix. Do not "solve" it by
excluding torch submodules - see the verification note above.

**The frozen app reports "PyTorch not installed" although torch builds fine**
See the verification section. Torch's own submodules must not be excluded.

**Qt platform plugin not found**
PyInstaller normally bundles the Qt plugins via its PyQt5 hook. If it does not:

```
--add-data "<venv>/Lib/site-packages/PyQt5/Qt5/plugins;PyQt5/Qt5/plugins"
```

**The build is enormous**
Use the CPU PyTorch build, and check `EXCLUDES` in the build script.

**`--onefile` starts very slowly**
Expected: it unpacks the whole bundle to a temporary directory on each launch.
Prefer one-folder for anything a user runs repeatedly.

**Antivirus flags the executable**
Common for unsigned PyInstaller output. Sign the binary, or submit it to the
vendor as a false positive.
