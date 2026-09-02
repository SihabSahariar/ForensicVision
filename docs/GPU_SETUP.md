# GPU setup

Neural restoration runs on CPU, but slowly. An NVIDIA GPU makes the difference
between seconds and minutes on a full-resolution frame.

**CUDA is optional.** Every classical operator and the whole analysis stage run
on CPU. The application starts, and reports honestly, with no GPU present.

---

## Requirements

- An NVIDIA GPU with compute capability 3.7 or newer (GTX 750 Ti onwards).
- A recent NVIDIA driver.
- The CUDA build of PyTorch.

A separate CUDA *toolkit* installation is **not** required: PyTorch ships its
own CUDA runtime. You only need a driver new enough for the build you install.

---

## Driver compatibility

| PyTorch wheel | CUDA runtime | Minimum driver (Windows) | Minimum driver (Linux) |
|---|---|---|---|
| `cu118` | 11.8 | 452.39 | 450.80.02 |
| `cu121` | 12.1 | 527.41 | 525.60.13 |
| `cu124` | 12.4 | 550.54 | 550.54.14 |
| `cu126` | 12.6 | 560.76 | 560.28.03 |

Check your driver:

```bash
nvidia-smi
```

The **CUDA Version** in that header is the highest runtime your driver
supports, not what is installed. A driver reporting 12.7 happily runs the
cu118, cu121, cu124 and cu126 builds.

`cu126` is this project's reference configuration.

---

## Installing

Uninstall any CPU build first - pip will not replace it automatically:

```bash
python -m pip uninstall -y torch torchvision
python -m pip install -r requirements-gpu.txt
```

Or pick a specific runtime:

```bash
# CUDA 12.1 - safe on older drivers
python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 - oldest supported
python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```bash
python main.py --check
```

```
Compute device
--------------
  Device: NVIDIA GeForce RTX 3070 Laptop GPU  |  CUDA 12.6  |  VRAM 1.0 / 8.0 GB
  GPU 0: NVIDIA GeForce RTX 3070 Laptop GPU (1.0 / 8.0 GB)  capability 8.6
```

---

## Memory and tiling

Large frames exceed the VRAM of a mid-range GPU. ForensicVision splits them
into overlapping tiles and recombines them with a raised-cosine weight ramp, so
no seam appears in the output - which matters here, because a seam would be an
artefact introduced by the tool and indistinguishable at a glance from one in
the evidence.

**Tools > Preferences > Processing**

| Setting | Default | Notes |
|---|---|---|
| Tile size | 512 | 0 processes the whole image in one pass |
| Tile overlap | 32 | Larger costs time, further reduces any seam risk |
| Half precision (FP16) | on | Roughly halves VRAM on CUDA |
| Auto-reduce on OOM | on | Halves the tile size and retries, down to 64 px |

Suggested starting points:

| VRAM | Tile size | FP16 |
|---|---|---|
| 4 GB | 256 | on |
| 6-8 GB | 512 | on |
| 12 GB+ | 768-1024 | optional |

Measured peak VRAM on this project's reference machine (RTX 3070 Laptop, 8 GB):

| Model | Input | Tile | Peak VRAM | Time |
|---|---|---|---|---|
| Real-ESRGAN x4plus | 320x213 | none | 0.64 GB | 3.0 s |
| Restormer motion deblur | 960x640 | 512 | 1.19 GB | 7.3 s |
| FBCNN colour | 960x640 | 512 | 1.1 GB | 7.1 s |
| DnCNN blind | 960x640 | 512 | 0.3 GB | 0.7 s |

If a run still exhausts memory at the minimum tile size the application says
so and suggests the CPU device or a smaller region, rather than crashing.

---

## Multiple GPUs

When several CUDA devices are visible, choose one in
**Tools > Preferences > Processing > CUDA device index**. A single restoration
runs on one device: splitting one image across GPUs would cost more in
transfers than it saves.

To pin a device for the whole session:

```bash
CUDA_VISIBLE_DEVICES=1 python main.py
```

Independent batch jobs can be spread across GPUs by launching one process per
device, each with its own `CUDA_VISIBLE_DEVICES` and its own case.

---

## Precision

FP16 halves memory and is roughly twice as fast on tensor-core hardware
(RTX 20-series onwards). The output differs from FP32 in the last bit or two -
far below the visual threshold, but **not bit-identical**. The provenance
record captures device, GPU model, CUDA version, PyTorch version and the
precision used, so a discrepancy between two runs can be explained rather than
merely observed.

For a result that must be reproducible on unlike hardware, use the CPU device,
or restrict the pipeline to classical operators, which are bit-reproducible
everywhere.

---

## Troubleshooting

**`torch.cuda.is_available()` is False**

```bash
python -c "import torch; print(torch.__version__)"
```

A version ending `+cpu` is the CPU build. Reinstall from the CUDA index.

**`CUDA error: no kernel image is available for execution on the device`**
The GPU is older than the wheel supports. Install the `cu118` build.

**Windows: CUDA works in a terminal but not from a shortcut**
The NVIDIA driver directory must be on `PATH`. Launch from a terminal, or use
the PyInstaller build, which bundles the runtime.

**Laptop with hybrid graphics**
Force the discrete GPU in the NVIDIA Control Panel (Windows) or run with
`__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` (Linux).

**Out of memory despite a small image**
Another process may hold VRAM - check `nvidia-smi`. The status bar shows live
usage, refreshed every four seconds.
