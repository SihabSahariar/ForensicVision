# Models

Two families, kept visibly distinct because the difference is what matters
forensically.

---

## Classical operators - always available

Deterministic signal processing. No download, no PyTorch, no GPU. Every output
sample is a function of measured input samples, so **none of these can
introduce structure that is not derivable from the evidence**.

| Operator | Task | Method | Watch for |
|---|---|---|---|
| Lanczos Upscale | Super-resolution | Windowed-sinc reconstruction | Adds pixels, never new frequencies - the honest baseline |
| Richardson-Lucy | Deblur | Iterative maximum-likelihood deconvolution | Amplifies noise as iterations rise; no automatic stopping rule |
| Wiener | Deblur | Regularised inverse filter | **Unsuitable for defocus disk** - pillbox zeros cannot be inverted |
| Unsharp Mask | Sharpen | Scaled high-pass residual | Edge halos; increases acutance, recovers nothing |
| Non-Local Means | Denoise | Weighted mean of self-similar patches | Flattens genuine fine texture at high strength |
| Bilateral | Denoise | Edge-preserving local averaging | "Watercolour" flattening at high settings |
| JPEG Deblocking | JPEG | Boundary-selective smoothing on the 8x8 lattice | Removes the grid, not the discarded detail |
| Dark Channel Dehaze | Dehaze | DCP transmission + guided filter | Strong noise amplification in dense regions |
| Exposure Correction | Exposure | Gamma + percentile level stretch | Cannot restore clipped samples |
| CLAHE | Contrast | Contrast-limited adaptive equalisation | Raises noise with signal; tile gradients at high clip |

## Neural models - explicit installation

Architectures are implemented natively in this repository with
**checkpoint-compatible layer naming**, so official upstream weights load
unmodified. There is no dependency on `basicsr` or `realesrgan`, which are
unmaintained and fail against torchvision >= 0.17.

| Model | Task | Params | Weights | Licence |
|---|---|---|---|---|
| Real-ESRGAN x4plus | SR x4 | 16.7 M | 64 MiB, direct | BSD-3-Clause |
| Real-ESRGAN x2plus | SR x2 | 16.7 M | 64 MiB, direct | BSD-3-Clause |
| Real-ESRGAN 6-block | SR x4 | 4.5 M | 17 MiB, direct | BSD-3-Clause |
| SwinIR Classical SR | SR x4 | 11.9 M | 65 MiB, direct | Apache-2.0 |
| SwinIR Real-World SR | SR x4 | 28.0 M | 136 MiB, direct | Apache-2.0 |
| SwinIR Colour Denoise | Denoise | 11.5 M | 117 MiB, direct | Apache-2.0 |
| SwinIR JPEG CAR | JPEG | 11.5 M | 98 MiB, direct | Apache-2.0 |
| Restormer Motion Deblur | Deblur | 26.1 M | 100 MiB, direct | **Non-commercial** |
| Restormer Defocus Deblur | Deblur | 26.1 M | 100 MiB, direct | **Non-commercial** |
| Restormer Denoise | Denoise | 26.1 M | 100 MiB, direct | **Non-commercial** |
| FBCNN colour | JPEG | 71.9 M | 274 MiB, direct | Apache-2.0 |
| FBCNN grayscale | JPEG | 71.9 M | 274 MiB, direct | Apache-2.0 |
| DnCNN colour blind | Denoise | 0.67 M | 2.6 MiB, direct | MIT |
| DnCNN sigma-25 | Denoise | 0.56 M | 2.1 MiB, direct | MIT |
| CodeFormer | Face restoration | 94.1 M | 359 MiB, direct | **Non-commercial** |
| YuNet (face detector) | Face alignment | 0.09 M | 227 KiB, direct | MIT |
| NAFNet deblur / denoise | Deblur / denoise | 17-68 M | **manual** | MIT code, NC weights |
| Zero-DCE | Exposure (low light) | 0.079 M | 313 KiB, direct | **CC BY-NC 4.0** |
| Zero-DCE++ | Exposure (low light) | 0.011 M | 51 KiB, direct | **CC BY-NC 4.0** |

**Every neural model can synthesise detail.** They are all marked
`may synthesise` and the application warns before running one.

### Checkpoint compatibility, verified

Each architecture was loaded against the real upstream checkpoint:

| Model | Keys | Missing | Unexpected | Params vs published |
|---|---|---|---|---|
| Real-ESRGAN x4plus | 702 | 0 | 0 | 16.70 M ✓ |
| Restormer | 494 | 0 | 0 | 26.13 M ✓ |
| SwinIR classical x4 | 550 | 0 | 0 | 11.90 M ✓ |
| FBCNN colour | 184 | 0 | 0 | 71.92 M ✓ |
| DnCNN blind | 40 | 0 | 0 | 0.67 M ✓ |
| DnCNN sigma-25 | 34 | 0 | 0 | 0.56 M ✓ |
| CodeFormer | 515 | 0 | 0 | 94.11 M ✓ |
| Zero-DCE | 14 | 0 | 0 | 0.079 M ✓ |
| Zero-DCE++ | 28 | 0 | 0 | 0.011 M ✓ |

Weights load with `strict` key checking. A mismatch raises rather than
silently producing a partially-initialised network, which would output
plausible-looking noise - the worst possible failure mode here.

Zero-DCE++ is additionally pinned against a transcription of the published
forward pass at every supported curve resolution, because this implementation
resizes by explicit target size where upstream crops the input to a multiple of
the scale factor. On a divisible input the two agree bit for bit; on any other
input this one keeps the full frame instead of discarding evidence at the right
and bottom edges.

### Neural is not the same question as synthesising

`kind` and `may_synthesise` are independent axes, and Zero-DCE is where they
come apart. Both Zero-DCE variants are `neural` - they are trained networks with
downloaded weights - and both are `may_synthesise=False`.

The reason is structural rather than a judgement call. These networks do not
output an image. They output the coefficients of a tone curve, which is then
applied eight times:

```
LE(x; r) = x + r * x * (x - 1),    r = tanh(...) in [-1, 1]
```

Its derivative `1 + r(2x - 1)` has minimum 0 over `x in [0,1], r in [-1,1]`, so
the curve is monotonically non-decreasing, and so is any composition of it. Each
output pixel is therefore a monotone function of *that pixel's own input value*.
The network chooses which monotone function; it cannot paint an edge, a
character or a facial feature that the input does not contain.
`tests/test_zerodce.py` asserts this both symbolically and against the published
weights, and the declaration is only valid while those tests pass.

DnCNN is declared the same way for a different reason: it is discriminative,
predicting and subtracting a noise residual with no learned image prior to draw
new structure from.

What Zero-DCE *can* do is covered in
[LIMITATIONS.md](LIMITATIONS.md#13-low-light-curve-models).

---

## Installing

### From the application

**Tools > Model Manager** (`Ctrl+M`). Each row shows task, kind, version,
licence, size and status. Selecting one shows the full record: authors, paper,
repository, what it does to the pixels, and the exact weight URL.

Press **Install** and the download URL, licence and size are shown for
confirmation. Nothing is fetched until you approve it. Downloads verify against
the published SHA-256 where upstream provides one, and always land atomically.

### From the command line

```bash
python scripts/download_models.py --list
python scripts/download_models.py --describe realesrgan_x4plus
python scripts/download_models.py --install realesrgan_x4plus fbcnn_color
python scripts/download_models.py --install-task super_resolution
python scripts/download_models.py --verify
```

Zero-DCE's weights live in the upstream Git repository rather than in a
release, so the declared URL is a `raw.githubusercontent.com` path pinned to the
default branch. Both files verify against a published SHA-256 recorded in this
repository:

| File | Bytes | SHA-256 |
|---|---|---|
| `zerodce_epoch99.pth` | 320,017 | `a4395acb874f3203...f74c3612` |
| `zerodce_pp_epoch99.pth` | 52,395 | `ca8855b90df9a80f...067f3b84` |

### Manually (air-gapped, or NAFNet)

NAFNet is published only via Google Drive. Rather than invent a mirror URL -
a weight file whose provenance cannot be stated is not admissible tooling - the
Model Manager shows the upstream page and offers **Install from file**.

Download the checkpoint yourself, then either drop it into the weights folder
under its declared filename, or use *Install from file*, which verifies the
digest before installing.

The NAFNet adapter reads the block layout back out of whatever checkpoint you
install (GoPro uses `[1,1,1,28]` with one middle block, SIDD uses `[2,2,4,8]`
with twelve), so any published variant loads correctly.

---

## Choosing a model

| Situation | Suggested |
|---|---|
| Establishing a defensible baseline | Lanczos Upscale - cannot invent anything |
| Known blur type, low noise | Richardson-Lucy with the matching PSF |
| Linear motion blur | Wiener (measured +6.6 dB, best of any option tested) |
| Defocus blur | Richardson-Lucy - **not** Wiener |
| Heavy JPEG on colour evidence | FBCNN colour |
| Real sensor noise | DnCNN blind, or Restormer denoise |
| Small CCTV frame, general enlargement | Real-ESRGAN x4plus |
| A face in the frame | CodeFormer — but read LIMITATIONS.md §5 first |
| Enlargement with the least invention | SwinIR Classical SR (PSNR-trained, not GAN) |

Always run the classical baseline alongside a neural result. If the neural
output shows detail the baseline does not, that detail was inferred.

---

## Face restoration

CodeFormer is fully integrated as a four-stage pipeline:

1. **Detect** — OpenCV YuNet locates faces and emits five landmarks (right eye,
   left eye, nose tip, right mouth corner, left mouth corner).
2. **Align** — a similarity transform warps each face into the canonical
   512×512 FFHQ frame the network was trained on. This stage is not optional:
   an unaligned crop produces plausible but geometrically wrong output.
3. **Restore** — the VQGAN encoder, codebook-lookup transformer and generator
   reconstruct the face, with a fidelity weight controlling how much encoder
   detail is mixed back in.
4. **Blend** — the result is warped back and feathered in proportional to face
   area, so the boundary does not itself become an artefact.

**This is the highest-risk operation in the application.** CodeFormer does not
sharpen a face — it replaces it with one reconstructed from a learned prior.
On the standard `astronaut` benchmark degraded to 128 px, it produced a sharp,
confident face **wearing eyeglasses the subject does not wear**, and also
altered apparent age, face shape and hairline.

Safeguards:

- a face-specific confirmation dialog before every run;
- the **inter-ocular distance** of each source face is measured and recorded,
  with a warning below 30 px;
- the fidelity weight is exposed so the range can be swept — variation between
  `w = 0` and `w = 1` is the prior, not the evidence;
- every derivative is marked `may synthesise` in the database, the provenance
  sidecar and the report.

It must never be used for identification. Read
[LIMITATIONS.md §5](LIMITATIONS.md) before using it at all.

## Not integrated

Declared in the Model Manager with licences and status; they refuse to run
rather than returning anything.

| Model | Blocker |
|---|---|
| GFPGAN | StyleGAN2 decoder needs fused-bias-activation and upfirdn2d CUDA extensions; pure-PyTorch fallbacks diverge numerically from the training. CodeFormer covers the same task. |
| LaMa | Distributed as a Hydra-configured training checkpoint, and the release archive named upstream no longer resolves. |

**NAFNet** is a partial case: the architecture reproduces the published
parameter counts exactly and the adapter infers its layout from whatever
checkpoint is installed, but because upstream ships weights only via Google
Drive it has **never been verified against the published checkpoint**.
Restormer covers the same tasks and is verified.

---

## Storage

Default: `models/weights/` in the repository, or
`%LOCALAPPDATA%\ForensicVision\weights` for a frozen build. Change it in
**Tools > Preferences > Folders** or with `FORENSICVISION_WEIGHTS_DIR`.

Installing everything with a direct URL is about 1.4 GB. The Model Manager
shows the current total.

---

## Adding a model

1. Put the architecture in `restoration/<name>/arch.py`, matching upstream
   layer names so official checkpoints load.
2. Subclass `TorchRestorationModel` in `restoration/<name>/model.py` and
   populate `ModelInfo` - including `license_name`, `repository`, `method` and
   `may_synthesise`, which are all surfaced to the investigator and printed in
   reports.
3. Add a `register_<name>()` function and list it in
   `restoration/__init__.py::register_all_models`.
4. Verify the checkpoint loads with zero missing and zero unexpected keys.

See [ARCHITECTURE.md](ARCHITECTURE.md).
