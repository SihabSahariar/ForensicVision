# Limitations

Read this before relying on any output from ForensicVision in a report,
disclosure or proceeding.

---

## 1. The fundamental limit

**No operation can recover information that is not in the file.**

Blur, downsampling and lossy compression are *lossy*: they drive spatial
frequencies to zero. Clipping at the black or white point discards samples
entirely. Once that has happened, the information is gone. Anything that
appears to restore it is producing an *estimate*, however convincing.

Concretely:

- A sample clipped to 255 could have been 255 or 4000. Nothing recovers which.
- A character stroke narrower than one pixel was never sampled. A model that
  renders a legible character there is inferring it from what characters
  usually look like.
- Frequencies removed by a blur kernel's transfer-function zeros cannot be
  inverted at all, by any algorithm.

The application marks operations that can invent content with `may synthesise`
and warns before running them. That warning is the substantive point, not a
formality.

## 2. Classical versus neural

| | Classical operators | Neural models |
|---|---|---|
| Output is | a deterministic function of the input samples | an estimate drawn from a learned prior |
| Can invent structure | **No** | **Yes** |
| Reproducible | Bit-identical on re-run | Bit-identical given identical hardware and precision |
| Failure mode | Visible artefacts - ringing, halos, flattening | *Plausible* wrong detail |

The neural failure mode is the dangerous one, because it does not look like a
failure. Always compare a neural result against the Lanczos Upscale baseline,
which cannot add anything.

## 3. Analysis indicators

Every degradation score is a **heuristic indicator** computed from classical
image statistics. None comes from a validated classification model. They exist
to direct attention and drive the pipeline recommender, not to support factual
claims about an image's provenance or history.

Known behaviours:

- **Blur** rises for genuinely low-texture scenes (fog, flat walls, night
  frames) that are perfectly in focus.
- **Motion blur** angle is estimated from gradient anisotropy, not from a
  measured camera trajectory. Strongly oriented scene content - fencing,
  blinds, text, brickwork - produces the same signature; the estimator gates on
  overall blur level to suppress this, but the gate is not perfect.
- **Noise** is inflated by fine scene texture. The reported sigma is an upper
  bound in textured regions.
- **JPEG** blocking is measured phase-selectively, so axis-aligned scene edges
  at a non-8-pixel pitch no longer produce a false positive. Content that
  genuinely repeats at exactly 8 pixels still will. Resampling destroys the
  block grid, so a resized JPEG scores low despite carrying the damage.
- **Low resolution** upscaling detection compares outer and mid spectral
  annuli. It reliably flags 2x-3x enlargement at any size and 4x above about
  600 px. **It cannot separate a heavily blurred native frame from an
  aggressively enlarged one** - blur and interpolation are both low-pass
  operations, and the two populations overlap. The threshold is deliberately
  biased towards missing an enlargement rather than falsely attributing one.
- **Haze** requires four signatures to agree (dark-channel lower quartile,
  saturation, relative local contrast, transmission uniformity). Dense uniform
  fog scores correctly; thin or localised haze scores low because its
  transmission is not uniform. The indicator is suppressed in largely clipped
  frames, where the prior has no colour information to work with.
- **Exposure** cannot distinguish a correctly exposed night scene from an
  under-exposed daylight one. It measures the histogram, not the scene.

## 4. Specific model limitations

- **Wiener deconvolution must not be used with the defocus-disk model.** A
  pillbox transfer function has exact zeros; the filter amplifies noise there
  instead of recovering anything, and can score *worse* than the blurred input.
  Use Richardson-Lucy for defocus. This is stated in the model's own
  documentation inside the application.
- **Richardson-Lucy amplifies noise as iterations rise.** On noisy input the
  result peaks and then degrades. There is no automatic stopping rule.
- **Every learned model carries its training distribution's biases.** Restormer
  motion-deblurring is trained on GoPro handheld footage; Restormer denoising on
  smartphone sensor noise; SwinIR classical SR on clean bicubic downsampling;
  DnCNN on additive white Gaussian noise. Surveillance encoder noise, thermal
  imagery, scanned film and document scans are all out of distribution, and
  behaviour there is not characterised.
- **Real-ESRGAN and the SwinIR GAN variant are adversarially trained.** They
  produce the most convincing texture and are correspondingly the most likely
  to invent it.
- **FBCNN's predicted quality factor can be wrong** on out-of-distribution
  content. The override exists so you can sweep the assumption and observe how
  sensitive the result is - a useful measurement in itself.
- **The grayscale FBCNN and DnCNN sigma-25 checkpoints are single-channel** and
  are applied per colour channel. This ignores chroma subsampling; prefer the
  colour variants for colour evidence.

## 5. Face restoration — read this before using it

CodeFormer is integrated and works. **It is the most dangerous feature in this
application**, and the danger is not that it fails — it is that it succeeds
convincingly at the wrong thing.

CodeFormer does not sharpen a face. It *replaces* the face with one
reconstructed from a learned codebook of high-quality face patches,
conditioned on the degraded input. The output is a plausible face consistent
with the input. It is not a measurement of the person depicted.

### A measured example

Run on `skimage.data.astronaut()` — a public-domain benchmark photograph —
degraded to 128 px with blur, sensor noise and JPEG quality 35, then restored:

| | Result |
|---|---|
| Detected face | 92 × 112 px, 44 px between the eyes |
| Restored output | Sharp, confident, entirely plausible |
| **Invented eyeglasses** | **The subject wears none in the ground truth** |
| Also changed | Apparent age, face shape, hairline, expression detail |

The model added eyewear that does not exist, and it did so with no visible
uncertainty. Nothing in the output signals which features were measured and
which were generated. Had this been a suspect image, an examiner comparing the
"restored" face to a reference photograph would have been comparing against
an invention.

This is a well-known CodeFormer behaviour, not a defect in this integration,
and it is the reason the application:

- requires an explicit, face-specific confirmation before every run;
- records the **inter-ocular distance** of each source face and warns below
  30 px, where a 512 px restoration is interpolating more than 17× linearly;
- exposes the fidelity weight so the range can be swept — if the face changes
  substantially between `w = 0` and `w = 1`, that variation is the prior, not
  the evidence;
- marks every derivative `may synthesise` in the database, the provenance
  sidecar and the report.

**Face restoration must never be used for identification.** Use it, if at all,
to judge whether a region merits further work — and report the output as a
synthesised reconstruction.

The detection stage (OpenCV YuNet) locates faces and produces five landmarks
solely to compute the alignment warp. It performs no recognition, no matching
and no identification.

## 6. Not integrated

Declared in the Model Manager with licences and status; they refuse to run
rather than returning anything:

- **GFPGAN** - its StyleGAN2 decoder depends on fused bias-activation and
  upfirdn2d CUDA extensions compiled at run time; the pure-PyTorch fallbacks
  are numerically divergent from the weights' training. CodeFormer covers the
  same task.
- **LaMa** - distributed as a Hydra-configured training checkpoint rather than
  a plain state dictionary, and the release archive named in the upstream
  README no longer resolves, so neither weights nor configuration can be
  obtained reproducibly.

**NAFNet** is a partial case. The architecture is implemented and its published
configurations reproduce the published parameter counts exactly (17.11 M and
67.89 M), and the adapter infers the block layout from whatever checkpoint is
installed. But upstream distributes the weights only through Google Drive, and
no attributable direct-download host exists — so unlike every other neural
model here, **NAFNet has never been verified against the published
checkpoint**. It is installed manually and its key-name compatibility with the
official weights is unverified. Restormer covers the same tasks and is
verified.

## 7. OCR

OCR is optional and, when installed, reads whatever image it is given. A
reading taken from an *enhanced derivative* reflects the derivative, not the
original. If the enhancement invented a character, OCR will confidently read
the invented character. Divergence between the before and after readings tells
you the enhancement changed what the engine saw - not which reading is correct.

## 8. Object detection

Class labels are the detector's estimate. `person` means the detector's
training distribution matched; it is not an identification, and it is not
evidence that a person is present.

## 9. Difference visualisations

Difference maps are rendered with a chosen gain and colour mapping. The
amplified view multiplies differences by 8; apparent intensity is a display
choice. Where the two images differ in size, the smaller is resampled for
comparison, which itself introduces differences.

Error Level Analysis is included as an inspection aid. It is heavily confounded
by local contrast and texture and is **not** evidence of manipulation.

## 10. Reproducibility

Classical pipelines are bit-reproducible on any machine. Neural results depend
on hardware, driver, cuDNN version and precision: an FP16 CUDA run and an FP32
CPU run of the same model on the same input produce visually identical but
not bit-identical output. The provenance record captures device, GPU model,
CUDA version and PyTorch version so a discrepancy can be explained.

Tiled inference blends overlapping tiles with a raised-cosine window. Output is
therefore not bit-identical to a single-pass run of the same model, though the
difference is far below the visual threshold.

## 11. Scope

ForensicVision does not:

- authenticate images or detect manipulation (ELA is an aid, not a verdict);
- perform photogrammetry, measurement or 3D reconstruction;
- process video (the engine is designed to be reusable for it; the application
  is not);
- perform facial recognition, face matching or identification of any kind;
- establish provenance beyond what it records itself from the point of import.

## 12. Chain of custody

The application records custody from the moment of import. It cannot attest to
anything before that: how the file was produced, transferred or handled prior
to import is outside its knowledge. Import records the source path and the
digest of the copy it stores, and nothing more.

---

> Algorithmic image enhancement modifies image data. AI-based restoration may
> infer or synthesize structures that are not directly represented in the
> source image. Enhanced imagery is a derivative representation and should not
> automatically be interpreted as an exact recovery of information absent from
> the original evidence.
