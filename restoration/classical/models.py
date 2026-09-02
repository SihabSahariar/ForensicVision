"""Classical operators exposed as :class:`~restoration.base.RestorationModel`.

These are deterministic signal-processing algorithms. They require no weight
downloads, run on any machine, and - critically for forensic use - cannot
introduce structures that are not derivable from the measured input samples.
Every one is registered with ``kind=ModelKind.CLASSICAL`` and
``may_synthesise=False``, and the UI labels them accordingly.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from app.constants import ModelKind, TaskType
from restoration.base import ModelInfo, ParamSpec, ProgressReporter, RestorationModel
from restoration.classical import deconvolution, enhance, psf
from restoration.registry import ModelRegistry

logger = logging.getLogger(__name__)

__all__ = ["register_classical_models"]

_LICENSE = "ForensicVision (Apache-2.0); algorithms implemented from published literature"


class _ClassicalModel(RestorationModel):
    """Base for classical operators: nothing to load or unload."""

    def load(self, device: str = "cpu", fp16: bool = False) -> None:
        """Mark the operator ready, always on the CPU.

        These operators run through NumPy and OpenCV regardless of what the
        pipeline runner selected for the neural steps. Recording the runner's
        device here would put "CUDA" in the provenance record for an operation
        that demonstrably ran on the CPU, so the requested device is ignored.
        """
        self.require_available()
        self._device = "cpu"
        self._fp16 = False
        self._loaded = True

    def _load(self) -> None:  # pragma: no cover - trivial
        return

    def _unload(self) -> None:  # pragma: no cover - trivial
        return


# --------------------------------------------------------------------------- #
# Super resolution
# --------------------------------------------------------------------------- #

class LanczosUpscaleModel(_ClassicalModel):
    """Windowed-sinc upscaling - the honest interpolation baseline."""

    info = ModelInfo(
        name="lanczos",
        display_name="Lanczos Upscale",
        task=TaskType.SUPER_RESOLUTION.value,
        kind=ModelKind.CLASSICAL.value,
        version="1.0",
        description=(
            "Resamples the image onto a larger pixel grid using a Lanczos-4 "
            "windowed-sinc kernel."
        ),
        method=(
            "Band-limited reconstruction of the existing samples. Produces more "
            "pixels but no new spatial frequencies - the correct baseline "
            "against which any learned super-resolution result should be judged."
        ),
        license_name=_LICENSE,
        paper="Duchon, 'Lanczos Filtering in One and Two Dimensions', 1979",
        parameters=(
            ParamSpec(
                name="scale",
                label="Scale",
                kind="choice",
                default=2,
                choices=((2, "2x"), (3, "3x"), (4, "4x"), (8, "8x")),
                help_text="Output size multiplier.",
            ),
        ),
        may_synthesise=False,
    )

    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        scale = int(params.get("scale", 2))
        if progress is not None:
            progress(20, f"Lanczos resample x{scale}")
        result = enhance.lanczos_resize(image, float(scale))
        if progress is not None:
            progress(100, "Resample complete")
        return result


# --------------------------------------------------------------------------- #
# Deblur
# --------------------------------------------------------------------------- #

class RichardsonLucyModel(_ClassicalModel):
    """Iterative maximum-likelihood deconvolution under an explicit PSF."""

    info = ModelInfo(
        name="richardson_lucy",
        display_name="Richardson-Lucy Deconvolution",
        task=TaskType.DEBLUR.value,
        kind=ModelKind.CLASSICAL.value,
        version="1.0",
        description=(
            "Iteratively inverts a user-specified blur model (Gaussian, linear "
            "motion or defocus disk)."
        ),
        method=(
            "Maximum-likelihood deconvolution for Poisson observations. The "
            "examiner selects the blur model, so the assumption being made is "
            "explicit and reviewable. Cannot recover frequencies the blur "
            "annihilated, and amplifies noise and ringing as iterations rise."
        ),
        license_name=_LICENSE,
        paper="Richardson, JOSA 1972; Lucy, AJ 1974",
        parameters=(
            ParamSpec(
                name="psf_type",
                label="Blur model",
                kind="choice",
                default="gaussian",
                choices=(
                    ("gaussian", "Gaussian (soft focus)"),
                    ("motion", "Linear motion"),
                    ("disk", "Defocus disk"),
                ),
                help_text="The point-spread function assumed when inverting.",
            ),
            ParamSpec(
                name="radius", label="Blur radius / sigma", kind="float",
                default=2.0, minimum=0.3, maximum=15.0, step=0.1,
                help_text="Gaussian sigma, disk radius, or motion length in pixels.",
            ),
            ParamSpec(
                name="angle", label="Motion angle (deg)", kind="float",
                default=0.0, minimum=-90.0, maximum=90.0, step=1.0,
                help_text="Direction of travel; used by the linear motion model only.",
            ),
            ParamSpec(
                name="iterations", label="Iterations", kind="int",
                default=20, minimum=1, maximum=200, step=1,
                help_text="More iterations sharpen further but amplify noise.",
            ),
            ParamSpec(
                name="damping", label="Damping", kind="float",
                default=0.0, minimum=0.0, maximum=0.9, step=0.05,
                help_text="Stabilises the update at the cost of sharpness.",
            ),
        ),
        supports_tiling=False,
        may_synthesise=False,
    )

    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        kernel = _build_psf(params)
        return deconvolution.richardson_lucy(
            image,
            kernel,
            iterations=int(params.get("iterations", 20)),
            damping=float(params.get("damping", 0.0)),
            progress=progress,
        )


class WienerDeblurModel(_ClassicalModel):
    """Single-pass regularised inverse filter."""

    info = ModelInfo(
        name="wiener",
        display_name="Wiener Deconvolution",
        task=TaskType.DEBLUR.value,
        kind=ModelKind.CLASSICAL.value,
        version="1.0",
        description=(
            "Frequency-domain minimum-mean-square-error inverse of a "
            "user-specified blur model."
        ),
        method=(
            "A single non-iterative linear inverse. Faster and more stable than "
            "Richardson-Lucy but softer; the noise-to-signal term sets the "
            "trade-off between sharpness and noise amplification. Not suitable "
            "for the defocus-disk model: a pillbox transfer function has exact "
            "zeros, and the filter amplifies noise and ringing at those "
            "frequencies instead of recovering anything. Use Richardson-Lucy "
            "for defocus."
        ),
        license_name=_LICENSE,
        paper="Wiener, 'Extrapolation, Interpolation, and Smoothing of "
        "Stationary Time Series', 1949",
        parameters=(
            ParamSpec(
                name="psf_type", label="Blur model", kind="choice",
                default="gaussian",
                choices=(
                    ("gaussian", "Gaussian (soft focus)"),
                    ("motion", "Linear motion"),
                    ("disk", "Defocus disk - prefer Richardson-Lucy"),
                ),
                help_text=(
                    "The defocus-disk transfer function has exact zeros, which "
                    "this filter cannot invert. Richardson-Lucy handles that "
                    "case correctly."
                ),
            ),
            ParamSpec(
                name="radius", label="Blur radius / sigma", kind="float",
                default=2.0, minimum=0.3, maximum=15.0, step=0.1,
            ),
            ParamSpec(
                name="angle", label="Motion angle (deg)", kind="float",
                default=0.0, minimum=-90.0, maximum=90.0, step=1.0,
            ),
            ParamSpec(
                name="noise_to_signal", label="Noise / signal", kind="float",
                default=0.01, minimum=0.0001, maximum=0.5, step=0.001,
                help_text="Higher suppresses noise; lower sharpens more.",
            ),
        ),
        may_synthesise=False,
        notes=(
            "Most effective on linear motion blur, where it typically "
            "outperforms Richardson-Lucy substantially. Do not use it with the "
            "defocus-disk model."
        ),
    )

    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        kernel = _build_psf(params)
        if progress is not None:
            progress(30, "Wiener inverse filtering")
        result = deconvolution.wiener_deconvolution(
            image, kernel, noise_to_signal=float(params.get("noise_to_signal", 0.01))
        )
        if progress is not None:
            progress(100, "Deconvolution complete")
        return result


class UnsharpMaskModel(_ClassicalModel):
    """Acutance enhancement via unsharp masking."""

    info = ModelInfo(
        name="unsharp",
        display_name="Unsharp Mask",
        task=TaskType.SHARPEN.value,
        kind=ModelKind.CLASSICAL.value,
        version="1.0",
        description="Boosts local contrast to increase apparent sharpness.",
        method=(
            "Adds a scaled high-pass residual back to the image. Increases "
            "acutance only; it recovers no lost detail and produces edge halos "
            "when over-applied, which must not be read as scene content."
        ),
        license_name=_LICENSE,
        parameters=(
            ParamSpec(
                name="radius", label="Radius", kind="float",
                default=1.5, minimum=0.3, maximum=20.0, step=0.1,
            ),
            ParamSpec(
                name="amount", label="Amount", kind="float",
                default=0.8, minimum=0.0, maximum=4.0, step=0.05,
            ),
            ParamSpec(
                name="threshold", label="Threshold", kind="float",
                default=0.0, minimum=0.0, maximum=0.2, step=0.005,
                help_text="Skip sharpening below this local contrast, sparing flat-area noise.",
            ),
        ),
        may_synthesise=False,
    )

    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        if progress is not None:
            progress(40, "Unsharp masking")
        result = deconvolution.unsharp_mask(
            image,
            radius=float(params.get("radius", 1.5)),
            amount=float(params.get("amount", 0.8)),
            threshold=float(params.get("threshold", 0.0)),
        )
        if progress is not None:
            progress(100, "Sharpening complete")
        return result


# --------------------------------------------------------------------------- #
# Denoise
# --------------------------------------------------------------------------- #

class NonLocalMeansModel(_ClassicalModel):
    """Non-local means denoising."""

    info = ModelInfo(
        name="nlm",
        display_name="Non-Local Means Denoise",
        task=TaskType.DENOISE.value,
        kind=ModelKind.CLASSICAL.value,
        version="1.0",
        description="Averages self-similar patches to suppress noise.",
        method=(
            "Every output sample is a weighted mean of measured input samples "
            "whose surrounding patches are similar. Preserves repeated "
            "structure such as text and brickwork; over-strong settings flatten "
            "genuine fine texture."
        ),
        license_name=_LICENSE + "; implementation from OpenCV (Apache-2.0)",
        paper="Buades, Coll & Morel, CVPR 2005",
        parameters=(
            ParamSpec(
                name="strength", label="Luminance strength", kind="float",
                default=6.0, minimum=0.5, maximum=30.0, step=0.5,
            ),
            ParamSpec(
                name="chroma_strength", label="Chroma strength", kind="float",
                default=8.0, minimum=0.5, maximum=30.0, step=0.5,
            ),
            ParamSpec(
                name="template", label="Patch size", kind="int",
                default=7, minimum=3, maximum=11, step=2,
            ),
            ParamSpec(
                name="search", label="Search window", kind="int",
                default=21, minimum=7, maximum=35, step=2,
            ),
        ),
        supports_tiling=True,
        may_synthesise=False,
    )

    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        if progress is not None:
            progress(20, "Non-local means denoising")
        result = enhance.nlm_denoise(
            image,
            strength=float(params.get("strength", 6.0)),
            chroma_strength=float(params.get("chroma_strength", 8.0)),
            template=int(params.get("template", 7)),
            search=int(params.get("search", 21)),
        )
        if progress is not None:
            progress(100, "Denoising complete")
        return result


class BilateralDenoiseModel(_ClassicalModel):
    """Edge-preserving bilateral smoothing."""

    info = ModelInfo(
        name="bilateral",
        display_name="Bilateral Denoise",
        task=TaskType.DENOISE.value,
        kind=ModelKind.CLASSICAL.value,
        version="1.0",
        description="Fast edge-preserving smoothing for mild noise.",
        method=(
            "Weighted local averaging with weights falling off in both space "
            "and intensity. Much faster than non-local means; flattens fine "
            "texture into a characteristic 'watercolour' look at high settings."
        ),
        license_name=_LICENSE + "; implementation from OpenCV (Apache-2.0)",
        paper="Tomasi & Manduchi, ICCV 1998",
        parameters=(
            ParamSpec(
                name="diameter", label="Diameter", kind="int",
                default=7, minimum=3, maximum=25, step=2,
            ),
            ParamSpec(
                name="sigma_color", label="Colour sigma", kind="float",
                default=0.08, minimum=0.005, maximum=0.5, step=0.005,
            ),
            ParamSpec(
                name="sigma_space", label="Spatial sigma", kind="float",
                default=5.0, minimum=1.0, maximum=30.0, step=0.5,
            ),
        ),
        supports_tiling=True,
        may_synthesise=False,
    )

    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        if progress is not None:
            progress(40, "Bilateral filtering")
        result = enhance.bilateral_denoise(
            image,
            diameter=int(params.get("diameter", 7)),
            sigma_color=float(params.get("sigma_color", 0.08)),
            sigma_space=float(params.get("sigma_space", 5.0)),
        )
        if progress is not None:
            progress(100, "Denoising complete")
        return result


# --------------------------------------------------------------------------- #
# JPEG
# --------------------------------------------------------------------------- #

class JpegDeblockModel(_ClassicalModel):
    """Block-grid-selective smoothing for JPEG artefacts."""

    info = ModelInfo(
        name="deblock",
        display_name="JPEG Deblocking",
        task=TaskType.JPEG_ARTIFACT.value,
        kind=ModelKind.CLASSICAL.value,
        version="1.0",
        description="Smooths discontinuities on the 8x8 JPEG block grid.",
        method=(
            "Applies edge-preserving smoothing only along the block boundary "
            "lattice, leaving block interiors untouched. Removes the visible "
            "grid without recovering the detail quantisation discarded."
        ),
        license_name=_LICENSE,
        parameters=(
            ParamSpec(
                name="strength", label="Strength", kind="float",
                default=0.6, minimum=0.0, maximum=1.0, step=0.05,
            ),
        ),
        may_synthesise=False,
    )

    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        if progress is not None:
            progress(40, "Deblocking")
        result = enhance.jpeg_deblock(
            image, strength=float(params.get("strength", 0.6))
        )
        if progress is not None:
            progress(100, "Deblocking complete")
        return result


# --------------------------------------------------------------------------- #
# Dehaze / exposure / contrast
# --------------------------------------------------------------------------- #

class DarkChannelDehazeModel(_ClassicalModel):
    """Dark channel prior dehazing."""

    info = ModelInfo(
        name="dcp_dehaze",
        display_name="Dark Channel Dehaze",
        task=TaskType.DEHAZE.value,
        kind=ModelKind.CLASSICAL.value,
        version="1.0",
        description="Inverts the atmospheric scattering model using the dark channel prior.",
        method=(
            "Estimates airlight and a guided-filter-refined transmission map, "
            "then solves I = J*t + A*(1-t) for the scene radiance. Strongly "
            "amplifies noise in dense regions; the transmission floor bounds it."
        ),
        license_name=_LICENSE,
        paper="He, Sun & Tang, CVPR 2009",
        parameters=(
            ParamSpec(
                name="omega", label="Haze removal", kind="float",
                default=0.90, minimum=0.1, maximum=1.0, step=0.05,
                help_text="Fraction of the veil removed; below 1 keeps aerial perspective.",
            ),
            ParamSpec(
                name="t_min", label="Transmission floor", kind="float",
                default=0.12, minimum=0.02, maximum=0.6, step=0.01,
                help_text="Raise to limit noise amplification in dense haze.",
            ),
            ParamSpec(
                name="patch", label="Patch size", kind="int",
                default=15, minimum=3, maximum=41, step=2,
            ),
        ),
        may_synthesise=False,
    )

    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        if progress is not None:
            progress(30, "Estimating transmission")
        result = enhance.dark_channel_dehaze(
            image,
            omega=float(params.get("omega", 0.90)),
            patch=int(params.get("patch", 15)),
            t_min=float(params.get("t_min", 0.12)),
        )
        if progress is not None:
            progress(100, "Dehazing complete")
        return result


class ExposureModel(_ClassicalModel):
    """Gamma and level correction."""

    info = ModelInfo(
        name="exposure",
        display_name="Exposure Correction",
        task=TaskType.EXPOSURE.value,
        kind=ModelKind.CLASSICAL.value,
        version="1.0",
        description="Applies a gamma curve and an optional percentile level stretch.",
        method=(
            "A monotonic per-pixel tone mapping. Reversible in principle and "
            "introduces no spatial structure, but amplifies existing noise in "
            "shadows and cannot restore clipped samples."
        ),
        license_name=_LICENSE,
        parameters=(
            ParamSpec(
                name="gamma", label="Gamma", kind="float",
                default=0.6, minimum=0.1, maximum=3.0, step=0.05,
                help_text="Below 1 brightens; above 1 darkens.",
            ),
            ParamSpec(
                name="auto_levels", label="Auto levels", kind="bool", default=True,
                help_text="Stretch the 0.5-99.5 luminance percentiles to full range.",
            ),
        ),
        may_synthesise=False,
    )

    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        result = image
        if params.get("auto_levels", True):
            if progress is not None:
                progress(30, "Auto levels")
            result = enhance.auto_levels(result)
        gamma = float(params.get("gamma", 0.6))
        if abs(gamma - 1.0) > 1e-3:
            if progress is not None:
                progress(70, "Gamma correction")
            result = enhance.gamma_correct(result, gamma)
        if progress is not None:
            progress(100, "Exposure correction complete")
        return result


class ClaheModel(_ClassicalModel):
    """Contrast-limited adaptive histogram equalisation."""

    info = ModelInfo(
        name="clahe",
        display_name="CLAHE Contrast",
        task=TaskType.CONTRAST.value,
        kind=ModelKind.CLASSICAL.value,
        version="1.0",
        description="Local contrast enhancement with a contrast limit.",
        method=(
            "Equalises the L* channel per tile with a clip limit, then "
            "interpolates between tiles. Reveals detail in shadows and "
            "highlights simultaneously; raises noise along with signal and can "
            "produce tile-boundary gradients at high clip limits."
        ),
        license_name=_LICENSE + "; implementation from OpenCV (Apache-2.0)",
        paper="Zuiderveld, 'Graphics Gems IV', 1994",
        parameters=(
            ParamSpec(
                name="clip_limit", label="Clip limit", kind="float",
                default=2.0, minimum=0.5, maximum=10.0, step=0.1,
            ),
            ParamSpec(
                name="tile_grid", label="Tile grid", kind="int",
                default=8, minimum=2, maximum=32, step=1,
            ),
        ),
        may_synthesise=False,
    )

    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        if progress is not None:
            progress(40, "Adaptive histogram equalisation")
        result = enhance.clahe(
            image,
            clip_limit=float(params.get("clip_limit", 2.0)),
            tile_grid=int(params.get("tile_grid", 8)),
        )
        if progress is not None:
            progress(100, "Contrast enhancement complete")
        return result


# --------------------------------------------------------------------------- #
# Helpers and registration
# --------------------------------------------------------------------------- #

def _build_psf(params: dict) -> np.ndarray:
    """Construct the PSF selected by ``params``."""
    kind = str(params.get("psf_type", "gaussian"))
    radius = float(params.get("radius", 2.0))
    if kind == "motion":
        return psf.motion_psf(length=max(2.0, radius * 2.0),
                              angle_deg=float(params.get("angle", 0.0)))
    if kind == "disk":
        return psf.disk_psf(radius=radius)
    return psf.gaussian_psf(sigma=radius)


_CLASSICAL_MODELS = (
    LanczosUpscaleModel,
    RichardsonLucyModel,
    WienerDeblurModel,
    UnsharpMaskModel,
    NonLocalMeansModel,
    BilateralDenoiseModel,
    JpegDeblockModel,
    DarkChannelDehazeModel,
    ExposureModel,
    ClaheModel,
)


def register_classical_models(replace: bool = False) -> int:
    """Register every classical operator with the shared registry.

    Args:
        replace: Overwrite existing registrations (used when reloading).

    Returns:
        The number of models registered.
    """
    count = 0
    for model_class in _CLASSICAL_MODELS:
        try:
            ModelRegistry.register(
                model_class.info, model_class, replace=replace
            )
            count += 1
        except ValueError:
            logger.debug("Model %s already registered", model_class.info.name)
    logger.info("Registered %d classical operators", count)
    return count
