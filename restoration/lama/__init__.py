"""LaMa inpainting adapter (declared, not integrated).

LaMa (resolution-robust large-mask inpainting with Fourier convolutions) is
declared here so its licence, provenance and status are visible in the Model
Manager, but it is not executable in this build. See
:class:`LaMaInpainting.integration_note` for the specific blocker.
"""

from __future__ import annotations

import logging

from app.constants import ModelKind, TaskType
from restoration.base import ModelInfo, WeightSpec
from restoration.registry import ModelRegistry
from restoration.unavailable import NotIntegratedModel

logger = logging.getLogger(__name__)

__all__ = ["LaMaInpainting", "register_lama"]

_REPO = "https://github.com/advimman/lama"


class LaMaInpainting(NotIntegratedModel):
    """Large-mask inpainting via fast Fourier convolutions."""

    integration_note = (
        "Not integrated. Upstream distributes LaMa as a Hydra-configured "
        "training checkpoint rather than a plain state dictionary, and the "
        "release archive referenced by the project README is no longer "
        "resolvable, so neither the weights nor their configuration can be "
        "obtained reproducibly. Integrating it would require vendoring the "
        "upstream FFC model definition together with its Hydra config, which "
        "has not been done rather than shipping an unverifiable path."
    )
    suggested_alternative = "none for inpainting; ROI cropping is the supported workflow"

    info = ModelInfo(
        name="lama",
        display_name="LaMa Inpainting",
        task=TaskType.INPAINTING.value,
        kind=ModelKind.NEURAL.value,
        version="1.0",
        description="Resolution-robust large-mask inpainting.",
        method=(
            "Fills masked regions with content invented from a learned prior. "
            "Inpainted areas contain no information from the evidence and are "
            "reconstruction, not recovery."
        ),
        license_name="Apache-2.0 (code); weights CC BY-NC-SA 4.0",
        repository=_REPO,
        paper="Suvorov et al., 'Resolution-robust Large Mask Inpainting with Fourier Convolutions', WACV 2022",
        authors="Roman Suvorov et al., Samsung AI Center Moscow",
        weights=(
            WeightSpec(
                filename="big-lama.pt",
                url="",
                size_bytes=0,
                license_name="CC BY-NC-SA 4.0",
                source=f"{_REPO}#pretrained-models",
            ),
        ),
        requires_packages=("torch",),
        may_synthesise=True,
        notes=(
            "Inpainting is the most synthetic operation in the application. "
            "Any inpainted region must be marked as such in any report."
        ),
    )


def register_lama(replace: bool = False) -> int:
    """Register the LaMa adapter."""
    try:
        ModelRegistry.register(LaMaInpainting.info, LaMaInpainting, replace=replace)
        return 1
    except ValueError:
        logger.debug("LaMa already registered")
        return 0
