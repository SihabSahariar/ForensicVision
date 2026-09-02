"""Face-restoration adapters.

CodeFormer is fully integrated: detect -> align to the canonical FFHQ frame ->
restore -> blend back. GFPGAN remains declared but not integrated; the reason
is recorded on the class.

This is the highest-risk operation in the application. CodeFormer does not
sharpen a face - it *replaces* it with a face reconstructed from a learned
codebook of high-quality face patches, conditioned on the degraded input. The
result is a plausible face consistent with the input, not a measurement of the
person depicted. Everything here is built to keep that fact in front of the
examiner: the adapter records the inter-ocular distance of each source face
(the honest measure of how much facial information actually existed), warns
when it is too small for the output to be evidence-led, and marks every
derivative as synthesised.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.constants import ModelKind, TaskType
from core.exceptions import InferenceError, ModelNotAvailableError
from restoration.base import ModelInfo, ParamSpec, ProgressReporter, WeightSpec
from restoration.registry import ModelRegistry
from restoration.torch_base import TorchRestorationModel, require_torch
from restoration.unavailable import NotIntegratedModel

logger = logging.getLogger(__name__)

__all__ = ["CodeFormer", "GFPGAN", "register_codeformer"]

_FACE_WARNING = (
    "Face restoration generates facial detail from a learned prior over human "
    "faces. The result is a plausible face consistent with the degraded input, "
    "not a measurement of the person depicted. It must never be used for "
    "identification, and any report including it must state that the facial "
    "detail is synthesised."
)

#: Inter-ocular distance below which the output is dominated by the prior.
#: Face recognition literature treats roughly 60 px between the eyes as the
#: floor for reliable work; below about 30 px a 512x512 restoration is
#: interpolating more than 17x linearly from the available samples.
_LOW_INFORMATION_IOD = 30.0


class CodeFormer(TorchRestorationModel):
    """Blind face restoration with a codebook lookup transformer."""

    info = ModelInfo(
        name="codeformer",
        display_name="CodeFormer (face)",
        task=TaskType.FACE_RESTORATION.value,
        kind=ModelKind.NEURAL.value,
        version="0.1.0",
        description=(
            "Detects faces, warps each into the canonical FFHQ frame, "
            "reconstructs it from a learned codebook and blends it back."
        ),
        method=_FACE_WARNING
        + " The fidelity weight controls the trade-off: 0 produces the "
        "highest-quality face and the least resemblance to the input; 1 stays "
        "closest to the measured pixels and restores least.",
        license_name=(
            "S-Lab License 1.0 - NON-COMMERCIAL research use only (CodeFormer); "
            "MIT (YuNet detector, OpenCV Zoo)"
        ),
        repository="https://github.com/sczhou/CodeFormer",
        paper=(
            "Zhou, Chan, Li & Loy, 'Towards Robust Blind Face Restoration with "
            "Codebook Lookup Transformer', NeurIPS 2022"
        ),
        authors="Shangchen Zhou, Kelvin C.K. Chan, Chongyi Li, Chen Change Loy",
        weights=(
            WeightSpec(
                filename="codeformer.pth",
                url=(
                    "https://github.com/sczhou/CodeFormer/releases/download/"
                    "v0.1.0/codeformer.pth"
                ),
                size_bytes=376_637_898,
                license_name="S-Lab License 1.0 (non-commercial)",
                source="https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0",
            ),
            WeightSpec(
                filename="face_detection_yunet_2023mar.onnx",
                url=(
                    "https://github.com/opencv/opencv_zoo/raw/main/models/"
                    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
                ),
                size_bytes=232_589,
                license_name="MIT (OpenCV Zoo)",
                source=(
                    "https://github.com/opencv/opencv_zoo/tree/main/models/"
                    "face_detection_yunet"
                ),
            ),
        ),
        parameters=(
            ParamSpec(
                name="fidelity",
                label="Fidelity weight",
                kind="float",
                default=0.7,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                help_text=(
                    "0 = highest quality, least faithful to the input. "
                    "1 = closest to the measured pixels, least restored. "
                    "For forensic work prefer high values, and sweep the range "
                    "to see how much of the output is the prior."
                ),
            ),
            ParamSpec(
                name="only_largest",
                label="Largest face only",
                kind="bool",
                default=False,
                help_text="Restore only the largest detected face.",
            ),
            ParamSpec(
                name="min_face_size",
                label="Minimum face size (px)",
                kind="int",
                default=12,
                minimum=6,
                maximum=256,
                step=2,
                help_text=(
                    "Ignore detections smaller than this. The default is "
                    "deliberately low so small faces are reported rather than "
                    "silently skipped; the inter-ocular measurement then says "
                    "how little information such a face actually carries."
                ),
            ),
            ParamSpec(
                name="detection_confidence",
                label="Detection confidence",
                kind="float",
                default=0.6,
                minimum=0.1,
                maximum=0.99,
                step=0.05,
            ),
        ),
        scale=1,
        supports_fp16=False,
        supports_tiling=False,
        requires_packages=("torch",),
        may_synthesise=True,
        notes=_FACE_WARNING,
    )

    def __init__(self, weights_dir=None) -> None:
        super().__init__(weights_dir)
        self._detector = None
        self._aligner = None
        self._last_faces: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ build
    def build_network(self):
        """Instantiate the CodeFormer network."""
        require_torch()
        from restoration.codeformer.arch import CodeFormerNet

        return CodeFormerNet(
            dim_embd=512, n_head=8, n_layers=9,
            codebook_size=1024, latent_size=256,
            connect_list=("32", "64", "128", "256"),
        )

    def primary_weight_spec(self):
        """Return the CodeFormer checkpoint, not the detector model."""
        return self.info.weights[0]

    def _load(self) -> None:
        super()._load()
        from detection.face import FaceAligner, YuNetDetector

        self._detector = YuNetDetector()
        self._detector.load()
        self._aligner = FaceAligner(512)

    def _unload(self) -> None:
        if self._detector is not None:
            self._detector.unload()
        self._detector = None
        self._aligner = None
        super()._unload()

    # ------------------------------------------------------------------ state
    @property
    def last_faces(self) -> List[Dict[str, Any]]:
        """Descriptions of the faces restored on the most recent run."""
        return list(self._last_faces)

    # -------------------------------------------------------------- execution
    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        """Detect, align, restore and blend back every qualifying face."""
        torch = require_torch()

        fidelity = float(params.get("fidelity", 0.7))
        only_largest = bool(params.get("only_largest", False))
        min_face_size = int(params.get("min_face_size", 12))
        confidence = float(params.get("detection_confidence", 0.6))
        cancelled = params.get("cancelled")

        self._last_faces = []
        source_u8 = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        if progress is not None:
            progress(5, "Detecting faces")

        self._detector._score_threshold = confidence  # noqa: SLF001
        self._detector.unload()
        detections = self._detector.detect(source_u8, min_size=min_face_size)

        if not detections:
            raise InferenceError(
                "No face was detected in this image, so face restoration was "
                "not applied. Lower the detection confidence or the minimum "
                "face size, or select a region containing a face and run the "
                "operation on that region."
            )

        if only_largest:
            detections = [max(detections, key=lambda d: d.width * d.height)]

        output = source_u8.copy()
        total = len(detections)

        for index, detection in enumerate(detections):
            if cancelled is not None and cancelled():
                from core.exceptions import OperationCancelled

                raise OperationCancelled("Face restoration cancelled")

            if progress is not None:
                progress(
                    10 + int(index * 85 / total),
                    f"Restoring face {index + 1}/{total}",
                )

            aligned = self._aligner.align(source_u8, detection)
            if aligned is None:
                logger.warning(
                    "Skipping a face: no stable alignment transform could be "
                    "estimated from its landmarks"
                )
                continue

            restored = self._restore_aligned(aligned.image, fidelity)
            output = self._aligner.paste_back(output, restored, aligned)

            record = detection.to_dict()
            record["fidelity"] = fidelity
            record["low_information"] = (
                detection.inter_ocular_distance < _LOW_INFORMATION_IOD
            )
            self._last_faces.append(record)

            if record["low_information"]:
                logger.warning(
                    "Face %d has an inter-ocular distance of only %.0f px. The "
                    "restored face is dominated by the learned prior rather "
                    "than by measured detail and must not be used for "
                    "identification.",
                    index + 1,
                    detection.inter_ocular_distance,
                )

        if progress is not None:
            progress(100, f"Restored {len(self._last_faces)} face(s)")

        logger.info(
            "CodeFormer restored %d face(s) at fidelity %.2f",
            len(self._last_faces), fidelity,
        )
        return (output.astype(np.float32) / 255.0).astype(np.float32)

    def _restore_aligned(self, face: np.ndarray, fidelity: float) -> np.ndarray:
        """Run the network over one canonical-frame face.

        Args:
            face: ``512x512x3`` uint8 RGB aligned face.
            fidelity: Fidelity weight in ``[0, 1]``.

        Returns:
            The restored face as ``512x512x3`` uint8 RGB.
        """
        torch = require_torch()

        tensor = torch.from_numpy(
            np.ascontiguousarray(face.transpose(2, 0, 1)[None])
        ).float().div_(255.0)
        # The network expects [-1, 1].
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(self._torch_device)

        with torch.inference_mode():
            output, _ = self._network(tensor, w=fidelity)

        result = output.detach().float().clamp_(-1.0, 1.0).cpu().numpy()[0]
        result = (result.transpose(1, 2, 0) + 1.0) / 2.0
        return (np.clip(result, 0.0, 1.0) * 255.0).round().astype(np.uint8)


class GFPGAN(NotIntegratedModel):
    """GAN-prior face restoration built on a StyleGAN2 decoder."""

    integration_note = (
        "Not integrated. GFPGAN's decoder is a StyleGAN2 generator whose "
        "reference implementation relies on fused bias-activation and "
        "upfirdn2d CUDA extensions compiled at runtime; the pure-PyTorch "
        "fallbacks are numerically divergent from the ones the weights were "
        "trained against."
    )
    suggested_alternative = "CodeFormer, which is fully integrated"

    info = ModelInfo(
        name="gfpgan",
        display_name="GFPGAN v1.4",
        task=TaskType.FACE_RESTORATION.value,
        kind=ModelKind.NEURAL.value,
        version="1.4",
        description="Face restoration using a generative facial prior.",
        method=_FACE_WARNING,
        license_name="Apache-2.0 (code); weights subject to upstream terms",
        repository="https://github.com/TencentARC/GFPGAN",
        paper=(
            "Wang et al., 'Towards Real-World Blind Face Restoration with "
            "Generative Facial Prior', CVPR 2021"
        ),
        authors="Xintao Wang, Yu Li, Honglun Zhang, Ying Shan",
        weights=(
            WeightSpec(
                filename="GFPGANv1.4.pth",
                url=(
                    "https://github.com/TencentARC/GFPGAN/releases/download/"
                    "v1.3.4/GFPGANv1.4.pth"
                ),
                size_bytes=348_632_874,
                license_name="Apache-2.0",
                source="https://github.com/TencentARC/GFPGAN/releases/tag/v1.3.4",
            ),
        ),
        requires_packages=("torch",),
        may_synthesise=True,
        notes=_FACE_WARNING,
    )


def register_codeformer(replace: bool = False) -> int:
    """Register the face-restoration adapters."""
    count = 0
    for model_class in (CodeFormer, GFPGAN):
        try:
            ModelRegistry.register(model_class.info, model_class, replace=replace)
            count += 1
        except ValueError:
            logger.debug("Model %s already registered", model_class.info.name)
    return count
