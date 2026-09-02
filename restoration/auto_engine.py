"""Automatic pipeline recommendation.

The engine maps degradation indicators onto an ordered pipeline and, crucially,
explains *why* each step was proposed. The recommendation is never executed
automatically: the investigator reviews it, edits it and approves it.

Ordering doctrine
-----------------
Operations are ordered by the physics of how the degradations were introduced,
inverting the capture chain last-in-first-out:

1. **Compression artefacts** first. Blocking and ringing are the most recently
   applied degradation and the most misleading to every later stage - a
   denoiser or deblurrer will happily treat block edges as scene structure.
2. **Noise** next, at native resolution where the noise model still holds.
3. **Blur** after noise, because deconvolution amplifies whatever noise remains.
4. **Exposure and contrast** once the signal is clean.
5. **Haze** before geometry changes, since the transmission estimate assumes
   native sampling.
6. **Super-resolution last**. Running it earlier forces every subsequent model
   to work on interpolated pixels, multiplying cost without adding information.
7. **Face restoration** last of all, and only when explicitly requested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from analysis.analyzer import AnalysisReport
from app.constants import (
    DEGRADATION_ACTION_THRESHOLD,
    DegradationKey,
    ModelKind,
    TaskType,
)
from restoration.base import ModelInfo
from restoration.pipeline import Pipeline, PipelineStep
from restoration.registry import ModelRegistry

logger = logging.getLogger(__name__)

__all__ = ["AutoRestorationEngine", "Recommendation", "RecommendedStep"]


@dataclass
class RecommendedStep:
    """One proposed step together with its justification."""

    step: PipelineStep
    reason: str
    trigger_key: str
    trigger_score: float
    alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "step": self.step.to_dict(),
            "reason": self.reason,
            "trigger": self.trigger_key,
            "trigger_score": round(self.trigger_score, 4),
            "alternatives": list(self.alternatives),
        }


@dataclass
class Recommendation:
    """A proposed pipeline plus the reasoning behind it."""

    pipeline: Pipeline
    steps: List[RecommendedStep] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether no operation was recommended."""
        return not self.pipeline.enabled_steps

    def rationale_text(self) -> str:
        """Return a readable explanation for the pipeline-review dialog."""
        lines: List[str] = []
        if self.is_empty:
            lines.append(
                "No degradation indicator exceeded the action threshold, so no "
                "restoration is proposed. Enhancement that is not indicated by "
                "the evidence adds risk without adding information."
            )
        else:
            for index, item in enumerate(self.steps, start=1):
                lines.append(f"{index}. {item.step.describe()}")
                lines.append(f"     Why: {item.reason}")
                if item.alternatives:
                    lines.append(
                        "     Alternatives: " + ", ".join(item.alternatives)
                    )
                lines.append("")
        if self.skipped:
            lines.append("Indicated but unavailable:")
            for name, reason in self.skipped:
                lines.append(f"  - {name}: {reason}")
            lines.append("")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return "\n".join(lines).strip()

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "pipeline": self.pipeline.to_dict(),
            "steps": [item.to_dict() for item in self.steps],
            "skipped": [{"model": n, "reason": r} for n, r in self.skipped],
            "warnings": list(self.warnings),
        }


class AutoRestorationEngine:
    """Turns an :class:`~analysis.analyzer.AnalysisReport` into a pipeline."""

    #: Task execution order; see the module docstring for the rationale.
    TASK_ORDER: Sequence[str] = (
        TaskType.JPEG_ARTIFACT.value,
        TaskType.DENOISE.value,
        TaskType.DEBLUR.value,
        TaskType.EXPOSURE.value,
        TaskType.CONTRAST.value,
        TaskType.DEHAZE.value,
        TaskType.SUPER_RESOLUTION.value,
        TaskType.FACE_RESTORATION.value,
    )

    #: Preferred model per task, most preferred first. Classical operators are
    #: listed as fallbacks so a machine with no weights installed still gets a
    #: working - and honestly labelled - pipeline.
    PREFERENCES: Dict[str, Sequence[str]] = {
        TaskType.JPEG_ARTIFACT.value: ("fbcnn_color", "swinir_car", "deblock"),
        TaskType.DENOISE.value: (
            "restormer_denoise", "nafnet_denoise", "dncnn_color_blind",
            "swinir_denoise", "nlm",
        ),
        TaskType.DEBLUR.value: (
            "restormer_motion_deblur", "nafnet_deblur",
            "restormer_defocus_deblur", "richardson_lucy", "wiener",
        ),
        TaskType.SUPER_RESOLUTION.value: (
            "realesrgan_x4plus", "swinir_real_sr", "realesrgan_x2plus",
            "realesrgan_anime6b", "lanczos",
        ),
        TaskType.DEHAZE.value: ("dcp_dehaze",),
        TaskType.EXPOSURE.value: ("exposure",),
        TaskType.CONTRAST.value: ("clahe",),
        TaskType.FACE_RESTORATION.value: ("codeformer", "gfpgan"),
    }

    def __init__(
        self,
        threshold: float = DEGRADATION_ACTION_THRESHOLD,
        prefer_classical: bool = False,
    ) -> None:
        """Create an engine.

        Args:
            threshold: Score above which an indicator triggers an operation.
            prefer_classical: Prefer deterministic operators over learned ones
                even when weights are installed. Useful when a result must be
                defensible without reference to a training distribution.
        """
        self._threshold = float(threshold)
        self._prefer_classical = bool(prefer_classical)

    # ------------------------------------------------------------------ public
    def recommend(
        self,
        report: AnalysisReport,
        include_face: bool = False,
        max_scale: int = 4,
    ) -> Recommendation:
        """Build a recommended pipeline from ``report``.

        Args:
            report: The degradation analysis to act on.
            include_face: Add a face-restoration step. Off by default because
                face restoration synthesises facial detail.
            max_scale: Upper bound on the super-resolution factor.

        Returns:
            A :class:`Recommendation`.
        """
        pipeline = Pipeline(name="Recommended pipeline")
        recommendation = Recommendation(pipeline=pipeline)

        for task in self.TASK_ORDER:
            if task == TaskType.FACE_RESTORATION.value and not include_face:
                continue
            proposal = self._propose_for_task(report, task, max_scale)
            if proposal is None:
                continue
            model_name, reason, trigger_key, score, alternatives = proposal

            selected = self._select_model(task)
            if selected is None:
                available_reason = self._unavailable_reason(task)
                recommendation.skipped.append(
                    (task.replace("_", " ").title(), available_reason)
                )
                continue

            parameters = self._parameters_for(selected, report, task, max_scale)
            step = PipelineStep(
                model_name=selected.name, parameters=parameters, note=reason
            )
            pipeline.add(step)
            recommendation.steps.append(
                RecommendedStep(
                    step=step,
                    reason=reason,
                    trigger_key=trigger_key,
                    trigger_score=score,
                    alternatives=alternatives,
                )
            )

        self._add_warnings(recommendation, report)
        pipeline.rationale = recommendation.rationale_text()
        logger.info(
            "Auto engine proposed %d step(s): %s",
            len(pipeline.enabled_steps),
            " -> ".join(s.display_name for s in pipeline.enabled_steps) or "(none)",
        )
        return recommendation

    # ---------------------------------------------------------------- triggers
    def _propose_for_task(
        self, report: AnalysisReport, task: str, max_scale: int
    ) -> Optional[Tuple[str, str, str, float, List[str]]]:
        """Decide whether ``task`` is indicated; return its justification."""
        threshold = self._threshold

        if task == TaskType.JPEG_ARTIFACT.value:
            score = report.score(DegradationKey.JPEG.value)
            if score < threshold:
                return None
            metric = report.get(DegradationKey.JPEG.value)
            quality = (metric.measurements.get("container_jpeg_quality")
                       if metric else None)
            detail = (
                f" The container's quantisation tables indicate an encode "
                f"quality of about {quality:.0f}/100."
                if quality is not None else
                " Blocking and ringing were measured in the pixel domain."
            )
            return (
                task,
                f"JPEG artefact indicator is {int(score * 100)}/100.{detail} "
                "Artefact removal runs first so later stages do not treat block "
                "edges as scene structure.",
                DegradationKey.JPEG.value, score, [],
            )

        if task == TaskType.DENOISE.value:
            score = report.score(DegradationKey.NOISE.value)
            if score < threshold:
                return None
            metric = report.get(DegradationKey.NOISE.value)
            sigma = metric.measurements.get("sigma_luma_8bit_equivalent", 0.0) if metric else 0.0
            return (
                task,
                f"Noise indicator is {int(score * 100)}/100 (estimated luminance "
                f"sigma about {sigma:.1f}/255). Denoising runs at native "
                "resolution, before any geometry change.",
                DegradationKey.NOISE.value, score, [],
            )

        if task == TaskType.DEBLUR.value:
            blur = report.score(DegradationKey.BLUR.value)
            motion = report.score(DegradationKey.MOTION_BLUR.value)
            score = max(blur, motion)
            if score < threshold:
                return None
            if motion >= threshold and motion >= blur * 0.8:
                metric = report.get(DegradationKey.MOTION_BLUR.value)
                angle = metric.measurements.get("estimated_angle_deg", 0.0) if metric else 0.0
                reason = (
                    f"Motion-blur indicator is {int(motion * 100)}/100 with a "
                    f"directional signature near {angle:.0f} degrees."
                )
            else:
                reason = f"Blur indicator is {int(blur * 100)}/100."
            return (
                task,
                reason + " Deblurring follows denoising because deconvolution "
                "and learned deblurring both amplify residual noise.",
                DegradationKey.BLUR.value, score, [],
            )

        if task == TaskType.EXPOSURE.value:
            under = report.score(DegradationKey.UNDEREXPOSURE.value)
            over = report.score(DegradationKey.OVEREXPOSURE.value)
            score = max(under, over)
            if score < threshold:
                return None
            which = "Under" if under >= over else "Over"
            key = (DegradationKey.UNDEREXPOSURE.value if under >= over
                   else DegradationKey.OVEREXPOSURE.value)
            return (
                task,
                f"{which}exposure indicator is {int(score * 100)}/100. Tone "
                "correction is a monotonic per-pixel mapping; it cannot restore "
                "samples that were clipped at capture.",
                key, score, [],
            )

        if task == TaskType.CONTRAST.value:
            score = report.score(DegradationKey.LOW_CONTRAST.value)
            if score < threshold:
                return None
            return (
                task,
                f"Low-contrast indicator is {int(score * 100)}/100. Adaptive "
                "equalisation raises local contrast, and raises noise with it.",
                DegradationKey.LOW_CONTRAST.value, score, [],
            )

        if task == TaskType.DEHAZE.value:
            score = report.score(DegradationKey.HAZE.value)
            if score < threshold:
                return None
            return (
                task,
                f"Haze indicator is {int(score * 100)}/100. Dehazing runs before "
                "any resampling because the transmission estimate assumes the "
                "native sampling grid.",
                DegradationKey.HAZE.value, score, [],
            )

        if task == TaskType.SUPER_RESOLUTION.value:
            score = report.score(DegradationKey.LOW_RESOLUTION.value)
            if score < threshold:
                return None
            metric = report.get(DegradationKey.LOW_RESOLUTION.value)
            if metric and metric.measurements.get("appears_upscaled"):
                # Refuse to stack interpolation on interpolation.
                return None
            long_edge = metric.measurements.get("long_edge", 0) if metric else 0
            scale = self._choose_scale(int(long_edge), max_scale)
            return (
                task,
                f"Resolution indicator is {int(score * 100)}/100 (long edge "
                f"{long_edge} px). A x{scale} enlargement is proposed, placed "
                "last so earlier stages work on native pixels.",
                DegradationKey.LOW_RESOLUTION.value, score,
                ["Lanczos Upscale for a non-generative baseline"],
            )

        if task == TaskType.FACE_RESTORATION.value:
            return (
                task,
                "Face restoration was explicitly requested. It reconstructs "
                "facial detail from a learned prior and must not be treated as "
                "recovery of the actual face.",
                DegradationKey.BLUR.value, 1.0, [],
            )

        return None

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _choose_scale(long_edge: int, max_scale: int) -> int:
        """Pick a scale factor that lands near a useful working resolution."""
        if long_edge <= 0:
            return min(2, max_scale)
        for candidate in (4, 3, 2):
            if candidate > max_scale:
                continue
            if long_edge * candidate <= 4096:
                return candidate
        return min(2, max_scale)

    def _candidates(self, task: str) -> List[ModelInfo]:
        """Return registered models for ``task`` in preference order."""
        preferred = list(self.PREFERENCES.get(task, ()))
        infos = {info.name: info for info in ModelRegistry.by_task(task)}
        ordered = [infos[name] for name in preferred if name in infos]
        ordered.extend(info for name, info in infos.items() if name not in preferred)
        if self._prefer_classical:
            ordered.sort(key=lambda i: 0 if i.kind == ModelKind.CLASSICAL.value else 1)
        return ordered

    def _select_model(self, task: str) -> Optional[ModelInfo]:
        """Return the first available model for ``task``."""
        for info in self._candidates(task):
            model = ModelRegistry.try_get(info.name)
            if model is not None and model.availability().ok:
                return info
        return None

    def _unavailable_reason(self, task: str) -> str:
        """Explain why no model could be selected for ``task``."""
        candidates = self._candidates(task)
        if not candidates:
            return f"No model is registered for this task."
        reasons = []
        for info in candidates[:3]:
            model = ModelRegistry.try_get(info.name)
            if model is None:
                continue
            reasons.append(f"{info.display_name}: {model.availability().reason}")
        return " | ".join(reasons) or "No usable model."

    def _parameters_for(
        self, info: ModelInfo, report: AnalysisReport, task: str, max_scale: int
    ) -> Dict[str, object]:
        """Choose sensible parameters for ``info`` given the analysis."""
        params: Dict[str, object] = dict(info.default_parameters())

        if task == TaskType.SUPER_RESOLUTION.value:
            metric = report.get(DegradationKey.LOW_RESOLUTION.value)
            long_edge = int(metric.measurements.get("long_edge", 0)) if metric else 0
            scale = self._choose_scale(long_edge, max_scale)
            if info.parameter("scale") is not None:
                allowed = [c[0] for c in (info.parameter("scale").choices or ())]
                params["scale"] = scale if scale in allowed else (
                    max(allowed) if allowed else info.scale
                )

        elif task == TaskType.DEBLUR.value and info.kind == ModelKind.CLASSICAL.value:
            motion = report.get(DegradationKey.MOTION_BLUR.value)
            blur = report.get(DegradationKey.BLUR.value)
            if motion is not None and motion.score >= self._threshold:
                params["psf_type"] = "motion"
                params["angle"] = float(motion.measurements.get("estimated_angle_deg", 0.0))
                params["radius"] = 2.0
            else:
                params["psf_type"] = "gaussian"
                severity = blur.score if blur else 0.5
                # Map severity onto a plausible sigma; the examiner can adjust.
                params["radius"] = round(1.0 + 3.0 * severity, 2)

        elif task == TaskType.DENOISE.value and info.name == "nlm":
            metric = report.get(DegradationKey.NOISE.value)
            sigma = metric.measurements.get("sigma_luma_8bit_equivalent", 6.0) if metric else 6.0
            params["strength"] = round(float(min(25.0, max(2.0, sigma * 0.9))), 1)
            params["chroma_strength"] = round(float(min(25.0, max(2.0, sigma * 1.1))), 1)

        elif task == TaskType.EXPOSURE.value:
            under = report.score(DegradationKey.UNDEREXPOSURE.value)
            over = report.score(DegradationKey.OVEREXPOSURE.value)
            if under >= over:
                params["gamma"] = round(max(0.35, 1.0 - 0.6 * under), 2)
            else:
                params["gamma"] = round(min(2.2, 1.0 + 1.0 * over), 2)

        return params

    def _add_warnings(
        self, recommendation: Recommendation, report: AnalysisReport
    ) -> None:
        """Attach cross-cutting cautions to ``recommendation``."""
        pipeline = recommendation.pipeline

        if pipeline.may_synthesise:
            recommendation.warnings.append(
                "This pipeline includes at least one generative step. The "
                "output may contain structures that are not present in the "
                "source evidence."
            )

        resolution = report.get(DegradationKey.LOW_RESOLUTION.value)
        if resolution is not None and resolution.measurements.get("appears_upscaled"):
            recommendation.warnings.append(
                "The source appears to have been interpolated up from a smaller "
                "original, so super-resolution was not proposed: it would only "
                "re-interpolate existing samples."
            )

        over = report.get(DegradationKey.OVEREXPOSURE.value)
        if over is not None:
            clipped = over.measurements.get("highlight_clipped_fraction", 0.0)
            if clipped > 0.05:
                recommendation.warnings.append(
                    f"{clipped * 100:.1f}% of pixels are clipped at the white "
                    "point. No operation can recover those samples; any detail "
                    "appearing there in the output is synthesised."
                )

        under = report.get(DegradationKey.UNDEREXPOSURE.value)
        if under is not None:
            crushed = under.measurements.get("shadow_clipped_fraction", 0.0)
            if crushed > 0.10:
                recommendation.warnings.append(
                    f"{crushed * 100:.1f}% of pixels are crushed at the black "
                    "point and carry no recoverable detail."
                )

        if len(pipeline.enabled_steps) >= 4:
            recommendation.warnings.append(
                "Long pipelines compound each stage's assumptions. Consider "
                "running and reviewing the steps individually."
            )
