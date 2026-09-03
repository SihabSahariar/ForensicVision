"""Restoration engine.

Deliberately independent of the GUI so the same engine can be driven from a
CLI, a batch runner or future video tooling.

Two families of operations are registered, and the distinction is surfaced
everywhere in the UI, the provenance records and the reports:

* **Classical** operators are deterministic signal processing. They always
  work, need no downloads, and cannot introduce structures absent from the
  measured samples.
* **Neural** models are trained networks. They usually look better and may
  synthesise detail from their training distribution rather than recovering it
  from the evidence.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["register_all_models", "REGISTRATION_REPORT"]

#: Populated by :func:`register_all_models` with any import failures, so the
#: Model Manager can explain why a family is absent instead of hiding it.
REGISTRATION_REPORT: dict = {}


def register_all_models(replace: bool = False) -> int:
    """Register every available model family with the shared registry.

    Import failures are recorded rather than raised: a broken optional
    dependency must never prevent the application from starting (S42).

    Args:
        replace: Replace existing registrations.

    Returns:
        The total number of models registered.
    """
    from restoration.registry import ModelRegistry

    families = (
        ("classical", "restoration.classical", "register_classical_models"),
        ("realesrgan", "restoration.realesrgan", "register_realesrgan"),
        ("restormer", "restoration.restormer", "register_restormer"),
        ("nafnet", "restoration.nafnet", "register_nafnet"),
        ("dncnn", "restoration.dncnn", "register_dncnn"),
        ("fbcnn", "restoration.fbcnn", "register_fbcnn"),
        ("swinir", "restoration.swinir", "register_swinir"),
        ("zerodce", "restoration.zerodce", "register_zerodce"),
        ("codeformer", "restoration.codeformer", "register_codeformer"),
        ("lama", "restoration.lama", "register_lama"),
    )

    total = 0
    REGISTRATION_REPORT.clear()
    for name, module_path, function_name in families:
        try:
            module = __import__(module_path, fromlist=[function_name])
            registrar = getattr(module, function_name)
            count = registrar(replace=replace)
            total += count
            REGISTRATION_REPORT[name] = {"status": "ok", "count": count}
        except Exception as exc:
            REGISTRATION_REPORT[name] = {"status": "error", "error": str(exc)}
            logger.warning("Could not register model family '%s': %s", name, exc)

    logger.info(
        "Model registry ready: %d model(s) across %d family(ies)",
        total,
        len(ModelRegistry.tasks()),
    )
    return total
