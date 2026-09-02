"""Deterministic classical restoration operators.

These algorithms are the forensic baseline: they run everywhere, need no
downloads, and cannot synthesise image content. Learned models are expected to
outperform them visually, but a result that a classical operator can reproduce
is far easier to defend.
"""

from restoration.classical.models import register_classical_models

__all__ = ["register_classical_models"]
