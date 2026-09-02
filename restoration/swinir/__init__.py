"""SwinIR adapters.

The architecture lives in :mod:`restoration.swinir.arch` and the adapters in
:mod:`restoration.swinir.model`.
"""

from restoration.swinir.model import register_swinir

__all__ = ["register_swinir"]
