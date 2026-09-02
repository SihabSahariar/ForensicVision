"""Adapters for models that are declared but not executable.

The specification is explicit (S14, S44): a model that cannot be integrated
must be *reported*, not hidden and not faked. These adapters therefore appear
in the Model Manager with their real licence, repository and weight
information, an accurate status, and a plain explanation of what is missing.
Calling :meth:`process` raises instead of returning a substitute result.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from app.constants import ModelStatus
from core.exceptions import ModelNotAvailableError
from restoration.base import (
    Availability,
    ModelInfo,
    ProgressReporter,
    RestorationModel,
)

logger = logging.getLogger(__name__)

__all__ = ["NotIntegratedModel"]


class NotIntegratedModel(RestorationModel):
    """A model whose adapter exists but whose implementation does not.

    Subclasses set :attr:`info` and :attr:`integration_note`. The note is what
    the investigator reads in the Model Manager, so it must say precisely what
    is missing and what would be required to complete the integration.
    """

    #: Plain explanation of the blocker.
    integration_note: str = "This model is not integrated in this build."

    #: Optional suggestion of a model that *is* available for the same task.
    suggested_alternative: str = ""

    def availability(self) -> Availability:
        """Always report ``NOT_INTEGRATED`` with the specific reason."""
        reason = self.integration_note
        if self.suggested_alternative:
            reason = f"{reason} Available alternative: {self.suggested_alternative}."
        return Availability(status=ModelStatus.NOT_INTEGRATED.value, reason=reason)

    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        """Always raise - this model produces no output of any kind."""
        raise ModelNotAvailableError(
            f"{self.info.display_name} is not integrated. {self.integration_note}"
        )
