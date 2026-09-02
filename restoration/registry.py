"""Model registry.

Models register a factory (not an instance) so that importing the registry does
not pull PyTorch into the process. Instances are created lazily on first use and
cached, which keeps application start-up fast on machines without a GPU.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Iterable, List, Optional

from app.constants import ModelKind, TaskType
from core.exceptions import ModelNotAvailableError
from restoration.base import ModelInfo, RestorationModel

logger = logging.getLogger(__name__)

__all__ = ["ModelRegistry", "register", "get", "available_models"]

ModelFactory = Callable[[], RestorationModel]


class _Registry:
    """Thread-safe registry mapping names to model factories."""

    def __init__(self) -> None:
        self._factories: Dict[str, ModelFactory] = {}
        self._infos: Dict[str, ModelInfo] = {}
        self._instances: Dict[str, RestorationModel] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------ registration
    def register(
        self, info: ModelInfo, factory: ModelFactory, replace: bool = False
    ) -> None:
        """Register a model factory under ``info.name``.

        Args:
            info: Static model description, used for listings without
                instantiating the model.
            factory: Zero-argument callable returning a
                :class:`~restoration.base.RestorationModel`.
            replace: Permit replacing an existing registration.

        Raises:
            ValueError: The name is already registered and ``replace`` is False.
        """
        with self._lock:
            if info.name in self._factories and not replace:
                raise ValueError(f"Model '{info.name}' is already registered")
            self._factories[info.name] = factory
            self._infos[info.name] = info
            self._instances.pop(info.name, None)
            logger.debug("Registered model %s (%s)", info.name, info.task)

    def unregister(self, name: str) -> None:
        """Remove a registration and drop any cached instance."""
        with self._lock:
            self._factories.pop(name, None)
            self._infos.pop(name, None)
            instance = self._instances.pop(name, None)
        if instance is not None:
            try:
                instance.unload()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Error unloading %s", name)

    # -------------------------------------------------------------- retrieval
    def get(self, name: str) -> RestorationModel:
        """Return the cached instance for ``name``, creating it if needed.

        Raises:
            ModelNotAvailableError: No model is registered under ``name``.
        """
        with self._lock:
            instance = self._instances.get(name)
            if instance is not None:
                return instance
            factory = self._factories.get(name)
            if factory is None:
                raise ModelNotAvailableError(f"No model registered as '{name}'")
        # Instantiate outside the lock: a factory may import torch, which is
        # slow, and must not block listings on other threads.
        instance = factory()
        with self._lock:
            self._instances.setdefault(name, instance)
            return self._instances[name]

    def try_get(self, name: str) -> Optional[RestorationModel]:
        """Return the model named ``name``, or ``None`` when absent."""
        try:
            return self.get(name)
        except ModelNotAvailableError:
            return None

    def info(self, name: str) -> Optional[ModelInfo]:
        """Return the static description for ``name`` without instantiating."""
        with self._lock:
            return self._infos.get(name)

    # --------------------------------------------------------------- listings
    def names(self) -> List[str]:
        """Return every registered model name, sorted."""
        with self._lock:
            return sorted(self._factories)

    def infos(self) -> List[ModelInfo]:
        """Return every registered :class:`ModelInfo`, display-name sorted."""
        with self._lock:
            values = list(self._infos.values())
        return sorted(values, key=lambda i: (i.task, i.kind, i.display_name))

    def by_task(
        self, task: str, kind: Optional[str] = None, only_available: bool = False
    ) -> List[ModelInfo]:
        """Return models handling ``task``.

        Args:
            task: A :class:`app.constants.TaskType` value.
            kind: Restrict to ``"neural"`` or ``"classical"``.
            only_available: Exclude models that cannot currently run. This
                instantiates candidates, so it is used for execution paths
                rather than for populating listings.
        """
        results = [i for i in self.infos() if i.task == task]
        if kind is not None:
            results = [i for i in results if i.kind == kind]
        if only_available:
            results = [i for i in results if self._is_available(i.name)]
        return results

    def tasks(self) -> List[str]:
        """Return the distinct tasks that have at least one registered model."""
        seen = []
        for info in self.infos():
            if info.task not in seen:
                seen.append(info.task)
        return seen

    def available(self) -> List[ModelInfo]:
        """Return models that can run right now."""
        return [i for i in self.infos() if self._is_available(i.name)]

    def _is_available(self, name: str) -> bool:
        model = self.try_get(name)
        if model is None:
            return False
        try:
            return model.availability().ok
        except Exception:  # pragma: no cover - defensive
            logger.exception("Availability check failed for %s", name)
            return False

    def status_table(self) -> List[dict]:
        """Return one row per model for the Model Manager table."""
        rows = []
        for info in self.infos():
            model = self.try_get(info.name)
            if model is None:
                continue
            try:
                state = model.availability()
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Availability check failed for %s", info.name)
                from app.constants import ModelStatus
                from restoration.base import Availability

                state = Availability(
                    status=ModelStatus.NOT_INTEGRATED.value, reason=str(exc)
                )
            rows.append(
                {
                    "name": info.name,
                    "display_name": info.display_name,
                    "task": info.task,
                    "task_label": info.task_label,
                    "kind": info.kind,
                    "version": info.version,
                    "license": info.license_name,
                    "repository": info.repository,
                    "status": state.status,
                    "status_label": state.label,
                    "reason": state.reason,
                    "may_synthesise": info.may_synthesise,
                    "weights": list(info.weights),
                    "missing_weights": list(state.missing_weights),
                    "missing_packages": list(state.missing_packages),
                    "info": info,
                    "model": model,
                }
            )
        return rows

    # ------------------------------------------------------------- management
    def unload_all(self) -> None:
        """Unload every instantiated model, freeing device memory."""
        with self._lock:
            instances = list(self._instances.values())
        for instance in instances:
            try:
                instance.unload()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Error unloading %s", instance.name)

    def clear(self) -> None:
        """Drop every registration (used by tests)."""
        self.unload_all()
        with self._lock:
            self._factories.clear()
            self._infos.clear()
            self._instances.clear()


#: Process-wide registry instance.
ModelRegistry = _Registry()


def register(info: ModelInfo, factory: ModelFactory, replace: bool = False) -> None:
    """Module-level shortcut for :meth:`_Registry.register`."""
    ModelRegistry.register(info, factory, replace=replace)


def get(name: str) -> RestorationModel:
    """Module-level shortcut for :meth:`_Registry.get`."""
    return ModelRegistry.get(name)


def available_models(task: Optional[str] = None) -> List[ModelInfo]:
    """Return currently usable models, optionally filtered by task."""
    models = ModelRegistry.available()
    if task is not None:
        models = [m for m in models if m.task == task]
    return models
