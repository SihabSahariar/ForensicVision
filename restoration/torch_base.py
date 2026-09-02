"""Shared plumbing for PyTorch-backed restoration models.

Handles the parts every neural adapter would otherwise duplicate: lazy torch
import, device and precision selection, numpy <-> tensor conversion, padding to
an architecture's required size multiple, tiled execution and OOM backoff.

``torch`` is imported only inside methods so that the application still starts,
and every classical operator still works, on an installation with no ML stack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from app.config import get_config
from core.exceptions import (
    DependencyMissingError,
    InferenceError,
    WeightsMissingError,
)
from restoration.base import ProgressReporter, RestorationModel, WeightSpec
from restoration.tiling import pad_to_multiple, tiled_process

logger = logging.getLogger(__name__)

__all__ = ["TorchRestorationModel", "load_state_dict", "require_torch"]


def require_torch():
    """Import and return the ``torch`` module.

    Raises:
        DependencyMissingError: PyTorch is not installed.
    """
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError as exc:  # pragma: no cover - torch is a soft dependency
        raise DependencyMissingError(
            "torch",
            "PyTorch is required for neural restoration models. Install the CPU "
            "build with 'pip install -r requirements.txt' or the CUDA build "
            "with 'pip install -r requirements-gpu.txt'.",
        ) from exc


def load_state_dict(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint and unwrap the common container keys.

    Upstream checkpoints variously store the tensors directly or nest them
    under ``params_ema``, ``params``, ``state_dict`` or ``model``.

    Args:
        path: Checkpoint file.
        map_location: Torch map location.

    Returns:
        The flat ``{name: tensor}`` mapping.

    Raises:
        WeightsMissingError: The file is absent.
        InferenceError: The file cannot be parsed as a checkpoint.
    """
    torch = require_torch()
    if not path.is_file():
        raise WeightsMissingError(path.name, f"Weight file not found: {path}")

    try:
        # weights_only=True refuses to unpickle arbitrary objects. Model weights
        # are downloaded from the internet, so this matters: it prevents a
        # tampered checkpoint from executing code during load.
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - torch < 2.0
        checkpoint = torch.load(path, map_location=map_location)
    except Exception as exc:
        raise InferenceError(f"Could not read checkpoint {path.name}: {exc}") from exc

    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict", "model"):
            inner = checkpoint.get(key)
            if isinstance(inner, dict) and inner:
                checkpoint = inner
                break

    if not isinstance(checkpoint, dict):
        raise InferenceError(f"Checkpoint {path.name} is not a state dictionary")

    # Strip a DataParallel prefix if present.
    if all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[len("module."):]: value for key, value in checkpoint.items()}
    return checkpoint


class TorchRestorationModel(RestorationModel):
    """Base class for models backed by a ``torch.nn.Module``.

    Subclasses implement :meth:`build_network` and may override
    :meth:`preprocess`/:meth:`postprocess` for non-RGB pipelines.
    """

    def __init__(self, weights_dir: Optional[Path] = None) -> None:
        super().__init__(weights_dir)
        self._network = None
        self._torch_device = None

    # ---------------------------------------------------------------- helpers
    @property
    def network(self):
        """The loaded ``torch.nn.Module``, or ``None``."""
        return self._network

    def primary_weight_spec(self) -> Optional[WeightSpec]:
        """Return the first required weight specification."""
        for spec in self.info.weights:
            if spec.required:
                return spec
        return None

    def primary_weight_path(self) -> Path:
        """Return the path of the primary weight file.

        Raises:
            WeightsMissingError: The model declares no required weight file.
        """
        spec = self.primary_weight_spec()
        if spec is None:
            raise WeightsMissingError(self.info.name, "Model declares no weight file")
        return self.weight_path(spec)

    # ------------------------------------------------------------- life cycle
    def build_network(self):
        """Construct the (untrained) network. Subclasses must implement."""
        raise NotImplementedError

    def load_weights(self, network) -> None:
        """Load the checkpoint into ``network``.

        The default loads the primary weight file with ``strict=True`` so a
        shape or naming mismatch fails loudly rather than silently producing a
        partially-initialised network - which would output plausible-looking
        noise, the single worst failure mode for this application.
        """
        state = load_state_dict(self.primary_weight_path())
        missing, unexpected = network.load_state_dict(state, strict=False)
        if missing:
            raise InferenceError(
                f"{self.info.display_name}: checkpoint is missing "
                f"{len(missing)} parameter(s), e.g. {missing[:3]}. The weight "
                "file does not match this architecture."
            )
        if unexpected:
            logger.warning(
                "%s: checkpoint has %d unexpected key(s), e.g. %s",
                self.info.display_name,
                len(unexpected),
                unexpected[:3],
            )

    def _load(self) -> None:
        torch = require_torch()
        from core.device import resolve_torch_device

        config = get_config()
        preference = self._device if self._device != "cpu" else config.device
        self._torch_device = resolve_torch_device(preference, config.cuda_index)
        self._device = str(self._torch_device)

        network = self.build_network()
        self.load_weights(network)
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)

        if self._fp16 and self._torch_device.type == "cuda":
            network = network.half()
        network = network.to(self._torch_device)
        self._network = network

    def _unload(self) -> None:
        self._network = None
        self._torch_device = None
        from core.device import empty_cache

        empty_cache()

    # ----------------------------------------------------------- tensor paths
    def to_tensor(self, image: np.ndarray):
        """Convert an ``HxWx3`` float array to a ``1x3xHxW`` tensor."""
        torch = require_torch()
        array = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        tensor = torch.from_numpy(array)
        if self._fp16:
            tensor = tensor.half()
        return tensor.to(self._torch_device)

    def to_numpy(self, tensor) -> np.ndarray:
        """Convert a ``1x3xHxW`` tensor back to an ``HxWx3`` float array."""
        array = tensor.detach().float().clamp_(0.0, 1.0).cpu().numpy()
        return np.ascontiguousarray(array[0].transpose(1, 2, 0))

    def forward_tile(self, tile: np.ndarray, **params: Any) -> np.ndarray:
        """Run the network over one tile, handling padding and precision.

        Subclasses that need extra network inputs (such as FBCNN's quality
        factor) override :meth:`run_network` rather than this method.
        """
        torch = require_torch()
        padded, (pad_h, pad_w) = pad_to_multiple(tile, self.info.size_multiple)
        with torch.inference_mode():
            tensor = self.to_tensor(padded)
            output = self.run_network(tensor, **params)
            result = self.to_numpy(output)

        scale = self.info.scale
        if pad_h or pad_w:
            height = (padded.shape[0] - pad_h) * scale
            width = (padded.shape[1] - pad_w) * scale
            result = result[:height, :width]
        return result

    def run_network(self, tensor, **params: Any):
        """Apply the network to ``tensor``. Override for custom signatures."""
        if self._network is None:  # pragma: no cover - guarded by process()
            raise InferenceError("Network is not loaded")
        return self._network(tensor)

    # -------------------------------------------------------------- execution
    def _process(
        self,
        image: np.ndarray,
        progress: Optional[ProgressReporter] = None,
        **params: Any,
    ) -> np.ndarray:
        config = get_config()
        tile_size = int(params.pop("tile_size", config.tile_size))
        overlap = int(params.pop("tile_overlap", config.tile_overlap))
        cancelled = params.pop("cancelled", None)

        def _run(tile: np.ndarray) -> np.ndarray:
            return self.forward_tile(tile, **params)

        return tiled_process(
            image,
            _run,
            scale=self.info.scale,
            tile_size=tile_size if self.info.supports_tiling else 0,
            overlap=overlap,
            progress=progress,
            cancelled=cancelled,
            auto_reduce=config.auto_reduce_tile,
            message=self.info.display_name,
        )
