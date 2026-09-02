"""Framework-agnostic core: image I/O, device discovery, shared exceptions."""

from core.exceptions import ForensicVisionError
from core.image_io import ImageData, load_image, save_image

__all__ = ["ForensicVisionError", "ImageData", "load_image", "save_image"]
