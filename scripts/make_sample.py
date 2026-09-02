"""Generate synthetic test evidence for development and integration tests.

Produces a small scene containing high-contrast text, a plate-like element,
smooth gradients and fine texture, then writes degraded variants (blur, noise,
JPEG, downscale, dark, hazy) so every analyzer and restoration path can be
exercised without shipping real case material.

Usage::

    python scripts/make_sample.py --out samples/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_scene(width: int = 960, height: int = 640, seed: int = 7) -> np.ndarray:
    """Render a deterministic synthetic scene as an RGB ``uint8`` array."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((height, width, 3), dtype=np.float32)

    # Sky-to-ground vertical gradient.
    gradient = np.linspace(0.72, 0.18, height, dtype=np.float32)[:, None]
    canvas[..., 0] = gradient * 0.86
    canvas[..., 1] = gradient * 0.90
    canvas[..., 2] = gradient * 1.00

    # Building blocks with distinct tones.
    blocks = [
        ((40, 240), (250, 560), (0.35, 0.33, 0.31)),
        ((300, 180), (520, 560), (0.52, 0.48, 0.44)),
        ((560, 300), (900, 560), (0.28, 0.30, 0.34)),
    ]
    for (x0, y0), (x1, y1), colour in blocks:
        canvas[y0:y1, x0:x1] = colour

    # Window grid: fine repetitive detail, sensitive to blur and resolution.
    for bx in range(320, 500, 30):
        for by in range(200, 540, 34):
            canvas[by : by + 18, bx : bx + 18] = (0.80, 0.78, 0.62)

    # A vehicle body with a light plate area.
    cv2.rectangle(canvas, (120, 400), (430, 520), (0.16, 0.20, 0.36), -1)
    cv2.rectangle(canvas, (150, 360), (390, 410), (0.20, 0.24, 0.40), -1)
    cv2.circle(canvas, (185, 522), 26, (0.08, 0.08, 0.09), -1)
    cv2.circle(canvas, (370, 522), 26, (0.08, 0.08, 0.09), -1)

    # Plate: the canonical fine-detail target.
    cv2.rectangle(canvas, (215, 452), (345, 495), (0.94, 0.94, 0.90), -1)
    cv2.rectangle(canvas, (215, 452), (345, 495), (0.10, 0.10, 0.10), 2)
    cv2.putText(
        canvas, "FV-1234", (222, 484),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0.05, 0.05, 0.05), 2, cv2.LINE_AA,
    )

    # A signboard with smaller text.
    cv2.rectangle(canvas, (600, 340), (880, 400), (0.92, 0.90, 0.84), -1)
    cv2.putText(
        canvas, "EVIDENCE 07", (612, 382),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0.12, 0.14, 0.30), 2, cv2.LINE_AA,
    )

    # Fine stochastic texture so denoisers have something to preserve.
    texture = rng.normal(0.0, 0.012, (height, width, 1)).astype(np.float32)
    canvas = canvas + texture

    # Subtle vignette.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 +
                     ((yy - height / 2) / (height / 2)) ** 2)
    canvas *= (1.0 - 0.18 * np.clip(radius - 0.55, 0.0, None))[..., None]

    return (np.clip(canvas, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def degrade(scene: np.ndarray, kind: str) -> np.ndarray:
    """Return a degraded copy of ``scene`` for the named degradation."""
    rng = np.random.default_rng(11)
    if kind == "pristine":
        return scene
    if kind == "blur":
        return cv2.GaussianBlur(scene, (9, 9), 3.0)
    if kind == "motion":
        length = 17
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        return cv2.filter2D(scene, -1, kernel)
    if kind == "noisy":
        noisy = scene.astype(np.float32) + rng.normal(0.0, 18.0, scene.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "lowres":
        small = cv2.resize(scene, (scene.shape[1] // 4, scene.shape[0] // 4),
                           interpolation=cv2.INTER_AREA)
        return small
    if kind == "dark":
        return np.clip(scene.astype(np.float32) * 0.22, 0, 255).astype(np.uint8)
    if kind == "bright":
        return np.clip(scene.astype(np.float32) * 2.1 + 40, 0, 255).astype(np.uint8)
    if kind == "hazy":
        depth = np.linspace(0.35, 0.85, scene.shape[0], dtype=np.float32)[:, None, None]
        airlight = np.array([230.0, 234.0, 240.0], dtype=np.float32)
        return np.clip(
            scene.astype(np.float32) * (1.0 - depth) + airlight * depth, 0, 255
        ).astype(np.uint8)
    if kind == "cctv":
        # The realistic composite: small, soft, noisy and heavily compressed.
        small = cv2.resize(scene, (scene.shape[1] // 3, scene.shape[0] // 3),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 1.4)
        small = np.clip(
            small.astype(np.float32) * 0.55 + rng.normal(0.0, 11.0, small.shape),
            0, 255,
        ).astype(np.uint8)
        return small
    raise ValueError(f"Unknown degradation: {kind}")


#: Variants written by :func:`main`, mapped to output extension and options.
VARIANTS = {
    "pristine": (".png", {}),
    "blur": (".png", {}),
    "motion": (".png", {}),
    "noisy": (".png", {}),
    "lowres": (".png", {}),
    "dark": (".png", {}),
    "bright": (".png", {}),
    "hazy": (".png", {}),
    "cctv": (".jpg", {"quality": 28}),
    "jpeg_low": (".jpg", {"quality": 18, "base": "pristine"}),
}


def main(argv=None) -> int:
    """Write all sample variants to the requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="samples", help="Output directory")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene(args.width, args.height)

    written = []
    for name, (extension, options) in VARIANTS.items():
        base = options.get("base", name)
        image = degrade(scene, base if base in
                        ("pristine", "blur", "motion", "noisy", "lowres",
                         "dark", "bright", "hazy", "cctv") else "pristine")
        path = out_dir / f"sample_{name}{extension}"
        bgr = image[..., ::-1]
        if extension == ".jpg":
            cv2.imwrite(str(path), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, int(options.get("quality", 90))])
        else:
            cv2.imwrite(str(path), bgr)
        written.append(path)

    for path in written:
        print(f"{path}  ({path.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
