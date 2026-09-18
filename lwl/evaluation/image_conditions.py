"""Test-time image conditions: the real image, three blur-and-downscale degradations, a flat
gray field and per-item noise. Rendered images are cached under a content hash.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


DEGRADATION_SETTINGS = {
    "mild": (0.75, 0.7),
    "medium": (0.50, 1.5),
    "severe": (0.25, 2.5),
}
IMAGE_MODES = ("real", "mild", "medium", "severe", "gray", "noise")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _degrade(image: Image.Image, scale: float, blur_radius: float) -> Image.Image:
    width, height = image.size
    reduced = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        resample=Image.Resampling.LANCZOS,
    )
    restored = reduced.resize((width, height), resample=Image.Resampling.BILINEAR)
    return restored.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def materialize_image(
    image_path: str,
    mode: str,
    cache_dir: Path,
    noise_seed: int = 0,
    condition_key: str | None = None,
) -> str:
    if mode not in IMAGE_MODES:
        raise ValueError(f"unsupported image mode: {mode}")
    if mode == "real":
        return image_path

    source = Path(image_path)
    source_hash = _sha256_file(source)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    identity = condition_key if mode == "noise" and condition_key is not None else source_hash
    digest = hashlib.sha256(f"{mode}:{noise_seed}:{identity}:{width}x{height}".encode("utf-8")).hexdigest()[:16]
    out_path = cache_dir / f"{digest}.png"
    if out_path.exists():
        return str(out_path)

    if mode == "gray":
        rendered = Image.new("RGB", (width, height), (128, 128, 128))
    elif mode == "noise":
        seed = int(hashlib.sha256(f"{noise_seed}:{identity}".encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        rendered = Image.fromarray(pixels, mode="RGB")
    else:
        rendered = _degrade(image, *DEGRADATION_SETTINGS[mode])

    cache_dir.mkdir(parents=True, exist_ok=True)
    rendered.save(out_path, format="PNG", optimize=False, compress_level=6)
    return str(out_path)
