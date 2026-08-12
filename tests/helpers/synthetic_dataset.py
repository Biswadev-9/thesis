"""Generate a miniature dataset with the same structure as the real one.

The real dataset is a multi-gigabyte Kaggle download, so tests and smoke runs build a
tiny stand-in instead. It reproduces the three structural properties the pipeline cares
about:

- the ``Training``/``Testing`` folder layout with four class folders;
- MRI-like content, a bright blob on a dark background, so Otsu thresholding and the
  background crop have something real to find;
- **deliberate cross-split duplicate files**, which is what makes the deduplication and
  leakage checks meaningful. Without them a passing test would prove nothing.
"""

import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from src.data.components.split_builder import CLASS_MAP


def _synthetic_slice(rng: np.random.Generator, size: int, brightness: float) -> np.ndarray:
    """Draw one MRI-like grayscale slice.

    :param rng: Seeded generator.
    :param size: Square edge length in pixels.
    :param brightness: Peak intensity of the bright region, in ``[0, 1]``.
    :return: ``uint8`` array of shape ``(size, size)``.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    centre = size / 2.0

    # Head: a filled ellipse occupying most of the frame.
    head = (((xx - centre) / (size * 0.42)) ** 2 + ((yy - centre) / (size * 0.46)) ** 2) <= 1.0

    # Lesion: a smaller off-centre blob whose position varies per image.
    lesion_x = centre + rng.uniform(-0.18, 0.18) * size
    lesion_y = centre + rng.uniform(-0.18, 0.18) * size
    radius = size * rng.uniform(0.08, 0.14)
    lesion = ((xx - lesion_x) ** 2 + (yy - lesion_y) ** 2) <= radius**2

    image = np.zeros((size, size), dtype=np.float32)
    image[head] = 0.45 + rng.normal(0, 0.03, size=head.sum())
    image[lesion & head] = brightness
    image += rng.normal(0, 0.01, size=image.shape)

    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def make_synthetic_dataset(
    root: Path,
    per_class_train: int = 14,
    per_class_test: int = 6,
    size: int = 48,
    duplicates_per_class: int = 2,
    seed: int = 0,
    overwrite: bool = True,
) -> Dict[str, int]:
    """Write a synthetic dataset tree to disk.

    :param root: Directory to create; receives ``Training/`` and ``Testing/``.
    :param per_class_train: Unique images per class under ``Training``.
    :param per_class_test: Unique images per class under ``Testing``.
    :param size: Square edge length in pixels.
    :param duplicates_per_class: Images copied byte-for-byte from ``Training`` into
        ``Testing``, creating the cross-split duplicates the dedup logic must catch.
    :param seed: Generator seed.
    :param overwrite: Delete ``root`` first if it exists.
    :return: Counts describing what was written.
    """
    root = Path(root)
    if overwrite and root.exists():
        shutil.rmtree(root)

    rng = np.random.default_rng(seed)
    # Distinct brightness per class gives the classes a learnable signal, so a smoke
    # training run produces a sane loss curve rather than pure noise.
    brightness = {"Glioma": 0.95, "Meningioma": 0.80, "Pituitary": 0.65, "No-tumor": 0.50}

    written = 0
    duplicated = 0

    for class_name in CLASS_MAP:
        train_dir = root / "Training" / class_name
        test_dir = root / "Testing" / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        train_paths: List[Path] = []
        for index in range(per_class_train):
            path = train_dir / f"{class_name.lower()}_train_{index:03d}.png"
            Image.fromarray(_synthetic_slice(rng, size, brightness[class_name])).save(path)
            train_paths.append(path)
            written += 1

        for index in range(per_class_test):
            path = test_dir / f"{class_name.lower()}_test_{index:03d}.png"
            Image.fromarray(_synthetic_slice(rng, size, brightness[class_name])).save(path)
            written += 1

        # Byte-identical copies straddling the vendor split boundary.
        for index in range(min(duplicates_per_class, len(train_paths))):
            shutil.copyfile(train_paths[index], test_dir / f"{class_name.lower()}_dup_{index}.png")
            written += 1
            duplicated += 1

    return {
        "root": str(root),
        "files_written": written,
        "duplicate_files": duplicated,
        "unique_images": written - duplicated,
        "classes": len(CLASS_MAP),
    }
