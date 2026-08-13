"""Controlled image degradations for robustness testing (Step 18).

Step 18 names the degradations to test:

    "Add controlled Gaussian noise to test images and measure performance degradation.
     Test contrast shifts, mild blur, resolution changes, and intensity normalization
     changes."

Each is a module-level callable class rather than a closure, so it survives pickling into
dataloader workers and into a Hydra config.

Two properties matter for the results to mean anything:

**Degradations are applied to the raw image, before the Step 5 pipeline.** Adding noise
after normalisation would change the noise's effective magnitude per image, so severities
would not be comparable across the dataset.

**They are deterministic given a seed.** Gaussian noise is the only stochastic
degradation; it takes a per-image seed derived from the image index, so every model in the
comparison sees the *same* corrupted pixels. Without that, a model could look more robust
purely by drawing an easier noise sample.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


class Degradation:
    """Base class for a named, severity-parameterised image degradation."""

    #: Short name used in result tables.
    name: str = "degradation"

    def __call__(self, image: Image.Image, index: int = 0) -> Image.Image:
        """Apply the degradation.

        :param image: Source image.
        :param index: Sample index, used to derive a deterministic per-image seed.
        :return: The degraded image.
        :raises NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError


class GaussianNoise(Degradation):
    """Additive Gaussian noise, the specification's first-named degradation.

    :param sigma: Noise standard deviation in 0-255 intensity units.
    :param seed: Base seed; combined with the sample index so each image is corrupted
        identically across models but differently from its neighbours.
    """

    name = "gaussian_noise"

    def __init__(self, sigma: float = 25.0, seed: int = 42) -> None:
        self.sigma = sigma
        self.seed = seed

    def __call__(self, image: Image.Image, index: int = 0) -> Image.Image:
        """:param image: Source image.
        :param index: Sample index.
        :return: Noisy image.
        """
        rng = np.random.default_rng(self.seed + index)
        array = np.asarray(image).astype(np.float32)
        noisy = array + rng.normal(0.0, self.sigma, array.shape)
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))

    def __repr__(self) -> str:
        return f"GaussianNoise(sigma={self.sigma})"


class ContrastShift(Degradation):
    """Scale intensities about their mean.

    :param factor: Below 1 reduces contrast, above 1 increases it.
    """

    name = "contrast_shift"

    def __init__(self, factor: float = 0.5) -> None:
        self.factor = factor

    def __call__(self, image: Image.Image, index: int = 0) -> Image.Image:
        """:param image: Source image.
        :param index: Unused; this degradation is deterministic.
        :return: Contrast-shifted image.
        """
        array = np.asarray(image).astype(np.float32)
        mean = array.mean()
        shifted = (array - mean) * self.factor + mean
        return Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))

    def __repr__(self) -> str:
        return f"ContrastShift(factor={self.factor})"


class Blur(Degradation):
    """Gaussian blur, simulating loss of acquisition sharpness.

    :param radius: Blur radius in pixels.
    """

    name = "blur"

    def __init__(self, radius: float = 2.0) -> None:
        self.radius = radius

    def __call__(self, image: Image.Image, index: int = 0) -> Image.Image:
        """:param image: Source image.
        :param index: Unused; this degradation is deterministic.
        :return: Blurred image.
        """
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))

    def __repr__(self) -> str:
        return f"Blur(radius={self.radius})"


class ResolutionLoss(Degradation):
    """Downsample then upsample, simulating a lower-resolution acquisition.

    The round trip is what matters: the image returns to its original size with detail
    permanently lost, so the model still receives its expected input shape.

    :param scale: Fraction of the original edge to downsample to.
    """

    name = "resolution"

    def __init__(self, scale: float = 0.25) -> None:
        self.scale = scale

    def __call__(self, image: Image.Image, index: int = 0) -> Image.Image:
        """:param image: Source image.
        :param index: Unused; this degradation is deterministic.
        :return: Resolution-degraded image at the original size.
        """
        small = (max(8, int(image.width * self.scale)), max(8, int(image.height * self.scale)))
        return image.resize(small, Image.BILINEAR).resize(image.size, Image.BILINEAR)

    def __repr__(self) -> str:
        return f"ResolutionLoss(scale={self.scale})"


class IntensityShift(Degradation):
    """Add a constant offset, simulating a scanner calibration difference.

    This is the "intensity normalization changes" case: it leaves structure untouched and
    moves only the absolute intensity level, which is exactly what per-image
    normalisation should absorb and fixed ImageNet normalisation should not.

    :param offset: Offset in 0-255 intensity units; may be negative.
    """

    name = "intensity_shift"

    def __init__(self, offset: float = 40.0) -> None:
        self.offset = offset

    def __call__(self, image: Image.Image, index: int = 0) -> Image.Image:
        """:param image: Source image.
        :param index: Unused; this degradation is deterministic.
        :return: Intensity-shifted image.
        """
        array = np.asarray(image).astype(np.float32) + self.offset
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

    def __repr__(self) -> str:
        return f"IntensityShift(offset={self.offset})"


#: The Step 18 sweep: each degradation at three increasing severities, plus a clean
#: reference. Severity labels are what appear in the result tables.
DEGRADATION_SWEEP: Dict[str, Tuple[Tuple[str, Optional[Degradation]], ...]] = {
    "clean": (("none", None),),
    "gaussian_noise": (
        ("sigma=10", GaussianNoise(10.0)),
        ("sigma=25", GaussianNoise(25.0)),
        ("sigma=40", GaussianNoise(40.0)),
    ),
    "contrast_shift": (
        ("0.7x", ContrastShift(0.7)),
        ("0.5x", ContrastShift(0.5)),
        ("0.3x", ContrastShift(0.3)),
    ),
    "blur": (
        ("radius=1", Blur(1.0)),
        ("radius=2", Blur(2.0)),
        ("radius=3", Blur(3.0)),
    ),
    "resolution": (
        ("0.5x", ResolutionLoss(0.5)),
        ("0.25x", ResolutionLoss(0.25)),
        ("0.125x", ResolutionLoss(0.125)),
    ),
    "intensity_shift": (
        ("+20", IntensityShift(20.0)),
        ("+40", IntensityShift(40.0)),
        ("-30", IntensityShift(-30.0)),
    ),
}


def build_degradation(category: str, severity: str) -> Optional[Degradation]:
    """Look up one degradation from the sweep.

    :param category: Key of :data:`DEGRADATION_SWEEP`.
    :param severity: Severity label within that category.
    :return: The degradation, or ``None`` for the clean condition.
    :raises ValueError: If the category or severity is unknown.
    """
    if category not in DEGRADATION_SWEEP:
        raise ValueError(
            f"Unknown degradation category {category!r}. Available: {sorted(DEGRADATION_SWEEP)}"
        )

    for label, degradation in DEGRADATION_SWEEP[category]:
        if label == severity:
            return degradation

    available = [label for label, _ in DEGRADATION_SWEEP[category]]
    raise ValueError(f"Unknown severity {severity!r} for {category!r}. Available: {available}")
