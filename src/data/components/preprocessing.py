"""Step 6: diffusion-based preprocessing and its comparators.

The specification prescribes anisotropic diffusion as the edge-preserving preprocessing
module, with the update written explicitly as::

    I(t+1) = I(t) + lambda * div( c(||grad I(t)||) * grad I(t) )

and two diffusion coefficients::

    c(s) = exp(-(s / kappa) ** 2)        # option 1, privileges high-contrast edges
    c(s) = 1 / (1 + (s / kappa) ** 2)    # option 2, privileges wide regions

It also requires the method to be compared against no preprocessing, Wiener filtering,
adaptive gamma correction, CLAHE and logarithmic transformation, and selected on
*validation performance and boundary preservation* rather than visual appearance.

Every filter is a module-level callable class rather than a closure, so it survives
pickling into dataloader workers and into a Hydra config.

Two things differ from the reference notebook:

**Single-channel processing.** The notebook replicated grayscale to RGB and then ran
diffusion three times over three identical channels - triple the cost for an identical
result. Filtering happens once on the luminance channel here, then replicates.

**One diffusion implementation.** The notebook defined ``anisotropic_diffusion`` twice
(cells 29 and 31) with the north/south ``np.roll`` directions swapped. Because all four
directional terms are summed and the coefficient depends only on ``|delta|``, the two
versions compute the *same* result - the discrepancy was cosmetic. One implementation
replaces both.
"""

from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.signal import wiener

# Explicit-scheme stability bound. The specification states lambda "should be small,
# commonly not greater than 0.25"; above it the discretised diffusion diverges.
MAX_LAMBDA = 0.25


def _to_gray_array(image: Image.Image) -> np.ndarray:
    """Convert any PIL image to a single-channel float array.

    :param image: Source image.
    :return: ``float32`` array of shape ``(H, W)`` with values in ``[0, 255]``.
    """
    return np.asarray(image.convert("L")).astype(np.float32)


def _to_rgb_image(array: np.ndarray) -> Image.Image:
    """Clip a single-channel array back to a 3-channel PIL image.

    The pipeline feeds ImageNet-pretrained backbones, which expect three channels, so
    the filtered luminance is replicated rather than colourised.

    :param array: Single-channel array in ``[0, 255]``.
    :return: RGB PIL image.
    """
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([clipped] * 3, axis=-1))


def anisotropic_diffusion(
    image: np.ndarray,
    num_iter: int = 10,
    kappa: float = 15.0,
    lam: float = 0.2,
    option: int = 1,
) -> np.ndarray:
    """Perona-Malik anisotropic diffusion on a single-channel image.

    Implements the specification's update directly. The divergence is discretised over
    the four nearest neighbours, each contributing ``c(|delta|) * delta``.

    :param image: Single-channel array.
    :param num_iter: Diffusion iterations. Step 6 sweeps 5, 10, 15 and 20.
    :param kappa: Edge threshold. Smaller values preserve stronger edges but leave more
        noise.
    :param lam: Update step, clamped to :data:`MAX_LAMBDA` for stability.
    :param option: 1 for the exponential coefficient, 2 for the reciprocal.
    :return: Diffused array, same shape and dtype range as the input.
    :raises ValueError: If ``option`` is not 1 or 2, or ``num_iter`` is negative.
    """
    if option not in (1, 2):
        raise ValueError(f"option must be 1 (exponential) or 2 (reciprocal), got {option}")
    if num_iter < 0:
        raise ValueError(f"num_iter must be non-negative, got {num_iter}")

    lam = min(float(lam), MAX_LAMBDA)
    result = image.astype(np.float32, copy=True)

    for _ in range(num_iter):
        # Neighbour differences. np.roll wraps at the borders; on MRI slices the border
        # is background, so the artefact is negligible and this matches the reference.
        deltas = (
            np.roll(result, 1, axis=0) - result,  # north
            np.roll(result, -1, axis=0) - result,  # south
            np.roll(result, -1, axis=1) - result,  # east
            np.roll(result, 1, axis=1) - result,  # west
        )

        update = np.zeros_like(result)
        for delta in deltas:
            ratio = delta / kappa
            coefficient = np.exp(-(ratio**2)) if option == 1 else 1.0 / (1.0 + ratio**2)
            update += coefficient * delta

        result += lam * update

    return result


class Identity:
    """Pass an image through untouched.

    Used for the ``raw`` and ``conventional`` recipes, which differ from each other only
    in the Step 5 intensity treatment applied later in the transform pipeline, not in any
    image-space filtering.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        """:param image: Source image.
        :return: The same image.
        """
        return image

    def __repr__(self) -> str:
        return "Identity()"


class AnisotropicDiffusion:
    """Edge-preserving diffusion filter, the specification's recommended module.

    :param num_iter: Diffusion iterations.
    :param kappa: Edge threshold.
    :param lam: Update step, clamped to :data:`MAX_LAMBDA`.
    :param option: 1 for the exponential coefficient, 2 for the reciprocal.
    """

    def __init__(
        self, num_iter: int = 10, kappa: float = 15.0, lam: float = 0.2, option: int = 1
    ) -> None:
        self.num_iter = num_iter
        self.kappa = kappa
        self.lam = min(float(lam), MAX_LAMBDA)
        self.option = option

    def __call__(self, image: Image.Image) -> Image.Image:
        """:param image: Source image.
        :return: Diffused RGB image.
        """
        diffused = anisotropic_diffusion(
            _to_gray_array(image), self.num_iter, self.kappa, self.lam, self.option
        )
        return _to_rgb_image(diffused)

    def __repr__(self) -> str:
        return (
            f"AnisotropicDiffusion(num_iter={self.num_iter}, kappa={self.kappa}, "
            f"lam={self.lam}, option={self.option})"
        )


class WienerFilter:
    """Adaptive Wiener denoising, one of the Step 6 comparators.

    :param size: Square window edge for the local statistics.
    """

    def __init__(self, size: int = 5) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        """:param image: Source image.
        :return: Filtered RGB image.
        """
        # scipy returns NaN where the local variance is zero, which happens across the
        # black background of an MRI slice.
        filtered = np.nan_to_num(wiener(_to_gray_array(image), mysize=self.size), nan=0.0)
        return _to_rgb_image(filtered)

    def __repr__(self) -> str:
        return f"WienerFilter(size={self.size})"


class AdaptiveGamma:
    """Gamma correction whose exponent adapts to each image's mean brightness.

    Chooses gamma so the mean intensity maps to mid-grey, then clamps it to avoid
    extreme corrections on very dark or very bright slices.

    :param target: Mean intensity to map to, in ``[0, 1]``.
    :param gamma_range: Lower and upper clamp on the exponent.
    """

    def __init__(self, target: float = 0.5, gamma_range: Tuple[float, float] = (0.3, 3.0)) -> None:
        self.target = target
        self.gamma_range = gamma_range

    def __call__(self, image: Image.Image) -> Image.Image:
        """:param image: Source image.
        :return: Gamma-corrected RGB image.
        """
        normalized = _to_gray_array(image) / 255.0
        gamma = np.log(self.target) / np.log(normalized.mean() + 1e-6)
        gamma = float(np.clip(gamma, *self.gamma_range))
        return _to_rgb_image(np.power(normalized, gamma) * 255.0)

    def __repr__(self) -> str:
        return f"AdaptiveGamma(target={self.target}, gamma_range={self.gamma_range})"


class CLAHE:
    """Contrast-limited adaptive histogram equalisation.

    :param clip_limit: Contrast ceiling; higher values amplify noise.
    :param tile_grid_size: Grid the image is equalised over.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)

    def __call__(self, image: Image.Image) -> Image.Image:
        """:param image: Source image.
        :return: Equalised RGB image.
        """
        operator = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return _to_rgb_image(operator.apply(_to_gray_array(image).astype(np.uint8)))

    def __repr__(self) -> str:
        return f"CLAHE(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})"


class LogTransform:
    """Logarithmic intensity transform, expanding dark-region detail."""

    def __call__(self, image: Image.Image) -> Image.Image:
        """:param image: Source image.
        :return: Log-transformed RGB image.
        """
        array = _to_gray_array(image)
        scale = 255.0 / np.log(1.0 + array.max() + 1e-6)
        return _to_rgb_image(scale * np.log(1.0 + array))

    def __repr__(self) -> str:
        return "LogTransform()"


def edge_preservation_score(original: Image.Image, processed: Image.Image) -> float:
    """Correlation between the Sobel edge maps of two images.

    Step 6 forbids selecting preprocessing on visual appearance alone and asks for a
    boundary/texture preservation check alongside validation performance. A score near 1
    means the filter left edge structure intact; a heavy blur drives it toward 0.

    :param original: Unprocessed image.
    :param processed: Filtered image.
    :return: Pearson correlation of the flattened Sobel responses, 0 if undefined.
    """
    original_edges = cv2.Sobel(_to_gray_array(original), cv2.CV_32F, 1, 1, ksize=3).ravel()
    processed_edges = cv2.Sobel(_to_gray_array(processed), cv2.CV_32F, 1, 1, ksize=3).ravel()

    if original_edges.std() == 0 or processed_edges.std() == 0:
        return 0.0

    correlation = float(np.corrcoef(original_edges, processed_edges)[0, 1])
    return 0.0 if np.isnan(correlation) else correlation


# --------------------------------------------------------------------------- registry

#: Recipes needing no image-space filtering. They read the raw tree directly, and differ
#: only in the Step 5 intensity treatment applied later (``normalize=none`` for A0's raw
#: condition, ``normalize=imagenet`` for A1's conventional pipeline).
IDENTITY_RECIPES = ("raw", "conventional")

_SIMPLE_RECIPES: Dict[str, Callable[[], Callable]] = {
    "raw": Identity,
    "conventional": Identity,
    "wiener": WienerFilter,
    "clahe": CLAHE,
    "gamma": AdaptiveGamma,
    "log": LogTransform,
}


def is_identity_recipe(name: str) -> bool:
    """:param name: Recipe name.
    :return: Whether the recipe needs no materialised mirror.
    """
    return name in IDENTITY_RECIPES


def diffusion_recipe_name(num_iter: int, kappa: float, option: int = 1) -> str:
    """Canonical directory name for a diffusion configuration.

    Encoding the parameters in the name is what keeps cached mirrors from being
    confused with one another.

    :param num_iter: Diffusion iterations.
    :param kappa: Edge threshold.
    :param option: Diffusion coefficient variant.
    :return: A name such as ``diffusion_i10_k15`` (option 2 appends ``_o2``).
    """
    kappa_text = f"{kappa:g}".replace(".", "p")
    suffix = "" if option == 1 else f"_o{option}"
    return f"diffusion_i{num_iter}_k{kappa_text}{suffix}"


def build_recipe(name: str) -> Callable[[Image.Image], Image.Image]:
    """Resolve a recipe name to its filter.

    :param name: One of :data:`IDENTITY_RECIPES`, ``wiener``, ``clahe``, ``gamma``,
        ``log``, or a diffusion name such as ``diffusion_i10_k15``.
    :return: A callable mapping a PIL image to a PIL image.
    :raises ValueError: If the name is not recognised.
    """
    if name in _SIMPLE_RECIPES:
        return _SIMPLE_RECIPES[name]()

    if name.startswith("diffusion_"):
        return _parse_diffusion_recipe(name)

    raise ValueError(
        f"Unknown recipe {name!r}. Expected one of {sorted(_SIMPLE_RECIPES)} "
        "or a diffusion name such as 'diffusion_i10_k15'."
    )


def _parse_diffusion_recipe(name: str) -> AnisotropicDiffusion:
    """Parse ``diffusion_i{iters}_k{kappa}[_o{option}]`` back into a filter.

    :param name: Recipe name.
    :return: The configured diffusion filter.
    :raises ValueError: If the name is malformed.
    """
    parts = name.split("_")[1:]
    num_iter: Optional[int] = None
    kappa: Optional[float] = None
    option = 1

    try:
        for part in parts:
            if part.startswith("i"):
                num_iter = int(part[1:])
            elif part.startswith("k"):
                kappa = float(part[1:].replace("p", "."))
            elif part.startswith("o"):
                option = int(part[1:])
    except ValueError as error:
        raise ValueError(f"Malformed diffusion recipe {name!r}: {error}") from error

    if num_iter is None or kappa is None:
        raise ValueError(
            f"Malformed diffusion recipe {name!r}; expected 'diffusion_i<iters>_k<kappa>'"
        )

    return AnisotropicDiffusion(num_iter=num_iter, kappa=kappa, option=option)


def default_diffusion_grid(
    iterations: Tuple[int, ...] = (5, 10, 15, 20),
    kappas: Tuple[float, ...] = (15.0, 30.0),
) -> Tuple[str, ...]:
    """The Step 6 diffusion sweep grid.

    The specification says to "start with 5, 10, 15, and 20 iterations" and to tune kappa
    on validation experiments.

    :param iterations: Iteration counts to sweep.
    :param kappas: Edge thresholds to sweep.
    :return: Recipe names covering the grid.
    """
    return tuple(
        diffusion_recipe_name(num_iter, kappa) for num_iter in iterations for kappa in kappas
    )
