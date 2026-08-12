"""Validated background cropping (Step 5).

The specification permits background cropping but constrains it: *"Apply background
cropping only if it does not remove tumor regions."* The reference notebook skipped
cropping entirely, so this module supplies it together with the validation the clause
demands.

The crop is **off by default**. Enabling it without running
:func:`validate_crop_preserves_foreground` over the training split would ignore the very
condition that makes it permissible.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label as connected_components
from skimage.measure import regionprops


def brain_bounding_box(
    image: Image.Image,
    background_threshold: int = 10,
    min_area_fraction: float = 0.05,
) -> Optional[Tuple[int, int, int, int]]:
    """Locate the head/brain region as the largest bright connected component.

    MRI slices sit on a near-black background, so a low absolute threshold isolates the
    head reliably without needing skull stripping (which the specification warns against
    unless segmentation quality is validated).

    :param image: Source image, any mode.
    :param background_threshold: Intensity at or below which a pixel is background.
    :param min_area_fraction: Reject a component covering less than this fraction of the
        frame; such a component is noise, not anatomy.
    :return: ``(left, upper, right, lower)`` box, or ``None`` if no region qualifies.
    """
    array = np.asarray(image.convert("L"))
    foreground = array > background_threshold
    if not foreground.any():
        return None

    labelled = connected_components(foreground)
    regions = regionprops(labelled)
    if not regions:
        return None

    largest = max(regions, key=lambda region: region.area)
    if largest.area < min_area_fraction * array.size:
        return None

    min_row, min_col, max_row, max_col = largest.bbox
    return int(min_col), int(min_row), int(max_col), int(max_row)


class BrainBoundingBoxCrop:
    """Crop to the brain bounding box with a safety margin.

    Defined at module level rather than as a lambda so it survives pickling into
    dataloader worker processes on Windows.

    :param margin: Fraction of the box's own width/height added on each side.
    :param background_threshold: Passed to :func:`brain_bounding_box`.
    :param min_area_fraction: Passed to :func:`brain_bounding_box`.
    """

    def __init__(
        self,
        margin: float = 0.02,
        background_threshold: int = 10,
        min_area_fraction: float = 0.05,
    ) -> None:
        self.margin = margin
        self.background_threshold = background_threshold
        self.min_area_fraction = min_area_fraction

    def __call__(self, image: Image.Image) -> Image.Image:
        """Crop the image, returning it untouched if no valid region is found.

        :param image: Source image.
        :return: The cropped image, or the original when cropping would be unsafe.
        """
        box = brain_bounding_box(image, self.background_threshold, self.min_area_fraction)
        if box is None:
            return image

        left, upper, right, lower = box
        pad_x = int(round(self.margin * (right - left)))
        pad_y = int(round(self.margin * (lower - upper)))
        return image.crop(
            (
                max(0, left - pad_x),
                max(0, upper - pad_y),
                min(image.width, right + pad_x),
                min(image.height, lower + pad_y),
            )
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(margin={self.margin}, "
            f"background_threshold={self.background_threshold}, "
            f"min_area_fraction={self.min_area_fraction})"
        )


def validate_crop_preserves_foreground(
    images: List[Image.Image],
    crop: BrainBoundingBoxCrop,
    tolerance: float = 0.001,
) -> Dict[str, object]:
    """Check that cropping never discards hyperintense tissue.

    Uses the Otsu threshold over in-brain pixels as a proxy for "tumour or other bright
    tissue" - the same proxy the notebook used for its Step 11 morphology analysis,
    since this dataset ships no segmentation masks. A crop that drops more than
    ``tolerance`` of those pixels on any image fails.

    :param images: Sample images, drawn from the training split only.
    :param crop: The configured crop to validate.
    :param tolerance: Maximum permitted fraction of bright pixels lost per image.
    :return: Report with ``passed``, ``n_images``, ``max_foreground_lost`` and
        ``worst_case_index``.
    """
    losses: List[float] = []

    for image in images:
        grey = np.asarray(image.convert("L")).astype(np.float32)
        in_brain = grey > crop.background_threshold
        if not in_brain.any():
            losses.append(0.0)
            continue

        bright = grey > threshold_otsu(grey[in_brain])
        bright_before = int(bright.sum())
        if bright_before == 0:
            losses.append(0.0)
            continue

        box = brain_bounding_box(image, crop.background_threshold, crop.min_area_fraction)
        if box is None:
            losses.append(0.0)
            continue

        left, upper, right, lower = box
        pad_x = int(round(crop.margin * (right - left)))
        pad_y = int(round(crop.margin * (lower - upper)))
        kept = bright[
            max(0, upper - pad_y) : min(image.height, lower + pad_y),
            max(0, left - pad_x) : min(image.width, right + pad_x),
        ]
        losses.append((bright_before - int(kept.sum())) / bright_before)

    max_loss = max(losses) if losses else 0.0
    return {
        "passed": bool(max_loss <= tolerance),
        "n_images": len(images),
        "tolerance": tolerance,
        "max_foreground_lost": float(max_loss),
        "mean_foreground_lost": float(np.mean(losses)) if losses else 0.0,
        "worst_case_index": int(np.argmax(losses)) if losses else -1,
    }
