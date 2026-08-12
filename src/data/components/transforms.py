"""Image transform pipelines (Steps 5 and 7).

Step 5 fixes the geometry and intensity handling shared by every model so that baseline
and proposed architectures are compared on identical inputs. Step 7 adds augmentation
that applies to the **training split only**.

Two configurable axes exist because the ablation table needs them:

- ``normalize`` selects the Step 5 intensity treatment. ``"none"`` is what makes
  ablation row A0 ("raw image + baseline CNN") genuinely distinct from A1
  ("conventional preprocessing + CNN"); the notebook collapsed these into one row
  because it had no way to express the difference.
- ``augment`` toggles the Step 7 block, which must never reach validation or test.

On input size: the specification suggests "such as 256 x 256". This project uses **224**
because ``torchvision``'s ``vit_b_16`` and ``swin_t`` carry positional embeddings baked
for 224 and cannot accept 256 without interpolation - which would confound Baselines 4
and 4b against the CNN baselines. ``image_size`` remains configurable.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
from torchvision.transforms import transforms

from src.data.components.cropping import BrainBoundingBoxCrop

# ImageNet channel statistics, required by every pretrained backbone in Step 9.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NORMALIZE_MODES = ("imagenet", "zscore", "minmax", "none")

# Step 7 augmentation ranges. Deliberately conservative: MRI anatomy carries clinical
# meaning, so nothing here warps tumour morphology. Exposed as a dict so the method
# section can report exact ranges rather than paraphrasing the code.
DEFAULT_AUGMENTATION: Dict[str, Any] = {
    "rotation_degrees": 10,
    "translate": (0.05, 0.05),
    "scale": (0.95, 1.05),
    "horizontal_flip_p": 0.5,
    "brightness": 0.1,
    "contrast": 0.1,
}


class PerImageZScore:
    """Standardise each image to zero mean and unit variance across all channels.

    Unlike ImageNet normalisation this uses the image's own statistics, which suits
    models trained from scratch on MRI where the ImageNet prior does not apply.

    :param eps: Floor on the standard deviation, guarding constant images.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """:param tensor: Image tensor in ``[0, 1]``.
        :return: Standardised tensor.
        """
        return (tensor - tensor.mean()) / (tensor.std() + self.eps)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(eps={self.eps})"


class PerImageMinMax:
    """Rescale each image to the full ``[0, 1]`` range using its own extremes.

    ``ToTensor`` already divides by 255, but MRI slices rarely span the full byte range;
    this stretches the actual dynamic range, which is what Step 5's "min-max scaling"
    means for this data.

    :param eps: Floor on the value range, guarding constant images.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """:param tensor: Image tensor in ``[0, 1]``.
        :return: Rescaled tensor.
        """
        minimum = tensor.amin()
        return (tensor - minimum) / (tensor.amax() - minimum + self.eps)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(eps={self.eps})"


def build_normalizer(normalize: str) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Resolve a normalisation mode to a tensor transform.

    :param normalize: One of ``imagenet``, ``zscore``, ``minmax``, ``none``.
    :return: The transform, or ``None`` for ``"none"``.
    :raises ValueError: If the mode is unknown.
    """
    if normalize == "imagenet":
        return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if normalize == "zscore":
        return PerImageZScore()
    if normalize == "minmax":
        return PerImageMinMax()
    if normalize == "none":
        return None
    raise ValueError(f"Unknown normalize mode {normalize!r}; expected one of {NORMALIZE_MODES}")


def build_transform(
    image_size: int = 224,
    normalize: str = "imagenet",
    augment: bool = False,
    crop_background: bool = False,
    augmentation: Optional[Dict[str, Any]] = None,
    crop_margin: float = 0.02,
) -> transforms.Compose:
    """Compose the full input pipeline.

    Order is deliberate: crop before resize (so the resize acts on anatomy rather than
    on padding), augment geometrically before converting to tensor, and normalise last.

    :param image_size: Square output edge in pixels.
    :param normalize: Step 5 intensity treatment; see :func:`build_normalizer`.
    :param augment: Apply the Step 7 training-only augmentation block.
    :param crop_background: Apply the validated Step 5 background crop.
    :param augmentation: Override for :data:`DEFAULT_AUGMENTATION`.
    :param crop_margin: Safety margin for the background crop.
    :return: The composed transform.
    """
    params = {**DEFAULT_AUGMENTATION, **(augmentation or {})}
    stages: List[Callable] = []

    if crop_background:
        stages.append(BrainBoundingBoxCrop(margin=crop_margin))

    stages.append(transforms.Resize((image_size, image_size)))

    if augment:
        stages.extend(
            [
                transforms.RandomRotation(degrees=params["rotation_degrees"]),
                transforms.RandomAffine(
                    degrees=0,
                    translate=tuple(params["translate"]),
                    scale=tuple(params["scale"]),
                ),
                transforms.RandomHorizontalFlip(p=params["horizontal_flip_p"]),
                transforms.ColorJitter(
                    brightness=params["brightness"], contrast=params["contrast"]
                ),
            ]
        )

    stages.append(transforms.ToTensor())

    normalizer = build_normalizer(normalize)
    if normalizer is not None:
        stages.append(normalizer)

    return transforms.Compose(stages)


def describe_pipeline(
    image_size: int,
    normalize: str,
    augment: bool,
    crop_background: bool,
    augmentation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialise the pipeline settings for the method section and run logs.

    Step 7 requires the exact augmentation ranges to be reported; this emits them in a
    form that can be written straight into a run artefact.

    :param image_size: Square output edge in pixels.
    :param normalize: Step 5 intensity treatment.
    :param augment: Whether the Step 7 block is active.
    :param crop_background: Whether the Step 5 crop is active.
    :param augmentation: Override for :data:`DEFAULT_AUGMENTATION`.
    :return: A JSON-serialisable description.
    """
    description: Dict[str, Any] = {
        "image_size": image_size,
        "normalize": normalize,
        "crop_background": crop_background,
        "augment": augment,
    }
    if augment:
        description["augmentation"] = {**DEFAULT_AUGMENTATION, **(augmentation or {})}
    return description
