"""Tests for the Step 5 preprocessing pipeline and Step 7 augmentation."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.components.cropping import (
    BrainBoundingBoxCrop,
    brain_bounding_box,
    validate_crop_preserves_foreground,
)
from src.data.components.transforms import (
    DEFAULT_AUGMENTATION,
    NORMALIZE_MODES,
    build_transform,
    describe_pipeline,
)


@pytest.fixture
def mri_like() -> Image.Image:
    """:return: A dark frame with an off-centre bright disc, mimicking an MRI slice."""
    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    array = np.zeros((size, size), dtype=np.uint8)
    array[((xx - 32) ** 2 + (yy - 32) ** 2) <= 22**2] = 120
    array[((xx - 38) ** 2 + (yy - 28) ** 2) <= 7**2] = 240
    return Image.fromarray(array).convert("RGB")


@pytest.mark.parametrize("normalize", NORMALIZE_MODES)
def test_every_normalize_mode_produces_a_correctly_shaped_tensor(mri_like, normalize):
    """All four Step 5 intensity treatments must be interchangeable in shape."""
    transform = build_transform(image_size=224, normalize=normalize, augment=False)
    output = transform(mri_like)

    assert output.shape == (3, 224, 224)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_normalize_none_leaves_the_zero_one_range(mri_like):
    """Ablation row A0's raw condition: ToTensor only, no intensity normalisation."""
    output = build_transform(normalize="none", augment=False)(mri_like)
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_minmax_stretches_to_the_full_range(mri_like):
    """MRI slices rarely span the full byte range; min-max must stretch them."""
    output = build_transform(normalize="minmax", augment=False)(mri_like)
    assert output.min() == pytest.approx(0.0, abs=1e-5)
    assert output.max() == pytest.approx(1.0, abs=1e-5)


def test_zscore_standardises_the_image(mri_like):
    """Per-image z-score must give roughly zero mean and unit variance."""
    output = build_transform(normalize="zscore", augment=False)(mri_like)
    assert output.mean().item() == pytest.approx(0.0, abs=1e-4)
    assert output.std().item() == pytest.approx(1.0, abs=1e-2)


def test_unknown_normalize_mode_is_rejected():
    """A typo in the data config must fail loudly rather than skip normalisation."""
    with pytest.raises(ValueError, match="Unknown normalize mode"):
        build_transform(normalize="standardise")


def test_augmentation_is_stochastic_but_eval_is_deterministic(mri_like):
    """Step 7 augmentation applies to training only; eval must be reproducible."""
    augmented = build_transform(augment=True)
    evaluation = build_transform(augment=False)

    torch.manual_seed(0)
    first = augmented(mri_like)
    torch.manual_seed(1)
    second = augmented(mri_like)
    assert not torch.allclose(first, second), "augmentation produced identical outputs"

    assert torch.allclose(evaluation(mri_like), evaluation(mri_like))


def test_image_size_is_configurable(mri_like):
    """224 is a justified deviation from the spec's suggested 256; both must work."""
    for size in (128, 224, 256):
        assert build_transform(image_size=size, augment=False)(mri_like).shape == (3, size, size)


def test_describe_pipeline_reports_exact_augmentation_ranges():
    """Step 7: "Report the exact augmentation ranges in the method section"."""
    described = describe_pipeline(224, "imagenet", augment=True, crop_background=False)

    assert described["image_size"] == 224
    assert described["normalize"] == "imagenet"
    assert described["augmentation"] == DEFAULT_AUGMENTATION
    assert described["augmentation"]["rotation_degrees"] == 10
    assert described["augmentation"]["horizontal_flip_p"] == 0.5

    # Eval pipelines must not advertise augmentation ranges they do not apply.
    assert "augmentation" not in describe_pipeline(224, "imagenet", False, False)


def test_bounding_box_finds_the_head_region(mri_like):
    """The crop must locate anatomy, not the whole frame."""
    box = brain_bounding_box(mri_like)
    assert box is not None

    left, upper, right, lower = box
    assert 0 <= left < right <= mri_like.width
    assert 0 <= upper < lower <= mri_like.height
    assert (right - left) < mri_like.width, "box should be tighter than the full frame"


def test_bounding_box_returns_none_for_an_empty_frame():
    """An all-black frame has no anatomy; cropping must decline rather than guess."""
    assert brain_bounding_box(Image.new("L", (64, 64), 0)) is None


def test_crop_falls_back_to_the_original_when_unsafe():
    """Step 5 permits cropping only when it is safe; unsafe input returns untouched."""
    blank = Image.new("RGB", (64, 64), 0)
    assert BrainBoundingBoxCrop()(blank).size == blank.size


def test_crop_validation_confirms_bright_tissue_is_preserved(mri_like):
    """Step 5: cropping is permitted "only if it does not remove tumor regions"."""
    report = validate_crop_preserves_foreground([mri_like] * 5, BrainBoundingBoxCrop())

    assert report["passed"] is True
    assert report["max_foreground_lost"] == pytest.approx(0.0, abs=1e-6)
    assert report["n_images"] == 5


def test_crop_runs_inside_the_transform_pipeline(mri_like):
    """Cropping happens before resize, so the output size is unchanged."""
    output = build_transform(image_size=224, augment=False, crop_background=True)(mri_like)
    assert output.shape == (3, 224, 224)
