"""Tests for the Step 6 preprocessing methods, recipe registry and caching."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from src.data.components.preprocessing import (
    MAX_LAMBDA,
    AdaptiveGamma,
    AnisotropicDiffusion,
    CLAHE,
    Identity,
    LogTransform,
    WienerFilter,
    anisotropic_diffusion,
    build_recipe,
    default_diffusion_grid,
    diffusion_recipe_name,
    edge_preservation_score,
    is_identity_recipe,
)


@pytest.fixture
def noisy_mri() -> Image.Image:
    """:return: A synthetic MRI-like slice with a sharp edge and additive noise."""
    rng = np.random.default_rng(0)
    size = 96
    yy, xx = np.mgrid[0:size, 0:size]

    array = np.zeros((size, size), dtype=np.float32)
    array[((xx - 48) ** 2 + (yy - 48) ** 2) <= 34**2] = 110.0
    array[((xx - 56) ** 2 + (yy - 40) ** 2) <= 11**2] = 235.0
    array += rng.normal(0, 12, array.shape)

    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")


# ------------------------------------------------------------------ diffusion core


def test_diffusion_reduces_noise_in_flat_regions(noisy_mri):
    """The point of the filter: suppress noise where the image should be uniform."""
    array = np.asarray(noisy_mri.convert("L")).astype(np.float32)
    diffused = anisotropic_diffusion(array, num_iter=10, kappa=15.0, lam=0.2)

    # A patch inside the bright disc but clear of the lesion boundary, which the
    # filter is meant to preserve rather than smooth.
    before = array[58:70, 28:40].std()
    after = diffused[58:70, 28:40].std()
    assert after < before, f"diffusion increased flat-region variance ({before} -> {after})"


def test_diffusion_preserves_edges_better_than_gaussian_blur(noisy_mri):
    """Edge preservation is the whole justification for choosing anisotropic diffusion."""
    diffused = AnisotropicDiffusion(num_iter=10, kappa=15.0)(noisy_mri)
    blurred = noisy_mri.filter(ImageFilter.GaussianBlur(radius=2))

    assert edge_preservation_score(noisy_mri, diffused) > edge_preservation_score(
        noisy_mri, blurred
    )


def test_zero_iterations_is_a_no_op(noisy_mri):
    """Guards the loop bounds; 0 iterations must not perturb the image."""
    array = np.asarray(noisy_mri.convert("L")).astype(np.float32)
    assert np.allclose(anisotropic_diffusion(array, num_iter=0), array)


def test_more_iterations_smooth_more(noisy_mri):
    """Monotonicity in the swept parameter, so the Step 6 grid means something."""
    array = np.asarray(noisy_mri.convert("L")).astype(np.float32)
    variances = [
        anisotropic_diffusion(array, num_iter=n, kappa=15.0)[58:70, 28:40].std()
        for n in (5, 10, 20)
    ]
    assert variances[0] > variances[1] > variances[2]


def test_lambda_is_clamped_for_stability():
    """Step 6: lambda "should be small, commonly not greater than 0.25"."""
    assert AnisotropicDiffusion(lam=0.9).lam == MAX_LAMBDA

    array = np.random.default_rng(0).uniform(0, 255, (32, 32)).astype(np.float32)
    result = anisotropic_diffusion(array, num_iter=20, lam=5.0)
    assert np.isfinite(result).all(), "unclamped lambda let the explicit scheme diverge"


@pytest.mark.parametrize("option", [1, 2])
def test_both_diffusion_coefficients_are_supported(noisy_mri, option):
    """The specification gives two choices of c(s); both must work."""
    array = np.asarray(noisy_mri.convert("L")).astype(np.float32)
    result = anisotropic_diffusion(array, num_iter=5, kappa=15.0, option=option)
    assert result.shape == array.shape
    assert np.isfinite(result).all()


def test_invalid_diffusion_options_are_rejected():
    """A bad config value must fail loudly rather than silently pick a coefficient."""
    array = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="option"):
        anisotropic_diffusion(array, option=3)
    with pytest.raises(ValueError, match="num_iter"):
        anisotropic_diffusion(array, num_iter=-1)


def test_north_south_roll_direction_is_cosmetic(noisy_mri):
    """The notebook defined diffusion twice with N/S swapped; prove it changed nothing.

    All four directional terms are summed and the coefficient depends only on |delta|,
    so swapping the north and south labels yields an identical result. This pins the
    claim made in docs/DEVIATIONS.md.
    """
    array = np.asarray(noisy_mri.convert("L")).astype(np.float32)
    kappa, lam = 15.0, 0.2

    swapped = array.copy()
    for _ in range(5):
        deltas = (
            np.roll(swapped, -1, axis=0) - swapped,  # notebook cell 31 called this north
            np.roll(swapped, 1, axis=0) - swapped,
            np.roll(swapped, -1, axis=1) - swapped,
            np.roll(swapped, 1, axis=1) - swapped,
        )
        update = np.zeros_like(swapped)
        for delta in deltas:
            update += np.exp(-((delta / kappa) ** 2)) * delta
        swapped += lam * update

    assert np.allclose(anisotropic_diffusion(array, 5, kappa, lam), swapped, atol=1e-4)


# ------------------------------------------------------------------- comparators


@pytest.mark.parametrize(
    "filter_fn",
    [Identity(), AnisotropicDiffusion(num_iter=3), WienerFilter(), AdaptiveGamma(), CLAHE(), LogTransform()],
    ids=["identity", "diffusion", "wiener", "gamma", "clahe", "log"],
)
def test_every_filter_returns_a_valid_rgb_image(noisy_mri, filter_fn):
    """Filters are interchangeable in the pipeline, so their contract must be uniform."""
    output = filter_fn(noisy_mri)

    assert isinstance(output, Image.Image)
    assert output.mode == "RGB"
    assert output.size == noisy_mri.size

    array = np.asarray(output)
    assert array.dtype == np.uint8
    assert np.isfinite(array).all()


def test_filters_operate_on_luminance_and_replicate(noisy_mri):
    """Grayscale MRI is filtered once, not three times over identical RGB channels."""
    array = np.asarray(AnisotropicDiffusion(num_iter=3)(noisy_mri))
    assert np.array_equal(array[:, :, 0], array[:, :, 1])
    assert np.array_equal(array[:, :, 1], array[:, :, 2])


def test_wiener_handles_the_black_background(noisy_mri):
    """scipy's Wiener yields NaN where local variance is zero - the MRI background."""
    output = WienerFilter()(Image.new("RGB", (32, 32), 0))
    assert np.isfinite(np.asarray(output)).all()


def test_adaptive_gamma_brightens_a_dark_image():
    """The exponent should adapt toward mid-grey rather than being fixed."""
    dark = Image.fromarray(np.full((32, 32), 40, dtype=np.uint8)).convert("RGB")
    assert np.asarray(AdaptiveGamma()(dark)).mean() > np.asarray(dark).mean()


def test_edge_preservation_score_is_one_for_an_unchanged_image(noisy_mri):
    """Calibrates the metric's upper end."""
    assert edge_preservation_score(noisy_mri, noisy_mri) == pytest.approx(1.0, abs=1e-6)


def test_edge_preservation_score_drops_under_heavy_blur(noisy_mri):
    """...and its lower end, so the reported numbers are interpretable."""
    heavy = noisy_mri.filter(ImageFilter.GaussianBlur(radius=6))
    assert edge_preservation_score(noisy_mri, heavy) < 0.5


def test_edge_preservation_score_is_defined_for_a_blank_image():
    """A constant image has no edges; the score must not be NaN."""
    blank = Image.new("RGB", (32, 32), 0)
    assert edge_preservation_score(blank, blank) == 0.0


# ---------------------------------------------------------------------- registry


@pytest.mark.parametrize(
    "name,expected",
    [("raw", Identity), ("conventional", Identity), ("wiener", WienerFilter), ("clahe", CLAHE), ("gamma", AdaptiveGamma), ("log", LogTransform)],
)
def test_named_recipes_resolve(name, expected):
    """Config strings must map to the right filter."""
    assert isinstance(build_recipe(name), expected)


def test_diffusion_recipe_names_round_trip():
    """Parameters are encoded in the directory name, so mirrors cannot be confused."""
    name = diffusion_recipe_name(10, 15.0)
    assert name == "diffusion_i10_k15"

    recipe = build_recipe(name)
    assert isinstance(recipe, AnisotropicDiffusion)
    assert recipe.num_iter == 10
    assert recipe.kappa == 15.0
    assert recipe.option == 1


def test_fractional_kappa_survives_the_name():
    """A '.' in a directory name is legal but fragile; 'p' encodes it instead."""
    name = diffusion_recipe_name(5, 2.5)
    assert name == "diffusion_i5_k2p5"
    assert build_recipe(name).kappa == 2.5


def test_option_two_is_encoded_distinctly():
    """The two diffusion coefficients must not share a cache directory."""
    assert diffusion_recipe_name(10, 15.0, option=2) == "diffusion_i10_k15_o2"
    assert build_recipe("diffusion_i10_k15_o2").option == 2


def test_unknown_recipes_are_rejected():
    """A typo in a config must not silently produce an unfiltered dataset."""
    with pytest.raises(ValueError, match="Unknown recipe"):
        build_recipe("bilateral")
    with pytest.raises(ValueError, match="Malformed diffusion recipe"):
        build_recipe("diffusion_i10")


def test_identity_recipes_need_no_mirror():
    """'raw' and 'conventional' differ only in Step 5 normalisation, not filtering."""
    assert is_identity_recipe("raw")
    assert is_identity_recipe("conventional")
    assert not is_identity_recipe("diffusion_i10_k15")


def test_default_grid_matches_the_specification():
    """Step 6: "Start with 5, 10, 15, and 20 iterations"."""
    grid = default_diffusion_grid()
    assert len(grid) == 8
    assert "diffusion_i5_k15" in grid
    assert "diffusion_i20_k30" in grid
    assert len(set(grid)) == len(grid)


# ----------------------------------------------------------------- materialisation


def test_materialise_recipe_mirrors_the_tree(tmp_path):
    """The cached mirror must reuse the raw tree's layout so one split table serves both."""
    from src.data.components.split_builder import build_split
    from src.prepare_dataset import materialise_recipe
    from tests.helpers.synthetic_dataset import make_synthetic_dataset

    raw_dir = tmp_path / "raw"
    make_synthetic_dataset(raw_dir, per_class_train=4, per_class_test=2, duplicates_per_class=1)
    split_csv = tmp_path / "split.csv"
    split_df, _ = build_split(raw_dir, split_csv)

    output_dir = tmp_path / "processed" / "clahe"
    manifest = materialise_recipe("clahe", raw_dir, output_dir, split_csv)

    assert manifest["images_written"] == len(split_df)
    assert manifest["images_failed"] == 0
    for rel_path in split_df["rel_path"]:
        assert (output_dir / rel_path).is_file(), f"missing mirrored image {rel_path}"
    assert (output_dir / "recipe_manifest.json").is_file()


def test_materialise_recipe_skips_existing_work(tmp_path):
    """Recipe caching is expensive; a re-run must not redo it."""
    from src.data.components.split_builder import build_split
    from src.prepare_dataset import materialise_recipe
    from tests.helpers.synthetic_dataset import make_synthetic_dataset

    raw_dir = tmp_path / "raw"
    make_synthetic_dataset(raw_dir, per_class_train=6, per_class_test=2, duplicates_per_class=0)
    split_csv = tmp_path / "split.csv"
    build_split(raw_dir, split_csv)
    output_dir = tmp_path / "processed" / "gamma"

    first = materialise_recipe("gamma", raw_dir, output_dir, split_csv)
    second = materialise_recipe("gamma", raw_dir, output_dir, split_csv)

    assert first["images_written"] > 0
    assert second["images_written"] == 0
    assert second["images_skipped"] == first["images_written"]


def test_materialised_images_actually_differ_from_the_raw_ones(tmp_path):
    """A cache that silently copies unfiltered images would invalidate every A2-A6 run."""
    from src.data.components.split_builder import build_split
    from src.prepare_dataset import materialise_recipe
    from tests.helpers.synthetic_dataset import make_synthetic_dataset

    raw_dir = tmp_path / "raw"
    make_synthetic_dataset(raw_dir, per_class_train=6, per_class_test=2, duplicates_per_class=0)
    split_csv = tmp_path / "split.csv"
    split_df, _ = build_split(raw_dir, split_csv)

    output_dir = tmp_path / "processed" / "diffusion_i10_k15"
    materialise_recipe("diffusion_i10_k15", raw_dir, output_dir, split_csv)

    rel_path = split_df.iloc[0]["rel_path"]
    original = np.asarray(Image.open(raw_dir / rel_path).convert("L")).astype(float)
    processed = np.asarray(Image.open(output_dir / rel_path).convert("L")).astype(float)
    assert np.abs(original - processed).mean() > 0.1
