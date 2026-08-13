"""Tests for the Step 16-18 evaluation stages."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.analysis.internal_test import LOCK_FILENAME, InternalTest
from src.analysis.robustness import RobustnessStudy
from src.data.components.degradations import (
    DEGRADATION_SWEEP,
    Blur,
    ContrastShift,
    GaussianNoise,
    IntensityShift,
    ResolutionLoss,
    build_degradation,
)
from src.data.components.external import EXTERNAL_CLASSES, FIGSHARE_LABELS
from src.data.components.split_builder import CLASS_MAP
from src.models.components.fusion import FusedFeatureClassifier
from src.models.components.multiscale import MultiscaleClassifier
from src.models.components.quantum import AdaptiveQuantumClassifier
from src.models.components.transfer import TransferBackbone
from src.models.full_pipeline import FullPipeline


@pytest.fixture
def sample_image() -> Image.Image:
    """:return: A small MRI-like image."""
    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    array = np.zeros((size, size), dtype=np.uint8)
    array[((xx - 32) ** 2 + (yy - 32) ** 2) <= 24**2] = 120
    array[((xx - 38) ** 2 + (yy - 28) ** 2) <= 8**2] = 230
    return Image.fromarray(array).convert("RGB")


# ----------------------------------------------------------------- degradations


@pytest.mark.parametrize(
    "degradation",
    [GaussianNoise(25.0), ContrastShift(0.5), Blur(2.0), ResolutionLoss(0.25), IntensityShift(40.0)],
    ids=["noise", "contrast", "blur", "resolution", "intensity"],
)
def test_degradations_preserve_shape_and_dtype(sample_image, degradation):
    """A degraded image must still be a valid model input."""
    output = degradation(sample_image, index=0)

    assert isinstance(output, Image.Image)
    assert output.size == sample_image.size
    array = np.asarray(output)
    assert array.dtype == np.uint8
    assert array.min() >= 0 and array.max() <= 255


@pytest.mark.parametrize(
    "degradation",
    [GaussianNoise(25.0), ContrastShift(0.5), Blur(2.0), ResolutionLoss(0.25), IntensityShift(40.0)],
    ids=["noise", "contrast", "blur", "resolution", "intensity"],
)
def test_every_degradation_actually_changes_the_image(sample_image, degradation):
    """A degradation that silently does nothing would report false robustness."""
    before = np.asarray(sample_image).astype(float)
    after = np.asarray(degradation(sample_image, index=0)).astype(float)
    assert np.abs(before - after).mean() > 0.5


def test_gaussian_noise_is_reproducible_across_models(sample_image):
    """Every model in the comparison must see identical corrupted pixels.

    Without this a model could look more robust purely by drawing an easier noise sample.
    """
    noise = GaussianNoise(25.0, seed=42)
    first = np.asarray(noise(sample_image, index=7))
    second = np.asarray(noise(sample_image, index=7))
    assert np.array_equal(first, second)


def test_gaussian_noise_differs_between_images(sample_image):
    """The whole test split must not receive one identical noise pattern."""
    noise = GaussianNoise(25.0, seed=42)
    assert not np.array_equal(
        np.asarray(noise(sample_image, index=0)), np.asarray(noise(sample_image, index=1))
    )


def test_higher_severity_degrades_more(sample_image):
    """Severity ordering must be monotonic or the sweep is uninterpretable."""
    reference = np.asarray(sample_image).astype(float)

    def distance(degradation) -> float:
        return float(np.abs(reference - np.asarray(degradation(sample_image, 0)).astype(float)).mean())

    assert distance(GaussianNoise(10.0)) < distance(GaussianNoise(40.0))
    assert distance(Blur(1.0)) < distance(Blur(3.0))
    assert distance(ResolutionLoss(0.5)) < distance(ResolutionLoss(0.125))


def test_resolution_loss_returns_the_original_size(sample_image):
    """The round trip must restore the shape, or the model cannot consume it."""
    assert ResolutionLoss(0.125)(sample_image, 0).size == sample_image.size


def test_sweep_covers_every_degradation_the_specification_names():
    """Step 18 names noise, contrast, blur, resolution and intensity normalisation."""
    assert set(DEGRADATION_SWEEP) == {
        "clean",
        "gaussian_noise",
        "contrast_shift",
        "blur",
        "resolution",
        "intensity_shift",
    }
    for category, severities in DEGRADATION_SWEEP.items():
        expected = 1 if category == "clean" else 3
        assert len(severities) == expected, f"{category} should have {expected} severities"


def test_build_degradation_round_trips_and_rejects_typos():
    """Conditions are addressed by name from config."""
    assert build_degradation("clean", "none") is None
    assert isinstance(build_degradation("gaussian_noise", "sigma=25"), GaussianNoise)

    with pytest.raises(ValueError, match="Unknown degradation category"):
        build_degradation("jpeg", "q=50")
    with pytest.raises(ValueError, match="Unknown severity"):
        build_degradation("blur", "radius=99")


# ----------------------------------------------------------------- full pipeline


def build_pipeline() -> FullPipeline:
    """:return: A small assembled pipeline for shape and gradient tests."""
    classical = TransferBackbone(arch="efficientnet_b0", num_classes=4, weights=None)
    quantum = AdaptiveQuantumClassifier(channels=8, num_classes=4, n_qubits=4)
    fusion = FusedFeatureClassifier(
        classical_dim=classical.feature_dim, spatial_dim=8, quantum_dim=4, num_classes=4
    )
    return FullPipeline(classical.eval(), quantum.eval(), fusion.eval())


def test_full_pipeline_exposes_every_intermediate_the_analyses_need():
    """Steps 16-20 read branch features, gate maps and quantum weights from here."""
    pipeline = build_pipeline().eval()

    with torch.no_grad():
        outputs = pipeline.extract(torch.randn(2, 3, 64, 64))

    assert outputs["logits"].shape == (2, 4)
    assert outputs["classical"].shape[1] == 1280
    assert outputs["spatial"].shape == (2, 8)
    assert outputs["quantum"].shape == (2, 4)
    assert outputs["quantum_weights"].shape == (2, 5)
    assert outputs["gate_maps"].shape[1] == 3


def test_gradients_reach_the_input_pixels():
    """Grad-CAM in Phase 7 depends on this.

    The branches are frozen by requires_grad=False, which stops their weights updating.
    Wrapping the forward in torch.no_grad as well would additionally block gradients from
    reaching the input - producing a blank saliency map rather than an error.
    """
    pipeline = build_pipeline().eval()
    images = torch.randn(2, 3, 64, 64, requires_grad=True)

    pipeline(images)[:, 0].sum().backward()

    assert images.grad is not None, "no gradient reached the input"
    assert torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0, "gradient reached the input but is identically zero"


def test_frozen_branches_do_not_accumulate_weight_gradients():
    """Frozen means frozen: only the fusion head should learn."""
    pipeline = build_pipeline()
    pipeline(torch.randn(2, 3, 64, 64)).sum().backward()

    assert all(p.grad is None for p in pipeline.classical_net.parameters())
    assert all(p.grad is None for p in pipeline.quantum_net.parameters())
    assert any(p.grad is not None for p in pipeline.fusion_net.parameters())


# --------------------------------------------------------------- external set


def test_figshare_label_mapping_goes_through_class_names():
    """Figshare is 1=meningioma, 2=glioma, 3=pituitary; ours is glioma=0, meningioma=1.

    Mapping raw index to raw index would silently swap glioma and meningioma and produce
    a plausible but meaningless confusion matrix.
    """
    assert FIGSHARE_LABELS == {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}

    assert CLASS_MAP[FIGSHARE_LABELS[1]] == 1  # meningioma
    assert CLASS_MAP[FIGSHARE_LABELS[2]] == 0  # glioma - note the index differs
    assert CLASS_MAP[FIGSHARE_LABELS[3]] == 2  # pituitary


def test_external_classes_exclude_the_absent_one():
    """Figshare has no non-tumour class; Step 17 evaluates the three present ones."""
    assert set(EXTERNAL_CLASSES) == {"Glioma", "Meningioma", "Pituitary"}
    assert "No-tumor" not in EXTERNAL_CLASSES
    assert set(EXTERNAL_CLASSES) < set(CLASS_MAP)


def test_missing_external_dataset_says_how_to_get_it(tmp_path):
    """Forgetting the --external download flag is the likely Step 17 mistake."""
    from src.data.components.external import scan_figshare

    with pytest.raises(FileNotFoundError, match="download_data"):
        scan_figshare(tmp_path / "nope")


# ------------------------------------------------------------- test-set lock


def test_second_evaluation_of_the_same_checkpoint_is_refused(tmp_path):
    """Step 16: "evaluate the final model once".

    The reference notebook read test metrics during Step 14 selection and then reported
    the same test set as its result. The lock makes a repeat deliberate.
    """
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (tmp_path / LOCK_FILENAME).write_text(json.dumps({"evaluations": []}), encoding="utf-8")

    analysis = InternalTest(fusion_ckpt=str(tmp_path))
    with pytest.raises(RuntimeError, match="already been evaluated"):
        analysis._check_lock()


def test_force_permits_re_evaluation_but_is_recorded(tmp_path):
    """An escape hatch must exist, and must leave a trace."""
    (tmp_path / LOCK_FILENAME).write_text(json.dumps({"evaluations": []}), encoding="utf-8")

    analysis = InternalTest(fusion_ckpt=str(tmp_path), force=True)
    assert analysis._check_lock() is not None  # does not raise


def test_first_evaluation_is_allowed_and_writes_the_lock(tmp_path):
    """The lock must not exist before the first run, and must exist after."""
    analysis = InternalTest(fusion_ckpt=str(tmp_path))
    lock_path = analysis._check_lock()
    assert not lock_path.is_file()

    analysis._output_dir = tmp_path
    analysis._write_lock(lock_path, {"overall": {"macro_f1": 0.9}})

    assert lock_path.is_file()
    recorded = json.loads(lock_path.read_text(encoding="utf-8"))["evaluations"]
    assert len(recorded) == 1
    assert recorded[0]["macro_f1"] == 0.9
    assert recorded[0]["forced"] is False


def test_lock_accumulates_a_history(tmp_path):
    """Repeat evaluations must all be visible, not overwrite one another."""
    analysis = InternalTest(fusion_ckpt=str(tmp_path), force=True)
    analysis._output_dir = tmp_path
    lock_path = tmp_path / LOCK_FILENAME

    analysis._write_lock(lock_path, {"overall": {"macro_f1": 0.9}})
    analysis._write_lock(lock_path, {"overall": {"macro_f1": 0.8}})

    assert len(json.loads(lock_path.read_text(encoding="utf-8"))["evaluations"]) == 2


# ------------------------------------------------------------- robustness maths


def _robustness_frame(rows: list):
    """Build a robustness results table.

    :param rows: ``(model, category, severity, macro_f1)`` tuples.
    :return: The table with drops added.
    """
    import pandas as pd

    frame = pd.DataFrame(rows, columns=["model", "category", "severity", "macro_f1"])
    return RobustnessStudy._add_drops(frame)


def test_drop_is_measured_against_each_model_s_own_clean_score():
    """Robustness is the drop, not the absolute score.

    A weaker model that degrades gently is more robust, and only the drop shows it.
    """
    results = _robustness_frame(
        [
            ("strong", "clean", "none", 0.99),
            ("strong", "blur", "radius=2", 0.60),
            ("weak", "clean", "none", 0.70),
            ("weak", "blur", "radius=2", 0.65),
        ]
    )
    drops = RobustnessStudy._mean_drops(results)

    assert drops["strong"] == pytest.approx(0.39)
    assert drops["weak"] == pytest.approx(0.05)
    assert drops["weak"] < drops["strong"], "the gentler degrader must rank as more robust"


def test_most_damaging_category_is_identified():
    """Step 18 asks which degradations the model is stable under."""
    results = _robustness_frame(
        [
            ("m", "clean", "none", 0.90),
            ("m", "blur", "radius=2", 0.85),
            ("m", "gaussian_noise", "sigma=25", 0.40),
        ]
    )
    worst = RobustnessStudy._most_damaging(results)

    assert worst["worst_category"] == "gaussian_noise"
    assert worst["worst_mean_drop"] == pytest.approx(0.50)


def test_diffusion_question_is_declined_when_it_cannot_be_answered():
    """Better to say the configuration cannot answer it than to imply a verdict."""
    study = RobustnessStudy(models={"proposed": {"kind": "pipeline"}})
    verdict = study._diffusion_verdict(_robustness_frame([("proposed", "clean", "none", 0.9)]))

    assert verdict["answered"] is False
    assert "preprocess" in verdict["reason"]


def test_diffusion_question_is_answered_per_noise_level():
    """The answer is per severity: denoising can help at high noise and hurt at low."""
    study = RobustnessStudy(
        models={
            "plain": {"kind": "pipeline", "preprocess": None},
            "diffused": {"kind": "pipeline", "preprocess": "diffusion_i10_k15"},
        }
    )
    results = _robustness_frame(
        [
            ("plain", "clean", "none", 0.90),
            ("diffused", "clean", "none", 0.88),
            ("plain", "gaussian_noise", "sigma=10", 0.80),
            ("diffused", "gaussian_noise", "sigma=10", 0.75),
            ("plain", "gaussian_noise", "sigma=40", 0.40),
            ("diffused", "gaussian_noise", "sigma=40", 0.60),
        ]
    )
    verdict = study._diffusion_verdict(results)

    assert verdict["answered"] is True
    assert verdict["helps_at_severities"] == ["sigma=40"]
    assert "some noise levels only" in verdict["verdict"]


def test_robustness_requires_at_least_one_model():
    """Step 18 mandates a baseline comparison; an empty config must fail."""
    study = RobustnessStudy(models={})
    with pytest.raises(ValueError, match="analysis.models is empty"):
        study.compute(datamodule=None)
