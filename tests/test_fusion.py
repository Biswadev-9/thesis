"""Tests for the Step 13 fusion strategies and the Step 14 final classifier."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis.loss_selection import REFERENCE_LOSSES, SELECTABLE_LOSSES, LossSelection
from src.data.bt_mri_feature_datamodule import BTMRIFeatureDataModule
from src.models.components.fusion import (
    BRANCH_NAMES,
    FUSION_STRATEGIES,
    ConcatFusion,
    FinalClassifier,
    FusedFeatureClassifier,
    GatedFusion,
    SEFusion,
)
from src.utils.metrics import (
    expected_calibration_error,
    multiclass_brier_score,
    specificity_per_class,
)

CLASSICAL_DIM, SPATIAL_DIM, QUANTUM_DIM = 1280, 32, 4
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No-tumor"]


@pytest.fixture
def features():
    """:return: A batch of ``(classical, spatial, quantum)`` features."""
    torch.manual_seed(0)
    return (
        torch.randn(6, CLASSICAL_DIM),
        torch.randn(6, SPATIAL_DIM),
        torch.randn(6, QUANTUM_DIM),
    )


# ------------------------------------------------------------------- strategies


@pytest.mark.parametrize("name", sorted(FUSION_STRATEGIES))
def test_every_strategy_classifies_and_exposes_the_fused_vector(name, features):
    """All three must satisfy the same contract to be interchangeable in the sweep."""
    # eval mode: these heads carry dropout, so two forward passes only agree once it is
    # disabled. The contract is that forward() *is* extract()["logits"], not that a
    # stochastic head is deterministic.
    net = FUSION_STRATEGIES[name](CLASSICAL_DIM, SPATIAL_DIM, QUANTUM_DIM, num_classes=4).eval()

    with torch.no_grad():
        outputs = net.extract(*features)

        assert outputs["logits"].shape == (6, 4)
        assert "fused" in outputs
        assert torch.isfinite(outputs["logits"]).all()
        assert torch.allclose(net(*features), outputs["logits"])


def test_projection_prevents_the_classical_branch_dominating_by_width(features):
    """Raw concatenation would make the classical branch 97 % of the input.

    Projecting first is what makes the branch comparison about information rather than
    dimensionality.
    """
    net = ConcatFusion(CLASSICAL_DIM, SPATIAL_DIM, QUANTUM_DIM, proj_dim=64)
    classical, spatial, quantum = net.projections(*features)

    assert classical.shape == spatial.shape == quantum.shape == (6, 64)
    assert net.extract(*features)["fused"].shape == (6, 192)


def test_gated_fusion_weights_are_a_distribution_over_branches(features):
    """Step 13: "report ... learned fusion weights" - they must be interpretable."""
    net = GatedFusion(CLASSICAL_DIM, SPATIAL_DIM, QUANTUM_DIM, num_classes=4)
    weights = net.extract(*features)["branch_weights"]

    assert weights.shape == (6, len(BRANCH_NAMES))
    assert torch.allclose(weights.sum(dim=1), torch.ones(6), atol=1e-5)
    assert (weights >= 0).all()


def test_gated_fusion_sums_rather_than_concatenates(features):
    """Its classifier sees one branch's width, which is worth knowing when comparing heads."""
    net = GatedFusion(CLASSICAL_DIM, SPATIAL_DIM, QUANTUM_DIM, proj_dim=64)
    assert net.extract(*features)["fused"].shape == (6, 64)


def test_se_fusion_weights_are_per_channel_not_per_branch(features):
    """SE attention cannot answer "which branch", which is why gated fusion also exists."""
    net = SEFusion(CLASSICAL_DIM, SPATIAL_DIM, QUANTUM_DIM, proj_dim=64)
    weights = net.extract(*features)["channel_weights"]

    assert weights.shape == (6, 192)
    assert ((weights >= 0) & (weights <= 1)).all(), "sigmoid gates must lie in [0, 1]"


def test_zeroing_a_branch_changes_the_prediction(features):
    """The ablation relies on a zeroed branch actually changing the output."""
    net = ConcatFusion(CLASSICAL_DIM, SPATIAL_DIM, QUANTUM_DIM).eval()
    classical, spatial, quantum = features

    with torch.no_grad():
        full = net(classical, spatial, quantum)
        without_quantum = net(classical, spatial, torch.zeros_like(quantum))

    assert not torch.allclose(full, without_quantum)


def test_gradients_reach_every_branch_projection(features):
    """A branch whose projection never learns would look uninformative for the wrong reason."""
    net = GatedFusion(CLASSICAL_DIM, SPATIAL_DIM, QUANTUM_DIM)
    net(*features).sum().backward()

    for branch in BRANCH_NAMES:
        weight = getattr(net.projections, branch)[0].weight
        assert weight.grad is not None, f"{branch} projection received no gradient"
        assert weight.grad.abs().sum() > 0


# -------------------------------------------------------------- final classifier


def test_final_classifier_returns_logits_and_probabilities():
    """Step 14 asks for a softmax output; the loss still needs raw logits."""
    head = FinalClassifier(input_dim=192, hidden_dims=(128, 64), num_classes=4).eval()
    fused = torch.randn(5, 192)

    logits = head(fused)
    assert logits.shape == (5, 4)

    probabilities = head.predict_proba(fused)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(5), atol=1e-5)
    assert (probabilities >= 0).all()


def test_final_classifier_uses_normalisation_and_dropout():
    """Step 14: "fully connected layers with normalization, dropout"."""
    head = FinalClassifier(input_dim=192, hidden_dims=(128, 64))
    kinds = {type(layer).__name__ for layer in head.hidden}

    assert "BatchNorm1d" in kinds
    assert "Dropout" in kinds
    assert "GELU" in kinds


def test_fused_feature_classifier_is_the_step14_deliverable(features):
    """Projections plus the deeper head, ready for Step 15's protocol runs."""
    net = FusedFeatureClassifier(CLASSICAL_DIM, SPATIAL_DIM, QUANTUM_DIM, num_classes=4).eval()
    outputs = net.extract(*features)

    assert outputs["logits"].shape == (6, 4)
    assert outputs["fused"].shape == (6, 192)

    probabilities = net.predict_proba(*features)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(6), atol=1e-5)


# ------------------------------------------------------------ feature datamodule


def write_cache(directory: Path, n: int = 40) -> None:
    """Write a synthetic feature cache.

    :param directory: Cache directory to create.
    :param n: Samples per split.
    """
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(0)

    for split in ("train", "val", "test"):
        torch.save(
            {
                "classical": torch.randn(n, CLASSICAL_DIM, generator=generator),
                "spatial": torch.randn(n, SPATIAL_DIM, generator=generator),
                "quantum": torch.randn(n, QUANTUM_DIM, generator=generator),
                "quantum_weights": torch.rand(n, 5, generator=generator),
                "labels": torch.arange(n) % 4,
            },
            directory / f"{split}.pt",
        )


def test_feature_datamodule_reads_dimensions_from_the_cache(tmp_path):
    """Head sizes must follow the cache, not a hard-coded guess."""
    write_cache(tmp_path / "features" / "default")
    datamodule = BTMRIFeatureDataModule(data_dir=str(tmp_path), tag="default", batch_size=8)
    datamodule.prepare_data()
    datamodule.setup()

    assert datamodule.classical_dim == CLASSICAL_DIM
    assert datamodule.spatial_dim == SPATIAL_DIM
    assert datamodule.quantum_dim == QUANTUM_DIM

    batch = next(iter(datamodule.train_dataloader()))
    assert len(batch) == 4
    assert batch[0].shape[1] == CLASSICAL_DIM


def test_zero_branches_zeroes_only_the_named_branch(tmp_path):
    """The contribution ablation depends on this being exact."""
    write_cache(tmp_path / "features" / "default")
    datamodule = BTMRIFeatureDataModule(
        data_dir=str(tmp_path), tag="default", zero_branches=["quantum"], batch_size=8
    )
    datamodule.prepare_data()
    datamodule.setup()

    classical, spatial, quantum, _ = next(iter(datamodule.val_dataloader()))
    assert torch.all(quantum == 0)
    assert not torch.all(classical == 0)
    assert not torch.all(spatial == 0)


def test_missing_feature_cache_says_how_to_build_it(tmp_path):
    """Forgetting the extraction step is the most likely Phase 5 mistake."""
    datamodule = BTMRIFeatureDataModule(data_dir=str(tmp_path), tag="default")
    with pytest.raises(FileNotFoundError, match="extract_features"):
        datamodule.prepare_data()


def test_incomplete_feature_cache_is_reported(tmp_path):
    """A half-written cache must fail loudly rather than train on two splits."""
    directory = tmp_path / "features" / "partial"
    write_cache(directory)
    (directory / "test.pt").unlink()

    datamodule = BTMRIFeatureDataModule(data_dir=str(tmp_path), tag="partial")
    with pytest.raises(FileNotFoundError, match="missing splits"):
        datamodule.prepare_data()


# ------------------------------------------------------------- loss selection


def _loss_results(rows: dict) -> pd.DataFrame:
    """Build a loss-selection results table.

    :param rows: Mapping of loss name to ``(macro_f1, balanced_accuracy, ece)``.
    :return: The table.
    """
    return pd.DataFrame(
        [
            {
                "loss": name,
                "selectable": name in SELECTABLE_LOSSES,
                "macro_f1": macro_f1,
                "balanced_accuracy": balanced,
                "ece": ece,
            }
            for name, (macro_f1, balanced, ece) in rows.items()
        ]
    )


def test_plain_cross_entropy_cannot_be_selected():
    """Step 14 permits only weighted CE or focal as the final loss.

    The reference notebook selected plain CE, which this rule forbids outright.
    """
    selected, _ = LossSelection._select(
        _loss_results(
            {
                "plain_ce": (0.99, 0.99, 0.01),  # best on every metric
                "weighted_ce": (0.95, 0.95, 0.05),
                "focal": (0.94, 0.94, 0.06),
            }
        )
    )
    assert selected == "weighted_ce"
    assert "plain_ce" in REFERENCE_LOSSES


def test_selection_uses_macro_f1_first():
    """The primary criterion, per the specification."""
    selected, rationale = LossSelection._select(
        _loss_results({"weighted_ce": (0.90, 0.80, 0.10), "focal": (0.95, 0.70, 0.20)})
    )
    assert selected == "focal"
    assert "macro_f1" in rationale


def test_ties_fall_through_to_balanced_accuracy():
    """Second criterion when macro-F1 is identical."""
    selected, rationale = LossSelection._select(
        _loss_results({"weighted_ce": (0.9897, 0.95, 0.10), "focal": (0.9897, 0.91, 0.05)})
    )
    assert selected == "weighted_ce"
    assert "balanced_accuracy" in rationale


def test_full_ties_are_broken_by_calibration_not_by_test_data():
    """The notebook's exact situation - a genuine tie - resolved without touching test.

    It broke its three-way tie by comparing test macro-F1 and then reported that same
    test set as the result. Calibration is used here instead.
    """
    selected, rationale = LossSelection._select(
        _loss_results({"weighted_ce": (0.9897, 0.9897, 0.08), "focal": (0.9897, 0.9897, 0.03)})
    )
    assert selected == "focal", "lower ECE should win a full tie"
    assert "ece" in rationale
    assert "test" not in rationale.lower() or "not consulted" in rationale.lower()


def test_selection_requires_at_least_one_permitted_candidate():
    """Running only the reference loss must fail rather than silently select it."""
    with pytest.raises(ValueError, match="only class-weighted"):
        LossSelection._select(_loss_results({"plain_ce": (0.99, 0.99, 0.01)}))


# --------------------------------------------------------------------- metrics


def test_perfect_confidence_on_correct_predictions_is_perfectly_calibrated():
    """Anchors the ECE scale at 0."""
    y_true = np.array([0, 1, 2, 3])
    y_prob = np.eye(4)[y_true]

    ece, bins = expected_calibration_error(y_true, y_prob)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert len(bins) == 1


def test_confident_and_wrong_scores_the_worst_possible_calibration():
    """The failure the notebook found: probability 1.000 on a wrong class."""
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.eye(4)[[1, 1, 1, 1]]

    ece, _ = expected_calibration_error(y_true, y_prob)
    assert ece == pytest.approx(1.0, abs=1e-9)


def test_brier_score_rewards_honest_probabilities():
    """A proper scoring rule: hedging beats being confidently wrong."""
    y_true = np.array([0, 0])
    confident_wrong = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    hedged = np.array([[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]])

    assert multiclass_brier_score(y_true, hedged, 4) < multiclass_brier_score(
        y_true, confident_wrong, 4
    )
    assert multiclass_brier_score(y_true, np.eye(4)[y_true], 4) == pytest.approx(0.0)


def test_specificity_is_one_when_no_false_positives():
    """Step 16 reports specificity alongside sensitivity; sklearn has no multiclass form."""
    confusion = np.diag([10, 10, 10, 10])
    values = specificity_per_class(confusion, CLASS_NAMES)

    assert set(values) == set(CLASS_NAMES)
    assert all(value == pytest.approx(1.0) for value in values.values())


def test_specificity_drops_when_a_class_absorbs_false_positives():
    """A class predicted too eagerly should show reduced specificity."""
    confusion = np.array(
        [[5, 5, 0, 0], [0, 10, 0, 0], [0, 5, 5, 0], [0, 0, 0, 10]]
    )
    values = specificity_per_class(confusion, CLASS_NAMES)

    assert values["Meningioma"] < 1.0
    assert values["No-tumor"] == pytest.approx(1.0)
