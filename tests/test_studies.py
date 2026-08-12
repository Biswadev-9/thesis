"""Tests for the Step 6 and Step 8 selection studies and the proxy datamodule."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis.imbalance_study import COMBINED_ARMS, ImbalanceStudy, build_loss
from src.analysis.preprocessing_study import PreprocessingStudy
from src.analysis.sweep_utils import classification_summary
from src.data.bt_mri_proxy_datamodule import BTMRIProxyDataModule
from src.data.components.preprocessing import AnisotropicDiffusion, CLAHE
from src.models.components.losses import CrossEntropyLoss, FocalLoss, LegacyFocalLoss
from tests.helpers.synthetic_dataset import make_synthetic_dataset

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No-tumor"]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """Build a synthetic data directory with its split table.

    :param tmp_path_factory: pytest temporary directory factory.
    :return: Path usable as a datamodule ``data_dir``.
    """
    from src.data.components.split_builder import build_split

    root = tmp_path_factory.mktemp("proxy_data")
    make_synthetic_dataset(root / "raw" / "bt_mri", per_class_train=20, per_class_test=8)
    build_split(root / "raw" / "bt_mri", root / "splits" / "dataset_split.csv")
    return root


# ------------------------------------------------------------- proxy datamodule


def test_proxy_subsets_are_balanced_and_capped(data_dir):
    """Every candidate must see the same balanced subset for the comparison to be fair."""
    datamodule = BTMRIProxyDataModule(
        data_dir=str(data_dir), per_class_train=5, per_class_val=2, image_size=32
    )
    datamodule.prepare_data()
    datamodule.setup()

    assert len(datamodule.data_train) == 5 * 4
    assert len(datamodule.data_val) == 2 * 4

    counts = pd.Series(datamodule.train_labels).value_counts()
    assert set(counts.index) == {0, 1, 2, 3}
    assert counts.nunique() == 1, "proxy training subset is not balanced across classes"


def test_proxy_subset_is_deterministic(data_dir):
    """Same seed, same subset - otherwise candidates are compared on different data."""
    def labels_for(seed: int) -> list:
        datamodule = BTMRIProxyDataModule(
            data_dir=str(data_dir), per_class_train=5, per_class_val=2, image_size=32, seed=seed
        )
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule.train_labels

    assert labels_for(42) == labels_for(42)


def test_proxy_validation_comes_from_the_validation_split(data_dir):
    """A selection study must never sample its validation data from training rows."""
    datamodule = BTMRIProxyDataModule(
        data_dir=str(data_dir), per_class_train=5, per_class_val=2, image_size=32
    )
    datamodule.prepare_data()
    datamodule.setup()

    val_paths = set(datamodule.data_val.dataset.df["rel_path"])
    train_paths = set(datamodule.data_train.dataset.df["rel_path"])
    assert not (val_paths & train_paths)


def test_proxy_never_exposes_the_internal_test_split(data_dir):
    """Step 16 requires the test set unseen until the final model is evaluated once."""
    datamodule = BTMRIProxyDataModule(
        data_dir=str(data_dir), per_class_train=5, per_class_val=2, image_size=32
    )
    datamodule.prepare_data()
    datamodule.setup()

    test_batch = next(iter(datamodule.test_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))
    assert torch.allclose(test_batch[0], val_batch[0])


def test_proxy_applies_preprocessing_on_the_fly(data_dir):
    """Step 6 sweeps filter in memory; only the winning recipe is cached to disk."""
    def first_image(preprocess) -> torch.Tensor:
        datamodule = BTMRIProxyDataModule(
            data_dir=str(data_dir),
            per_class_train=2,
            per_class_val=1,
            image_size=32,
            preprocess=preprocess,
        )
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule.data_train[0][0]

    assert not torch.allclose(first_image(None), first_image(CLAHE()))


def test_proxy_requires_an_existing_split(tmp_path):
    """It must sample the pipeline's split, never invent its own."""
    datamodule = BTMRIProxyDataModule(data_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="step04_audit"):
        datamodule.prepare_data()


def test_proxy_class_weights_come_from_the_subset(data_dir):
    """Weights must describe the data actually trained on."""
    datamodule = BTMRIProxyDataModule(
        data_dir=str(data_dir), per_class_train=5, per_class_val=2, image_size=32
    )
    datamodule.prepare_data()
    datamodule.setup()

    weights = datamodule.class_weights
    assert weights.shape == (4,)
    # A balanced subset yields all-ones weights.
    assert torch.allclose(weights, torch.ones(4), atol=1e-5)


# ------------------------------------------------------------------- Step 6 study


def test_candidate_list_covers_the_specified_grid_and_comparators():
    """Step 6 names both the iteration grid and the methods to compare against."""
    study = PreprocessingStudy(
        diffusion_iterations=[5, 10, 15, 20],
        diffusion_kappas=[15.0, 30.0],
        comparators=["conventional", "wiener", "gamma", "clahe", "log"],
    )
    recipes = study.candidate_recipes()

    assert len(recipes) == 5 + 8
    assert len(set(recipes)) == len(recipes)
    for required in ("conventional", "wiener", "gamma", "clahe", "log"):
        assert required in recipes
    for iterations in (5, 10, 15, 20):
        assert f"diffusion_i{iterations}_k15" in recipes


def test_identity_recipes_score_one_for_edge_preservation():
    """A filter that changes nothing cannot degrade edges; anchors the metric."""
    study = PreprocessingStudy()
    assert study._edge_score([], None, "conventional") == 1.0


# ------------------------------------------------------------------- Step 8 study


@pytest.mark.parametrize(
    "name,expected,weighted",
    [
        ("plain_ce", CrossEntropyLoss, False),
        ("weighted_ce", CrossEntropyLoss, True),
        ("focal", FocalLoss, True),
        ("focal_legacy", LegacyFocalLoss, True),
    ],
)
def test_strategy_losses_resolve(name, expected, weighted):
    """Each Step 8 arm must map to the loss it claims to test."""
    loss = build_loss(name)
    assert isinstance(loss, expected)
    assert loss.use_class_weights is weighted


def test_unknown_strategy_loss_is_rejected():
    """A typo must not silently downgrade an arm to plain cross-entropy."""
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss("dice")


def _results(scores: dict) -> pd.DataFrame:
    """Build a ranked results table.

    :param scores: Mapping of strategy name to either a macro-F1 float, or a
        ``(macro_f1, min_class_recall)`` pair.
    :return: Table sorted by macro-F1 descending.
    """
    rows = []
    for strategy, value in scores.items():
        macro_f1, min_recall = value if isinstance(value, tuple) else (value, value)
        rows.append(
            {"strategy": strategy, "macro_f1": macro_f1, "min_class_recall": min_recall}
        )
    return (
        pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    )


def test_individual_winner_is_selected_directly():
    """The ordinary case: the best single strategy wins."""
    selected, rationale = ImbalanceStudy._select(
        _results({"weighted_sampler": 0.80, "class_weighting": 0.75, "baseline": 0.70})
    )
    assert selected == "weighted_sampler"
    assert "highest validation macro-F1" in rationale


def test_combined_strategy_is_rejected_unless_it_beats_every_component():
    """Step 8: "Use more than one strategy only when ablation confirms benefit."

    This is the exact situation the reference notebook hit - the combined arm led on one
    metric but did not beat both components, and combining was correctly not adopted.
    """
    selected, rationale = ImbalanceStudy._select(
        _results(
            {
                # Leads on macro-F1 but collapses a class, exactly as the notebook's
                # combined arm did (Meningioma recall 0.373).
                "combined_sampler_weighting": (0.90, 0.37),
                "weighted_sampler": (0.82, 0.74),
                "class_weighting": (0.78, 0.70),
                "baseline": (0.65, 0.40),
            }
        )
    )
    assert selected == "weighted_sampler"
    assert "did not beat all of its components" in rationale


def test_combined_strategy_is_adopted_when_ablation_confirms_it():
    """The converse: when combining genuinely helps, it is selected."""
    selected, rationale = ImbalanceStudy._select(
        _results(
            {
                "combined_sampler_weighting": (0.90, 0.85),
                "weighted_sampler": (0.82, 0.74),
                "class_weighting": (0.78, 0.70),
                "baseline": (0.65, 0.40),
            }
        )
    )
    assert selected == "combined_sampler_weighting"
    assert "confirmed beneficial" in rationale


def test_combined_arms_reference_real_strategies():
    """The selection rule is only meaningful if its component names exist."""
    from src.analysis.imbalance_study import STRATEGIES

    for combined, components in COMBINED_ARMS.items():
        assert combined in STRATEGIES
        for component in components:
            assert component in STRATEGIES


def test_focal_correction_delta_is_reported_when_both_arms_run():
    """Quantifying what F6 changed is the point of carrying the legacy arm."""
    delta = ImbalanceStudy._focal_correction_delta(
        _results({"focal_loss": 0.80, "focal_loss_legacy": 0.74})
    )
    assert delta == pytest.approx(0.06)

    assert ImbalanceStudy._focal_correction_delta(_results({"focal_loss": 0.80})) is None


# ------------------------------------------------------------------- shared metrics


def test_classification_summary_reports_the_specified_metrics():
    """Step 8 judges on macro-F1, balanced accuracy and class-wise recall."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    summary = classification_summary(y_true, y_pred, CLASS_NAMES)
    assert summary["macro_f1"] == pytest.approx(1.0)
    assert summary["balanced_accuracy"] == pytest.approx(1.0)
    assert summary["min_class_recall"] == pytest.approx(1.0)
    for name in CLASS_NAMES:
        assert summary[f"recall_{name}"] == pytest.approx(1.0)


def test_classification_summary_exposes_class_collapse():
    """A collapsed class is the failure mode class-wise recall exists to catch."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 0, 0, 0, 2, 2, 3, 3])  # Meningioma never predicted

    summary = classification_summary(y_true, y_pred, CLASS_NAMES)
    assert summary["recall_Meningioma"] == 0.0
    assert summary["min_class_recall"] == 0.0
    assert summary["macro_f1"] < 1.0
