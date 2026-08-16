"""Step 24: the receptive-field strategy ablation, tested without training anything.

Step 24 asks one question - does *spatially adaptive selection among* several receptive
fields beat committing to one, and beat combining them without a gate? Every one of the
five conditions already exists as a Step 11 arm, so this file guards the experiment rather
than the architecture. Four ways it could quietly stop answering that question:

* **Inherited settings.** ``configs/experiment/step11_arm_ablation.yaml`` declares
  ``augment: true`` and ``use_weighted_sampler: true``, and the pipeline then overrides both
  with Step 8's selection. An arm run that way is not comparable with another run under a
  different selection. Step 24 pins all five conditions explicitly and asserts it.
* **A gate leaking into a control.** If ``MULTISCALE_NO_GATE`` acquired a gate, or the
  fixed conditions acquired extra paths, H24 would compare the adaptive model with itself.
* **A capacity claim that is not true.** The five conditions are *not* parameter-matched.
  The adaptive model has 65% more parameters than the fixed 3x3 one - and 2.6% *fewer* than
  the ungated multi-scale model it is primarily compared against. Both facts must be
  recorded, not assumed.
* **Family inflation.** Step 23 protects a registered four-hypothesis family. Step 24
  declares its own single formal hypothesis; if it ever appended to Phase 8's family, four
  corrected tests would silently become five.

Nothing here trains or loads a checkpoint: predictions are injected.
"""

import json

import numpy as np
import pytest
import torch

from src.analysis.ablation_rows import PROTOCOL_SEEDS
from src.analysis.receptive_field_rows import (
    CONDITIONS,
    PINNED,
    PRIMARY_COMPARISON,
    SUPPORTING_COMPARISONS,
    ReceptiveFieldContext,
    condition_overrides,
    get_condition,
)
from src.analysis.receptive_field_study import (
    CONDITION_IDS,
    TABLE_COLUMNS,
    ReceptiveFieldStudy,
)

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No-tumor"]

#: A deliberately neutral placeholder. Structural tests need *a* recipe to compose with,
#: and naming a real candidate here would read as an assumed winner. Which recipe Step 24
#: actually uses is decided by Step 6's confirmation and is exercised separately, across
#: several unrelated recipes, so no test encodes a preprocessing decision.
CONTEXT = ReceptiveFieldContext(recipe="recipe_under_confirmation")


def compose_condition(condition_id, seed=42, context=CONTEXT):
    """Compose the training config a condition would actually run under.

    :param condition_id: Condition identifier.
    :param seed: Protocol seed.
    :param context: Resolved preprocessing choice.
    :return: The composed DictConfig.
    """
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    overrides = condition_overrides(get_condition(condition_id), seed=seed, context=context)
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=list(overrides))
    GlobalHydra.instance().clear()
    return cfg


# ------------------------------------------------------------- the five conditions


def test_exactly_five_conditions_are_registered():
    """The ladder is fixed: three fixed kernels, one ungated multi-scale, one adaptive."""
    assert [c.condition_id for c in CONDITIONS] == [
        "FIXED_3X3", "FIXED_5X5", "FIXED_DILATED_3X3", "MULTISCALE_NO_GATE",
        "ADAPTIVE_MULTISCALE",
    ]
    assert list(CONDITION_IDS) == [c.condition_id for c in CONDITIONS]
    assert len(CONDITIONS) == len({c.condition_id for c in CONDITIONS}) == 5


def test_every_condition_reuses_an_existing_step11_arm():
    """No new architecture: Step 11 already implements all five."""
    from src.models.components.multiscale import ARMS

    for condition in CONDITIONS:
        assert condition.arm in ARMS, f"{condition.condition_id} names an unknown arm"

    assert [c.arm for c in CONDITIONS] == [
        "arm1_fixed_3x3", "arm2_fixed_5x5", "arm3_fixed_dilated",
        "arm4_concat_nogate", "arm6_spatial_gate",
    ]


def test_every_condition_is_resolvable_and_builds():
    """:return: None."""
    from src.models.components.multiscale import MultiscaleClassifier

    for condition in CONDITIONS:
        model = MultiscaleClassifier.from_arm(condition.arm)
        assert model.branch.variant == condition.variant


# ------------------------------------------------------- the architectures themselves


def test_fixed_3x3_uses_a_single_3x3_kernel():
    """:return: None."""
    from src.models.components.multiscale import MultiscaleClassifier

    branch = MultiscaleClassifier.from_arm(get_condition("FIXED_3X3").arm).branch
    conv = branch.path[0]

    assert conv.kernel_size == (3, 3)
    assert conv.dilation == (1, 1)
    assert not hasattr(branch, "gate")
    assert not hasattr(branch, "path_medium")


def test_fixed_5x5_uses_a_single_5x5_kernel():
    """:return: None."""
    from src.models.components.multiscale import MultiscaleClassifier

    conv = MultiscaleClassifier.from_arm(get_condition("FIXED_5X5").arm).branch.path[0]

    assert conv.kernel_size == (5, 5)
    assert conv.dilation == (1, 1)


def test_fixed_dilated_uses_a_dilated_3x3_not_a_literal_7x7():
    """The specification's "7x7" is reached by dilation here, at 3x3 parameter cost.

    Silently substituting a literal 7x7 would change the parameter budget of the condition
    and break its comparability with the same path inside the multi-scale conditions.
    """
    from src.models.components.multiscale import MultiscaleClassifier

    conv = MultiscaleClassifier.from_arm(get_condition("FIXED_DILATED_3X3").arm).branch.path[0]

    assert conv.kernel_size == (3, 3), "must remain a 3x3 kernel"
    assert conv.dilation == (3, 3)
    effective = conv.dilation[0] * (conv.kernel_size[0] - 1) + 1
    assert effective == 7


def test_ungated_multiscale_has_all_three_paths_and_no_gate():
    """The control for H24: same receptive fields, no adaptive selection among them."""
    from src.models.components.multiscale import MultiscaleClassifier

    branch = MultiscaleClassifier.from_arm(get_condition("MULTISCALE_NO_GATE").arm).branch

    assert branch.path_fine[0].kernel_size == (3, 3)
    assert branch.path_medium[0].kernel_size == (5, 5)
    assert branch.path_broad[0].dilation == (3, 3)

    assert not hasattr(branch, "gate")
    assert not any("gate" in name for name, _ in branch.named_modules())


def test_ungated_multiscale_fuses_by_concatenation_and_a_learned_projection():
    """Not equal-weight averaging - and the difference matters for how it is described.

    The arm concatenates the three paths and learns a 1x1 projection back to the shared
    width. Calling that "equal-weight fusion" in the write-up would misdescribe a learned,
    input-independent mixer as a fixed one.
    """
    from src.models.components.multiscale import MultiscaleClassifier

    branch = MultiscaleClassifier.from_arm(get_condition("MULTISCALE_NO_GATE").arm).branch

    assert isinstance(branch.project, torch.nn.Conv2d)
    assert branch.project.kernel_size == (1, 1)
    assert branch.project.in_channels == 3 * branch.channels
    assert branch.project.out_channels == branch.channels

    condition = get_condition("MULTISCALE_NO_GATE")
    assert "concat" in condition.fusion.lower()
    assert "equal" not in condition.fusion.lower()


def test_adaptive_multiscale_has_the_spatial_gate():
    """:return: None."""
    from src.models.components.multiscale import MultiscaleClassifier, SpatialMultiScaleGate

    branch = MultiscaleClassifier.from_arm(get_condition("ADAPTIVE_MULTISCALE").arm).branch

    assert isinstance(branch.gate, SpatialMultiScaleGate)
    assert branch.variant == "spatial_gate"


def test_the_adaptive_gate_emits_a_normalised_distribution_over_the_three_paths():
    """Non-negative, summing to one along the path axis, at every spatial location."""
    from src.models.components.multiscale import MultiscaleClassifier

    model = MultiscaleClassifier.from_arm(get_condition("ADAPTIVE_MULTISCALE").arm).eval()
    with torch.no_grad():
        outputs = model.extract(torch.randn(2, 3, 64, 64))

    weights = outputs["gate_maps"]
    assert weights.shape[1] == 3
    assert (weights >= 0).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones_like(weights.sum(dim=1)), atol=1e-5)


def test_the_gate_weights_depend_on_the_input():
    """"Adaptive" has to mean input-dependent, not merely learned-once.

    Step 11's own suite checks the map varies across space within one input; nothing there
    checks it varies BETWEEN inputs, which is the property the word "adaptive" claims.
    """
    from src.models.components.multiscale import MultiscaleClassifier

    torch.manual_seed(0)
    model = MultiscaleClassifier.from_arm(get_condition("ADAPTIVE_MULTISCALE").arm).eval()

    with torch.no_grad():
        first = model.extract(torch.randn(1, 3, 64, 64))["gate_maps"]
        second = model.extract(torch.randn(1, 3, 64, 64) * 4.0)["gate_maps"]

    assert (first - second).abs().max() > 1e-4, "gate ignored the input"


def test_only_the_adaptive_condition_is_marked_adaptive():
    """A fixed-kernel condition that claimed adaptivity would invert the whole ladder."""
    adaptive = [c.condition_id for c in CONDITIONS if c.adaptive]

    assert adaptive == ["ADAPTIVE_MULTISCALE"]

    from src.models.components.multiscale import MultiscaleClassifier

    for condition in CONDITIONS:
        model = MultiscaleClassifier.from_arm(condition.arm).eval()
        with torch.no_grad():
            outputs = model.extract(torch.randn(2, 3, 64, 64))
        assert ("gate_maps" in outputs) == condition.adaptive


# --------------------------------------------------------------- shared everything


def test_every_condition_produces_the_same_feature_width():
    """Different widths would mean different classifier capacity, not different kernels."""
    from src.models.components.multiscale import MultiscaleClassifier

    widths = set()
    for condition in CONDITIONS:
        model = MultiscaleClassifier.from_arm(condition.arm).eval()
        with torch.no_grad():
            widths.add(model.extract(torch.randn(2, 3, 64, 64))["features"].shape[1])

    assert widths == {32}


def test_every_condition_has_an_identical_classifier_head():
    """:return: None."""
    from src.models.components.multiscale import MultiscaleClassifier

    heads = {
        (m.classifier.in_features, m.classifier.out_features, type(m.classifier).__name__)
        for m in (MultiscaleClassifier.from_arm(c.arm) for c in CONDITIONS)
    }

    assert heads == {(32, 4, "Linear")}


def test_every_condition_shares_the_same_stem():
    """The stem is upstream of the receptive-field choice and must not vary with it."""
    from src.models.components.multiscale import MultiscaleClassifier

    shapes = {
        tuple(tuple(p.shape) for p in MultiscaleClassifier.from_arm(c.arm).branch.stem.parameters())
        for c in CONDITIONS
    }

    assert len(shapes) == 1, "the shared stem differs between conditions"


@pytest.mark.parametrize("condition_id", [c.condition_id for c in CONDITIONS])
def test_every_condition_pins_the_same_data_handling(condition_id):
    """Step 11's experiment config sets augment/sampler true and the pipeline overrides them.

    Inheriting that would make two conditions incomparable for a reason unrelated to
    receptive fields, so Step 24 pins all of it and asserts the composed result.
    """
    cfg = compose_condition(condition_id)

    assert cfg.data.recipe == CONTEXT.recipe
    assert cfg.data.normalize == PINNED["normalize"]
    assert cfg.data.augment is PINNED["augment"]
    assert cfg.data.use_weighted_sampler is PINNED["use_weighted_sampler"]
    assert cfg.model.criterion.use_class_weights is False


@pytest.mark.parametrize("condition_id", [c.condition_id for c in CONDITIONS])
def test_every_condition_pins_its_settings_explicitly(condition_id):
    """Emitted as overrides, so nothing is left to recipe_override/imbalance_overrides."""
    overrides = condition_overrides(get_condition(condition_id), seed=42, context=CONTEXT)

    for key in ("data.recipe=", "data.normalize=", "data.augment=",
                "data.use_weighted_sampler=", "loss@model.criterion="):
        assert sum(o.startswith(key) for o in overrides) == 1, f"{condition_id} lacks {key}"


@pytest.mark.parametrize("condition_id", [c.condition_id for c in CONDITIONS])
def test_every_condition_composes_the_fixed_protocol(condition_id):
    """Read from Step 15's own composition rather than retyped."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        reference = compose(config_name="train.yaml",
                            overrides=["experiment=step15_final_protocol"])
    GlobalHydra.instance().clear()

    cfg = compose_condition(condition_id)

    assert cfg.trainer.max_epochs == reference.trainer.max_epochs
    assert cfg.trainer.min_epochs == reference.trainer.min_epochs
    assert cfg.callbacks.early_stopping.patience == reference.callbacks.early_stopping.patience
    assert cfg.callbacks.early_stopping.monitor == reference.callbacks.early_stopping.monitor
    assert cfg.callbacks.model_checkpoint.monitor == reference.callbacks.model_checkpoint.monitor
    assert cfg.callbacks.model_checkpoint.save_top_k == reference.callbacks.model_checkpoint.save_top_k
    assert cfg.data.batch_size == reference.data.batch_size
    assert cfg.model.optimizer.lr == reference.model.optimizer.lr
    assert cfg.model.optimizer.weight_decay == reference.model.optimizer.weight_decay
    assert cfg.optimized_metric == reference.optimized_metric


def test_the_seed_set_is_the_studys_own():
    """Not a new seed policy: the same three every other stage uses."""
    from src.analysis.receptive_field_rows import SEEDS

    assert SEEDS == PROTOCOL_SEEDS == (42, 123, 7)


@pytest.mark.parametrize("condition_id", [c.condition_id for c in CONDITIONS])
def test_every_condition_runs_every_seed(condition_id):
    """:param condition_id: Condition under test."""
    for seed in PROTOCOL_SEEDS:
        assert compose_condition(condition_id, seed=seed).seed == seed


def test_all_conditions_share_one_preprocessing_recipe():
    """The ladder varies receptive fields; nothing else may vary with it."""
    recipes = {compose_condition(c.condition_id).data.recipe for c in CONDITIONS}

    assert len(recipes) == 1


# ------------------------------------------------------------------- the hypothesis


def test_the_single_formal_hypothesis_is_adaptive_against_ungated():
    """Both contain the same three receptive fields, so the gate is the only difference."""
    assert PRIMARY_COMPARISON["id"] == "H24"
    assert PRIMARY_COMPARISON["condition_a"] == "ADAPTIVE_MULTISCALE"
    assert PRIMARY_COMPARISON["condition_b"] == "MULTISCALE_NO_GATE"


def test_the_fixed_kernel_comparisons_are_supporting_not_formal():
    """They change capacity and receptive field together, so they cannot isolate gating."""
    assert [c["id"] for c in SUPPORTING_COMPARISONS] == [
        "S24a_vs_fixed_3x3", "S24b_vs_fixed_5x5", "S24c_vs_fixed_dilated",
    ]
    assert all(c["condition_a"] == "ADAPTIVE_MULTISCALE" for c in SUPPORTING_COMPARISONS)
    assert all(c["condition_b"] != "MULTISCALE_NO_GATE" for c in SUPPORTING_COMPARISONS)


def test_step24_does_not_touch_phase8s_hypothesis_family():
    """Four corrected tests must not silently become five."""
    from src.analysis.statistical_report import PRIMARY_FAMILY

    assert [h["id"] for h in PRIMARY_FAMILY] == ["H1", "H2", "H3", "H4"]

    step24_ids = {PRIMARY_COMPARISON["id"]} | {c["id"] for c in SUPPORTING_COMPARISONS}
    assert not step24_ids & {h["id"] for h in PRIMARY_FAMILY}


def test_step24_declares_its_own_family_of_one():
    """One formal hypothesis, so Holm is the identity - stated rather than left implicit."""
    from src.analysis.receptive_field_rows import FAMILY_SIZE

    assert FAMILY_SIZE == 1


# ---------------------------------------------------------------------- the study


def synth(seed, n=200, skill=1.4):
    """Deterministic synthetic predictions.

    :param seed: Generator seed.
    :param n: Sample count.
    :param skill: Probability mass on the true class.
    :return: Prediction dict.
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, len(CLASS_NAMES), n)
    logits = rng.normal(size=(n, len(CLASS_NAMES)))
    logits[np.arange(n), y_true] += skill
    y_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return {"y_true": y_true, "y_pred": y_prob.argmax(axis=1), "y_prob": y_prob,
            "class_names": list(CLASS_NAMES)}


@pytest.fixture
def study(tmp_path):
    """A study wired to a synthetic run tree, with prediction injected.

    :param tmp_path: Per-test directory.
    :return: The study.
    """
    import zlib

    runs = tmp_path / "runs"
    for condition in CONDITIONS:
        for seed in PROTOCOL_SEEDS:
            directory = runs / condition.condition_id / f"seed_{seed}" / "checkpoints"
            directory.mkdir(parents=True)
            (directory / "epoch_004.ckpt").touch()

    analysis = ReceptiveFieldStudy(run_root=str(runs), recipe="recipe_under_confirmation",
                                   n_resamples=200)
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    # Shared labels across conditions, exactly as a shared test split gives.
    labels = synth(0)["y_true"]

    def predict(condition, seed, checkpoint):
        payload = synth(zlib.crc32(f"{condition.condition_id}:{seed}".encode()) % 2**32)
        payload["y_true"] = labels
        return payload

    analysis.predict_condition = predict
    return analysis


@pytest.fixture
def result(study):
    """:param study: The wired study.

    :return: The computed summary.
    """
    return study.compute(datamodule=None)


def test_every_condition_is_evaluated_at_every_seed(result):
    """:param result: The computed summary."""
    assert list(result["conditions"]) == list(CONDITION_IDS)

    for record in result["conditions"].values():
        assert sorted(record["per_seed"], key=int) == ["7", "42", "123"]
        assert record["across_seeds"]["macro_f1"]["n"] == 3


def test_the_primary_metric_is_the_studys_established_one(result):
    """macro-F1, as Steps 21 and 23 use. Not accuracy."""
    assert result["parameters"]["primary_metric"] == "macro_f1"
    assert result["primary"]["metric"] == "macro_f1"


def test_the_full_metric_battery_is_reported_for_every_condition(result):
    """Same code as Step 16, so Step 24's numbers are comparable with the headline."""
    expected = {"accuracy", "macro_f1", "balanced_accuracy", "macro_precision",
                "macro_recall_sensitivity", "weighted_f1", "mcc"}

    for record in result["conditions"].values():
        for metrics in record["per_seed"].values():
            assert expected <= set(metrics["overall"])
            assert len(metrics["per_class"]) == len(CLASS_NAMES)
            assert {"precision", "recall_sensitivity", "f1"} <= set(metrics["per_class"][0])
            assert metrics["confusion"]["class_names"] == CLASS_NAMES


def test_the_primary_comparison_is_formal_and_paired(result):
    """:param result: The computed summary."""
    primary = result["primary"]

    assert primary["comparison"] == "H24"
    assert primary["family"] == "primary"
    assert primary["statistical_status"] == "formal"
    assert primary["mcnemar_method"] in (
        "exact binomial", "chi-square with continuity correction", "none",
    )
    assert primary["ci_low"] is not None and primary["ci_high"] is not None
    assert primary["raw_p_value"] is not None
    assert primary["adjusted_p_value"] is not None
    assert primary["family_size"] == 1


def test_holm_over_a_family_of_one_leaves_the_p_value_unchanged(result):
    """Stated explicitly so nobody reads the adjusted value as a correction that happened."""
    primary = result["primary"]

    assert primary["adjusted_p_value"] == pytest.approx(primary["raw_p_value"])
    assert "family of one" in primary["correction_note"].lower()


def test_the_supporting_comparisons_carry_no_significance_claim(result):
    """They change capacity and receptive field together; a p-value would overclaim."""
    assert [c["comparison"] for c in result["supporting"]] == [
        "S24a_vs_fixed_3x3", "S24b_vs_fixed_5x5", "S24c_vs_fixed_dilated",
    ]

    for comparison in result["supporting"]:
        assert comparison["statistical_status"] == "descriptive"
        assert comparison["raw_p_value"] is None
        assert comparison["adjusted_p_value"] is None
        assert comparison["significant"] is None
        assert comparison["ci_low"] is not None, "a descriptive interval is still reported"


def test_the_comparison_can_return_any_of_the_three_outcomes(study, tmp_path):
    """The design must not guarantee the adaptive model wins.

    Driven with the ungated condition made deliberately stronger, the verdict has to follow
    the data rather than the label.
    """
    import zlib

    labels = synth(0)["y_true"]

    def predict(condition, seed, checkpoint):
        skill = 2.2 if condition.condition_id == "MULTISCALE_NO_GATE" else 0.8
        payload = synth(zlib.crc32(f"{condition.condition_id}:{seed}".encode()) % 2**32,
                        skill=skill)
        payload["y_true"] = labels
        return payload

    study.predict_condition = predict
    primary = study.compute(datamodule=None)["primary"]

    assert primary["observed_delta"] < 0
    assert "MULTISCALE_NO_GATE" in primary["effect_direction"]


def test_seed_spread_is_descriptive_and_wilcoxon_is_refused(result):
    """Three seeds cannot support a significance claim at any effect size."""
    seeds = result["primary"]["seeds"]

    assert seeds["role"] == "descriptive"
    assert seeds["wilcoxon"]["p_value"] is None
    assert seeds["delta_across_seeds"]["n"] == 3


def test_predictions_are_computed_here_not_harvested(result):
    """``trainer.test()`` writes test/* into every run's metrics.csv; it is never read."""
    for record in result["conditions"].values():
        for metrics in record["per_seed"].values():
            assert metrics["provenance"].startswith("computed by step24")
            assert metrics["selection"] == "val/f1_macro (training-time ModelCheckpoint)"


def test_the_study_reads_no_training_csv():
    """:return: None."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/analysis/receptive_field_study.py").read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert "read_csv" not in called


def test_a_missing_checkpoint_is_reported_not_silently_dropped(study, tmp_path):
    """:param study: The wired study.

    :param tmp_path: Per-test directory.
    """
    import shutil

    shutil.rmtree(tmp_path / "runs" / "FIXED_5X5" / "seed_7")
    summary = study.compute(datamodule=None)

    assert summary["conditions"]["FIXED_5X5"]["missing_seeds"] == [7]
    assert summary["integrity"]["complete"] is False


def test_a_missing_primary_condition_is_fatal(study, tmp_path):
    """H24 needs both sides; a one-sided family would be reported as if it were whole."""
    import shutil

    for seed in PROTOCOL_SEEDS:
        shutil.rmtree(tmp_path / "runs" / "MULTISCALE_NO_GATE" / f"seed_{seed}")

    with pytest.raises(FileNotFoundError, match="MULTISCALE_NO_GATE"):
        study.compute(datamodule=None)


def test_misaligned_predictions_are_refused(study):
    """A paired test over unaligned samples returns a confident wrong answer, not an error."""
    import zlib

    def predict(condition, seed, checkpoint):
        payload = synth(zlib.crc32(condition.condition_id.encode()) % 2**32)
        if condition.condition_id == "MULTISCALE_NO_GATE":
            payload["y_true"] = payload["y_true"][::-1]
        return payload

    study.predict_condition = predict

    with pytest.raises(ValueError, match="not aligned"):
        study.compute(datamodule=None)


# ------------------------------------------------------------- parameters and fairness


def test_parameter_counts_are_measured_not_assumed(result):
    """The five conditions are NOT parameter-matched, and the report must say so."""
    capacity = result["capacity"]

    assert set(capacity) == set(CONDITION_IDS)
    for record in capacity.values():
        assert isinstance(record["total_parameters"], int)
        assert record["total_parameters"] > 0
        assert "delta_vs_adaptive" in record

    assert capacity["ADAPTIVE_MULTISCALE"]["delta_vs_adaptive"] == 0
    assert not result["parameters"]["parameter_matched"]


def test_the_capacity_record_reflects_the_real_modules(result):
    """Counted by instantiating the arms, not written down."""
    from src.models.components.multiscale import MultiscaleClassifier

    for condition in CONDITIONS:
        actual = sum(
            p.numel() for p in MultiscaleClassifier.from_arm(condition.arm).parameters()
        )
        assert result["capacity"][condition.condition_id]["total_parameters"] == actual


def test_the_primary_comparison_is_not_capacity_confounded_in_the_adaptive_favour(result):
    """The ungated control is the LARGER model, which is what makes H24 defensible.

    If the adaptive condition were the bigger of the two, a win could be capacity rather
    than gating - so this relationship is asserted rather than left to a footnote.
    """
    capacity = result["capacity"]
    ungated = capacity["MULTISCALE_NO_GATE"]["total_parameters"]
    adaptive = capacity["ADAPTIVE_MULTISCALE"]["total_parameters"]

    assert ungated > adaptive
    assert result["primary"]["capacity_note"]


def test_the_fixed_conditions_are_recorded_as_smaller(result):
    """Their comparisons are descriptive precisely because capacity moves with them."""
    capacity = result["capacity"]
    adaptive = capacity["ADAPTIVE_MULTISCALE"]["total_parameters"]

    for condition_id in ("FIXED_3X3", "FIXED_5X5", "FIXED_DILATED_3X3"):
        assert capacity[condition_id]["total_parameters"] < adaptive
        assert capacity[condition_id]["delta_vs_adaptive"] < 0


# ----------------------------------------------------------------- the result matrix


def test_the_matrix_records_the_strategy_and_adaptivity_of_every_condition(result):
    """:param result: The computed summary."""
    matrix = {row["condition"]: row for row in result["matrix"]}

    assert set(matrix) == set(CONDITION_IDS)
    assert matrix["FIXED_3X3"]["adaptive"] is False
    assert matrix["ADAPTIVE_MULTISCALE"]["adaptive"] is True
    for row in matrix.values():
        assert row["receptive_field_strategy"]
        assert row["fusion"]
        assert row["total_parameters"] > 0


def test_the_flat_table_schema_is_pinned(study, result):
    """:param study: The wired study.

    :param result: The computed summary.
    """
    import pandas as pd

    frame = pd.read_csv(study.output_dir / "step24_receptive_field.csv")

    assert list(frame.columns) == list(TABLE_COLUMNS)
    assert set(frame["condition"]) == set(CONDITION_IDS)
    assert len(frame) == 5 * 3


def test_the_table_records_the_controlled_settings_per_row(study, result):
    """Provenance for the fairness claim: every row states what it was trained under."""
    import pandas as pd

    frame = pd.read_csv(study.output_dir / "step24_receptive_field.csv")

    assert frame["recipe"].nunique() == 1
    assert frame["loss"].nunique() == 1
    assert set(frame["augment"]) == {False}
    assert set(frame["use_weighted_sampler"]) == {False}
    assert set(frame["seed"]) == set(PROTOCOL_SEEDS)


def test_an_empty_matrix_keeps_the_full_schema(study):
    """:param study: The wired study."""
    frame = study._flat_table({c.condition_id: {} for c in CONDITIONS})

    assert len(frame) == 0
    assert list(frame.columns) == list(TABLE_COLUMNS)


def test_per_seed_predictions_are_saved_for_the_paired_test(study, result):
    """:param study: The wired study.

    :param result: The computed summary.
    """
    saved = list(study.output_dir.glob("step24_predictions_*.npz"))

    assert len(saved) == 5 * 3
    payload = np.load(study.output_dir / "step24_predictions_ADAPTIVE_MULTISCALE_seed42.npz")
    assert set(payload) == {"y_true", "y_pred", "y_prob"}


# ------------------------------------------------------------------ Hydra composition


def test_the_step24_analysis_stage_composes():
    """:return: None."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="analyze.yaml", overrides=["analysis=step24_receptive_field"])
    GlobalHydra.instance().clear()

    assert cfg.analysis._target_ == "src.analysis.receptive_field_study.ReceptiveFieldStudy"
    assert cfg.analysis.name == "step24_receptive_field"
    assert cfg.analysis.n_resamples == 2000
    assert list(cfg.analysis.seeds) == list(PROTOCOL_SEEDS)


def test_the_step24_experiment_declares_no_data_defaults():
    """A default here is a value some condition silently inherits."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml",
                      overrides=["experiment=step24_receptive_field"])
    GlobalHydra.instance().clear()

    assert cfg.model.net._target_.endswith("MultiscaleClassifier.from_arm")
    assert cfg.trainer.max_epochs == 30


# ------------------------------------------------------- nothing validated is touched


def test_no_validated_artefact_was_touched():
    """:return: None."""
    import subprocess

    changed = subprocess.run(
        ["git", "status", "--short", "--", "configs/protocol",
         "configs/experiment/step15_final_protocol.yaml", "data/splits"],
        capture_output=True, text=True,
    ).stdout.strip()

    assert changed == ""


def test_step24_does_not_alter_the_phase8_ablation_matrix():
    """Steps 21-23 keep their rows; Step 24 is a separate experiment."""
    from src.analysis.ablation_rows import ROWS

    assert [r.row_id for r in ROWS] == [
        "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "P",
    ]


# ======================================================================================
# Step 6 confirmation -> Step 24
#
# Step 24 must not decide its own preprocessing, and must not inherit the reduced-scale
# proxy's ranking. It consumes the authoritative real-backbone confirmation, and stops if
# that does not exist. These tests cover both halves.
# ======================================================================================


def write_confirmation(directory, recipe, status="confirmed"):
    """Write a Step 6 confirmation summary.

    :param directory: Where to write it.
    :param recipe: The confirmed recipe.
    :param status: Confirmation status.
    :return: Path to the summary.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "step06_confirm_summary.json"
    path.write_text(
        json.dumps({"selected_recipe": recipe, "confirmation_status": status,
                    "selection_metric": "val/f1_macro_best"}),
        encoding="utf-8",
    )
    return path


# ------------------------------------------------------- Step 24 blocks without it


def test_step24_refuses_to_run_without_a_confirmed_recipe(tmp_path):
    """Fifteen training runs must not start on an unconfirmed preprocessing.

    The proxy ranking is explicitly not a substitute: it ranks candidates with a SmallCNN
    at 128px and its own summary asks for real-backbone confirmation before committing.
    """
    from src.analysis.preprocessing_confirmation import ConfirmationIncomplete

    analysis = ReceptiveFieldStudy(run_root=str(tmp_path / "runs"))
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    with pytest.raises(ConfirmationIncomplete, match="authoritative selected_recipe"):
        analysis.compute(datamodule=None)


def test_step24_refuses_a_missing_confirmation_file(tmp_path):
    """:param tmp_path: Per-test directory."""
    from src.analysis.preprocessing_confirmation import ConfirmationIncomplete

    analysis = ReceptiveFieldStudy(confirmation_summary=str(tmp_path / "nope.json"))

    with pytest.raises(ConfirmationIncomplete, match="does not exist"):
        _ = analysis.recipe


def test_step24_refuses_an_unconfirmed_status(tmp_path):
    """A summary that exists but is not marked confirmed is not a decision."""
    from src.analysis.preprocessing_confirmation import ConfirmationIncomplete

    path = write_confirmation(tmp_path / "s6", "clahe", status="pending")
    analysis = ReceptiveFieldStudy(confirmation_summary=str(path))

    with pytest.raises(ConfirmationIncomplete, match="not 'confirmed'"):
        _ = analysis.recipe


def test_step24_never_falls_back_to_the_proxy_ranking():
    """The proxy summary must not be reachable from Step 24's resolution path."""
    import ast
    from pathlib import Path

    for module in ("src/analysis/receptive_field_rows.py",
                   "src/analysis/receptive_field_study.py"):
        source = Path(module).read_text(encoding="utf-8")
        tree = ast.parse(source)
        literals = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        docstrings = {
            ast.get_docstring(n) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
        }
        for text in literals - docstrings:
            assert "step06_preprocessing" not in text, (
                f"{module} references the proxy summary in code"
            )


# ------------------------------------------------------ Step 24 reads the confirmation


@pytest.mark.parametrize("recipe", ["clahe", "diffusion_i5_k15", "gamma", "wiener"])
def test_step24_uses_whatever_step6_confirmed(tmp_path, recipe):
    """No recipe is assumed. Whatever the confirmation says is what all five conditions get.

    Parametrised across unrelated recipes precisely so no test encodes a winner.
    """
    path = write_confirmation(tmp_path / "s6", recipe)
    context = ReceptiveFieldContext.from_confirmation(str(path))

    assert context.recipe == recipe
    assert "step06_confirm" in context.source

    for condition in CONDITIONS:
        overrides = condition_overrides(condition, seed=42, context=context)
        assert f"data.recipe={recipe}" in overrides


def test_an_identity_recipe_becomes_a_null_override(tmp_path):
    """conventional reads the raw tree, so it must not be sent as a mirror name."""
    path = write_confirmation(tmp_path / "s6", "conventional")
    context = ReceptiveFieldContext.from_confirmation(str(path))

    assert context.recipe is None
    assert "data.recipe=null" in condition_overrides(CONDITIONS[0], 42, context)


@pytest.mark.parametrize("recipe", ["clahe", "diffusion_i5_k15", "log"])
def test_all_five_conditions_receive_exactly_one_identical_recipe(tmp_path, recipe):
    """The fairness property, asserted across every condition and every seed."""
    path = write_confirmation(tmp_path / "s6", recipe)
    context = ReceptiveFieldContext.from_confirmation(str(path))

    emitted = set()
    for condition in CONDITIONS:
        for seed in PROTOCOL_SEEDS:
            overrides = condition_overrides(condition, seed=seed, context=context)
            recipes = [o for o in overrides if o.startswith("data.recipe=")]
            assert len(recipes) == 1, f"{condition.condition_id} must pin exactly one recipe"
            emitted.add(recipes[0])

    assert emitted == {f"data.recipe={recipe}"}


def test_no_condition_carries_a_recipe_of_its_own(tmp_path):
    """One context supplies all five, so a per-condition recipe cannot diverge."""
    from dataclasses import fields

    assert "recipe" not in {f.name for f in fields(CONDITIONS[0])}

    path = write_confirmation(tmp_path / "s6", "clahe")
    context = ReceptiveFieldContext.from_confirmation(str(path))
    base = set(condition_overrides(CONDITIONS[0], 42, context))
    for condition in CONDITIONS[1:]:
        difference = set(condition_overrides(condition, 42, context)) - base
        assert difference == {f"model.net.arm={condition.arm}"}


def test_an_explicit_override_is_preserved_and_recorded(tmp_path):
    """--recipe is an explicit operator decision, unlike an implicit proxy fallback."""
    analysis = ReceptiveFieldStudy(recipe="wiener",
                                   confirmation_summary=str(tmp_path / "absent.json"))

    assert analysis.recipe == "wiener"
    assert "override" in analysis._context().source


# ------------------------------------------------ the confirmation analysis itself


def make_run(directory, recipe, seed, val_score, experiment="step06_confirm",
             test_score=0.99):
    """Write a fake Hydra training run.

    :param directory: Run directory.
    :param recipe: Recipe override to record.
    :param seed: Seed override.
    :param val_score: Best validation macro-F1.
    :param experiment: Experiment override.
    :param test_score: A test metric, present so the reader can be shown to ignore it.
    :return: The run directory.
    """
    import pandas as pd

    (directory / ".hydra").mkdir(parents=True, exist_ok=True)
    (directory / ".hydra" / "overrides.yaml").write_text(
        json.dumps([f"experiment={experiment}", "model=baseline_efficientnet_b0",
                    f"seed={seed}", f"data.recipe={recipe}"]),
        encoding="utf-8",
    )
    (directory / "csv" / "version_0").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "epoch": [0, 1],
        "val/f1_macro": [val_score - 0.05, val_score],
        "val/f1_macro_best": [val_score - 0.05, val_score],
        "test/f1_macro": [test_score, test_score],
    }).to_csv(directory / "csv" / "version_0" / "metrics.csv", index=False)
    return directory


@pytest.fixture
def sweep(tmp_path):
    """A completed confirmation sweep with three candidates.

    :param tmp_path: Per-test directory.
    :return: The sweep root.
    """
    root = tmp_path / "multiruns" / "ts"
    make_run(root / "0", "null", 42, 0.71)
    make_run(root / "1", "clahe", 42, 0.83)
    make_run(root / "2", "diffusion_i5_k15", 42, 0.78)
    return root


def test_the_confirmation_picks_the_best_validation_score(sweep, tmp_path):
    """:param sweep: A completed sweep.

    :param tmp_path: Per-test directory.
    """
    from src.analysis.preprocessing_confirmation import PreprocessingConfirmation

    analysis = PreprocessingConfirmation(run_root=str(sweep))
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()
    summary = analysis.compute(datamodule=None)

    assert summary["selected_recipe"] == "clahe"
    assert summary["confirmation_status"] == "confirmed"
    assert summary["n_candidates"] == 3
    assert summary["selection_metric"] == "val/f1_macro_best"
    assert summary["model"] == "baseline_efficientnet_b0"


def test_the_confirmation_follows_the_data_not_a_preferred_recipe(tmp_path):
    """Reverse the scores and the winner reverses. No recipe is favoured by the code."""
    from src.analysis.preprocessing_confirmation import PreprocessingConfirmation

    root = tmp_path / "sweep"
    make_run(root / "0", "clahe", 42, 0.60)
    make_run(root / "1", "diffusion_i5_k15", 42, 0.88)

    analysis = PreprocessingConfirmation(run_root=str(root))
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    assert analysis.compute(datamodule=None)["selected_recipe"] == "diffusion_i5_k15"


def test_the_conventional_reference_is_named_not_null(sweep, tmp_path):
    """data.recipe=null is the conventional reference and belongs in the table."""
    from src.analysis.preprocessing_confirmation import PreprocessingConfirmation

    analysis = PreprocessingConfirmation(run_root=str(sweep))
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()
    summary = analysis.compute(datamodule=None)

    assert "conventional" in {c["recipe"] for c in summary["candidates"]}


def test_the_confirmation_reads_no_test_metric():
    """It decides preprocessing, which precedes Step 16; test metrics must stay unread."""
    from src.analysis.preprocessing_confirmation import PreprocessingConfirmation

    with pytest.raises(ValueError, match="validation only"):
        PreprocessingConfirmation(metric="test/f1_macro").compute(datamodule=None)


def test_the_confirmation_ignores_runs_from_another_experiment(tmp_path):
    """A Step 9 sweep also varies by recipe; reading it would answer another question."""
    from src.analysis.preprocessing_confirmation import (
        ConfirmationIncomplete,
        PreprocessingConfirmation,
    )

    root = tmp_path / "sweep"
    make_run(root / "0", "clahe", 42, 0.9, experiment="step09_baselines")
    make_run(root / "1", "gamma", 42, 0.8, experiment="step09_baselines")

    analysis = PreprocessingConfirmation(run_root=str(root))
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    with pytest.raises(ConfirmationIncomplete, match="at least 2 distinct recipes"):
        analysis.compute(datamodule=None)


def test_the_confirmation_refuses_a_single_candidate(tmp_path):
    """One training run is not a preprocessing comparison."""
    from src.analysis.preprocessing_confirmation import (
        ConfirmationIncomplete,
        PreprocessingConfirmation,
    )

    root = tmp_path / "sweep"
    make_run(root / "0", "clahe", 42, 0.9)

    analysis = PreprocessingConfirmation(run_root=str(root))
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    with pytest.raises(ConfirmationIncomplete, match="at least 2 distinct"):
        analysis.compute(datamodule=None)


def test_the_confirmation_refuses_an_absent_sweep(tmp_path):
    """No winner exists before the runs do."""
    from src.analysis.preprocessing_confirmation import (
        ConfirmationIncomplete,
        PreprocessingConfirmation,
    )

    analysis = PreprocessingConfirmation(run_root=str(tmp_path / "never_ran"))
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    with pytest.raises(ConfirmationIncomplete, match="does not exist"):
        analysis.compute(datamodule=None)


def test_the_confirmation_aggregates_across_seeds(tmp_path):
    """Multiple seeds per recipe are averaged, and the spread is recorded."""
    from src.analysis.preprocessing_confirmation import PreprocessingConfirmation

    root = tmp_path / "sweep"
    for index, (recipe, seed, score) in enumerate([
        ("clahe", 42, 0.80), ("clahe", 123, 0.84),
        ("gamma", 42, 0.90), ("gamma", 123, 0.70),
    ]):
        make_run(root / str(index), recipe, seed, score)

    analysis = PreprocessingConfirmation(run_root=str(root))
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()
    summary = analysis.compute(datamodule=None)

    by_recipe = {c["recipe"]: c for c in summary["candidates"]}
    assert by_recipe["clahe"]["mean"] == pytest.approx(0.82)
    assert by_recipe["gamma"]["mean"] == pytest.approx(0.80)
    assert by_recipe["clahe"]["n_runs"] == 2
    assert summary["selected_recipe"] == "clahe"
    assert summary["seeds"] == [42, 123]


def test_the_confirmation_records_its_provenance(sweep, tmp_path):
    """:param sweep: A completed sweep.

    :param tmp_path: Per-test directory.
    """
    from src.analysis.preprocessing_confirmation import PreprocessingConfirmation

    analysis = PreprocessingConfirmation(run_root=str(sweep))
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()
    summary = analysis.compute(datamodule=None)

    assert summary["provenance"]["source_stage"] == "step06_confirm"
    assert len(summary["runs"]) == 3
    for record in summary["runs"]:
        assert record["run_dir"]
        assert record["metric"].startswith("val/")
    assert "proxy" in summary["supersedes"]


def test_the_confirmation_config_composes():
    """:return: None."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="analyze.yaml", overrides=["analysis=step06_confirm"])
    GlobalHydra.instance().clear()

    assert cfg.analysis._target_ == (
        "src.analysis.preprocessing_confirmation.PreprocessingConfirmation"
    )
    assert cfg.analysis.metric.startswith("val/")
    assert cfg.analysis.min_candidates == 2
    assert cfg.analysis.require_experiment == "step06_confirm"
    assert cfg.analysis.run_root is None, "no winner may be implied by the config"


def test_the_step24_config_names_no_recipe():
    """The config must not encode a preprocessing decision."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="analyze.yaml",
                      overrides=["analysis=step24_receptive_field"])
    GlobalHydra.instance().clear()

    assert cfg.analysis.recipe is None
    assert cfg.analysis.confirmation_summary is None


def test_no_step24_source_file_names_a_recipe():
    """No preprocessing winner is assumed anywhere in the implementation."""
    from pathlib import Path

    forbidden = ("clahe", "diffusion_i5_k15", "diffusion_i10_k15", "gamma", "wiener")
    for module in ("src/analysis/receptive_field_rows.py",
                   "src/analysis/receptive_field_study.py",
                   "src/analysis/preprocessing_confirmation.py"):
        source = Path(module).read_text(encoding="utf-8").lower()
        for name in forbidden:
            assert name not in source, f"{module} names the recipe {name!r}"
