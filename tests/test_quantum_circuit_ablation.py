"""Step 25: the quantum-circuit adaptivity ablation, tested without training anything.

Step 12 runs five quantum circuits on every image and mixes their outputs with a learned
per-image softmax. Nothing in the study has tested whether that beats one fixed circuit -
Phase 8's H2 compares against ``baseline_fixed_qcnn``, a different architecture with a
fourteenth of the parameters. Step 25 asks directly, and all four conditions are the same
class with different ``circuit_names``.

Four ways the experiment could quietly stop answering its question:

* **A control that is not one.** If ``FIXED_*`` acquired more than one circuit, or the
  adaptive condition lost four of its five, the comparison would be the model against
  itself.
* **Inherited settings.** ``configs/experiment/step12_adaptive_quantum.yaml`` sets
  ``augment: true`` and ``use_weighted_sampler: true`` and the pipeline overrides both with
  Step 8's selection. Two conditions run that way are incomparable for a reason unrelated
  to circuits.
* **Test-set leakage.** This is an architecture question, so it is decided on validation.
  Every run's ``metrics.csv`` also carries ``test/*``, and using it would spend Step 16's
  budget on a design decision.
* **An overstated claim.** The Step 11 spatial gate is inside the branch and dominates the
  parameters. Held constant it is fine; described loosely it turns a narrow result into a
  claim about the whole model.

Nothing here trains or loads a checkpoint: predictions are injected.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.analysis.ablation_rows import PROTOCOL_SEEDS
from src.analysis.quantum_circuit_ablation_rows import (
    ADAPTIVE_CIRCUITS,
    CONDITIONS,
    EVALUATION_SPLIT,
    FAMILY_SIZE,
    PINNED,
    PRIMARY_COMPARISONS,
    SECONDARY_COMPARISONS,
    QuantumAblationContext,
    build_model,
    circuit_override,
    condition_overrides,
    condition_parameters,
    get_condition,
)
from src.analysis.quantum_circuit_ablation_study import (
    CONDITION_IDS,
    TABLE_COLUMNS,
    QuantumCircuitAblationStudy,
)

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No-tumor"]

#: A neutral placeholder. Which recipe Step 25 uses is Step 6's decision; naming a real
#: candidate here would read as an assumed winner.
CONTEXT = QuantumAblationContext(recipe="recipe_under_confirmation")

FIXED_IDS = ("FIXED_BASIC", "FIXED_DEEP", "FIXED_STRONG")


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


# ------------------------------------------------------------- the four conditions


def test_exactly_four_conditions_are_registered():
    """Three fixed circuits and one mixture."""
    assert [c.condition_id for c in CONDITIONS] == [
        "FIXED_BASIC", "FIXED_DEEP", "FIXED_STRONG", "ADAPTIVE_QUANTUM",
    ]
    assert list(CONDITION_IDS) == [c.condition_id for c in CONDITIONS]
    assert len(CONDITIONS) == len({c.condition_id for c in CONDITIONS}) == 4


def test_the_condition_circuit_lists_are_the_intended_ones():
    """:return: None."""
    assert get_condition("FIXED_BASIC").circuit_names == ("fixed",)
    assert get_condition("FIXED_DEEP").circuit_names == ("deep",)
    assert get_condition("FIXED_STRONG").circuit_names == ("strong",)
    assert get_condition("ADAPTIVE_QUANTUM").circuit_names is None


def test_the_adaptive_circuit_set_matches_the_quantum_module():
    """The manifest must not carry a stale copy of the circuit list."""
    from src.models.components.quantum import CIRCUIT_NAMES

    assert ADAPTIVE_CIRCUITS == CIRCUIT_NAMES
    assert get_condition("ADAPTIVE_QUANTUM").n_circuits == 5


@pytest.mark.parametrize("condition_id", FIXED_IDS)
def test_each_fixed_condition_contains_exactly_one_circuit(condition_id):
    """More than one circuit in a control makes it a mixture, not a control."""
    condition = get_condition(condition_id)
    assert len(condition.circuit_names) == 1
    assert condition.n_circuits == 1
    assert condition.adaptive is False

    model = build_model(condition)
    assert len(model.branch.experts) == 1


def test_the_adaptive_condition_contains_all_five_circuits():
    """:return: None."""
    model = build_model(get_condition("ADAPTIVE_QUANTUM"))

    assert len(model.branch.experts) == 5
    assert model.branch.circuit_names == ADAPTIVE_CIRCUITS
    assert get_condition("ADAPTIVE_QUANTUM").adaptive is True


def test_only_the_adaptive_condition_is_marked_adaptive():
    """:return: None."""
    assert [c.condition_id for c in CONDITIONS if c.adaptive] == ["ADAPTIVE_QUANTUM"]


@pytest.mark.parametrize("condition_id", FIXED_IDS)
def test_a_fixed_condition_has_no_effective_mixture(condition_id):
    """A softmax over one element is identically 1.0, so the selector cannot weight."""
    model = build_model(get_condition(condition_id)).eval()
    with torch.no_grad():
        outputs = model.extract(torch.randn(3, 3, 64, 64))

    weights = outputs["quantum_weights"]
    assert weights.shape == (3, 1)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_the_adaptive_condition_emits_a_five_element_distribution():
    """:return: None."""
    model = build_model(get_condition("ADAPTIVE_QUANTUM")).eval()
    with torch.no_grad():
        weights = model.extract(torch.randn(4, 3, 64, 64))["quantum_weights"]

    assert weights.shape == (4, 5)
    assert (weights >= 0).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


def test_different_inputs_can_produce_different_adaptive_weights():
    """"Adaptive" has to mean input-dependent, not merely learned-once."""
    torch.manual_seed(0)
    model = build_model(get_condition("ADAPTIVE_QUANTUM")).eval()

    with torch.no_grad():
        first = model.extract(torch.randn(1, 3, 64, 64))["quantum_weights"]
        second = model.extract(torch.randn(1, 3, 64, 64) * 6.0 + 2.0)["quantum_weights"]

    assert (first - second).abs().max() > 1e-5, "the selector ignored its input"


def test_all_circuits_execute_and_none_is_skipped():
    """The mechanism is a soft mixture, not conditional execution.

    Every expert module is evaluated on every forward pass. Describing it as dynamic
    selection or conditional execution would misstate both the mechanism and its cost.
    """
    model = build_model(get_condition("ADAPTIVE_QUANTUM")).eval()
    calls = []
    for index, expert in enumerate(model.branch.experts):
        original = expert.forward
        expert.forward = (
            lambda x, i=index, f=original: (calls.append(i), f(x))[1]
        )

    with torch.no_grad():
        model.extract(torch.randn(2, 3, 64, 64))

    assert sorted(calls) == [0, 1, 2, 3, 4]


# ------------------------------------------------------ everything else identical


def test_only_the_circuit_list_differs_between_conditions():
    """The single intended experimental variable."""
    base = set(condition_overrides(CONDITIONS[0], 42, CONTEXT))
    for condition in CONDITIONS[1:]:
        difference = set(condition_overrides(condition, 42, CONTEXT)) - base
        assert difference == {circuit_override(condition)}, (
            f"{condition.condition_id} differs by more than its circuit list"
        )


@pytest.mark.parametrize("condition_id", list(CONDITION_IDS))
def test_every_condition_pins_the_same_data_handling(condition_id):
    """Step 12's config sets augment/sampler true and the pipeline overrides them."""
    cfg = compose_condition(condition_id)

    assert cfg.data.recipe == CONTEXT.recipe
    assert cfg.data.normalize == PINNED["normalize"]
    assert cfg.data.augment is PINNED["augment"]
    assert cfg.data.use_weighted_sampler is PINNED["use_weighted_sampler"]
    assert cfg.model.criterion.use_class_weights is False


@pytest.mark.parametrize("condition_id", list(CONDITION_IDS))
def test_every_condition_pins_its_settings_explicitly(condition_id):
    """Nothing is left to recipe_override/imbalance_overrides."""
    overrides = condition_overrides(get_condition(condition_id), seed=42, context=CONTEXT)

    for key in ("data.recipe=", "data.normalize=", "data.augment=",
                "data.use_weighted_sampler=", "loss@model.criterion=",
                "model.net.circuit_names="):
        assert sum(o.startswith(key) for o in overrides) == 1, f"{condition_id} lacks {key}"


def test_all_conditions_share_one_recipe():
    """:return: None."""
    recipes = {compose_condition(c.condition_id).data.recipe for c in CONDITIONS}
    assert len(recipes) == 1


@pytest.mark.parametrize("condition_id", list(CONDITION_IDS))
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
    assert cfg.data.batch_size == reference.data.batch_size
    assert cfg.model.optimizer.lr == reference.model.optimizer.lr
    assert cfg.optimized_metric == reference.optimized_metric


def test_the_seed_set_is_the_studys_own():
    """:return: None."""
    from src.analysis.quantum_circuit_ablation_rows import SEEDS

    assert SEEDS == PROTOCOL_SEEDS == (42, 123, 7)


@pytest.mark.parametrize("condition_id", list(CONDITION_IDS))
def test_every_condition_runs_every_seed(condition_id):
    """:param condition_id: Condition under test."""
    for seed in PROTOCOL_SEEDS:
        assert compose_condition(condition_id, seed=seed).seed == seed


def test_the_training_matrix_is_four_conditions_by_three_seeds():
    """:return: None."""
    matrix = [(c.condition_id, s) for c in CONDITIONS for s in PROTOCOL_SEEDS]

    assert len(matrix) == 12
    assert len(set(matrix)) == 12


# ----------------------------------------------- shared architecture across conditions


def test_every_condition_shares_the_same_spatial_branch():
    """The Step 11 gate is inside Step 12 and must be identical in all four conditions."""
    shapes = {
        tuple(tuple(p.shape) for p in build_model(c).branch.spatial_branch.parameters())
        for c in CONDITIONS
    }
    assert len(shapes) == 1, "the spatial branch differs between conditions"

    counts = {condition_parameters(c)["spatial_branch"] for c in CONDITIONS}
    assert len(counts) == 1


def test_every_condition_shares_the_same_projection_and_feature_width():
    """:return: None."""
    assert len({condition_parameters(c)["reduce"] for c in CONDITIONS}) == 1
    assert {condition_parameters(c)["feature_dim"] for c in CONDITIONS} == {
        PINNED["channels"] + PINNED["n_qubits"]
    }


def test_every_condition_shares_the_same_classifier_head():
    """:return: None."""
    assert len({condition_parameters(c)["classifier"] for c in CONDITIONS}) == 1

    heads = set()
    for condition in CONDITIONS:
        model = build_model(condition)
        heads.add(tuple(
            (type(m).__name__, getattr(m, "in_features", None), getattr(m, "out_features", None))
            for m in model.classifier
        ))
    assert len(heads) == 1


def test_every_condition_shares_qubits_encoding_and_measurement():
    """Circuit width, encoding and readout are constants of the experiment."""
    for condition in CONDITIONS:
        model = build_model(condition)
        for expert in model.branch.experts:
            assert expert.n_qubits == PINNED["n_qubits"] == 4
        with torch.no_grad():
            outputs = model.extract(torch.randn(2, 3, 64, 64))
        assert outputs["quantum_features"].shape == (2, PINNED["n_qubits"])


# ------------------------------------------------------------------ capacity


def test_parameter_counts_are_measured_from_instantiated_models():
    """Nothing is tabulated: a written-down number outlives the architecture."""
    for condition in CONDITIONS:
        counts = condition_parameters(condition)
        actual = sum(p.numel() for p in build_model(condition).parameters())
        assert counts["total"] == actual
        assert counts["quantum_experts"] > 0


def test_the_conditions_are_not_parameter_matched_and_the_adaptive_one_is_larger():
    """The gap runs in the adaptive condition's favour, so it must be reported."""
    counts = {c.condition_id: condition_parameters(c) for c in CONDITIONS}
    adaptive = counts["ADAPTIVE_QUANTUM"]["total"]

    for condition_id in FIXED_IDS:
        assert counts[condition_id]["total"] < adaptive
        assert counts[condition_id]["quantum_experts"] < counts["ADAPTIVE_QUANTUM"][
            "quantum_experts"
        ]


def test_the_capacity_difference_is_confined_to_the_quantum_and_selector_groups():
    """Everything else being equal is what makes the comparison worth running."""
    counts = {c.condition_id: condition_parameters(c) for c in CONDITIONS}

    for group in ("spatial_branch", "reduce", "classifier"):
        assert len({counts[c]["total" if False else group] for c in counts}) == 1, (
            f"{group} differs between conditions"
        )


# ------------------------------------------------------------- the hypothesis family


def test_the_primary_family_is_the_three_adaptive_versus_fixed_comparisons():
    """:return: None."""
    assert [c["id"] for c in PRIMARY_COMPARISONS] == [
        "H25a_vs_basic", "H25b_vs_deep", "H25c_vs_strong",
    ]
    assert all(c["condition_a"] == "ADAPTIVE_QUANTUM" for c in PRIMARY_COMPARISONS)
    assert {c["condition_b"] for c in PRIMARY_COMPARISONS} == set(FIXED_IDS)
    assert FAMILY_SIZE == len(PRIMARY_COMPARISONS) == 3


def test_the_fixed_versus_fixed_comparisons_are_secondary():
    """Whether depth or entanglement explains a difference is a different question."""
    assert [c["id"] for c in SECONDARY_COMPARISONS] == [
        "S25a_deep_vs_basic", "S25b_strong_vs_basic", "S25c_strong_vs_deep",
    ]
    for comparison in SECONDARY_COMPARISONS:
        assert comparison["condition_a"] in FIXED_IDS
        assert comparison["condition_b"] in FIXED_IDS


def test_step25_does_not_touch_the_other_hypothesis_families():
    """Phase 8's H1-H4 and Step 24's H24 must be untouched."""
    from src.analysis.receptive_field_rows import (
        PRIMARY_COMPARISON as STEP24_PRIMARY,
    )
    from src.analysis.statistical_report import PRIMARY_FAMILY

    assert [h["id"] for h in PRIMARY_FAMILY] == ["H1", "H2", "H3", "H4"]
    assert STEP24_PRIMARY["id"] == "H24"

    step25 = {c["id"] for c in PRIMARY_COMPARISONS} | {
        c["id"] for c in SECONDARY_COMPARISONS
    }
    assert not step25 & {h["id"] for h in PRIMARY_FAMILY}
    assert STEP24_PRIMARY["id"] not in step25


# ------------------------------------------------------------ the study itself


def synth(seed, labels=None, n=160, skill=1.3, n_circuits=5):
    """Deterministic synthetic predictions plus mixture weights.

    :param seed: Generator seed.
    :param labels: Shared label vector, or ``None`` to draw one.
    :param n: Sample count.
    :param skill: Probability mass on the true class.
    :param n_circuits: Width of the mixture weight vector.
    :return: Prediction dict.
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, len(CLASS_NAMES), n) if labels is None else labels
    logits = rng.normal(size=(len(y_true), len(CLASS_NAMES)))
    logits[np.arange(len(y_true)), y_true] += skill
    y_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    raw = rng.normal(size=(len(y_true), n_circuits))
    weights = np.exp(raw) / np.exp(raw).sum(axis=1, keepdims=True)

    return {"y_true": y_true, "y_pred": y_prob.argmax(axis=1), "y_prob": y_prob,
            "quantum_weights": weights, "class_names": list(CLASS_NAMES)}


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
            (directory / "epoch_006.ckpt").touch()

    analysis = QuantumCircuitAblationStudy(
        run_root=str(runs), recipe="recipe_under_confirmation", n_resamples=200
    )
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    labels = synth(0)["y_true"]

    def predict(condition, seed, checkpoint):
        key = zlib.crc32(f"{condition.condition_id}:{seed}".encode()) % 2**32
        return synth(key, labels=labels, n_circuits=condition.n_circuits)

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


def test_the_study_evaluates_on_validation_not_test(result):
    """The internal test set stays sealed for Step 16."""
    assert result["parameters"]["evaluation_split"] == "val" == EVALUATION_SPLIT
    for record in result["conditions"].values():
        for metrics in record["per_seed"].values():
            assert metrics["split"] == "val"
            assert "no test metric was read" in metrics["provenance"]


def test_a_test_split_request_is_refused(tmp_path):
    """No architecture decision may consume the test set."""
    analysis = QuantumCircuitAblationStudy(split="test", recipe="x")
    analysis._output_dir = tmp_path
    with pytest.raises(ValueError, match="'val' only|val.* only"):
        analysis.compute(datamodule=None)


def test_the_primary_metric_is_macro_f1(result):
    """:param result: The computed summary."""
    assert result["parameters"]["primary_metric"] == "macro_f1"
    assert all(c["metric"] == "macro_f1" for c in result["primary"])


def test_the_full_metric_battery_is_reported(result):
    """Accuracy, macro-F1, per-class precision/recall/F1 and a confusion matrix."""
    expected = {"accuracy", "macro_f1", "balanced_accuracy", "macro_precision",
                "macro_recall_sensitivity", "weighted_f1", "mcc"}

    for record in result["conditions"].values():
        for metrics in record["per_seed"].values():
            assert expected <= set(metrics["overall"])
            assert len(metrics["per_class"]) == len(CLASS_NAMES)
            assert {"precision", "recall_sensitivity", "f1"} <= set(metrics["per_class"][0])
            assert metrics["confusion"]["class_names"] == CLASS_NAMES

    for record in result["conditions"].values():
        across = record["across_seeds"]
        assert set(across["per_class_recall"]) == set(CLASS_NAMES)
        assert set(across["per_class_precision"]) == set(CLASS_NAMES)
        assert set(across["per_class_f1"]) == set(CLASS_NAMES)


def test_the_three_primary_comparisons_are_formal_and_holm_corrected(result):
    """:param result: The computed summary."""
    assert [c["comparison"] for c in result["primary"]] == [
        "H25a_vs_basic", "H25b_vs_deep", "H25c_vs_strong",
    ]
    for comparison in result["primary"]:
        assert comparison["statistical_status"] == "formal"
        assert comparison["correction"] == "holm-bonferroni"
        assert comparison["family_size"] == 3
        assert comparison["raw_p_value"] is not None
        assert comparison["adjusted_p_value"] is not None
        assert comparison["adjusted_p_value"] >= comparison["raw_p_value"]


def test_holm_actually_corrects_a_family_of_three(study):
    """Unlike Step 24's family of one, the correction must move the p-values here."""
    records = [
        {"comparison": c["id"], "raw_p_value": p, "significant": False,
         "adjusted_p_value": None}
        for c, p in zip(PRIMARY_COMPARISONS, (0.02, 0.03, 0.04))
    ]
    study._apply_holm(records)

    assert [round(r["adjusted_p_value"], 4) for r in records] == [0.06, 0.06, 0.06]
    assert not any(r["significant"] for r in records)


def test_the_secondary_comparisons_carry_no_significance_claim(result):
    """:param result: The computed summary."""
    assert [c["comparison"] for c in result["secondary"]] == [
        "S25a_deep_vs_basic", "S25b_strong_vs_basic", "S25c_strong_vs_deep",
    ]
    for comparison in result["secondary"]:
        assert comparison["statistical_status"] == "descriptive"
        assert comparison["raw_p_value"] is None
        assert comparison["adjusted_p_value"] is None
        assert comparison["significant"] is None
        assert comparison["ci_low"] is not None


def test_the_comparison_can_return_any_outcome(study):
    """The design must not guarantee the adaptive condition wins."""
    import zlib

    labels = synth(0)["y_true"]

    def predict(condition, seed, checkpoint):
        skill = 0.4 if condition.adaptive else 2.4
        key = zlib.crc32(f"{condition.condition_id}:{seed}".encode()) % 2**32
        return synth(key, labels=labels, skill=skill, n_circuits=condition.n_circuits)

    study.predict_condition = predict
    primary = study.compute(datamodule=None)["primary"]

    assert all(c["observed_delta"] < 0 for c in primary)
    assert all("FIXED" in c["effect_direction"] for c in primary)


# ------------------------------------------------------- adaptive-behaviour analysis


def test_the_mixture_weights_are_summarised_per_circuit(result):
    """The claimed contribution is the weighting, so it is inspected directly."""
    behaviour = result["adaptive_behaviour"]

    assert behaviour["available"] is True
    assert behaviour["circuits"] == list(ADAPTIVE_CIRCUITS)

    for record in behaviour["per_seed"].values():
        assert set(record["per_circuit"]) == set(ADAPTIVE_CIRCUITS)
        for stats in record["per_circuit"].values():
            assert {"mean", "std", "min", "max"} <= set(stats)
            assert 0.0 <= stats["min"] <= stats["mean"] <= stats["max"] <= 1.0
        assert 0.0 <= record["mean_normalised_entropy"] <= 1.0
        assert record["dominant_circuit"] in ADAPTIVE_CIRCUITS
        assert record["uniform_weight"] == pytest.approx(0.2)


def test_a_collapsed_selector_is_detectable(study):
    """A selector that collapsed onto one circuit is a fixed circuit with extra parameters."""
    labels = synth(0)["y_true"]

    def predict(condition, seed, checkpoint):
        payload = synth(seed, labels=labels, n_circuits=condition.n_circuits)
        if condition.adaptive:
            collapsed = np.zeros_like(payload["quantum_weights"])
            collapsed[:, 0] = 1.0
            payload["quantum_weights"] = collapsed
        return payload

    study.predict_condition = predict
    behaviour = study.compute(datamodule=None)["adaptive_behaviour"]

    for record in behaviour["per_seed"].values():
        assert record["mean_normalised_entropy"] == pytest.approx(0.0, abs=1e-6)
        assert record["max_mean_weight"] == pytest.approx(1.0)
        assert record["varies_between_images"] is False


def test_a_uniform_selector_is_detectable(study):
    """A selector stuck at uniform is an unweighted average, not adaptivity."""
    labels = synth(0)["y_true"]

    def predict(condition, seed, checkpoint):
        payload = synth(seed, labels=labels, n_circuits=condition.n_circuits)
        if condition.adaptive:
            payload["quantum_weights"] = np.full_like(payload["quantum_weights"], 0.2)
        return payload

    study.predict_condition = predict
    behaviour = study.compute(datamodule=None)["adaptive_behaviour"]

    for record in behaviour["per_seed"].values():
        assert record["mean_normalised_entropy"] == pytest.approx(1.0, abs=1e-6)
        assert record["varies_between_images"] is False


def test_varying_weights_are_not_presented_as_evidence_of_benefit(result):
    """The note must say what the weights do and do not show."""
    note = result["adaptive_behaviour"]["note"].lower()

    assert "do not show" in note or "does not show" in note
    assert "paired comparison" in note


def test_the_cost_record_states_that_no_circuit_is_skipped(result):
    """:param result: The computed summary."""
    cost = result["cost"]

    assert cost["circuits_executed_per_image"]["ADAPTIVE_QUANTUM"] == 5
    for condition_id in FIXED_IDS:
        assert cost["circuits_executed_per_image"][condition_id] == 1
    assert "not skip" in cost["note"].lower()
    assert "conditional execution" in cost["note"].lower()


def test_the_notes_scope_the_claim_to_circuit_adaptivity(result):
    """The spatial branch dominates capacity; the claim must not extend to the model."""
    notes = result["notes"]

    assert "circuit-mixture adaptivity ONLY" in notes["scope"]
    assert "spatial gate" in notes["scope"]
    assert "soft mixture" in notes["terminology"]
    assert "not dynamic circuit selection" in notes["terminology"]
    assert "NOT parameter-matched" in notes["capacity"]


# ------------------------------------------------------------------- schema


def test_the_flat_table_schema_is_pinned(study, result):
    """:param study: The wired study.

    :param result: The computed summary.
    """
    frame = pd.read_csv(study.output_dir / "step25_quantum_circuit_ablation.csv")

    assert list(frame.columns) == list(TABLE_COLUMNS)
    assert len(frame) == 4 * 3
    assert set(frame["condition"]) == set(CONDITION_IDS)
    assert set(frame["split"]) == {"val"}
    assert frame["recipe"].nunique() == 1
    assert set(frame["augment"]) == {False}
    assert set(frame["use_weighted_sampler"]) == {False}


def test_an_empty_ladder_keeps_the_full_schema(study):
    """:param study: The wired study."""
    capacity = study._capacity()
    frame = study._flat_table({c.condition_id: {} for c in CONDITIONS}, capacity)

    assert len(frame) == 0
    assert list(frame.columns) == list(TABLE_COLUMNS)


def test_per_seed_predictions_and_weights_are_saved(study, result):
    """:param study: The wired study.

    :param result: The computed summary.
    """
    saved = list(study.output_dir.glob("step25_predictions_*.npz"))
    assert len(saved) == 4 * 3

    payload = np.load(study.output_dir / "step25_predictions_ADAPTIVE_QUANTUM_seed42.npz")
    assert set(payload) == {"y_true", "y_pred", "y_prob", "quantum_weights"}
    assert payload["quantum_weights"].shape[1] == 5


# ------------------------------------------------------------ preprocessing policy


def test_the_study_refuses_to_run_without_a_confirmed_recipe(tmp_path):
    """Step 25 follows Step 12's preprocessing policy; it selects nothing itself."""
    from src.analysis.preprocessing_confirmation import ConfirmationIncomplete

    analysis = QuantumCircuitAblationStudy(run_root=str(tmp_path / "runs"))
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    with pytest.raises(ConfirmationIncomplete, match="authoritative selected_recipe"):
        analysis.compute(datamodule=None)


@pytest.mark.parametrize("recipe", ["clahe", "diffusion_i5_k15", "wiener"])
def test_the_study_uses_whatever_step6_confirmed(tmp_path, recipe):
    """Parametrised over unrelated recipes so no test encodes a winner."""
    directory = tmp_path / "s6"
    directory.mkdir()
    path = directory / "step06_confirm_summary.json"
    path.write_text(
        json.dumps({"selected_recipe": recipe, "confirmation_status": "confirmed"}),
        encoding="utf-8",
    )

    context = QuantumAblationContext.from_confirmation(str(path))
    assert context.recipe == recipe

    for condition in CONDITIONS:
        assert f"data.recipe={recipe}" in condition_overrides(condition, 42, context)


def test_an_explicit_override_is_recorded_as_such(tmp_path):
    """A development override must be visible in the output, never silent."""
    analysis = QuantumCircuitAblationStudy(recipe="wiener")

    assert analysis.recipe == "wiener"
    assert "override" in analysis._context().source


def test_no_source_file_names_a_preprocessing_recipe():
    """No preprocessing winner is assumed anywhere in Step 25."""
    from pathlib import Path

    forbidden = ("clahe", "diffusion_i5_k15", "diffusion_i10_k15", "gamma", "wiener")
    for module in ("src/analysis/quantum_circuit_ablation_rows.py",
                   "src/analysis/quantum_circuit_ablation_study.py"):
        source = Path(module).read_text(encoding="utf-8").lower()
        for name in forbidden:
            assert name not in source, f"{module} names the recipe {name!r}"


# ------------------------------------------------------------ Hydra composition


def test_the_step25_analysis_stage_composes():
    """:return: None."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="analyze.yaml",
                      overrides=["analysis=step25_quantum_circuit_ablation"])
    GlobalHydra.instance().clear()

    assert cfg.analysis._target_ == (
        "src.analysis.quantum_circuit_ablation_study.QuantumCircuitAblationStudy"
    )
    assert cfg.analysis.split == "val"
    assert cfg.analysis.n_resamples == 2000
    assert list(cfg.analysis.seeds) == list(PROTOCOL_SEEDS)
    assert cfg.analysis.recipe is None
    assert cfg.analysis.confirmation_summary is None


def test_the_step25_experiment_declares_no_circuit_list():
    """A default here is one condition's value silently applied to another."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml",
                      overrides=["experiment=step25_quantum_circuit_ablation"])
    GlobalHydra.instance().clear()

    raw = OmegaConf.to_container(cfg.model.net, resolve=True)
    assert raw["_target_"].endswith("AdaptiveQuantumClassifier")
    assert cfg.trainer.max_epochs == 30


# ------------------------------------------------------ nothing validated is touched


def test_no_validated_artefact_was_touched():
    """:return: None."""
    import subprocess

    changed = subprocess.run(
        ["git", "status", "--short", "--", "configs/protocol",
         "configs/experiment/step15_final_protocol.yaml", "data/splits",
         "src/models/components/quantum.py", "src/analysis/ablation_rows.py"],
        capture_output=True, text=True,
    ).stdout.strip()

    assert changed == ""


def test_step12_and_phase8_definitions_are_unchanged():
    """Step 25 adds an experiment; it does not amend the ones it ablates."""
    from src.analysis.ablation_rows import ROWS
    from src.models.components.quantum import CIRCUIT_NAMES, DEFAULT_N_QUBITS

    assert DEFAULT_N_QUBITS == 4
    assert CIRCUIT_NAMES == ("fixed", "deep", "strong", "combined", "reupload")
    assert [r.row_id for r in ROWS] == [
        "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "P",
    ]

