"""Step 21: the ablation evaluation, tested without training anything.

The scientific claim Step 21 makes is that its nine rows are *comparable* - to each other
and to the study's headline. Three things could break that silently:

1. **A different metric implementation.** "The same metrics for every configuration" has
   to mean the same code, or two implementations drift apart the first time one changes a
   zero-division policy and the table compares numbers that only look alike.

2. **Test-set contamination.** ``test: True`` in ``configs/train.yaml`` is never
   overridden, so ``trainer.test()`` runs after every training run and every
   ``metrics.csv`` already carries ``test/*`` columns. Harvesting those would be quick and
   would make the provenance of an ablation number unauditable - and worse, would not look
   wrong in the output. Checkpoints must come from validation and predictions must be
   recomputed here.

3. **A partial matrix reported as a full one.** A missing seed or an absent row has to
   surface in the summary rather than quietly shrink the table.

Nothing here trains, and nothing loads a real checkpoint: prediction is injected, so the
aggregation, schema and provenance logic are exercised on synthetic outputs.
"""

import json

import numpy as np
import pytest

from src.analysis import metric_battery as battery
from src.analysis.ablation_rows import ROWS, AblationContext
from src.analysis.ablation_study import (
    PRIMARY_METRICS,
    TABLE_COLUMNS,
    AblationStudy,
)

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No-tumor"]

CONTEXT = AblationContext(
    diffusion_recipe="diffusion_i10_k15",
    selected_recipe="clahe",
    step14_loss="weighted_ce",
)


def fake_predictions(seed: int, n: int = 120, skill: float = 1.4):
    """Deterministic synthetic predictions standing in for a model's output.

    :param seed: Varies the draw, so seeds differ as real ones would.
    :param n: Sample count.
    :param skill: How much probability mass lands on the true class.
    :return: ``{"y_true", "y_pred", "y_prob", "class_names"}``.
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, len(CLASS_NAMES), n)
    logits = rng.normal(size=(n, len(CLASS_NAMES)))
    logits[np.arange(n), y_true] += skill
    y_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    return {
        "y_true": y_true,
        "y_pred": y_prob.argmax(axis=1),
        "y_prob": y_prob,
        "class_names": list(CLASS_NAMES),
    }


@pytest.fixture
def study(tmp_path):
    """A study wired to a synthetic run tree, so nothing real is read or written.

    :param tmp_path: Per-test directory.
    :return: The study, with prediction and checkpoint lookup stubbed.
    """
    step06 = tmp_path / "step06.json"
    step06.write_text(
        json.dumps(
            {
                "selected_recipe": "clahe",
                "ranking": [
                    {"recipe": "clahe", "macro_f1": 0.257},
                    {"recipe": "diffusion_i10_k15", "macro_f1": 0.247},
                ],
            }
        ),
        encoding="utf-8",
    )
    step14 = tmp_path / "step14.json"
    step14.write_text(json.dumps({"selected_loss": "weighted_ce"}), encoding="utf-8")

    outputs = fake_predictions(0)
    metrics = battery.full_battery(
        outputs["y_true"], outputs["y_pred"], outputs["y_prob"], CLASS_NAMES
    )
    step16 = tmp_path / "step16.json"
    step16.write_text(
        json.dumps(
            {
                "checkpoint": "logs/train/runs/step15_final/seed_42/checkpoints/epoch_007.ckpt",
                "n_test_samples": metrics["n_samples"],
                "overall": metrics["overall"],
                "per_class": metrics["per_class"],
                "confusion": metrics["confusion"],
                "calibration": metrics["calibration"],
            }
        ),
        encoding="utf-8",
    )

    runs = tmp_path / "runs"
    for row in ROWS:
        if not row.trains and row.row_id != "A8":
            continue
        for seed in (42, 123, 7):
            (runs / row.row_id / f"seed_{seed}" / "checkpoints").mkdir(parents=True)
            (runs / row.row_id / f"seed_{seed}" / "checkpoints" / "epoch_003.ckpt").touch()

    analysis = AblationStudy(
        step06_summary=str(step06),
        step14_summary=str(step14),
        step16_summary=str(step16),
        run_root=str(runs),
    )
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    # Injected: the whole point is to exercise aggregation without a checkpoint or a GPU.
    analysis.predict_row = lambda row, seed, checkpoint, context: fake_predictions(
        seed + len(row.row_id)
    )
    return analysis


@pytest.fixture
def result(study):
    """:param study: The wired study.

    :return: The computed matrix.
    """
    return study.compute(datamodule=None)


# ------------------------------------------------------------------- completeness


def test_every_row_is_evaluated_exactly_once(result):
    """A0-A8 and P, no gaps, no duplicates, in the specification's order."""
    assert list(result["rows"]) == ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "P"]
    assert result["coverage"]["expected_rows"] == list(result["rows"])


def test_all_three_seeds_are_represented_for_the_trained_rows(result):
    """Step 15's seed protocol carries into the ablation, or the spreads are not comparable."""
    for row_id in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        assert sorted(result["rows"][row_id]["per_seed"], key=int) == ["7", "42", "123"]
        assert result["rows"][row_id]["across_seeds"]["macro_f1"]["n"] == 3


def test_a_missing_seed_is_reported_rather_than_hidden(study, tmp_path):
    """A thinner matrix must never pass for a complete one."""
    import shutil

    shutil.rmtree(tmp_path / "runs" / "A3" / "seed_7")
    summary = study.compute(datamodule=None)

    assert summary["rows"]["A3"]["missing_seeds"] == [7]
    assert summary["coverage"]["complete"] is False
    assert "A3" in summary["coverage"]["incomplete_rows"]


def test_a_missing_checkpoint_can_be_made_fatal(study, tmp_path):
    """``strict`` is for the final run, where an absent row is a failure, not a footnote."""
    import shutil

    shutil.rmtree(tmp_path / "runs" / "A3")
    study.strict = True

    with pytest.raises(FileNotFoundError):
        study.compute(datamodule=None)


# ------------------------------------------------------------- the test-set rule


def test_predictions_are_computed_here_not_harvested(result):
    """Every trained row records that its numbers came from this analysis.

    ``trainer.test()`` already wrote test/* into each run's metrics.csv. Reading those
    would be easy and would leave no trace in the output - so the provenance is asserted.
    """
    for row_id in ("A0", "A4", "A7"):
        for metrics in result["rows"][row_id]["per_seed"].values():
            assert metrics["provenance"].startswith("computed by step21")
            assert "validation-selected" in metrics["provenance"]


def test_checkpoint_selection_is_recorded_as_validation_based(result):
    """The selection rule travels with the number, so the table proves its own hygiene."""
    for row in result["rows"].values():
        for metrics in row["per_seed"].values():
            assert metrics["selection"] == "val/f1_macro (training-time ModelCheckpoint)"


def test_the_analysis_never_reads_a_training_metrics_csv():
    """A source-level guard: no path in Step 21 leads to a training run's metrics.csv."""
    from pathlib import Path

    import ast

    tree = ast.parse(Path("src/analysis/ablation_study.py").read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    # A training run's metrics.csv is the only CSV Step 21 could harvest from, and this
    # module never reads a CSV at all - it writes one. Checked on calls rather than on
    # string literals, because the module's prose legitimately names the file it refuses
    # to read.
    assert "read_csv" not in called
    assert not {"read_excel", "read_parquet"} & called

    # The one reader it does have is JSON, for the three summaries it is configured with.
    assert "_read_json" in called


def test_the_checkpoint_looked_up_is_the_best_not_the_last(study, tmp_path):
    """``last.ckpt`` is the final epoch; the best one is what validation selected."""
    directory = tmp_path / "runs" / "A0" / "seed_42" / "checkpoints"
    (directory / "last.ckpt").touch()

    seen = {}
    study.predict_row = lambda row, seed, checkpoint, context: (
        seen.__setitem__((row.row_id, seed), checkpoint) or fake_predictions(seed)
    )
    study.compute(datamodule=None)

    assert seen[("A0", 42)].name == "epoch_003.ckpt"


# ----------------------------------------------------------------- rows P and A8


def test_row_p_is_read_from_step16_and_not_re_evaluated(result, study):
    """The shipped model already spent the once-only test budget."""
    p = result["rows"]["P"]

    assert p["reused_from"] == "step16_internal"
    assert list(p["per_seed"]) == ["42"], "P should carry Step 16's single evaluated seed"
    assert p["per_seed"]["42"]["provenance"].startswith("reused from Step 16")
    assert p["recipe"] == "clahe"
    assert p["loss"] == "weighted_ce"


def test_row_p_reports_the_headline_numbers_unchanged(result, study):
    """P's macro-F1 must be Step 16's, to the digit."""
    step16 = json.loads(open(study.step16_summary, encoding="utf-8").read())

    assert result["rows"]["P"]["per_seed"]["42"]["overall"]["macro_f1"] == (
        step16["overall"]["macro_f1"]
    )


def test_row_p_flags_its_single_seed_coverage(result):
    """P has one seed and the A-rows have three; the asymmetry is stated, not hidden."""
    assert result["rows"]["P"]["missing_seeds"] == [123, 7]
    assert "single-seed" in result["notes"]["row_p_seeds"]


def test_a8_trains_nothing_and_mirrors_a7(result):
    """Explanations change no weights, so A8's metrics are A7's by construction."""
    a7, a8 = result["rows"]["A7"], result["rows"]["A8"]

    assert a8["trains"] is False
    assert a8["mirrors"] == "A7"
    assert a8["across_seeds"]["macro_f1"]["mean"] == a7["across_seeds"]["macro_f1"]["mean"]

    for metrics in a8["per_seed"].values():
        assert "identical to A7 by construction" in metrics["provenance"]


# --------------------------------------------------------------- rows and context


def test_the_diffusion_rows_carry_the_recipe_step6_ranked(result):
    """A2-A7 are diffusion rows; row P is not."""
    for row_id in ("A2", "A3", "A4", "A5", "A6", "A7"):
        assert result["rows"][row_id]["recipe"] == "diffusion_i10_k15"

    assert result["rows"]["P"]["recipe"] == "clahe"
    assert result["context"]["diffusion_recipe"] == "diffusion_i10_k15"


def test_a6_uses_plain_ce_and_a7_uses_step14s_loss(result):
    """The A6/A7 delta is the imbalance-aware loss and nothing else."""
    assert result["rows"]["A6"]["loss"] == "plain_ce"
    assert result["rows"]["A7"]["loss"] == "weighted_ce"


def test_the_fusion_rows_read_the_diffusion_cache(result):
    """Sharing the shipped model's cache would train them on CLAHE features."""
    assert result["rows"]["A6"]["feature_tag"] == "a6_diffusion"
    assert result["rows"]["A7"]["feature_tag"] == "a6_diffusion"
    assert result["rows"]["P"]["feature_tag"] == "default"


def test_the_fusion_rows_get_the_feature_datamodule(study):
    """A6/A7 are evaluated on cached features; A0-A5 on images at their own recipe."""
    from src.analysis.ablation_rows import get_row
    from src.data.bt_mri_datamodule import BTMRIDataModule
    from src.data.bt_mri_feature_datamodule import BTMRIFeatureDataModule

    fusion = study.build_datamodule(get_row("A6"), CONTEXT)
    assert isinstance(fusion, BTMRIFeatureDataModule)
    assert fusion.hparams.tag == "a6_diffusion"

    image = study.build_datamodule(get_row("A2"), CONTEXT)
    assert isinstance(image, BTMRIDataModule)
    assert image.hparams.recipe == "diffusion_i10_k15"


def test_a0_is_evaluated_without_normalization(study):
    """A0's raw condition has to survive into evaluation, not just training."""
    from src.analysis.ablation_rows import get_row

    assert study.build_datamodule(get_row("A0"), CONTEXT).hparams.normalize == "none"
    assert study.build_datamodule(get_row("A1"), CONTEXT).hparams.normalize == "imagenet"


def test_evaluation_never_augments_or_resamples(study):
    """Augmentation and balanced sampling are training-time only."""
    from src.analysis.ablation_rows import get_row

    for row_id in ("A0", "A1", "A2", "A5"):
        datamodule = study.build_datamodule(get_row(row_id), CONTEXT)
        assert datamodule.hparams.augment is False
        assert datamodule.hparams.use_weighted_sampler is False


# ------------------------------------------------------------------ the schema


def test_the_primary_metrics_are_the_ones_step21_names(result):
    """"macro-F1 and class-wise recall as primary selection metrics"."""
    assert result["primary_metrics"] == ["macro_f1", "per_class_recall"]
    assert PRIMARY_METRICS == ("macro_f1", "per_class_recall")

    for row_id in ("A0", "A7"):
        across = result["rows"][row_id]["across_seeds"]
        assert "macro_f1" in across
        assert set(across["per_class_recall"]) == set(CLASS_NAMES)


def test_the_full_step16_battery_is_reported_for_every_row(result):
    """Comparability with the headline requires the same metrics, not a subset."""
    expected = {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall_sensitivity",
        "macro_f1",
        "weighted_f1",
        "macro_specificity",
        "mcc",
        "auc_ovr_macro",
    }

    for row in result["rows"].values():
        for metrics in row["per_seed"].values():
            assert expected <= set(metrics["overall"])
            assert {"expected_calibration_error", "brier_score"} <= set(metrics["calibration"])
            assert metrics["confusion"]["class_names"] == CLASS_NAMES
            assert len(metrics["per_class"]) == len(CLASS_NAMES)


def test_the_metrics_are_step16s_own_code(result, study):
    """Not a reimplementation that happens to agree today.

    Both call src/analysis/metric_battery.py, so this asserts the numbers are identical to
    what Step 16 would produce on the same predictions.
    """
    from src.analysis.internal_test import InternalTest

    outputs = fake_predictions(42 + len("A0"))
    reference = InternalTest(name="step16_internal")
    reference._output_dir = study.output_dir

    assert result["rows"]["A0"]["per_seed"]["42"]["overall"] == reference._overall_metrics(
        outputs["y_true"], outputs["y_pred"], outputs["y_prob"], CLASS_NAMES
    )


def test_the_flat_table_schema_is_deterministic(study, result):
    """Downstream steps key on these columns; the order must not depend on dict iteration."""
    table = study.output_dir / "step21_ablation_matrix.csv"
    assert table.is_file()

    import pandas as pd

    frame = pd.read_csv(table)
    assert list(frame.columns) == list(TABLE_COLUMNS)
    assert list(frame.columns[:3]) == ["row_id", "label", "seed"]

    # Eight trained rows and A8 contribute one line per seed; P contributes its single
    # one. A8 is listed rather than omitted - it is a row of the specification's table -
    # and its provenance column says the numbers are A7's by construction.
    assert len(frame) == 9 * 3 + 1
    assert set(frame["row_id"]) == {row.row_id for row in ROWS}


def test_the_table_carries_the_worst_class_recall(study, result):
    """Class-wise recall is primary, so the minority-class number is a column, not a dig."""
    import pandas as pd

    frame = pd.read_csv(study.output_dir / "step21_ablation_matrix.csv")

    assert frame["min_class_recall"].notna().all()
    assert frame["worst_class"].isin(CLASS_NAMES).all()
    assert (frame["min_class_recall"] <= frame["macro_recall_sensitivity"] + 1e-9).all()


def test_per_seed_predictions_are_saved_for_the_paired_tests(study, result):
    """Step 23's paired bootstrap and McNemar need the predictions, not just the metrics."""
    saved = list(study.output_dir.glob("step21_predictions_*.npz"))

    # Eight trained rows x three seeds. A8 saves none: its predictions ARE A7's, and
    # writing them twice would imply two evaluations where there was one.
    assert len(saved) == 8 * 3
    assert not any("A8" in path.name for path in saved)
    payload = np.load(study.output_dir / "step21_predictions_A2_seed42.npz")
    assert set(payload) == {"y_true", "y_pred", "y_prob"}


def test_the_summary_states_that_a6_is_not_the_proposed_model(result):
    """The one thing a reader must not assume from the table's shape."""
    note = result["notes"]["a6_vs_p"]

    assert "diffusion_i10_k15" in note
    assert "clahe" in note
    assert "A6 is not the proposed model" in note


# -------------------------------------------------------------- across-seed stats


def test_seed_summaries_report_spread_and_an_interval():
    """Step 23: mean, standard deviation and a 95% interval for the main metrics."""
    summary = battery.summarise_seeds([0.90, 0.92, 0.94])

    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(0.92)
    assert summary["std"] == pytest.approx(0.02)
    assert summary["ci_low"] < summary["mean"] < summary["ci_high"]


def test_a_single_seed_reports_no_interval():
    """One point cannot support one, and pretending otherwise would understate uncertainty."""
    summary = battery.summarise_seeds([0.9])

    assert summary["n"] == 1
    assert summary["std"] == 0.0
    assert summary["ci_low"] is None and summary["ci_high"] is None


def test_missing_metrics_do_not_poison_the_summary():
    """AUC is None when a class is absent from a split; that must not become NaN."""
    assert battery.summarise_seeds([0.9, None, 0.7])["n"] == 2
    assert battery.summarise_seeds([None, None])["mean"] is None


# ---------------------------------------------------------------- required inputs


def test_a_missing_step6_summary_fails_clearly(tmp_path):
    """Without Step 6 there is no diffusion recipe, so there are no diffusion rows."""
    analysis = AblationStudy(step06_summary=None, step14_summary=None)

    with pytest.raises(FileNotFoundError, match="Step 6"):
        analysis.context()


def test_a_missing_step14_summary_fails_clearly(tmp_path):
    """Without Step 14 row A7 has no loss, and A6 vs A7 measures nothing."""
    step06 = tmp_path / "s6.json"
    step06.write_text(
        json.dumps({"selected_recipe": "clahe", "ranking": [{"recipe": "diffusion_i10_k15"}]}),
        encoding="utf-8",
    )
    analysis = AblationStudy(step06_summary=str(step06), step14_summary=str(tmp_path / "nope.json"))

    with pytest.raises(FileNotFoundError, match="Step 14"):
        analysis.context()


# ------------------------------------------------------------- Hydra composition


def compose_step21(*overrides: str):
    """Compose the Step 21 analysis config.

    :param overrides: Extra Hydra overrides.
    :return: The composed DictConfig.
    """
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="analyze.yaml",
            overrides=["analysis=step21_ablation", *overrides],
        )
    GlobalHydra.instance().clear()
    return cfg


def test_the_step21_stage_composes():
    """A stage that fails to compose fails hours into a run, not here."""
    analysis = compose_step21().analysis

    assert analysis._target_ == "src.analysis.ablation_study.AblationStudy"
    assert list(analysis.seeds) == [42, 123, 7]
    assert analysis.name == "step21_ablation"


def test_every_trained_row_has_a_module_config():
    """A row without one cannot be rebuilt from its checkpoint."""
    row_models = compose_step21().analysis.row_models

    for row in ROWS:
        if row.trains or row.row_id == "A8":
            assert row.row_id in row_models, f"{row.row_id} has no module config"


def test_the_module_configs_come_from_configs_model():
    """Restated architectures drift from the ones actually trained.

    The first draft of this config hand-copied two net blocks and got both ``_target_``s
    wrong - MultiScaleClassifier for MultiscaleClassifier.from_arm, and a FixedQCNNClassifier
    that does not exist. Composing configs/model/ instead makes that class of error
    impossible.
    """
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    row_models = compose_step21().analysis.row_models

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        real = {
            name: compose(config_name="train.yaml", overrides=[f"model={name}"]).model
            for name in (
                "branch_classical",
                "branch_multiscale",
                "baseline_fixed_qcnn",
                "branch_adaptive_quantum",
                "final_classifier",
            )
        }
    GlobalHydra.instance().clear()

    assert row_models.A0.net == real["branch_classical"].net
    assert row_models.A3.net == real["branch_multiscale"].net
    assert row_models.A4.net == real["baseline_fixed_qcnn"].net
    assert row_models.A5.net == real["branch_adaptive_quantum"].net
    assert row_models.A7.net == real["final_classifier"].net


def test_rows_sharing_an_architecture_share_a_config():
    """A0-A2 are the same network at three recipes; A6-A8 the same head."""
    row_models = compose_step21().analysis.row_models

    assert row_models.A0 == row_models.A1 == row_models.A2
    assert row_models.A6 == row_models.A7 == row_models.A8


def test_the_summaries_are_paths_not_baked_in_values():
    """Steps 6, 14 and 16 stay the source of truth; Step 21 only points at them."""
    analysis = compose_step21().analysis

    assert analysis.step06_summary is None
    assert analysis.step14_summary is None
    assert analysis.step16_summary is None


# ------------------------------------------------------- nothing validated is touched


def test_step21_changes_no_validated_artefact():
    """Phase 8 adds an analysis; it does not amend the protocol or the splits."""
    import subprocess

    changed = subprocess.run(
        ["git", "status", "--short", "--", "configs/protocol", "configs/experiment/step15_final_protocol.yaml", "data/splits"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert changed == "", f"validated artefacts were modified:\n{changed}"


def test_an_unexpected_metric_key_cannot_leak_into_the_schema(study):
    """The column list is the authority, not whatever the record dict happened to carry.

    Without the pinned ``columns=`` argument the table's shape would follow dict insertion
    order, so adding a field anywhere upstream would silently change the schema that
    Steps 22 and 23 read.
    """
    from src.analysis.ablation_rows import get_row

    rows = {
        row.row_id: {
            "row_id": row.row_id,
            "recipe": "x",
            "loss": "y",
            "per_seed": {
                "42": {
                    "overall": {"macro_f1": 0.5},
                    "per_class": [{"class_name": "Glioma", "recall_sensitivity": 0.5}],
                    "calibration": {},
                    "surprise_column": "should not appear",
                }
            },
        }
        for row in ROWS
    }

    frame = study._flat_table(rows, CONTEXT)

    assert list(frame.columns) == list(TABLE_COLUMNS)
    assert "surprise_column" not in frame.columns
    assert get_row("A0").row_id in set(frame["row_id"])


def test_an_empty_matrix_still_has_the_full_schema(study):
    """Before anything is trained the table is empty - but it is not shapeless.

    Steps 22 and 23 read these columns. An empty DataFrame without them turns "nothing has
    run yet" into a KeyError several steps downstream, far from the cause.
    """
    frame = study._flat_table({row.row_id: {} for row in ROWS}, CONTEXT)

    assert len(frame) == 0
    assert list(frame.columns) == list(TABLE_COLUMNS)
