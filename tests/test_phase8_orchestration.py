"""Phase 8 orchestration: the ablation ladder wired into the runner.

Phases 8.1-8.4 built the pieces and pinned their scientific content. This file guards the
wiring, where a different class of mistake lives: the components are correct and the runner
feeds them the wrong thing.

The specific hazard is inheritance. Every training stage in ``scripts/kaggle_pipeline.py``
applies ``recipe_override()`` and ``imbalance_overrides()``, which inject Step 6's and
Step 8's *selections*. That is right for the main study and catastrophic for an ablation:
row A2 would be labelled "diffusion" and trained on whatever won Step 6, and the label
would still read "diffusion" in the final table. The real run already demonstrates the
mechanism - Step 9's EfficientNet trained with ``data.recipe=clahe`` despite its experiment
config declaring ``recipe: null``.

So every A-row stage is composed here, under Hydra, and its recipe, normalization,
augmentation, sampler, loss, seed, cache tag and protocol are asserted against the row
manifest rather than against the pipeline's selections.

The other half is order. Step 21 may not read checkpoints that do not exist, Step 23
consumes Step 21, and Step 22 consumes both. The dependency graph is asserted positionally
and by the failures each stage raises when its inputs are absent.

Nothing here launches training: stages are built and composed, never executed.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.analysis.ablation_rows import DIFFUSION_FEATURE_TAG, PROTOCOL_SEEDS, ROWS
from tests.helpers.completed_runs import mark_run_complete

ROOT = Path(__file__).resolve().parents[1]

#: Rows the pipeline must train. A8 and P train nothing.
TRAINING_ROWS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")

#: The recipe Step 6's ranking supplies in the fixture, and which A2-A7 must use.
DIFFUSION = "diffusion_i10_k15"


def load_pipeline():
    """:return: The runner module, imported from ``scripts/`` which is not a package."""
    path = ROOT / "scripts" / "kaggle_pipeline.py"
    spec = importlib.util.spec_from_file_location("kaggle_pipeline_phase8", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kp = load_pipeline()


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    """A pipeline rooted in a temporary tree, with Steps 6 and 14 already decided.

    Step 6 selects CLAHE and ranks diffusion second - the real run's outcome, and the case
    that matters: the A-rows must use diffusion anyway.

    :param tmp_path: Per-test directory.
    :param monkeypatch: Used to redirect the runner's ROOT.
    :return: The pipeline.
    """
    monkeypatch.setattr(kp, "ROOT", tmp_path)
    args = kp.parse_args([])
    args.profile = "full"
    pipeline = kp.Pipeline(args)

    def summary(stage, payload):
        path = pipeline.summary_path(stage, "analyze", f"{stage}_summary.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    summary("step06_preprocessing", {
        "selected_recipe": "clahe",
        "ranking": [{"recipe": "clahe", "macro_f1": 0.257},
                    {"recipe": DIFFUSION, "macro_f1": 0.247},
                    {"recipe": "conventional", "macro_f1": 0.100}],
    })
    summary("step14_loss_selection", {"selected_loss": "weighted_ce"})
    summary("step08_imbalance", {"selected_strategy": "combined_sampler_weighting"})
    return pipeline


def stages(pipeline):
    """:param pipeline: The pipeline.

    :return: The full stage graph.
    """
    return kp.build_stages(pipeline)


def stage_by_id(pipeline, stage_id):
    """:param pipeline: The pipeline.

    :param stage_id: Stage identifier.
    :return: That stage.
    """
    return next(s for s in stages(pipeline) if s.id == stage_id)


def compose_stage(stage, pipeline, config_name="train.yaml"):
    """Compose the config a stage would actually run under.

    :param stage: The stage.
    :param pipeline: The pipeline.
    :param config_name: Root config for the stage's entry point.
    :return: The composed DictConfig.
    """
    overrides = stage.build(pipeline)
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    GlobalHydra.instance().clear()
    return cfg


def row_stage(pipeline, row_id, seed=42):
    """:param pipeline: The pipeline.

    :param row_id: Ablation row.
    :param seed: Protocol seed.
    :return: That row's training stage.
    """
    return stage_by_id(pipeline, f"step21_ablation/{row_id}/seed_{seed}")


# ------------------------------------------------------------------ the stages


def test_every_trainable_row_has_a_stage_for_every_protocol_seed(pipe):
    """Eight rows times three seeds; a missing one would silently thin the ladder."""
    ids = {s.id for s in stages(pipe) if s.is_train}

    for row_id in TRAINING_ROWS:
        for seed in PROTOCOL_SEEDS:
            assert f"step21_ablation/{row_id}/seed_{seed}" in ids


def test_a8_creates_no_training_stage(pipe):
    """Explanations change no weights; there is nothing to train."""
    assert not any("A8" in s.id for s in stages(pipe))


def test_p_creates_no_training_stage(pipe):
    """P is the shipped model, already trained by Step 15 and tested by Step 16."""
    ablation = [s for s in stages(pipe) if s.id.startswith("step21_ablation/")]

    assert not any(s.id.endswith("/P") or "/P/" in s.id for s in ablation)


def test_the_ablation_stages_use_the_protocol_seed_set(pipe):
    """:param pipe: The pipeline."""
    assert pipe.seeds == list(PROTOCOL_SEEDS)

    for row_id in TRAINING_ROWS:
        seeds = {
            int(s.id.rsplit("_", 1)[1])
            for s in stages(pipe)
            if s.id.startswith(f"step21_ablation/{row_id}/seed_")
        }
        assert seeds == set(PROTOCOL_SEEDS)


# ----------------------------------------------------- no inherited selections


@pytest.mark.parametrize("row_id", TRAINING_ROWS)
def test_no_a_row_inherits_step6s_recipe_selection(pipe, row_id):
    """Step 6 chose CLAHE. The diffusion rows must train on diffusion regardless.

    This is the failure the whole ablation exists to avoid: a row labelled "diffusion"
    trained on whatever won Step 6, with nothing in the output to show it.
    """
    from src.analysis.ablation_rows import get_row

    overrides = row_stage(pipe, row_id).build(pipe)

    if get_row(row_id).feature_tag is None:
        assert sum(o.startswith("data.recipe=") for o in overrides) == 1, (
            f"{row_id} must pin exactly one recipe"
        )
    else:
        # A6/A7 read cached features; their preprocessing is fixed by which cache they read.
        assert sum(o.startswith("data.tag=") for o in overrides) == 1
    assert "data.recipe=clahe" not in overrides


@pytest.mark.parametrize("row_id", TRAINING_ROWS)
def test_no_a_row_inherits_step8s_imbalance_selection(pipe, row_id):
    """The fixture's Step 8 selected combined_sampler_weighting; no row may follow it."""
    overrides = row_stage(pipe, row_id).build(pipe)

    assert sum(o.startswith("loss@model.criterion=") for o in overrides) == 1
    assert sum(o.startswith("data.use_weighted_sampler=") for o in overrides) == 1
    assert "data.use_weighted_sampler=true" not in overrides
    assert "loss@model.criterion=weighted_ce" not in overrides or row_id == "A7"


@pytest.mark.parametrize("row_id", TRAINING_ROWS)
def test_a_row_composition_pins_augmentation_off(pipe, row_id):
    """Augmentation is a Step 8 lever; the rows hold it fixed to match the shipped model."""
    cfg = compose_stage(row_stage(pipe, row_id), pipe)

    assert cfg.data.use_weighted_sampler is False
    if row_id not in ("A6", "A7"):
        assert cfg.data.augment is False


# --------------------------------------------------- what each row composes to


def test_a0_and_a1_remain_different(pipe):
    """The distinction the reference notebook could not express."""
    a0 = compose_stage(row_stage(pipe, "A0"), pipe)
    a1 = compose_stage(row_stage(pipe, "A1"), pipe)

    assert a0.data.normalize == "none"
    assert a1.data.normalize == "imagenet"
    assert a0.data.recipe is None and a1.data.recipe is None
    assert a0.data.normalize != a1.data.normalize


@pytest.mark.parametrize("row_id", ["A2", "A3", "A4", "A5"])
def test_the_diffusion_image_rows_compose_the_diffusion_recipe(pipe, row_id):
    """:param pipe: The pipeline.

    :param row_id: Ablation row.
    """
    cfg = compose_stage(row_stage(pipe, row_id), pipe)

    assert cfg.data.recipe == DIFFUSION
    assert cfg.data.normalize == "imagenet"


@pytest.mark.parametrize("row_id", ["A6", "A7"])
def test_the_fusion_rows_compose_the_diffusion_cache(pipe, row_id):
    """Sharing the shipped model's cache would train them on CLAHE-derived features."""
    cfg = compose_stage(row_stage(pipe, row_id), pipe)

    assert cfg.data.tag == DIFFUSION_FEATURE_TAG
    assert cfg.data.tag != "default"


def test_a6_uses_plain_ce_and_a7_uses_step14s_loss(pipe):
    """The A6/A7 delta is the loss and nothing else."""
    a6 = compose_stage(row_stage(pipe, "A6"), pipe)
    a7 = compose_stage(row_stage(pipe, "A7"), pipe)

    assert a6.model.criterion.use_class_weights is False
    assert a7.model.criterion.use_class_weights is True
    assert a6.data.tag == a7.data.tag
    assert a6.model.net == a7.model.net


@pytest.mark.parametrize("row_id", TRAINING_ROWS)
def test_every_row_composes_the_fixed_protocol(pipe, row_id):
    """One protocol for every row, read from Step 15's own composition."""
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        reference = compose(config_name="train.yaml",
                            overrides=["experiment=step15_final_protocol"])
    GlobalHydra.instance().clear()

    cfg = compose_stage(row_stage(pipe, row_id), pipe)

    assert cfg.trainer.max_epochs == reference.trainer.max_epochs
    assert cfg.callbacks.early_stopping.patience == reference.callbacks.early_stopping.patience
    assert cfg.callbacks.model_checkpoint.monitor == reference.callbacks.model_checkpoint.monitor
    assert cfg.model.optimizer.lr == reference.model.optimizer.lr
    assert cfg.data.batch_size == reference.data.batch_size


@pytest.mark.parametrize("row_id", TRAINING_ROWS)
def test_every_row_composes_its_declared_model(pipe, row_id):
    """:param pipe: The pipeline.

    :param row_id: Ablation row.
    """
    from src.analysis.ablation_rows import get_row

    cfg = compose_stage(row_stage(pipe, row_id), pipe)

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        declared = compose(config_name="train.yaml",
                           overrides=[f"model={get_row(row_id).model}"]).model
    GlobalHydra.instance().clear()

    assert cfg.model.net._target_ == declared.net._target_


@pytest.mark.parametrize("row_id", TRAINING_ROWS)
def test_every_row_composes_its_own_seed(pipe, row_id):
    """:param pipe: The pipeline.

    :param row_id: Ablation row.
    """
    for seed in PROTOCOL_SEEDS:
        assert compose_stage(row_stage(pipe, row_id, seed), pipe).seed == seed


def test_the_quantum_rows_stay_on_the_cpu_simulator(pipe):
    """A4 and A5 run circuits; moving them to a GPU shuttles tensors every batch."""
    for row_id in ("A4", "A5"):
        assert "trainer=default" in row_stage(pipe, row_id).build(pipe)


# ------------------------------------------------------------ prerequisites


def test_the_diffusion_mirror_is_materialised_before_the_rows_that_need_it(pipe):
    """A2-A5 read data/processed/<diffusion>; something has to write it."""
    order = [s.id for s in stages(pipe)]
    mirror = "step21_materialise_diffusion"

    assert mirror in order
    assert order.index(mirror) < order.index("step21_ablation/A2/seed_42")
    assert compose_stage(stage_by_id(pipe, mirror), pipe, "prepare_dataset.yaml").recipe == DIFFUSION


def test_the_a6_feature_cache_is_built_from_the_diffusion_branches(pipe):
    """A6's branches must be A2's and A5's, not the shipped CLAHE ones."""
    stage = stage_by_id(pipe, "step21_ablation/features")
    order = [s.id for s in stages(pipe)]

    assert order.index(stage.id) > order.index("step21_ablation/A5/seed_42")
    assert order.index(stage.id) < order.index("step21_ablation/A6/seed_42")

    for row in ("A2", "A5"):
        mark_run_complete(
            pipe.log_root / "train" / "runs" / "step21_ablation" / row / "seed_42"
        )

    overrides = stage.build(pipe)
    assert f"tag={DIFFUSION_FEATURE_TAG}" in overrides
    assert any(o.startswith("classical_ckpt=") and "/A2/seed_42" in o for o in overrides)
    assert any(o.startswith("quantum_ckpt=") and "/A5/seed_42" in o for o in overrides)
    assert f"data.recipe={DIFFUSION}" in overrides


def test_the_feature_stage_refuses_to_run_without_its_branches(pipe):
    """Building the cache from absent checkpoints would produce an empty or stale cache."""
    with pytest.raises(kp.StageFailed, match="A2|A5"):
        stage_by_id(pipe, "step21_ablation/features").build(pipe)


# ------------------------------------------------------------------- ordering


def test_the_phase8_stage_order_is_the_declared_dependency_graph(pipe):
    """A0-A7, then Step 21, then Step 23, then Step 22."""
    order = [s.id for s in stages(pipe)]

    def at(stage_id):
        return order.index(stage_id)

    for row_id in TRAINING_ROWS:
        for seed in PROTOCOL_SEEDS:
            assert at(f"step21_ablation/{row_id}/seed_{seed}") < at("step21_ablation")

    assert at("step21_ablation") < at("step23_statistics")
    assert at("step23_statistics") < at("step22_rq_mapping")


def test_the_rows_are_ordered_a0_through_a7(pipe):
    """A ladder read out of order is a ladder a reader has to reassemble."""
    order = [s.id for s in stages(pipe)]
    positions = [
        order.index(f"step21_ablation/{row_id}/seed_42") for row_id in TRAINING_ROWS
    ]

    assert positions == sorted(positions)


def test_phase8_runs_after_the_rest_of_the_study(pipe):
    """Nothing in Phase 8 may inform an earlier selection."""
    order = [s.group for s in stages(pipe)]

    for earlier in ("step15", "step16", "step18", "step20"):
        assert order.index(earlier) < order.index("step21")


def test_the_graph_is_deterministic(pipe):
    """Same inputs, same graph - including order."""
    assert [s.id for s in stages(pipe)] == [s.id for s in stages(pipe)]


def test_step21_refuses_to_evaluate_before_the_rows_are_trained(pipe):
    """No evaluation may read checkpoints that do not exist."""
    with pytest.raises(kp.StageFailed, match="A0|checkpoint"):
        stage_by_id(pipe, "step21_ablation").build(pipe)


# --------------------------------------------------- the three analysis stages


def _train_all_rows(pipe):
    """Mark every trainable row and seed as a training run that finished.

    A checkpoint directory alone is what an interrupted run leaves; Step 21 asks for
    completion, so these fixtures have to record it.

    :param pipe: The pipeline.
    """
    for row_id in TRAINING_ROWS:
        for seed in PROTOCOL_SEEDS:
            mark_run_complete(
                pipe.log_root / "train" / "runs" / "step21_ablation" / row_id / f"seed_{seed}"
            )


def test_step21_is_wired_to_the_ablation_run_root_and_the_prior_summaries(pipe):
    """:param pipe: The pipeline."""
    _train_all_rows(pipe)
    cfg = compose_stage(stage_by_id(pipe, "step21_ablation"), pipe, "analyze.yaml")

    assert cfg.analysis._target_ == "src.analysis.ablation_study.AblationStudy"
    assert cfg.analysis.run_root.endswith("step21_ablation")
    assert cfg.analysis.step06_summary.endswith("step06_preprocessing_summary.json")
    assert cfg.analysis.step14_summary.endswith("step14_loss_selection_summary.json")
    assert cfg.analysis.step16_summary.endswith("step16_internal_summary.json")
    assert list(cfg.analysis.seeds) == list(PROTOCOL_SEEDS)


def test_step21_reuses_p_from_step16_rather_than_retraining_it(pipe):
    """P is the shipped model; the pipeline points Step 21 at Step 16's summary."""
    _train_all_rows(pipe)
    overrides = stage_by_id(pipe, "step21_ablation").build(pipe)

    assert any("step16_internal_summary.json" in o for o in overrides)
    assert not any("/P/" in o for o in overrides)


def _finish(pipe, stage_id):
    """Write the summary a downstream stage waits on.

    :param pipe: The pipeline.
    :param stage_id: Stage whose output to fake.
    """
    directory = pipe.log_root / "analyze" / "runs" / stage_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stage_id}_summary.json").write_text("{}", encoding="utf-8")


def test_step23_is_wired_to_step21s_output(pipe):
    """:param pipe: The pipeline."""
    _finish(pipe, "step21_ablation")
    cfg = compose_stage(stage_by_id(pipe, "step23_statistics"), pipe, "analyze.yaml")

    assert cfg.analysis._target_ == "src.analysis.statistical_report.StatisticalReport"
    assert cfg.analysis.ablation_dir.endswith("step21_ablation")
    assert [(h.id, h.row_a, h.row_b) for h in cfg.analysis.primary_comparisons] == [
        ("H1", "A2", "A1"), ("H2", "A5", "A4"), ("H3", "A7", "A6"), ("H4", "A6", "A3"),
    ]


def test_step22_is_wired_to_the_analysis_root(pipe):
    """:param pipe: The pipeline."""
    _finish(pipe, "step21_ablation")
    _finish(pipe, "step23_statistics")
    cfg = compose_stage(stage_by_id(pipe, "step22_rq_mapping"), pipe, "analyze.yaml")

    assert cfg.analysis._target_ == "src.analysis.rq_mapping.RQMapping"
    assert cfg.analysis.analyze_root.endswith("runs")
    assert len(cfg.analysis.research_questions) == 10


def test_step23_refuses_before_step21_has_produced_anything(pipe):
    """:param pipe: The pipeline."""
    with pytest.raises(kp.StageFailed, match="Step 21"):
        stage_by_id(pipe, "step23_statistics").build(pipe, )


def test_step22_refuses_before_step23_has_produced_anything(pipe):
    """:param pipe: The pipeline."""
    _finish(pipe, "step21_ablation")

    with pytest.raises(kp.StageFailed, match="Step 23"):
        stage_by_id(pipe, "step22_rq_mapping").build(pipe)


def test_the_pipeline_does_not_reimplement_the_statistics_or_the_mapping():
    """Those belong to Steps 22 and 23; the runner only points at them."""
    import ast

    tree = ast.parse((ROOT / "scripts" / "kaggle_pipeline.py").read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    # Checked on machinery rather than on words: the runner's comments legitimately explain
    # what Steps 22 and 23 do, and a substring scan cannot tell prose from logic.
    assert "scipy" not in imported and "sklearn" not in imported
    assert not {"holm_bonferroni", "mcnemar_test", "paired_bootstrap", "wilcoxon_paired",
                "bootstrap_ci"} & called

    # The manifest is the one thing it does import from src, and only to emit overrides.
    assert not {"statistical_report", "rq_mapping", "ablation_study"} & {
        (node.module or "").rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }


# ---------------------------------------------------------- output namespace


def test_phase8_writes_only_under_the_phase8_namespace(pipe):
    """:param pipe: The pipeline."""
    for stage in stages(pipe):
        if stage.group in ("step21", "step22", "step23"):
            assert str(pipe.out_dir(stage)).startswith(str(pipe.log_root))


def test_phase8_never_writes_into_an_existing_result_bundle(pipe):
    """The shipped bundles are immutable evidence; nothing may land inside them."""
    for stage in stages(pipe):
        target = str(pipe.out_dir(stage))
        assert "thesis_results_20260813_090056" not in target
        assert "thesis_results_20260814_075721" not in target


def test_the_existing_result_bundles_are_untouched_on_disk():
    """A committed bundle that changed would corrupt the study's record."""
    import subprocess

    changed = subprocess.run(
        ["git", "status", "--short", "--", "thesis_results_20260813_090056"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout

    assert not any(line.startswith(" M") for line in changed.splitlines())


def test_no_validated_artefact_was_touched():
    """:return: None."""
    import subprocess

    changed = subprocess.run(
        ["git", "status", "--short", "--", "configs/protocol",
         "configs/experiment/step15_final_protocol.yaml", "data/splits"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout.strip()

    assert changed == ""


# ------------------------------------------------------------------- hygiene


def test_building_the_graph_launches_nothing(pipe, monkeypatch):
    """Stage construction must be pure; only run() may start a process."""
    def forbidden(*args, **kwargs):
        raise AssertionError("building the stage graph must not launch a subprocess")

    monkeypatch.setattr(kp.subprocess, "Popen", forbidden)
    monkeypatch.setattr(kp.subprocess, "run", forbidden)

    _train_all_rows(pipe)
    for stage in stages(pipe):
        if stage.group in ("step21", "step22", "step23"):
            try:
                stage.build(pipe)
            except kp.StageFailed:
                pass


def test_the_full_profile_now_covers_phase8(pipe):
    """:param pipe: The pipeline."""
    groups = {s.group for s in stages(pipe)}

    assert {"step21", "step22", "step23"} <= groups


def test_every_ablation_row_in_the_manifest_is_accounted_for(pipe):
    """A row added to the manifest must reach the pipeline or be explicitly untrained."""
    ids = {s.id for s in stages(pipe)}

    for row in ROWS:
        trained = any(f"step21_ablation/{row.row_id}/seed_" in sid for sid in ids)
        assert trained == row.trains, f"{row.row_id}: trains={row.trains} but staged={trained}"
