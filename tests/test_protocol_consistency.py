"""Regression tests binding the evaluation stages to their configured inputs.

Two classes of silent failure are guarded here, both of which produce a plausible-looking
result rather than an error:

1. **Step 20's control drifting from the Step 15 protocol.** If the control is trained with
   a different learning rate, epoch budget, stopping rule, checkpoint-selection rule or
   loss than the model it is compared against, the measured "quantum contribution" is
   partly a training-schedule artefact. Every one of the differences originally present
   handicapped the control, which biases toward a false positive for quantum advantage.

2. **Step 18 reading a different dataset than the one configured.** If the degraded loader
   falls back to default paths, robustness gets reported for a tree or a split the model
   was never evaluated on.

Step 15 is the source of truth throughout. These tests read the real composed Step 15
config and assert Step 20 matches it, so amending the protocol fails the suite instead of
invalidating Step 20 unnoticed.
"""

from pathlib import Path
from typing import Any, Dict

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


def compose_config(config_name: str, *overrides: str):
    """Compose a config in isolation from other tests.

    :param config_name: Root config file name.
    :param overrides: Hydra override strings.
    :return: The composed DictConfig.
    """
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    GlobalHydra.instance().clear()
    return cfg


@pytest.fixture(scope="module")
def step15():
    """:return: The real, fully composed Step 15 training config - the source of truth."""
    return compose_config("train.yaml", "experiment=step15_final_protocol")


@pytest.fixture(scope="module")
def step20():
    """:return: The composed Step 20 analysis config."""
    return compose_config("analyze.yaml", "analysis=step20_quantum_advantage")


# ------------------------------------------------------- Step 20 protocol parity


def test_step20_epoch_budget_matches_step15(step15, step20):
    """A shorter budget would under-train the control and inflate the quantum delta."""
    assert step20.analysis.protocol.max_epochs == step15.trainer.max_epochs
    assert step20.analysis.protocol.min_epochs == step15.trainer.min_epochs


def test_step20_early_stopping_matches_step15(step15, step20):
    """Training the control without early stopping is a different protocol."""
    assert step20.analysis.protocol.patience == step15.callbacks.early_stopping.patience
    assert step20.analysis.protocol.monitor == step15.callbacks.early_stopping.monitor
    assert step20.analysis.protocol.mode == step15.callbacks.early_stopping.mode


def test_step20_checkpoint_selection_matches_step15(step15, step20):
    """The full model is the best-validation epoch, so the control must be too.

    Scoring the control's final epoch against the full model's best epoch is a different
    selection rule, and one that can only disadvantage the control.
    """
    assert step20.analysis.protocol.monitor == step15.callbacks.model_checkpoint.monitor
    assert step20.analysis.protocol.mode == step15.callbacks.model_checkpoint.mode


def test_step20_batch_size_and_sampler_match_step15(step15, step20):
    """Both change the effective optimisation problem, so both must be held fixed."""
    assert step20.analysis.protocol.batch_size == step15.data.batch_size
    assert step20.analysis.protocol.use_weighted_sampler == step15.data.use_weighted_sampler


def test_step20_optimizer_matches_step15(step15, step20):
    """The original control ran at 1e-3 against Step 15's 1e-4 - a 10x difference."""
    optimizer = step20.analysis.fusion_model.optimizer

    assert optimizer._target_ == step15.model.optimizer._target_
    assert optimizer.lr == step15.model.optimizer.lr
    assert optimizer.weight_decay == step15.model.optimizer.weight_decay


def test_step20_scheduler_matches_step15(step15, step20):
    """A cosine cycle of a different length decays the rate differently."""
    scheduler = step20.analysis.fusion_model.scheduler

    assert scheduler._target_ == step15.model.scheduler._target_
    assert scheduler.T_max == step15.model.scheduler.T_max


def test_step20_loss_matches_step15(step15, step20):
    """The control previously trained on an UNWEIGHTED cross-entropy.

    Step 15 trains with class weighting, so the control was solving a different objective -
    the single most consequential of the original mismatches.
    """
    criterion = step20.analysis.fusion_model.criterion

    assert criterion._target_ == step15.model.criterion._target_
    assert criterion.use_class_weights == step15.model.criterion.use_class_weights


def test_step20_architecture_matches_step15(step15, step20):
    """The control must differ from the proposed model ONLY by the quantum branch.

    Anything else - projection width, hidden dims, dropout - would confound the ablation
    with a capacity change.
    """
    control = step20.analysis.fusion_model.net
    proposed = step15.model.net

    assert control._target_ == proposed._target_
    for field in ("proj_dim", "hidden_dims", "num_classes", "dropout"):
        assert control[field] == proposed[field], f"{field} differs between control and model"


def test_step20_scheduler_cycle_tracks_its_own_epoch_budget(step20):
    """A cosine cycle shorter or longer than the run decays at the wrong rate."""
    assert step20.analysis.fusion_model.scheduler.T_max == step20.analysis.protocol.max_epochs


# ------------------------------------------------- Step 18 dataset-path propagation


class _RecordingDataModule:
    """Captures the keyword arguments a degraded datamodule is constructed with."""

    captured: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).captured = dict(kwargs)
        self.hparams = type("H", (), kwargs)

    def setup(self, stage: Any = None) -> None:
        """:param stage: Unused."""

    def test_dataloader(self):
        """:return: An empty loader; this test only inspects construction."""
        return []


def test_step18_propagates_the_configured_dataset_paths(monkeypatch):
    """The degraded loader must read the configured tree and split, not the defaults.

    With a custom `raw_subdir` or `split_subpath`, falling back to defaults would evaluate
    robustness on a different dataset - or a different split - than the rest of the study,
    and report it as though it were the same.
    """
    import src.analysis.robustness as robustness

    monkeypatch.setattr(robustness, "DegradedTestDataModule", _RecordingDataModule)
    monkeypatch.setattr(robustness, "predict", lambda *a, **k: {
        "y_true": __import__("numpy").array([0, 1]),
        "y_pred": __import__("numpy").array([0, 1]),
    })

    study = robustness.RobustnessStudy(models={"m": {"kind": "simple"}})
    study._evaluate(
        model=None,
        spec={},
        dataset_paths={
            "data_dir": "/custom/data",
            "raw_subdir": "raw/somewhere_else",
            "split_subpath": "splits/custom_split.csv",
        },
        degradation=None,
        device="cpu",
    )

    captured = _RecordingDataModule.captured
    assert captured["data_dir"] == "/custom/data"
    assert captured["raw_subdir"] == "raw/somewhere_else"
    assert captured["split_subpath"] == "splits/custom_split.csv"


def test_step18_dataset_paths_come_from_the_datamodule(monkeypatch, tmp_path):
    """`compute` must read all three paths off the configured datamodule."""
    import src.analysis.robustness as robustness

    recorded = {}

    def fake_evaluate(self, model, spec, dataset_paths, degradation, device):
        recorded.update(dataset_paths)
        return 0.5

    monkeypatch.setattr(robustness.RobustnessStudy, "_evaluate", fake_evaluate)
    monkeypatch.setattr(robustness.RobustnessStudy, "_load", lambda self, spec, device: None)
    monkeypatch.setattr(robustness.RobustnessStudy, "_plot", lambda self, results: None)

    class Source:
        hparams = type(
            "H",
            (),
            {
                "data_dir": "/data/root",
                "raw_subdir": "raw/custom_tree",
                "split_subpath": "splits/custom.csv",
            },
        )
        class_names = ["Glioma", "Meningioma", "Pituitary", "No-tumor"]

    study = robustness.RobustnessStudy(
        models={"proposed": {"kind": "simple"}}, categories=["clean"]
    )
    # Into tmp_path, not the repo root: the study writes its results table on the way
    # through.
    study._output_dir = tmp_path
    study.compute(Source())

    assert recorded == {
        "data_dir": "/data/root",
        "raw_subdir": "raw/custom_tree",
        "split_subpath": "splits/custom.csv",
    }


# ---------------------------------------------- Step 20 loss and seed provenance


def _study(**kwargs):
    """Build a Step 20 study with the given overrides.

    :param kwargs: Constructor overrides.
    :return: The study.
    """
    from src.analysis.quantum_advantage import QuantumAdvantageStudy

    return QuantumAdvantageStudy(**kwargs)


def _write_step14_summary(tmp_path, selected: str):
    """Write a minimal Step 14 summary.

    :param tmp_path: Directory to write into.
    :param selected: The loss Step 14 selected.
    :return: Path to the summary file.
    """
    import json

    path = tmp_path / "step14_loss_selection_summary.json"
    path.write_text(json.dumps({"selected_loss": selected}), encoding="utf-8")
    return path


def test_step20_follows_step14s_actual_selection(tmp_path):
    """Step 15's loss is read from Step 14 at run time, so Step 20 must be too.

    `scripts/kaggle_pipeline.py` calls `selected_loss()` and passes
    `loss@model.criterion=<name>` to Step 15, raising if Step 14 never ran. A hard-coded
    loss in Step 20 would silently diverge the moment Step 14's answer changed.
    """
    from src.models.components.losses import FocalLoss

    summary = _write_step14_summary(tmp_path, "focal")
    study = _study(loss_summary=str(summary))

    name, source = study._resolve_loss()
    assert name == "focal"
    assert "Step 14" in source

    config = {"criterion": {"_target_": "src.models.components.losses.CrossEntropyLoss"}}
    record = study._apply_loss(config)

    assert isinstance(config["criterion"], FocalLoss), "criterion was not replaced"
    assert record["loss"] == "focal"


def test_step20_weighted_ce_selection_is_honoured(tmp_path):
    """The case the current smoke run produced: Step 14 chose weighted_ce."""
    from src.models.components.losses import CrossEntropyLoss

    study = _study(loss_summary=str(_write_step14_summary(tmp_path, "weighted_ce")))
    config = {"criterion": {}}
    record = study._apply_loss(config)

    assert record["loss"] == "weighted_ce"
    assert isinstance(config["criterion"], CrossEntropyLoss)
    assert config["criterion"].use_class_weights is True


def test_step20_explicit_loss_override_wins(tmp_path):
    """Mirrors the pipeline's `--loss` flag taking precedence over the summary."""
    study = _study(loss="focal", loss_summary=str(_write_step14_summary(tmp_path, "weighted_ce")))
    name, source = study._resolve_loss()

    assert name == "focal"
    assert "override" in source


def test_step20_warns_when_the_loss_is_unverified(caplog):
    """Falling back to the config is permitted but must never be silent."""
    study = _study()
    name, _ = study._resolve_loss()
    assert name is None

    record = study._apply_loss({"criterion": {"_target_": "x"}})
    assert record["loss"] is None
    assert "unverified" in record["source"]


def test_step20_detects_a_seed_mismatch_with_the_checkpoint():
    """The pipeline evaluates a FIXED seed, so the control must train at that same seed.

    `_pipeline_ckpts` uses `pipe.seeds[0]`, not the best of the three Step 15 seeds - so a
    single-seed control is fair, but only while the two seeds agree. Reordering `--seeds`
    would otherwise compare a seed-7 checkpoint against a seed-42 control and fold a seed
    effect into the reported quantum delta.
    """
    matched = _study(seed=42, fusion_ckpt="logs/train/runs/step15_final/seed_42")
    assert matched._check_seed_matches_checkpoint()["seeds_match"] is True

    mismatched = _study(seed=42, fusion_ckpt="logs/train/runs/step15_final/seed_7")
    check = mismatched._check_seed_matches_checkpoint()
    assert check["seeds_match"] is False
    assert check["checkpoint_seed"] == 7
    assert check["control_seed"] == 42


def test_step20_seed_check_tolerates_an_unlabelled_checkpoint():
    """A hand-specified path carries no seed; that must not be reported as a mismatch."""
    check = _study(seed=42, fusion_ckpt="/some/where/model.ckpt")._check_seed_matches_checkpoint()

    assert check["checkpoint_seed"] is None
    assert check["seeds_match"] is True


def test_pipeline_evaluates_a_fixed_seed_not_the_best_one():
    """Guards the claim that a single-seed control is fair.

    If the runner ever switched to picking the best-performing seed for downstream
    evaluation, comparing it against one control would give the proposed model a selection
    advantage, and this test should fail to force a rethink.
    """
    source = Path("scripts/kaggle_pipeline.py").read_text(encoding="utf-8")

    assert "seed = pipe.seeds[0]" in source, "downstream evaluation is no longer a fixed seed"
    assert '{"smoke": [42], "fast": [42], "full": [42, 123, 7]}' in source


# ------------------------------------------------ Step 14 -> Step 20 through the pipeline


def _kp():
    """:return: The pipeline driver, imported from ``scripts/`` which is not a package."""
    import importlib.util
    import sys

    path = Path("scripts/kaggle_pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("kaggle_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wired_pipeline(tmp_path, monkeypatch, selected=None, **overrides):
    """Build a pipeline whose log tree lives under ``tmp_path``.

    :param tmp_path: Directory to use as the repo root, so nothing is written to the repo.
    :param monkeypatch: Fixture used to redirect the driver's ROOT.
    :param selected: Loss to write into a Step 14 summary, or ``None`` to write no summary.
    :param overrides: Command-line argument values to change.
    :return: ``(module, pipeline)``.
    """
    import json

    kp = _kp()
    monkeypatch.setattr(kp, "ROOT", tmp_path)
    args = kp.parse_args([])
    for key, value in {"profile": "full", **overrides}.items():
        setattr(args, key, value)
    pipe = kp.Pipeline(args)

    if selected is not None:
        run = pipe.summary_path(*kp.STEP14_SUMMARY)
        run.parent.mkdir(parents=True, exist_ok=True)
        run.write_text(json.dumps({"selected_loss": selected}), encoding="utf-8")

    # The three checkpoints Step 20 compares; only their presence is checked.
    for stage in ("step10_classical", "step12_adaptive_quantum", "step15_final"):
        (pipe.log_root / "train" / "runs" / stage / "seed_42" / "checkpoints").mkdir(
            parents=True, exist_ok=True
        )
    return kp, pipe


def _step20_overrides(kp, pipe):
    """:param kp: The driver module.

    :param pipe: The pipeline.
    :return: The Hydra overrides the Step 20 stage would be launched with.
    """
    stage = next(s for s in kp.build_stages(pipe) if s.group == "step20")
    return stage.build(pipe)


def test_pipeline_hands_step20_the_loss_step14_actually_chose(tmp_path, monkeypatch):
    """The run that exists today: Step 14 chose weighted_ce, so Step 20 must use it.

    The pipeline passes the summary *path* rather than the resolved name, so Step 14 stays
    the single source of truth and Step 20 records where its loss came from.
    """
    kp, pipe = _wired_pipeline(tmp_path, monkeypatch, selected="weighted_ce")
    overrides = _step20_overrides(kp, pipe)

    summary_arg = next(o for o in overrides if o.startswith("analysis.loss_summary="))
    path = Path(summary_arg.split("=", 1)[1])

    assert path.is_file()
    import json

    assert json.loads(path.read_text(encoding="utf-8"))["selected_loss"] == "weighted_ce"

    # And Step 15 is fed the same file, so the two provably agree.
    assert pipe.selected_loss() == "weighted_ce"


def test_a_future_step14_choice_reaches_step20_without_a_code_change(tmp_path, monkeypatch):
    """The margin in the real run was 0.001 macro-F1, so this case is not hypothetical."""
    kp, pipe = _wired_pipeline(tmp_path, monkeypatch, selected="focal")
    overrides = _step20_overrides(kp, pipe)

    summary_arg = next(o for o in overrides if o.startswith("analysis.loss_summary="))
    import json

    payload = json.loads(Path(summary_arg.split("=", 1)[1]).read_text(encoding="utf-8"))

    assert payload["selected_loss"] == "focal"
    assert pipe.selected_loss() == "focal"

    # End to end: the study resolves that file to focal and builds the matching criterion.
    from src.models.components.losses import FocalLoss

    study = _study(loss_summary=summary_arg.split("=", 1)[1])
    config = {"criterion": {}}
    assert study._apply_loss(config)["loss"] == "focal"
    assert isinstance(config["criterion"], FocalLoss)


def test_the_pipeline_refuses_to_run_step20_without_step14(tmp_path, monkeypatch):
    """Failing loudly beats a control trained on an unverified objective.

    Step 15 raises the same way. If Step 20 fell back to its config instead, the run would
    finish and produce a quantum-advantage number nobody could defend.
    """
    kp, pipe = _wired_pipeline(tmp_path, monkeypatch, selected=None)

    with pytest.raises(kp.StageFailed, match="Step 14"):
        _step20_overrides(kp, pipe)

    with pytest.raises(kp.StageFailed, match="Step 14"):
        next(s for s in kp.build_stages(pipe) if s.group == "step15").build(pipe)


def test_an_explicit_loss_override_reaches_step20_too(tmp_path, monkeypatch):
    """``--loss`` bypasses Step 14 for Step 15, so it must bypass it for Step 20 as well."""
    kp, pipe = _wired_pipeline(tmp_path, monkeypatch, selected="weighted_ce", loss="focal")
    overrides = _step20_overrides(kp, pipe)

    assert "analysis.loss=focal" in overrides
    assert not any(o.startswith("analysis.loss_summary=") for o in overrides)
    assert pipe.selected_loss() == "focal"


def test_the_pipeline_names_no_loss_of_its_own_for_step15_or_step20(tmp_path, monkeypatch):
    """Neither stage may carry a literal loss name; both must read Step 14's answer.

    A hard-coded ``weighted_ce`` would be correct today and silently wrong the first time
    Step 14 chose otherwise - which is exactly the failure this wiring exists to prevent.
    """
    kp, pipe = _wired_pipeline(tmp_path, monkeypatch, selected="focal")

    step20 = _step20_overrides(kp, pipe)
    step15 = next(s for s in kp.build_stages(pipe) if s.group == "step15").build(pipe)

    assert not any("weighted_ce" in o for o in step20)
    assert "loss@model.criterion=focal" in step15, "Step 15 did not follow Step 14 either"


def test_step20_is_wired_to_the_same_fixed_seed_as_every_other_stage(tmp_path, monkeypatch):
    """The seed-matched comparison has to survive the stage being run automatically."""
    kp, pipe = _wired_pipeline(tmp_path, monkeypatch, selected="weighted_ce")
    overrides = _step20_overrides(kp, pipe)

    ckpt = next(o for o in overrides if o.startswith("analysis.fusion_ckpt="))

    assert ckpt.endswith(f"step15_final/seed_{pipe.seeds[0]}")
    assert pipe.seeds[0] == 42


def test_step19_and_step20_run_after_the_evaluation_steps(tmp_path, monkeypatch):
    """Ordering is the protocol: nothing downstream may inform an earlier selection.

    Both stages only read finalized checkpoints, so running them automatically cannot
    change any result - but only while they stay at the end of the graph.
    """
    kp, pipe = _wired_pipeline(tmp_path, monkeypatch, selected="weighted_ce")
    groups = [s.group for s in kp.build_stages(pipe)]

    for earlier in ("step14", "step15", "step16", "step18"):
        assert groups.index(earlier) < groups.index("step19")
    assert groups.index("step19") < groups.index("step20")


@pytest.mark.parametrize("group", ["step19", "step20"])
def test_the_wired_stages_actually_compose(tmp_path, monkeypatch, group):
    """Every override the pipeline emits must exist in the config it targets.

    Found this way: Step 20 was initially given the classical and quantum checkpoints too,
    but its config carries only ``fusion_ckpt`` - the branches reach it through the feature
    cache. Hydra rejects an unknown key, so the stage would have failed on Kaggle, hours
    in, rather than here.
    """
    kp, pipe = _wired_pipeline(tmp_path, monkeypatch, selected="weighted_ce")
    stage = next(s for s in kp.build_stages(pipe) if s.group == group)

    cfg = compose_config("analyze.yaml", *stage.build(pipe))

    assert cfg.analysis.name == stage.id


def test_the_shortened_profiles_never_weaken_step20s_control(tmp_path, monkeypatch):
    """A smaller epoch budget for the control would fake a quantum advantage.

    Smoke and fast thin the bootstrap, which only widens the interval. The control's
    training protocol stays at Step 15's in every profile, because a control trained for
    fewer epochs loses for a reason that has nothing to do with the quantum branch.
    """
    full = compose_config("analyze.yaml", "analysis=step20_quantum_advantage").analysis.protocol

    for profile in ("smoke", "fast"):
        kp, pipe = _wired_pipeline(
            tmp_path / profile, monkeypatch, selected="weighted_ce", profile=profile
        )
        stage = next(s for s in kp.build_stages(pipe) if s.group == "step20")
        cfg = compose_config("analyze.yaml", *stage.build(pipe))

        assert cfg.analysis.protocol == full, f"{profile} altered the control's protocol"
        assert cfg.analysis.n_resamples < full_resamples(), f"{profile} did not thin the bootstrap"


def full_resamples() -> int:
    """:return: The bootstrap resample count Step 20 uses when nothing is shortened."""
    return compose_config("analyze.yaml", "analysis=step20_quantum_advantage").analysis.n_resamples
