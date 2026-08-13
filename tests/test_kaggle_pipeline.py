"""Tests for the one-command study driver, ``scripts/kaggle_pipeline.py``.

The driver restates things that live elsewhere in the repository: the seven baselines,
the eight ablation arms, the Step 8 strategy table, the identity recipes. Restating them
is what makes the driver readable, but it is also how a graph silently stops matching the
study - add a ninth arm and the pipeline would just never train it. These tests fail when
the two drift apart.

The rest checks the mechanics that are easy to get wrong and expensive to discover on
Kaggle at hour eleven: stage-id uniqueness, the ``--only``/``--from``/``--until``/``--skip``
selectors, and Lightning's ``max_time`` format.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from src.analysis.imbalance_study import STRATEGIES
from src.data.components.preprocessing import IDENTITY_RECIPES
from src.models.components.multiscale import ARMS

ROOT = Path(__file__).resolve().parents[1]


def _load_pipeline():
    """:return: The driver module, imported from ``scripts/`` which is not a package."""
    path = ROOT / "scripts" / "kaggle_pipeline.py"
    spec = importlib.util.spec_from_file_location("kaggle_pipeline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so register before executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kp = _load_pipeline()


def _pipeline(**overrides):
    """:param overrides: Argument values to change from the defaults.

    :return: A pipeline configured as if from the command line.
    """
    args = kp.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return kp.Pipeline(args)


# --------------------------------------------------------------------------- drift


def test_arms_match_the_ablation_definition():
    """Every arm the study defines is trained, and no arm that does not exist."""
    assert list(kp.ARMS) == list(ARMS)


def test_quantum_arms_are_the_ones_marked_quantum():
    """The arms kept off the accelerator are exactly the ones running circuits."""
    expected = {name for name, (_, quantum) in ARMS.items() if quantum}
    assert kp.QUANTUM_ARMS == expected


def test_baselines_match_the_configs_on_disk():
    """The seven Step 9 baselines are the seven ``baseline_*`` configs."""
    configs = {p.stem for p in (ROOT / "configs" / "model").glob("baseline_*.yaml")}
    assert set(kp.BASELINES) == configs


def test_imbalance_table_matches_the_study():
    """Step 8 names a strategy; the driver turns it into overrides. They must agree."""
    for name, spec in STRATEGIES.items():
        assert name in kp.IMBALANCE_OVERRIDES, f"{name} has no override mapping"
        loss, sampler, augment = kp.IMBALANCE_OVERRIDES[name]
        assert loss == spec["loss"]
        assert sampler == spec["sampler"]
        assert augment == spec["augment"]


def test_identity_recipes_match_the_preprocessing_module():
    """A recipe needing no mirror must not be sent to prepare_dataset."""
    assert tuple(kp.IDENTITY_RECIPES) == tuple(IDENTITY_RECIPES)


def test_entrypoints_exist():
    """Every entry point the graph invokes is a real file."""
    for relative in kp.ENTRYPOINTS.values():
        assert (ROOT / relative).is_file(), relative


# ---------------------------------------------------------------------- the graph


def test_stage_ids_are_unique():
    """Two stages sharing an id would share a run directory and overwrite each other."""
    stages = kp.build_stages(_pipeline(profile="full"))
    ids = [s.id for s in stages]
    assert len(ids) == len(set(ids))


def test_full_profile_covers_every_step():
    """The graph runs the whole implemented study, Steps 4 through 18."""
    groups = {s.group for s in kp.build_stages(_pipeline(profile="full"))}
    assert groups == {
        "step04",
        "step06",
        "step08",
        "step09",
        "step10",
        "step11",
        "step12",
        "features",
        "step13",
        "step14",
        "step15",
        "step16",
        "step17",
        "step18",
    }


def test_full_profile_trains_three_seeds():
    """Step 15: "run at least three seeds for the final model and major baselines"."""
    pipe = _pipeline(profile="full")
    assert pipe.seeds == [42, 123, 7]
    finals = [s for s in kp.build_stages(pipe) if s.group == "step15"]
    assert len(finals) == 3


def test_shortened_profiles_are_flagged_as_unreportable():
    """Only the full profile leaves the fixed protocol alone."""
    assert _pipeline(profile="full").protocol_intact
    assert not _pipeline(profile="fast").protocol_intact
    assert not _pipeline(profile="smoke").protocol_intact

    assert _pipeline(profile="full").train_shape() == []
    assert any("max_epochs" in o for o in _pipeline(profile="fast").train_shape())


def test_shortened_profiles_cannot_be_mistaken_for_the_real_study():
    """A smoke run must not leave completion markers a later full run would honour.

    The markers are what make the pipeline resumable, so they are also how a one-epoch
    run could silently satisfy `--profile full`: every stage would report itself already
    done. Separate roots make that impossible, and the feature cache is separated for the
    same reason - Steps 13 to 15 train over whatever tensors sit in data/features/<tag>.
    """
    full = _pipeline(profile="full")
    assert full.log_root == ROOT / "logs"
    assert full.tag == "default"

    for profile in ("smoke", "fast"):
        short = _pipeline(profile=profile)
        assert short.log_root == ROOT / "logs" / f"_{profile}"
        assert short.tag == profile

        stage = next(s for s in kp.build_stages(short) if s.id == "step04_audit")
        assert full.out_dir(stage) != short.out_dir(stage)


def test_an_explicit_tag_is_always_respected():
    """``--tag`` is how a user separates caches by hand; the profile must not override it."""
    assert _pipeline(profile="smoke", tag="experiment7").tag == "experiment7"


def test_quantum_models_stay_on_the_cpu_simulator():
    """Sending a circuit model to the GPU moves tensors back and forth every batch."""
    pipe = _pipeline()
    pipe.gpu = True
    assert pipe.trainer_override(quantum=False) == ["trainer=gpu"]
    assert pipe.trainer_override(quantum=True) == ["trainer=default"]


def test_feature_datamodule_gets_no_augment_flag():
    """``bt_mri_features`` has no ``augment`` key; overriding it would abort the run."""
    pipe = _pipeline(imbalance="augmentation")
    assert any("data.augment" in o for o in pipe.imbalance_overrides(with_data=True))
    assert not any("data.augment" in o for o in pipe.imbalance_overrides(with_data=False))


def test_selections_can_be_forced_from_the_command_line():
    """``--recipe``/``--imbalance`` win over whatever the studies wrote."""
    pipe = _pipeline(recipe="clahe", imbalance="class_weighting")
    assert pipe.recipe_override() == ["data.recipe=clahe"]
    assert "loss@model.criterion=weighted_ce" in pipe.imbalance_overrides()

    assert _pipeline(recipe="null").recipe_override() == ["data.recipe=null"]


def test_selections_are_ignored_when_switched_off():
    """``--no-apply-selections`` leaves each stage on its config's own defaults."""
    pipe = _pipeline(apply_selections=False)
    assert pipe.imbalance_overrides() == []


# ------------------------------------------------------------------- the selectors


@pytest.fixture
def stages():
    """:return: The full stage graph."""
    return kp.build_stages(_pipeline(profile="full"))


def test_only_accepts_a_group(stages):
    """``--only step09`` selects the whole Step 9 group."""
    args = kp.parse_args(["--only", "step09"])
    chosen = kp.select_stages(stages, args)
    assert chosen and all(s.group == "step09" for s in chosen)


def test_only_accepts_a_single_stage(stages):
    """``--only <id>`` selects exactly that stage."""
    args = kp.parse_args(["--only", "step16_internal"])
    assert [s.id for s in kp.select_stages(stages, args)] == ["step16_internal"]


def test_from_starts_partway_down(stages):
    """``--from`` drops everything before the first match."""
    args = kp.parse_args(["--from", "step13_fusion"])
    chosen = kp.select_stages(stages, args)
    assert chosen[0].id == "step13_fusion"
    assert chosen[-1].id == "step18_robustness"


def test_until_keeps_the_last_match(stages):
    """``--until`` on a group stops after that group's final stage."""
    args = kp.parse_args(["--until", "step09"])
    chosen = kp.select_stages(stages, args)
    assert chosen[-1].group == "step09"
    assert "step10_embeddings" not in {s.id for s in chosen}


def test_skip_removes_groups(stages):
    """``--skip`` takes the same tokens as ``--only``."""
    args = kp.parse_args(["--skip", "step11,step17"])
    groups = {s.group for s in kp.select_stages(stages, args)}
    assert "step11" not in groups and "step17" not in groups
    assert "step12" in groups


def test_unknown_selector_fails_loudly(stages):
    """A typo in ``--from`` must not silently run the whole study."""
    with pytest.raises(SystemExit):
        kp.select_stages(stages, kp.parse_args(["--from", "step99_nonexistent"]))


# ------------------------------------------------------------------------ details


@pytest.mark.parametrize(
    "seconds,expected",
    [(90, "00:00:01:30"), (3600, "00:01:00:00"), (86_400 + 61, "01:00:01:01"), (5, "00:00:01:00")],
)
def test_max_time_uses_lightnings_format(seconds, expected):
    """Lightning parses ``DD:HH:MM:SS``; anything else is silently ignored."""
    assert kp._hms(seconds) == expected


def test_checkpoints_are_never_bundled():
    """The results archive must stay small enough to download."""
    assert ".ckpt" not in kp.BUNDLE_SUFFIXES
    assert ".pt" not in kp.BUNDLE_SUFFIXES


def test_summary_headlines_can_read_nested_fields():
    """Step 16 nests its metrics; a flat lookup would report nothing."""
    payload = {"overall": {"macro_f1": 0.9}, "flat": 1}
    assert kp._dig(payload, "overall.macro_f1") == 0.9
    assert kp._dig(payload, "flat") == 1
    assert kp._dig(payload, "overall.missing") is None
    assert kp._dig(payload, "missing.deeper") is None


def test_report_reads_the_pinned_directory_not_whatever_it_can_find(tmp_path):
    """A stale summary from an earlier timestamped run must not reach the report.

    Every stage runs into a pinned directory, but ``logs/`` also holds run directories
    from manual invocations. Globbing for ``step14_loss_selection_summary.json`` would
    find several and report contradictory answers to the same question.
    """
    pinned = tmp_path / "logs" / "analyze" / "runs" / "step14_loss_selection"
    pinned.mkdir(parents=True)
    (pinned / "step14_loss_selection_summary.json").write_text(
        '{"selected_loss": "weighted_ce", "rationale": "current"}'
    )

    stale = tmp_path / "logs" / "analyze" / "runs" / "2026-01-01_00-00-00"
    stale.mkdir(parents=True)
    (stale / "step14_loss_selection_summary.json").write_text(
        '{"selected_loss": "focal", "rationale": "from an older run"}'
    )

    pipe = _pipeline()
    pipe.root = tmp_path
    pipe.log_root = tmp_path / "logs"
    pipe.pipeline_dir = tmp_path / "logs" / "pipeline"
    pipe.pipeline_dir.mkdir(parents=True)

    report = kp.write_report(pipe).read_text(encoding="utf-8")
    assert "weighted_ce" in report
    assert "focal" not in report


def test_preflight_refuses_to_start_without_the_dataset(tmp_path):
    """A missing dataset must stop the run, not produce twenty identical failures.

    Every stage reads the same image tree. Without this guard a session with no data
    attached burns twenty-odd stages twelve seconds apart and writes a report that reads
    like the study collapsed rather than like a setup mistake.
    """
    pipe = _pipeline()
    pipe.root = tmp_path

    with pytest.raises(SystemExit, match="Preflight failed"):
        pipe.preflight()

    # An empty directory left behind by a failed download must not count as success.
    (tmp_path / "data" / "raw" / "bt_mri").mkdir(parents=True)
    with pytest.raises(SystemExit, match="holds 0 images"):
        pipe.preflight()


def test_preflight_passes_once_images_are_present(tmp_path):
    """:param tmp_path: Temporary project root."""
    classes = tmp_path / "data" / "raw" / "bt_mri" / "Training" / "Glioma"
    classes.mkdir(parents=True)
    for index in range(100):
        (classes / f"{index}.jpg").write_bytes(b"")

    pipe = _pipeline()
    pipe.root = tmp_path
    pipe.preflight()  # must not raise


def test_out_dir_is_pinned_and_derived_from_the_stage_id():
    """Downstream stages address checkpoints by path, so the path cannot be a timestamp."""
    pipe = _pipeline(profile="full")
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step15_final/seed_42")
    assert pipe.out_dir(stage) == ROOT / "logs" / "train" / "runs" / "step15_final" / "seed_42"
