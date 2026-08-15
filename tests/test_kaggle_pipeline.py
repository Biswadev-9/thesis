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
import os
import shutil
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
    """The graph runs the whole implemented study, Steps 4 through 20."""
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
        "step19",
        "step20",
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
    assert chosen[-1].id == stages[-1].id  # and keeps everything after it


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


def test_only_a_full_run_resumes_from_a_checkpoint(tmp_path):
    """Mid-run resume is for stages measured in hours, not seconds.

    A shortened profile's leftover last.ckpt comes from a run that did not finish, so
    restoring it reproduces whatever went wrong. Restarting costs seconds.
    """
    for profile, expect_resume in (("full", True), ("smoke", False), ("fast", False)):
        root = tmp_path / profile
        pipe = _pipeline(profile=profile)
        pipe.root = root
        pipe.log_root = root / "logs"
        pipe.pipeline_dir = root / "logs" / "pipeline"

        stage = next(s for s in kp.build_stages(pipe) if s.is_train)
        (pipe.out_dir(stage) / "checkpoints").mkdir(parents=True)
        (pipe.out_dir(stage) / "checkpoints" / "last.ckpt").write_bytes(b"")

        seen = []
        pipe._spawn = lambda argv, log_path: (seen.append(argv), (0, False))[1]
        pipe.run_stage(stage)

        resumed = any(a.startswith("ckpt_path=") for a in seen[0])
        assert resumed is expect_resume, f"{profile} should resume={expect_resume}"


def test_dataloader_workers_default_to_zero():
    """Workers pass tensors through /dev/shm, which is small in Kaggle's container.

    When it fills they block rather than raise, and the stage hangs silently until the
    session is killed. Throughput is not worth a lost session, so the default is safe and
    raising it is a deliberate act.
    """
    assert kp.parse_args([]).num_workers == 0
    assert kp.parse_args(["--num-workers", "4"]).num_workers == 4


def test_only_the_smoke_profile_polices_stage_duration():
    """A smoke stage runs one epoch on three batches; hours means a deadlock.

    A full run's quantum stages legitimately take hours, so a timeout there would kill
    real work.
    """
    assert _pipeline(profile="smoke").stage_timeout == 20 * 60
    assert _pipeline(profile="fast").stage_timeout is None
    assert _pipeline(profile="full").stage_timeout is None


def test_stage_timeout_can_be_set_or_disabled_by_hand():
    """:return: None."""
    assert _pipeline(profile="full", stage_timeout=45).stage_timeout == 45 * 60
    assert _pipeline(profile="smoke", stage_timeout=0).stage_timeout is None


def test_a_hung_stage_is_killed_and_reported(tmp_path):
    """The watchdog must end a deadlocked stage rather than let it eat the session."""
    pipe = _pipeline(profile="smoke", stage_timeout=0.02)  # ~1.2 s
    pipe.root = tmp_path
    pipe.log_root = tmp_path / "logs"
    pipe.pipeline_dir = tmp_path / "logs" / "pipeline"

    kp.HEARTBEAT_SECONDS, original = 0.2, kp.HEARTBEAT_SECONDS
    try:
        code, hung = pipe._spawn(
            [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "stage.log"
        )
    finally:
        kp.HEARTBEAT_SECONDS = original

    assert hung, "the watchdog should have killed it"
    assert code != 0


def test_stage_output_goes_to_the_log_not_the_console(tmp_path, capsys):
    """Forwarding every stage's output to a notebook wedges it.

    Jupyter rate-limits stdout. Once the kernel stops consuming, the pipeline blocks on
    write, stops draining the child's pipe, the child fills the 64 KB buffer and blocks
    too - both frozen for good. The full output still reaches stage.log.
    """
    pipe = _pipeline(profile="smoke")
    pipe.root = tmp_path
    pipe.log_root = tmp_path / "logs"
    pipe.pipeline_dir = tmp_path / "logs" / "pipeline"
    log = tmp_path / "stage.log"

    shout = "print('x' * 200)\n" * 50
    code, hung = pipe._spawn([sys.executable, "-c", shout], log)

    assert (code, hung) == (0, False)
    assert log.read_text().count("x" * 200) == 50, "everything must reach the log"
    assert "x" * 200 not in capsys.readouterr().out, "and none of it the console"


def test_verbose_forwards_stage_output(tmp_path, capsys):
    """The escape hatch for debugging a single stage."""
    pipe = _pipeline(profile="smoke", verbose=True)
    pipe.root = tmp_path
    pipe.log_root = tmp_path / "logs"
    pipe.pipeline_dir = tmp_path / "logs" / "pipeline"

    pipe._spawn([sys.executable, "-c", "print('visible')"], tmp_path / "stage.log")
    assert "visible" in capsys.readouterr().out


def test_a_failing_stage_still_shows_its_error(tmp_path, capsys):
    """Silencing stage output must not silence the reason a stage failed."""
    pipe = _pipeline(profile="smoke")
    pipe.root = tmp_path
    pipe.log_root = tmp_path / "logs"
    pipe.pipeline_dir = tmp_path / "logs" / "pipeline"

    log = tmp_path / "stage.log"
    log.write_text("Traceback (most recent call last):\nValueError: the real reason\n")
    pipe._tail(log)

    assert "the real reason" in capsys.readouterr().out


def test_the_config_tree_dump_is_switched_off(tmp_path):
    """~120 lines per stage of config Hydra already saved to disk. Volume is the hazard."""
    pipe = _pipeline(profile="smoke")
    pipe.root = tmp_path
    pipe.log_root = tmp_path / "logs"
    pipe.pipeline_dir = tmp_path / "logs" / "pipeline"

    stage = next(s for s in kp.build_stages(pipe) if s.is_train)
    seen = []
    pipe._spawn = lambda argv, log_path: (seen.append(argv), (0, False))[1]
    pipe.run_stage(stage)

    assert "extras.print_config=false" in seen[0]


def test_a_healthy_stage_is_not_killed(tmp_path):
    """:param tmp_path: Temporary directory."""
    pipe = _pipeline(profile="smoke")
    pipe.root = tmp_path
    pipe.log_root = tmp_path / "logs"
    pipe.pipeline_dir = tmp_path / "logs" / "pipeline"

    code, hung = pipe._spawn(
        [sys.executable, "-c", "print('done')"], tmp_path / "stage.log"
    )
    assert (code, hung) == (0, False)
    assert "done" in (tmp_path / "stage.log").read_text()


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


def test_report_distinguishes_two_metrics_of_the_same_name(tmp_path):
    """Step 17 reports a restricted and an unrestricted macro-F1.

    Labelling both `macro_f1` printed the same name twice with no way to tell which was
    which - and the two answer different questions about the same model.
    """
    pinned = tmp_path / "logs" / "analyze" / "runs" / "step17_external"
    pinned.mkdir(parents=True)
    (pinned / "step17_external_summary.json").write_text(
        '{"restricted": {"macro_f1": 0.61}, "unrestricted": {"macro_f1": 0.42}}'
    )

    pipe = _pipeline()
    pipe.root = tmp_path
    pipe.log_root = tmp_path / "logs"
    pipe.pipeline_dir = tmp_path / "logs" / "pipeline"

    report = kp.write_report(pipe).read_text(encoding="utf-8")
    assert "restricted.macro_f1: `0.61`" in report
    assert "unrestricted.macro_f1: `0.42`" in report


# ------------------------------------------------------------- finding the dataset


def _make_dataset(root, images_per_class=30, classes=("Glioma", "Meningioma", "Pituitary", "No-tumor")):
    """Build a dataset tree shaped like the real archive.

    :param root: Directory that becomes the dataset root.
    :param images_per_class: Files written per class per split.
    :param classes: Class folder names to create.
    :return: The dataset root.
    """
    for split in ("Training", "Testing"):
        for name in classes:
            folder = root / split / name
            folder.mkdir(parents=True)
            for index in range(images_per_class):
                (folder / f"{name}_{index}.jpg").write_bytes(b"")
    return root


def _make_kaggle_mount(tmp_path):
    """Reproduce the mount layout observed on Kaggle.

    ``/kaggle/input/datasets/<owner>/<slug>/BT-MRI Dataset/BT-MRI Dataset/`` - six levels
    down, and one level past what the loader can descend on its own. This exact shape is
    what produced "No images found" from a correctly attached dataset.

    :param tmp_path: Temporary directory standing in for ``/kaggle/input``.
    :return: ``(input_root, true_dataset_root)``.
    """
    mount = tmp_path / "input"
    nested = (
        mount
        / "datasets"
        / "mohamadabouali1"
        / "mri-brain-tumor-dataset-4-class-7023-images"
        / "BT-MRI Dataset"
        / "BT-MRI Dataset"
    )
    nested.mkdir(parents=True)
    return mount, _make_dataset(nested)


def test_finds_the_dataset_however_deeply_kaggle_nests_it(tmp_path):
    """The search must reach the real Kaggle layout, not just the tidy one."""
    mount, expected = _make_kaggle_mount(tmp_path)
    found, inspected = kp.find_dataset_root(mount)
    assert found == expected
    assert inspected, "the inspected list feeds the error message"


def test_finds_the_dataset_at_the_shallow_layout_too(tmp_path):
    """Some accounts mount at /kaggle/input/<slug>/ with no extra nesting."""
    mount = tmp_path / "input"
    expected = _make_dataset(mount / "mri-brain-tumor-dataset-4-class-7023-images")
    found, _ = kp.find_dataset_root(mount)
    assert found == expected


def test_a_root_needs_both_split_folders(tmp_path):
    """Training alone is not a dataset root."""
    partial = tmp_path / "input" / "ds"
    for name in ("Glioma", "Meningioma", "Pituitary", "No-tumor"):
        (partial / "Training" / name).mkdir(parents=True)
    assert not kp.is_dataset_root(partial)
    assert kp.find_dataset_root(tmp_path / "input")[0] is None


def test_a_root_needs_all_four_classes(tmp_path):
    """Three classes means the wrong tree, or a truncated download."""
    partial = tmp_path / "input" / "ds"
    partial.mkdir(parents=True)
    _make_dataset(partial, classes=("Glioma", "Meningioma", "Pituitary"))
    assert not kp.is_dataset_root(partial)


def test_the_degraded_copies_are_never_selected(tmp_path):
    """The archive ships blurred copies under the same four class names.

    They have no Training/Testing division, which is the only thing separating them from
    the real data. Selecting one would train the study on deliberately corrupted images.
    """
    mount = tmp_path / "input"
    challenging = mount / "ds" / "Challenging Datasets" / "Blurred Dataset"
    for name in ("Glioma", "Meningioma", "Pituitary", "No-tumor"):
        folder = challenging / name
        folder.mkdir(parents=True)
        (folder / "blur.jpg").write_bytes(b"")

    real = _make_dataset(mount / "ds" / "BT-MRI Dataset")
    assert kp.find_dataset_root(mount)[0] == real


def test_class_folder_spellings_agree_with_the_loader():
    """The driver restates the loader's class-name normalisation; they must not drift."""
    from src.data.components.split_builder import normalize_class_folder

    for name in ("Glioma", "glioma", "No-tumor", "notumor", "no_tumor", "NO TUMOR",
                 "meningioma_tumor", "Pituitary", "pituitary tumour"):
        canonical = normalize_class_folder(name)
        assert canonical is not None, name
        assert kp._class_key(name) == kp._class_key(canonical), name

    for name in ("Training", "Testing", "Challenging Datasets", "README"):
        assert normalize_class_folder(name) is None
        assert kp._class_key(name) not in kp._CLASS_KEYS


def test_missing_dataset_error_says_where_it_looked(tmp_path):
    """"Not found" is only actionable if it names the mount and the expected shape."""
    mount = tmp_path / "input"
    (mount / "some-other-dataset").mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as excinfo:
        kp.setup_kaggle_data(tmp_path / "project", mount)

    message = str(excinfo.value)
    assert "some-other-dataset" in message          # what is mounted
    assert "Directories inspected" in message       # where it looked
    assert "Training" in message and "No-tumor" in message  # the expected structure
    assert "Add Input" in message                   # how to fix it


# ------------------------------------------------------------------------ linking


def test_linking_points_at_the_root_the_loader_can_reach(tmp_path):
    """The link must land on the split folders' parent, not the mount point.

    Linking `/kaggle/input/datasets` instead put the split folders six levels down, and
    the loader descends three. That one-level gap is the whole bug.
    """
    mount, expected = _make_kaggle_mount(tmp_path)
    project = tmp_path / "project"

    report = kp.setup_kaggle_data(project, mount)
    assert report["primary_action"] in ("linked", "copied")
    assert Path(report["primary_source"]) == expected

    linked = project / "data" / "raw" / "bt_mri"
    assert kp.is_dataset_root(linked), "loader must see Training/Testing immediately"
    found, _ = kp.find_dataset_root(linked, max_depth=kp.LOADER_MAX_DEPTH)
    assert found == linked


def test_linking_replaces_an_empty_stale_directory(tmp_path):
    """A failed download leaves data/raw/bt_mri behind and it blocks the fix."""
    mount, _ = _make_kaggle_mount(tmp_path)
    project = tmp_path / "project"
    stale = project / "data" / "raw" / "bt_mri"
    stale.mkdir(parents=True)

    kp.setup_kaggle_data(project, mount)
    assert kp.is_dataset_root(stale)


def test_linking_keeps_a_directory_that_is_already_correct(tmp_path):
    """Re-running setup must be free, not destructive."""
    mount, _ = _make_kaggle_mount(tmp_path)
    project = tmp_path / "project"

    target = project / "data" / "raw" / "bt_mri"
    target.mkdir(parents=True)
    _make_dataset(target)
    marker = target / "Training" / "Glioma" / "keep_me.jpg"
    marker.write_bytes(b"")

    assert kp.setup_kaggle_data(project, mount)["primary_action"] == "already_present"
    assert marker.is_file(), "an already-correct directory must survive untouched"


def test_a_normally_downloaded_dataset_is_kept_despite_its_nesting(tmp_path):
    """download_data.sh leaves the archive's own doubly-nested folder in place.

    The loader descends into it without complaint, so setup must too. Requiring
    Training/ at the top level would condemn a valid local dataset as junk and tell the
    user to delete it.
    """
    mount, _ = _make_kaggle_mount(tmp_path)
    project = tmp_path / "project"

    target = project / "data" / "raw" / "bt_mri"
    nested = target / "BT-MRI Dataset" / "BT-MRI Dataset"
    nested.mkdir(parents=True)
    _make_dataset(nested)

    assert kp.setup_kaggle_data(project, mount)["primary_action"] == "already_present"
    assert kp.is_dataset_root(nested), "the existing download must survive untouched"


def _supports_symlinks(tmp_path):
    """:param tmp_path: A writable directory.

    :return: Whether this platform lets the test process create symlinks.
    """
    probe, destination = tmp_path / "probe_link", tmp_path / "probe_dir"
    destination.mkdir(exist_ok=True)
    try:
        probe.symlink_to(destination, target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    probe.unlink()
    return True


def test_setup_is_idempotent(tmp_path):
    """``--setup-data`` runs on every session, including resumed ones.

    The second call previously died: the guard chain fell through to ``symlink_to`` on
    an existing path, and the resulting FileExistsError - itself an OSError - was caught
    by the copy fallback, which then failed the same way.
    """
    mount, expected = _make_kaggle_mount(tmp_path)
    project = tmp_path / "project"

    first = kp.setup_kaggle_data(project, mount)
    second = kp.setup_kaggle_data(project, mount)
    third = kp.setup_kaggle_data(project, mount)

    assert first["primary_action"] in ("linked", "copied")
    assert second["primary_action"] in ("already_linked", "already_present")
    assert second["primary_action"] == third["primary_action"]

    linked = project / "data" / "raw" / "bt_mri"
    assert kp.is_dataset_root(linked), "the dataset must still be reachable afterwards"
    assert Path(third["primary_source"]) == expected


def test_linking_a_missing_target_creates_the_link(tmp_path):
    """Case 1: nothing there yet."""
    source = _make_dataset(tmp_path / "source")
    target = tmp_path / "project" / "data" / "raw" / "bt_mri"

    assert kp.link_dataset(source, target) in ("linked", "copied")
    assert kp.is_dataset_root(target)


def test_linking_leaves_a_correct_symlink_alone(tmp_path):
    """Case 2: already pointing where it should - do not churn it."""
    if not _supports_symlinks(tmp_path):
        pytest.skip("this platform does not allow symlink creation")

    source = _make_dataset(tmp_path / "source")
    target = tmp_path / "bt_mri"
    target.symlink_to(source, target_is_directory=True)
    before = os.readlink(target)

    assert kp.link_dataset(source, target) == "already_linked"
    assert target.is_symlink(), "the link must not have been recreated"
    assert os.readlink(target) == before


def test_linking_repoints_a_symlink_aimed_at_the_wrong_place(tmp_path):
    """Case 3: an old link from a previous session must be replaced, not the dataset."""
    if not _supports_symlinks(tmp_path):
        pytest.skip("this platform does not allow symlink creation")

    stale = _make_dataset(tmp_path / "stale")
    source = _make_dataset(tmp_path / "source")
    target = tmp_path / "bt_mri"
    target.symlink_to(stale, target_is_directory=True)

    assert kp.link_dataset(source, target) == "relinked"
    assert os.path.realpath(target) == os.path.realpath(source)
    assert kp.is_dataset_root(stale), "only the link is removed, never a dataset"


def test_linking_replaces_a_broken_symlink(tmp_path):
    """A dangling link is invisible to exists() but still occupies the path."""
    if not _supports_symlinks(tmp_path):
        pytest.skip("this platform does not allow symlink creation")

    source = _make_dataset(tmp_path / "source")
    target = tmp_path / "bt_mri"
    target.symlink_to(tmp_path / "gone", target_is_directory=True)
    assert not target.exists() and target.is_symlink()

    assert kp.link_dataset(source, target) == "relinked"
    assert kp.is_dataset_root(target)


def test_linking_replaces_a_stray_file_at_the_target(tmp_path):
    """Neither a directory nor a link: the case that used to fall through.

    Kaggle's working-directory persistence restores a symlink that pointed into
    /kaggle/input as a plain file, so this is a state users actually reach. A file cannot
    be a dataset, so it is replaced rather than reported - refusing would strand the user
    with a path only a shell can clear.
    """
    source = _make_dataset(tmp_path / "source")
    target = tmp_path / "bt_mri"
    target.write_text("dead symlink residue")

    assert kp.link_dataset(source, target) in ("replaced", "copied")
    assert kp.is_dataset_root(target)


def test_a_stray_file_is_described_not_guessed_at(tmp_path):
    """Messages should say what was found, so the next report needs no round trip."""
    target = tmp_path / "bt_mri"
    target.write_text("12345")
    assert _describe_path_of(target) == "a file of 5 bytes"

    assert "unreadable" in _describe_path_of(tmp_path / "nothing_here")


def _describe_path_of(path):
    """:param path: Path to describe.

    :return: The driver's description of it.
    """
    return kp._describe_path(path)


def test_copy_fallback_never_runs_onto_an_existing_target(tmp_path, monkeypatch):
    """The core defect: FileExistsError is an OSError, so the copy fallback caught it.

    ``copytree`` was then handed a target that still existed and failed the same way,
    burying the real problem under a second traceback. A refusal to link must never be
    retried as a copy over live content.
    """
    source = _make_dataset(tmp_path / "source")
    target = tmp_path / "bt_mri"  # absent on entry, so the guard does not fire

    def occupy_then_fail(self, *args, **kwargs):
        """Stand in for a target that exists by the time the link is attempted."""
        self.mkdir()
        raise FileExistsError(17, "File exists")

    monkeypatch.setattr(Path, "symlink_to", occupy_then_fail)
    monkeypatch.setattr(
        shutil, "copytree", lambda *a, **k: pytest.fail("copytree ran on a live target")
    )

    with pytest.raises(RuntimeError, match="not safe"):
        kp._create_link(source, target)


def _fake_symlink(monkeypatch, link: Path, destination: Path):
    """Make ``link`` behave like a symlink to ``destination`` without OS privileges.

    Windows refuses symlink creation without developer mode, so the three tests above
    skip there - and those are precisely the cases that fire on Kaggle. This substitutes
    for them by faking only the four primitives ``link_dataset`` inspects, and only for
    this one path; every other path falls through to the real implementation. It proves
    which branch is taken, which is where the bug was, not what the OS does with links.

    :param monkeypatch: Pytest's patcher.
    :param link: The path to present as a symlink.
    :param destination: Where it appears to point.
    :return: A list that records ``("unlink" | "symlink_to", path)`` calls.
    """
    calls = []
    real_symlink, real_unlink = Path.symlink_to, Path.unlink
    real_lexists, real_realpath = os.path.lexists, os.path.realpath
    state = {"exists": True}

    monkeypatch.setattr(
        os.path, "lexists", lambda p: state["exists"] if Path(p) == link else real_lexists(p)
    )
    monkeypatch.setattr(
        os.path,
        "realpath",
        lambda p, **kw: str(destination) if Path(p) == link else real_realpath(p, **kw),
    )
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == link)

    def unlink(self, *args, **kwargs):
        if self == link:
            calls.append(("unlink", self))
            state["exists"] = False
            return None
        return real_unlink(self, *args, **kwargs)

    def symlink_to(self, target, target_is_directory=False):
        if self == link:
            calls.append(("symlink_to", Path(target)))
            state["exists"] = True
            return None
        return real_symlink(self, target, target_is_directory)

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(Path, "symlink_to", symlink_to)
    return calls


def test_a_correct_link_is_left_untouched_on_any_platform(tmp_path, monkeypatch):
    """Case 2 without needing symlink privileges: nothing is unlinked or recreated."""
    source = _make_dataset(tmp_path / "source")
    link = tmp_path / "bt_mri"
    calls = _fake_symlink(monkeypatch, link, source)

    assert kp.link_dataset(source, link) == "already_linked"
    assert calls == [], "an already-correct link must not be churned"


def test_a_wrong_link_is_unlinked_then_recreated_on_any_platform(tmp_path, monkeypatch):
    """Case 3 without needing symlink privileges: the link goes, the dataset stays."""
    stale = _make_dataset(tmp_path / "stale")
    source = _make_dataset(tmp_path / "source")
    link = tmp_path / "bt_mri"
    calls = _fake_symlink(monkeypatch, link, stale)

    assert kp.link_dataset(source, link) == "relinked"
    assert calls == [("unlink", link), ("symlink_to", source)]
    assert kp.is_dataset_root(stale), "only the link is removed, never a dataset"


def test_create_link_never_overwrites_an_existing_path(tmp_path):
    """The guard in front of the link, independent of what the platform supports."""
    source = _make_dataset(tmp_path / "source")
    target = tmp_path / "bt_mri"
    target.mkdir()

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        kp._create_link(source, target)
    assert target.is_dir()


def test_linking_refuses_to_destroy_unrelated_content(tmp_path):
    """A non-empty directory that is not a dataset root is never silently replaced."""
    mount, _ = _make_kaggle_mount(tmp_path)
    project = tmp_path / "project"
    target = project / "data" / "raw" / "bt_mri"
    target.mkdir(parents=True)
    (target / "something_important.txt").write_text("do not delete me")

    with pytest.raises(RuntimeError, match="Refusing to replace"):
        kp.setup_kaggle_data(project, mount)
    assert (target / "something_important.txt").is_file()


def test_figshare_is_optional_and_found_by_its_mat_files(tmp_path):
    """:param tmp_path: Temporary directory."""
    mount, _ = _make_kaggle_mount(tmp_path)
    project = tmp_path / "project"

    assert kp.setup_kaggle_data(project, mount)["external_source"] is None

    external = mount / "figshare-brain-tumor-dataset" / "brainTumorDataPublic"
    external.mkdir(parents=True)
    (external / "1.mat").write_bytes(b"")
    report = kp.setup_kaggle_data(project, mount)
    assert Path(report["external_source"]) == external
    assert (project / "data" / "raw" / "figshare" / "1.mat").exists()


# ---------------------------------------------------------------------- preflight


def test_preflight_refuses_to_start_without_the_dataset(tmp_path):
    """A missing dataset must stop the run, not produce twenty identical failures.

    Every stage reads the same image tree. Without this guard a session with no data
    attached burns twenty-odd stages twelve seconds apart and writes a report that reads
    like the study collapsed rather than like a setup mistake.
    """
    pipe = _pipeline()
    pipe.root = tmp_path

    with pytest.raises(FileNotFoundError, match="does not exist"):
        pipe.preflight()

    # An empty directory left behind by a failed download must not count as success.
    (tmp_path / "data" / "raw" / "bt_mri").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="not a dataset root"):
        pipe.preflight()


def test_preflight_rejects_a_tree_the_loader_cannot_reach(tmp_path):
    """Structure alone is not enough - it has to be within the loader's descent limit."""
    raw = tmp_path / "data" / "raw" / "bt_mri"
    buried = raw / "a" / "b" / "c" / "d" / "BT-MRI Dataset"
    buried.mkdir(parents=True)
    _make_dataset(buried)

    pipe = _pipeline()
    pipe.root = tmp_path
    with pytest.raises(FileNotFoundError, match="nested too deeply"):
        pipe.preflight()


def test_preflight_rejects_a_truncated_dataset(tmp_path):
    """Right folders, almost no images: a partial download, not a usable dataset."""
    raw = tmp_path / "data" / "raw" / "bt_mri"
    raw.mkdir(parents=True)
    _make_dataset(raw, images_per_class=1)

    pipe = _pipeline()
    pipe.root = tmp_path
    with pytest.raises(FileNotFoundError, match="only 8 images"):
        pipe.preflight()


def test_preflight_passes_on_a_complete_dataset(tmp_path):
    """:param tmp_path: Temporary project root."""
    raw = tmp_path / "data" / "raw" / "bt_mri"
    raw.mkdir(parents=True)
    _make_dataset(raw, images_per_class=40)

    pipe = _pipeline()
    pipe.root = tmp_path
    pipe.log_root = tmp_path / "logs"
    pipe.pipeline_dir = tmp_path / "logs" / "pipeline"
    pipe.preflight()  # must not raise


def test_preflight_passes_when_the_root_is_one_level_down(tmp_path):
    """The archive's own nesting, which the loader does handle."""
    raw = tmp_path / "data" / "raw" / "bt_mri"
    nested = raw / "BT-MRI Dataset" / "BT-MRI Dataset"
    nested.mkdir(parents=True)
    _make_dataset(nested, images_per_class=40)

    pipe = _pipeline()
    pipe.root = tmp_path
    pipe.log_root = tmp_path / "logs"
    pipe.pipeline_dir = tmp_path / "logs" / "pipeline"
    pipe.preflight()  # must not raise


def test_out_dir_is_pinned_and_derived_from_the_stage_id():
    """Downstream stages address checkpoints by path, so the path cannot be a timestamp."""
    pipe = _pipeline(profile="full")
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step15_final/seed_42")
    assert pipe.out_dir(stage) == ROOT / "logs" / "train" / "runs" / "step15_final" / "seed_42"
