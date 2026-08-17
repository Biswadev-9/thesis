"""Interruption safety and resume correctness.

Every stage of this study runs on a machine that will be killed: Kaggle stops a session at
twelve hours whether or not the work is finished. The driver is built to be re-run, and
these tests cover the three ways "re-run it" can quietly go wrong.

**A resumed run must be read whole.** Lightning's ``CSVLogger`` never writes into a version
directory that already exists, so a run killed and resumed into the same pinned Hydra
directory leaves ``csv/version_0/`` holding the epochs before the crash and
``csv/version_1/`` the epochs after. Step 6's confirmation reads those metrics to decide
preprocessing for Steps 24, 25 and the shipped model; reading one segment scores the
candidate on a fraction of its training, and does it silently.

**A completion marker must be validated, not counted.** The marker is written after the
stage exits zero, which is the right ordering - but existence alone said nothing about
whether the marker survived the kill that ended the session, whether the outputs came back
with ``--restore-from``, or whether the inputs an aggregate was built from have since
grown.

**A checkpoint directory is not a finished run.** A run killed at epoch two leaves
``checkpoints/epoch_002.ckpt``, which ``find_checkpoint`` cannot distinguish from a
converged run's best epoch.

Nothing here trains anything: stages are simulated by replacing ``_spawn``, and every
artefact is fabricated in ``tmp_path``.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.preprocessing_confirmation import (
    ConfirmationIncomplete,
    PreprocessingConfirmation,
    _metrics_segments,
    _read_metric,
)
from src.utils.atomic import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]


def _load_pipeline():
    """:return: The driver module, imported from ``scripts/`` which is not a package."""
    path = ROOT / "scripts" / "kaggle_pipeline.py"
    spec = importlib.util.spec_from_file_location("kaggle_pipeline_resume", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kp = _load_pipeline()


# --------------------------------------------------------------------------- helpers


def write_segment(run_dir, version, rows, column="val/f1_macro_best"):
    """Fabricate one CSVLogger segment.

    :param run_dir: The run directory.
    :param version: Version number the logger would have chosen.
    :param rows: Metric values, one per logged epoch.
    :param column: Metric column name.
    :return: The metrics file written.
    """
    directory = Path(run_dir) / "csv" / f"version_{version}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metrics.csv"
    pd.DataFrame({"epoch": range(len(rows)), column: rows}).to_csv(path, index=False)
    return path


def write_run(run_dir, recipe, seed, rows, versions=None, experiment="step06_confirm"):
    """Fabricate a confirmation training run.

    :param run_dir: Where the run lives.
    :param recipe: The ``data.recipe`` it trained on.
    :param seed: Its seed.
    :param rows: Metric values for ``version_0``.
    :param versions: ``{version: rows}`` for a resumed run, instead of ``rows``.
    :param experiment: The experiment its overrides declare.
    :return: The run directory.
    """
    run_dir = Path(run_dir)
    (run_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    (run_dir / ".hydra" / "overrides.yaml").write_text(
        f"- experiment={experiment}\n- data.recipe={recipe}\n- seed={seed}\n",
        encoding="utf-8",
    )
    for version, values in (versions or {0: rows}).items():
        write_segment(run_dir, version, values)
    return run_dir


def complete_run(pipe, stage_id, entry="train", artifact="checkpoints/epoch_003.ckpt"):
    """Fabricate a stage that genuinely finished.

    :param pipe: The pipeline.
    :param stage_id: Stage identifier.
    :param entry: Entry point it ran under.
    :param artifact: Artefact to leave behind, relative to the run directory.
    :return: The run directory.
    """
    out = pipe.out_dir_for(entry, stage_id)
    target = out / artifact
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    atomic_write_json(
        out / ".pipeline_done.json", {"stage": stage_id, "status": "done", "returncode": 0}
    )
    return out


def interrupted_run(pipe, stage_id, entry="train"):
    """Fabricate a run killed mid-training: checkpoints, but no completion marker.

    :param pipe: The pipeline.
    :param stage_id: Stage identifier.
    :param entry: Entry point it ran under.
    :return: The run directory.
    """
    out = pipe.out_dir_for(entry, stage_id)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints" / "epoch_002.ckpt").write_bytes(b"x")
    (out / "checkpoints" / "last.ckpt").write_bytes(b"x")
    return out


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    """A full-profile pipeline rooted in a temporary tree, with the proxy already run.

    :param tmp_path: Per-test directory.
    :param monkeypatch: Used to redirect the driver's ROOT.
    :return: The pipeline.
    """
    monkeypatch.setattr(kp, "ROOT", tmp_path)
    args = kp.parse_args([])
    args.profile = "full"
    pipeline = kp.Pipeline(args)

    for stage, payload in (
        (
            "step06_preprocessing",
            {
                "selected_recipe": "cand_a",
                "ranking": [
                    {"recipe": "cand_a"},
                    {"recipe": "cand_b"},
                    {"recipe": "diffusion_i10_k15"},
                    {"recipe": "conventional"},
                ],
            },
        ),
        ("step14_loss_selection", {"selected_loss": "weighted_ce"}),
    ):
        path = pipeline.summary_path(stage, "analyze", f"{stage}_summary.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload)
    return pipeline


@pytest.fixture
def simulate(monkeypatch):
    """Replace a pipeline's subprocess call with one that fabricates the stage's outputs.

    :param monkeypatch: Unused directly; keeps the fixture per-test.
    :return: A factory taking the pipeline and returning the list of executed argv.
    """

    def install(pipeline, exit_code=0, artifacts=True):
        calls = []

        def fake(argv, log_path):
            calls.append(argv)
            out = Path(
                next(a for a in argv if a.startswith("hydra.run.dir=")).split("=", 1)[1]
            )
            (out / ".hydra").mkdir(parents=True, exist_ok=True)
            if artifacts and exit_code == 0:
                if "src/train.py" in " ".join(argv).replace("\\", "/"):
                    (out / "checkpoints").mkdir(exist_ok=True)
                    (out / "checkpoints" / "epoch_003.ckpt").write_bytes(b"x")
                    (out / "checkpoints" / "last.ckpt").write_bytes(b"x")
                else:
                    atomic_write_json(out / f"{out.name}_summary.json", {"ok": True})
            return exit_code, False

        pipeline._spawn = fake
        pipeline.echo = lambda message: None
        return calls

    return install


def ran(calls):
    """:param calls: Recorded argv lists.

    :return: The run directories they targeted.
    """
    return [
        next(a for a in c if a.startswith("hydra.run.dir=")).split("=", 1)[1] for c in calls
    ]


# ------------------------------------------------- 1. resumed runs: every CSV segment


def test_a_resumed_run_contributes_every_segment(tmp_path):
    """version_0 alone is the training done before the crash, not the run's result."""
    run = write_run(tmp_path / "run", "cand_a", 42, None, versions={0: [0.40, 0.55], 1: [0.72]})

    value, column, segments = _read_metric(run, "val/f1_macro_best")

    assert segments == ["version_0", "version_1"]
    assert value == pytest.approx(0.72), "the post-resume epochs decide the run's best"
    assert column == "val/f1_macro_best"


def test_segments_are_ordered_numerically_not_lexically(tmp_path):
    """version_10 sorts before version_2 as a string, which would reorder a long resume."""
    run = tmp_path / "run"
    for version in (0, 2, 10):
        write_segment(run, version, [0.1])

    assert [p.parent.name for p in _metrics_segments(run)] == [
        "version_0",
        "version_2",
        "version_10",
    ]


def test_a_resumed_run_that_got_worse_keeps_its_best(tmp_path):
    """`val/f1_macro_best` is a running best; a resumed segment cannot lower the run."""
    run = write_run(tmp_path / "run", "cand_a", 42, None, versions={0: [0.81], 1: [0.60]})

    value, _, segments = _read_metric(run, "val/f1_macro_best")

    assert len(segments) == 2
    assert value == pytest.approx(0.81)


def test_an_empty_resumed_segment_is_ignored_not_fatal(tmp_path):
    """A run killed before its second segment flushed still has its first."""
    run = write_run(tmp_path / "run", "cand_a", 42, [0.66])
    (run / "csv" / "version_1").mkdir(parents=True)  # directory, no metrics.csv

    value, _, segments = _read_metric(run, "val/f1_macro_best")

    assert segments == ["version_0"]
    assert value == pytest.approx(0.66)


def test_an_unreadable_segment_does_not_lose_the_readable_one(tmp_path):
    """A truncated CSV from a kill mid-flush must not discard the rest of the run."""
    run = write_run(tmp_path / "run", "cand_a", 42, [0.5])
    bad = run / "csv" / "version_1"
    bad.mkdir(parents=True)
    (bad / "metrics.csv").write_bytes(b'"unterminated\x00\x00')

    value, _, _ = _read_metric(run, "val/f1_macro_best")

    assert value == pytest.approx(0.5)


def test_a_run_with_no_metrics_at_all_is_reported_as_unusable(tmp_path):
    """:return: None."""
    assert _read_metric(tmp_path / "empty", "val/f1_macro_best") == (None, None, [])


def test_the_confirmation_scores_a_resumed_candidate_on_its_whole_run(tmp_path):
    """End to end: a resumed candidate must not lose to its own interruption."""
    runs = tmp_path / "runs"
    # cand_b is the better recipe, but its seed-123 run was killed and resumed. Reading
    # version_0 alone would score that run at 0.50 and hand the decision to cand_a.
    for seed, values in ((42, [0.80]), (123, [0.81]), (7, [0.80])):
        write_run(runs / f"cand_a_{seed}", "cand_a", seed, values)
    write_run(runs / "cand_b_42", "cand_b", 42, [0.90])
    write_run(runs / "cand_b_123", "cand_b", 123, None, versions={0: [0.50], 1: [0.91]})
    write_run(runs / "cand_b_7", "cand_b", 7, [0.90])

    analysis = PreprocessingConfirmation(run_root=str(runs))
    summary = analysis.run(output_dir=tmp_path / "out")

    assert summary["selected_recipe"] == "cand_b"
    resumed = [r for r in summary["runs"] if r["resumed"]]
    assert len(resumed) == 1
    assert resumed[0]["metric_segments"] == ["version_0", "version_1"]
    assert resumed[0]["metric_value"] == pytest.approx(0.91)


# ------------------------------------------------------- 4. the design must be complete


def test_an_unbalanced_confirmation_refuses_to_decide(tmp_path):
    """One candidate on three seeds against another on one is a draw, not a comparison."""
    runs = tmp_path / "runs"
    for seed in (42, 123, 7):
        write_run(runs / f"cand_a_{seed}", "cand_a", seed, [0.80])
    write_run(runs / "cand_b_42", "cand_b", 42, [0.95])  # the crash landed here

    with pytest.raises(ConfirmationIncomplete, match="not balanced"):
        PreprocessingConfirmation(run_root=str(runs)).run(output_dir=tmp_path / "out")


def test_no_summary_is_written_when_the_design_is_unbalanced(tmp_path):
    """An unbalanced sweep must leave no artefact for Steps 24 and 25 to consume."""
    runs = tmp_path / "runs"
    for seed in (42, 123):
        write_run(runs / f"cand_a_{seed}", "cand_a", seed, [0.80])
    write_run(runs / "cand_b_42", "cand_b", 42, [0.95])
    out = tmp_path / "out"

    with pytest.raises(ConfirmationIncomplete):
        PreprocessingConfirmation(run_root=str(runs)).run(output_dir=out)

    assert not (out / "step06_confirm_summary.json").exists()


def test_a_balanced_confirmation_records_its_design(tmp_path):
    """The completed design is stated in the summary, not left implicit."""
    runs = tmp_path / "runs"
    for recipe in ("cand_a", "cand_b"):
        for seed in (42, 123, 7):
            write_run(runs / f"{recipe}_{seed}", recipe, seed, [0.80])

    summary = PreprocessingConfirmation(run_root=str(runs)).run(output_dir=tmp_path / "out")

    assert summary["design"]["balanced"] is True
    assert summary["design"]["seeds_observed"] == [7, 42, 123]
    assert summary["design"]["candidates_missing_seeds"] == {}


def test_imbalance_can_be_recorded_instead_of_raised_but_is_marked(tmp_path):
    """The escape hatch exists, and it labels the result rather than hiding the gap."""
    runs = tmp_path / "runs"
    for seed in (42, 123):
        write_run(runs / f"cand_a_{seed}", "cand_a", seed, [0.80])
    write_run(runs / "cand_b_42", "cand_b", 42, [0.95])

    summary = PreprocessingConfirmation(run_root=str(runs), require_balanced=False).run(
        output_dir=tmp_path / "out"
    )

    assert summary["design"]["balanced"] is False
    assert summary["design"]["candidates_missing_seeds"] == {"cand_b": [123]}


# ------------------------------------------------------ 2. completion marker validation


def test_a_completed_stage_is_skipped(pipe, simulate):
    """The baseline the rest of these tests are measured against."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step04_audit")
    calls = simulate(pipe)

    assert pipe.run_stage(stage) == "done"
    assert pipe.run_stage(stage) == "cached"
    assert len(calls) == 1


def test_an_empty_marker_does_not_count_as_completed(pipe, simulate):
    """A kill during the marker's own write used to leave a stage permanently 'done'."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step04_audit")
    out = pipe.out_dir(stage)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".pipeline_done.json").write_text("", encoding="utf-8")
    calls = simulate(pipe)

    complete, reason = pipe.stage_is_complete(stage)
    assert not complete and "unreadable" in reason
    assert pipe.run_stage(stage) == "done"
    assert len(calls) == 1, "the stage must actually re-run"


def test_a_truncated_marker_does_not_count_as_completed(pipe):
    """:return: None."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step04_audit")
    out = pipe.out_dir(stage)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".pipeline_done.json").write_text('{"status": "do', encoding="utf-8")

    assert pipe.stage_is_complete(stage)[0] is False


def test_a_marker_recording_a_failure_does_not_count_as_completed(pipe):
    """:return: None."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step04_audit")
    out = pipe.out_dir(stage)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / ".pipeline_done.json", {"status": "failed", "returncode": 1})
    atomic_write_json(out / "step04_audit_summary.json", {"ok": True})

    complete, reason = pipe.stage_is_complete(stage)
    assert not complete and "status" in reason


def test_a_marker_without_its_summary_does_not_count_as_completed(pipe, simulate):
    """Kaggle's filesystem does not survive the session; --restore-from can miss files."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step04_audit")
    out = pipe.out_dir(stage)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / ".pipeline_done.json", {"status": "done", "returncode": 0})
    calls = simulate(pipe)

    complete, reason = pipe.stage_is_complete(stage)
    assert not complete and "step04_audit_summary.json" in reason
    assert pipe.run_stage(stage) == "done"
    assert len(calls) == 1


def test_a_training_marker_without_a_checkpoint_does_not_count_as_completed(pipe):
    """:return: None."""
    stage = next(s for s in kp.build_stages(pipe) if s.is_train)
    out = pipe.out_dir(stage)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / ".pipeline_done.json", {"status": "done", "returncode": 0})

    complete, reason = pipe.stage_is_complete(stage)
    assert not complete and "checkpoints" in reason


def test_a_stage_exiting_zero_without_its_artifact_fails_rather_than_caching(pipe, simulate):
    """Exit zero and no summary means the stage did not do its job; do not enshrine it."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step04_audit")
    simulate(pipe, artifacts=False)

    with pytest.raises(kp.StageFailed, match="produced no"):
        pipe.run_stage(stage)
    assert not (pipe.out_dir(stage) / ".pipeline_done.json").exists()


def test_every_analyze_stage_is_named_for_the_analysis_it_runs(pipe):
    """The expected-artefact rule derives the summary name from the stage id.

    If a stage is ever added whose id differs from its ``analysis=`` config, that rule
    would look for a file no analysis writes and the stage would re-run forever.
    """
    for stage in kp.build_stages(pipe):
        if stage.entry != "analyze":
            continue
        try:
            overrides = stage.build(pipe)
        except kp.StageFailed:
            continue  # its inputs have not been produced in this fixture
        if overrides is None:
            continue
        declared = next(o.split("=", 1)[1] for o in overrides if o.startswith("analysis="))
        assert declared == stage.id, f"{stage.id} runs analysis={declared}"


# ---------------------------------------- 3. a checkpoint directory is not a finished run


def test_an_interrupted_run_is_not_reported_as_a_trained_branch(pipe):
    """`checkpoints/` exists from epoch one; completion is a different question."""
    interrupted_run(pipe, "step15_final/seed_42")

    assert pipe.branch_ckpt("step15_final/seed_42") is None


def test_a_completed_run_is_reported_as_a_trained_branch(pipe):
    """:return: None."""
    complete_run(pipe, "step15_final/seed_42")

    assert pipe.branch_ckpt("step15_final/seed_42") is not None


@pytest.mark.parametrize(
    "group,namespace,total",
    [
        ("step24", kp.RECEPTIVE_FIELD_NAMESPACE, 15),
        ("step25", kp.QUANTUM_CIRCUIT_NAMESPACE, 12),
    ],
)
def test_an_evaluation_refuses_a_half_trained_condition(pipe, group, namespace, total):
    """A condition killed at epoch two must not be scored as a result of the study."""
    path = pipe.summary_path("step06_confirm", "analyze", "step06_confirm_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path,
        {
            "confirmation_status": "confirmed",
            "selection_metric": "val/f1_macro_best",
            "selected_recipe": "cand_a",
        },
    )

    trains = [s for s in kp.build_stages(pipe) if s.group == group and s.is_train]
    assert len(trains) == total

    for stage in trains[:-1]:
        complete_run(pipe, stage.id)
    interrupted_run(pipe, trains[-1].id)  # the one the session died in

    evaluation = next(s for s in kp.build_stages(pipe) if s.id == namespace)
    with pytest.raises(kp.StageFailed, match="missing"):
        evaluation.build(pipe)

    complete_run(pipe, trains[-1].id)
    assert evaluation.build(pipe) is not None


# --------------------------------------------------------------- 4. aggregator staleness


def test_the_confirmation_summary_will_not_build_from_a_partial_sweep(pipe):
    """The decision feeding Steps 24, 25 and the shipped model needs the whole design."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step06_confirm")
    required = stage.inputs(pipe)
    assert len(required) == 12, "four candidates at three seeds"

    for stage_id in required[:4]:
        complete_run(pipe, stage_id)

    with pytest.raises(kp.StageFailed, match="4/12 complete"):
        stage.build(pipe)

    for stage_id in required[4:]:
        complete_run(pipe, stage_id)
    assert stage.build(pipe) is not None


def test_an_aggregate_built_from_a_partial_sweep_is_recomputed_when_the_rest_lands(
    pipe, simulate
):
    """A marker written over four of twelve runs must not stand as the answer forever."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step06_confirm")
    required = stage.inputs(pipe)
    for stage_id in required:
        complete_run(pipe, stage_id)
    calls = simulate(pipe)
    assert pipe.run_stage(stage) == "done"
    assert pipe.run_stage(stage) == "cached"

    # Simulate the marker having been written when only four runs existed: the fingerprint
    # recorded then cannot match the twelve that exist now.
    marker = pipe.out_dir(stage) / ".pipeline_done.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    record["inputs_fingerprint"] = "4/12:" + ",".join(sorted(required)[:4])
    atomic_write_json(marker, record)

    complete, reason = pipe.stage_is_complete(stage)
    assert not complete and "inputs changed" in reason

    calls.clear()
    assert pipe.run_stage(stage) == "done"
    assert len(calls) == 1, "the decision must be recomputed over the full sweep"


@pytest.mark.parametrize(
    "namespace,total",
    [
        (kp.ABLATION_NAMESPACE, None),
        (kp.RECEPTIVE_FIELD_NAMESPACE, 15),
        (kp.QUANTUM_CIRCUIT_NAMESPACE, 12),
    ],
)
def test_every_evaluation_stage_declares_the_runs_it_aggregates(pipe, namespace, total):
    """Without declared inputs, an evaluation's marker can never go stale."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == namespace)

    assert stage.inputs is not None, f"{namespace} declares no inputs"
    declared = stage.inputs(pipe)
    assert declared, f"{namespace} declares an empty input set"
    if total is not None:
        assert len(declared) == total
    trains = {s.id for s in kp.build_stages(pipe) if s.is_train}
    assert set(declared) <= trains, "every declared input must be a training stage"


def test_a_completed_training_stage_is_never_retrained_by_the_invalidation(pipe, simulate):
    """Invalidation is for aggregates. Retraining is the expensive thing it must not do."""
    trains = [s for s in kp.build_stages(pipe) if s.group == "step06" and s.is_train]
    calls = simulate(pipe)
    for stage in trains[:4]:
        pipe.run_stage(stage)
    calls.clear()

    for stage in trains:
        pipe.run_stage(stage)

    assert len(calls) == len(trains) - 4
    finished = {str(pipe.out_dir(s)) for s in trains[:4]}
    assert not finished & set(ran(calls))


# ------------------------------------------------------------------- 5. atomic writes


def test_atomic_write_leaves_no_temporary_behind(tmp_path):
    """:return: None."""
    target = tmp_path / "summary.json"
    atomic_write_json(target, {"a": 1})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["summary.json"]


def test_a_failed_atomic_write_keeps_the_previous_file(tmp_path):
    """Half a new file must never replace a whole old one."""
    from src.utils.atomic import atomic_write

    target = tmp_path / "summary.json"
    atomic_write_text(target, "original")

    def explode(temporary):
        temporary.write_text("half a file", encoding="utf-8")
        raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError):
        atomic_write(target, explode)

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target], "the temporary must be cleaned up"


def test_the_analysis_base_writes_summaries_atomically(tmp_path):
    """:return: None."""
    from src.analysis.base import Analysis

    analysis = Analysis(name="probe")
    analysis._output_dir = tmp_path
    analysis.save_json({"selected_recipe": "cand_a"}, "probe_summary.json")
    analysis.save_table(pd.DataFrame({"a": [1]}), "probe_table.csv")

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "probe_summary.json",
        "probe_table.csv",
    ]


def test_the_completion_marker_is_written_atomically(pipe, simulate):
    """:return: None."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step04_audit")
    simulate(pipe)
    pipe.run_stage(stage)

    leftovers = [p.name for p in pipe.out_dir(stage).iterdir() if p.name.endswith(".tmp")]
    assert not leftovers
    assert json.loads((pipe.out_dir(stage) / ".pipeline_done.json").read_text())["status"] == "done"
