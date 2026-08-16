"""Step 6 confirmation -> Steps 24 and 25: the dependency chain.

The proxy *ranks* preprocessing candidates with a SmallCNN at 128px. The real-backbone
confirmation *decides*. Everything downstream must consume the decision, and there must be
no path by which the ranking reaches a fifteen- or twelve-run experiment instead.

This file guards the whole chain rather than either end of it:

* both consumers read the same artefact, and neither reads the proxy;
* the proxy cannot override the confirmation even when both exist;
* a missing, malformed, unconfirmed or test-derived confirmation stops both consumers;
* every condition in both experiments receives one identical recipe;
* the pipeline graph puts the confirmation before both consumers, so Run All cannot reach
  Step 24 with the artefact absent.

No test names a winner. Which recipe wins is not established until the confirmation runs,
and encoding a guess here would be the very failure the chain exists to prevent.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Recipes used to prove propagation. Deliberately varied and deliberately not a prediction.
CANDIDATE_RECIPES = ["clahe", "diffusion_i5_k15", "gamma", "wiener"]


def load_pipeline():
    """:return: The runner module, imported from ``scripts/`` which is not a package."""
    path = Path("scripts/kaggle_pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("kaggle_pipeline_dep", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kp = load_pipeline()


def write_confirmation(directory, recipe, status="confirmed", **extra):
    """Write a Step 6 confirmation summary.

    :param directory: Where to write it.
    :param recipe: The confirmed recipe, or ``None`` to omit the field.
    :param status: Confirmation status.
    :param extra: Extra top-level fields.
    :return: Path to the summary.
    """
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"confirmation_status": status, "selection_metric": "val/f1_macro_best",
               **extra}
    if recipe is not None:
        payload["selected_recipe"] = recipe

    path = directory / "step06_confirm_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def step24_recipes(summary_path):
    """Every recipe the Step 24 conditions would train on.

    :param summary_path: Confirmation summary.
    :return: The set of ``data.recipe=`` overrides emitted.
    """
    from src.analysis.receptive_field_rows import (
        CONDITIONS,
        PROTOCOL_SEEDS,
        ReceptiveFieldContext,
        condition_overrides,
    )

    context = ReceptiveFieldContext.from_confirmation(str(summary_path))
    return {
        override
        for condition in CONDITIONS
        for seed in PROTOCOL_SEEDS
        for override in condition_overrides(condition, seed, context)
        if override.startswith("data.recipe=")
    }


def step25_recipes(summary_path):
    """Every recipe the Step 25 conditions would train on.

    :param summary_path: Confirmation summary.
    :return: The set of ``data.recipe=`` overrides emitted.
    """
    from src.analysis.quantum_circuit_ablation_rows import (
        CONDITIONS,
        SEEDS,
        QuantumAblationContext,
        condition_overrides,
    )

    context = QuantumAblationContext.from_confirmation(str(summary_path))
    return {
        override
        for condition in CONDITIONS
        for seed in SEEDS
        for override in condition_overrides(condition, seed, context)
        if override.startswith("data.recipe=")
    }


# ------------------------------------------------- Test A: confirmed recipe propagates


@pytest.mark.parametrize("recipe", CANDIDATE_RECIPES)
def test_the_confirmed_recipe_reaches_every_step24_condition(tmp_path, recipe):
    """Whatever the confirmation says is what all five conditions get.

    Parametrised across unrelated recipes precisely so no test encodes a winner.
    """
    path = write_confirmation(tmp_path / "s6", recipe)

    assert step24_recipes(path) == {f"data.recipe={recipe}"}


@pytest.mark.parametrize("recipe", CANDIDATE_RECIPES)
def test_the_confirmed_recipe_reaches_every_step25_condition(tmp_path, recipe):
    """:param tmp_path: Per-test directory.

    :param recipe: The confirmed recipe.
    """
    path = write_confirmation(tmp_path / "s6", recipe)

    assert step25_recipes(path) == {f"data.recipe={recipe}"}


@pytest.mark.parametrize("recipe", CANDIDATE_RECIPES)
def test_both_experiments_resolve_the_same_recipe(tmp_path, recipe):
    """One decision, two consumers - they must not drift apart."""
    path = write_confirmation(tmp_path / "s6", recipe)

    assert step24_recipes(path) == step25_recipes(path) == {f"data.recipe={recipe}"}


# --------------------------------------- Test B: the proxy cannot override confirmation


def test_the_proxy_winner_cannot_override_the_confirmation(tmp_path):
    """The proxy ranks; the confirmation decides. Both present, the confirmation wins."""
    analyze = tmp_path / "analyze" / "runs"
    proxy = analyze / "step06_preprocessing"
    proxy.mkdir(parents=True)
    (proxy / "step06_preprocessing_summary.json").write_text(
        json.dumps({"selected_recipe": "PROXY_WINNER",
                    "ranking": [{"recipe": "PROXY_WINNER", "macro_f1": 0.99}]}),
        encoding="utf-8",
    )
    confirmed = write_confirmation(analyze / "step06_confirm", "CONFIRMED_WINNER")

    assert step24_recipes(confirmed) == {"data.recipe=CONFIRMED_WINNER"}
    assert step25_recipes(confirmed) == {"data.recipe=CONFIRMED_WINNER"}


def test_neither_consumer_reads_the_proxy_summary():
    """A source-level guard: the proxy artefact is unreachable from either resolver."""
    import ast

    for module in ("src/analysis/receptive_field_rows.py",
                   "src/analysis/receptive_field_study.py",
                   "src/analysis/quantum_circuit_ablation_rows.py",
                   "src/analysis/quantum_circuit_ablation_study.py"):
        tree = ast.parse(Path(module).read_text(encoding="utf-8"))
        docstrings = {
            ast.get_docstring(n) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
        }
        literals = {
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        for text in literals - docstrings:
            assert "step06_preprocessing" not in text, (
                f"{module} references the proxy summary in code"
            )


# --------------------------------------------------- Test C: missing confirmation


def test_step24_refuses_without_a_confirmation(tmp_path):
    """Fifteen runs must not start on a preprocessing nobody confirmed."""
    from src.analysis.preprocessing_confirmation import ConfirmationIncomplete
    from src.analysis.receptive_field_rows import ReceptiveFieldContext

    with pytest.raises(ConfirmationIncomplete, match="authoritative selected_recipe"):
        ReceptiveFieldContext.from_confirmation(str(tmp_path / "absent.json"))


def test_step25_refuses_without_a_confirmation(tmp_path):
    """:param tmp_path: Per-test directory."""
    from src.analysis.preprocessing_confirmation import ConfirmationIncomplete
    from src.analysis.quantum_circuit_ablation_rows import QuantumAblationContext

    with pytest.raises(ConfirmationIncomplete, match="authoritative selected_recipe"):
        QuantumAblationContext.from_confirmation(str(tmp_path / "absent.json"))


def test_a_present_proxy_does_not_rescue_a_missing_confirmation(tmp_path):
    """The failure mode this whole chain exists to prevent."""
    from src.analysis.preprocessing_confirmation import ConfirmationIncomplete
    from src.analysis.quantum_circuit_ablation_rows import QuantumAblationContext
    from src.analysis.receptive_field_rows import ReceptiveFieldContext

    proxy = tmp_path / "analyze" / "runs" / "step06_preprocessing"
    proxy.mkdir(parents=True)
    (proxy / "step06_preprocessing_summary.json").write_text(
        json.dumps({"selected_recipe": "PROXY_WINNER"}), encoding="utf-8"
    )
    missing = tmp_path / "analyze" / "runs" / "step06_confirm" / "step06_confirm_summary.json"

    for resolver in (ReceptiveFieldContext, QuantumAblationContext):
        with pytest.raises(ConfirmationIncomplete):
            resolver.from_confirmation(str(missing))


# --------------------------------------------------- Test D: invalid confirmation


@pytest.mark.parametrize(
    "recipe,status,label",
    [
        (None, "confirmed", "no selected_recipe"),
        ("clahe", "pending", "unconfirmed status"),
        ("clahe", "failed", "failed status"),
        ("", "confirmed", "empty recipe"),
    ],
)
def test_an_invalid_confirmation_stops_both_consumers(tmp_path, recipe, status, label):
    """:param tmp_path: Per-test directory.

    :param recipe: Recipe to write, or ``None`` to omit.
    :param status: Confirmation status.
    :param label: What is wrong, for the failure message.
    """
    from src.analysis.preprocessing_confirmation import ConfirmationIncomplete
    from src.analysis.quantum_circuit_ablation_rows import QuantumAblationContext
    from src.analysis.receptive_field_rows import ReceptiveFieldContext

    path = write_confirmation(tmp_path / "s6", recipe, status=status)

    for resolver in (ReceptiveFieldContext, QuantumAblationContext):
        with pytest.raises(ConfirmationIncomplete):
            resolver.from_confirmation(str(path))


def test_a_malformed_confirmation_stops_both_consumers(tmp_path):
    """:param tmp_path: Per-test directory."""
    from src.analysis.preprocessing_confirmation import ConfirmationIncomplete
    from src.analysis.quantum_circuit_ablation_rows import QuantumAblationContext
    from src.analysis.receptive_field_rows import ReceptiveFieldContext

    directory = tmp_path / "s6"
    directory.mkdir()
    path = directory / "step06_confirm_summary.json"
    path.write_text("{ not json", encoding="utf-8")

    for resolver in (ReceptiveFieldContext, QuantumAblationContext):
        with pytest.raises(ConfirmationIncomplete, match="not valid JSON"):
            resolver.from_confirmation(str(path))


def test_the_confirmation_cannot_be_produced_from_a_test_metric():
    """Preprocessing is decided before Step 16; the test set may not inform it."""
    from src.analysis.preprocessing_confirmation import PreprocessingConfirmation

    with pytest.raises(ValueError, match="validation only"):
        PreprocessingConfirmation(metric="test/f1_macro").compute(datamodule=None)


def test_an_incomplete_confirmation_sweep_produces_no_winner(tmp_path):
    """One candidate is not a comparison, so no decision is written."""
    from src.analysis.preprocessing_confirmation import (
        ConfirmationIncomplete,
        PreprocessingConfirmation,
    )

    analysis = PreprocessingConfirmation(run_root=str(tmp_path / "never_ran"))
    analysis._output_dir = tmp_path / "out"
    analysis._output_dir.mkdir()

    with pytest.raises(ConfirmationIncomplete):
        analysis.compute(datamodule=None)


# ------------------------------------------- Test E: one recipe across every condition


@pytest.mark.parametrize("recipe", CANDIDATE_RECIPES)
def test_every_condition_in_both_experiments_shares_one_recipe(tmp_path, recipe):
    """Five Step 24 conditions and four Step 25 conditions, one recipe between them."""
    from src.analysis.quantum_circuit_ablation_rows import CONDITIONS as Q_CONDITIONS
    from src.analysis.receptive_field_rows import CONDITIONS as RF_CONDITIONS

    path = write_confirmation(tmp_path / "s6", recipe)

    assert len(RF_CONDITIONS) == 5
    assert len(Q_CONDITIONS) == 4
    assert len(step24_recipes(path)) == 1
    assert len(step25_recipes(path)) == 1
    assert step24_recipes(path) == step25_recipes(path)


def test_an_identity_recipe_becomes_null_for_both(tmp_path):
    """``conventional`` reads the raw tree and must not be sent as a mirror name."""
    path = write_confirmation(tmp_path / "s6", "conventional")

    assert step24_recipes(path) == {"data.recipe=null"}
    assert step25_recipes(path) == {"data.recipe=null"}


# ------------------------------------------------------ Test F: no hard-coded winner


def test_no_consumer_module_names_a_preprocessing_recipe():
    """Until the confirmation runs, selected_recipe is NOT YET ESTABLISHED."""
    forbidden = ("clahe", "diffusion_i5_k15", "diffusion_i10_k15", "gamma", "wiener")

    for module in ("src/analysis/receptive_field_rows.py",
                   "src/analysis/receptive_field_study.py",
                   "src/analysis/quantum_circuit_ablation_rows.py",
                   "src/analysis/quantum_circuit_ablation_study.py",
                   "src/analysis/preprocessing_confirmation.py"):
        source = Path(module).read_text(encoding="utf-8").lower()
        for name in forbidden:
            assert name not in source, f"{module} names the recipe {name!r}"


def test_neither_experiment_config_names_a_recipe():
    """:return: None."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    for analysis in ("step24_receptive_field", "step25_quantum_circuit_ablation"):
        GlobalHydra.instance().clear()
        with initialize(version_base="1.3", config_path="../configs"):
            cfg = compose(config_name="analyze.yaml", overrides=[f"analysis={analysis}"])
        GlobalHydra.instance().clear()

        assert cfg.analysis.recipe is None, f"{analysis} encodes a recipe"
        assert cfg.analysis.confirmation_summary is None


def test_the_confirmation_config_implies_no_winner():
    """:return: None."""
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="analyze.yaml", overrides=["analysis=step06_confirm"])
    GlobalHydra.instance().clear()

    assert cfg.analysis.run_root is None
    assert cfg.analysis.metric.startswith("val/")


# ------------------------------------------------------ Test G: dependency ordering


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    """A pipeline whose proxy has run, so the confirmation candidates are enumerable.

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

    # Deliberately opaque names: the graph must not depend on which recipe won.
    summary("step06_preprocessing", {
        "selected_recipe": "cand_a",
        "ranking": [{"recipe": "cand_a"}, {"recipe": "cand_b"}, {"recipe": "cand_c"},
                    {"recipe": "conventional"}],
    })
    summary("step14_loss_selection", {"selected_loss": "weighted_ce"})
    return pipeline


def order(pipeline):
    """:param pipeline: The pipeline.

    :return: Stage ids in graph order.
    """
    return [s.id for s in kp.build_stages(pipeline)]


def test_the_confirmation_stage_is_in_the_graph(pipe):
    """Absent from the graph, Run All would reach Step 24 and stop."""
    ids = order(pipe)

    assert "step06_confirm" in ids
    assert any(i.startswith("step06_confirm/") for i in ids), "no confirmation training runs"


def test_the_confirmation_runs_after_the_proxy(pipe):
    """The proxy supplies the candidates, so it must finish first."""
    ids = order(pipe)

    assert ids.index("step06_preprocessing") < ids.index("step06_confirm")
    for stage_id in ids:
        if stage_id.startswith("step06_confirm/"):
            assert ids.index("step06_preprocessing") < ids.index(stage_id)


def test_the_confirmation_summary_follows_its_training_runs(pipe):
    """The summary is assembled from the runs; it cannot precede them."""
    ids = order(pipe)
    runs = [i for i in ids if i.startswith("step06_confirm/")]

    assert runs
    assert max(ids.index(i) for i in runs) < ids.index("step06_confirm")


def test_step24_and_step25_are_downstream_of_the_confirmation(pipe):
    """The dependency the pipeline must express, not merely happen to satisfy."""
    ids = order(pipe)

    assert ids.index("step06_confirm") < ids.index("step24_receptive_field")
    assert ids.index("step06_confirm") < ids.index("step25_quantum_circuit_ablation")

    for stage_id in ids:
        if stage_id.startswith(("step24_", "step25_")):
            assert ids.index("step06_confirm") < ids.index(stage_id)


def test_the_consumers_are_not_merely_downstream_of_the_proxy(pipe):
    """Ordering after the proxy is not the dependency; ordering after the decision is.

    Both hold, so this checks the stronger claim: the confirmation sits strictly between
    the proxy and the consumers.
    """
    ids = order(pipe)
    proxy, confirm = ids.index("step06_preprocessing"), ids.index("step06_confirm")

    assert proxy < confirm < ids.index("step24_receptive_field")
    assert proxy < confirm < ids.index("step25_quantum_circuit_ablation")


def test_the_confirmation_writes_where_the_consumers_read(pipe):
    """A dependency that writes to the wrong path is not a dependency."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step06_confirm")
    written = pipe.out_dir(stage) / "step06_confirm_summary.json"

    assert written == pipe.summary_path(*kp.STEP06_CONFIRM_SUMMARY)


def test_run_all_stops_at_step24_when_the_confirmation_is_absent(pipe):
    """Run All must fail at the consumer rather than proceed on a proxy winner.

    Checked on the TRAINING stages: they are what resolve the recipe, and they are what a
    Run All would reach first. The analysis stages fail earlier still, on their missing
    checkpoints.
    """
    stages = kp.build_stages(pipe)
    training = [
        s for s in stages
        if s.is_train and s.id.startswith(("step24_", "step25_"))
    ]
    assert len(training) == 15 + 12, "expected every Step 24 and Step 25 training stage"

    for stage in training:
        with pytest.raises(kp.StageFailed, match="confirm|authoritative|unconfirmed"):
            stage.build(pipe)


def test_the_consumers_build_once_the_confirmation_exists(pipe):
    """And the same graph proceeds once the artefact is there."""
    write_confirmation(
        pipe.log_root / "analyze" / "runs" / "step06_confirm", "CONFIRMED_WINNER"
    )

    for stage in kp.build_stages(pipe):
        if stage.is_train and stage.id.startswith(("step24_", "step25_")):
            overrides = stage.build(pipe)
            assert "data.recipe=CONFIRMED_WINNER" in overrides, stage.id


def test_every_training_condition_gets_the_confirmed_recipe_through_the_pipeline(pipe):
    """End to end: the artefact reaches all nine training conditions' overrides."""
    write_confirmation(
        pipe.log_root / "analyze" / "runs" / "step06_confirm", "CONFIRMED_WINNER"
    )
    stages = kp.build_stages(pipe)

    recipes = set()
    for stage in stages:
        if stage.is_train and stage.id.startswith(("step24_", "step25_")):
            recipes |= {
                o for o in stage.build(pipe) if o.startswith("data.recipe=")
            }

    assert recipes == {"data.recipe=CONFIRMED_WINNER"}


def test_the_candidate_set_comes_from_the_proxy_ranking_not_a_literal(pipe):
    """Which candidates are confirmed is data-driven, and overridable explicitly."""
    assert pipe.confirm_candidates() == ["cand_a", "cand_b", "cand_c", "null"]

    pipe.args.confirm_recipes = "only_this"
    assert pipe.confirm_candidates() == ["only_this", "null"]


def test_the_conventional_reference_is_always_confirmed(pipe):
    """A confirmation that cannot say "no preprocessing was as good" confirms nothing."""
    assert "null" in pipe.confirm_candidates()

    pipe.args.confirm_recipes = "a,b"
    assert "null" in pipe.confirm_candidates()


def test_the_confirmation_is_resumable_rather_than_rerun(pipe):
    """It is GPU-expensive, so a completed run must not repeat on the next invocation."""
    stage = next(s for s in kp.build_stages(pipe) if s.id == "step06_confirm")
    marker = pipe.out_dir(stage) / ".pipeline_done.json"

    assert not marker.exists()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")

    # The runner skips any stage carrying this marker; --list renders it as complete.
    assert marker.exists()
    assert pipe.confirmation_is_available() is False  # marker is not the artefact

    write_confirmation(pipe.log_root / "analyze" / "runs" / "step06_confirm", "X")
    assert pipe.confirmation_is_available() is True
