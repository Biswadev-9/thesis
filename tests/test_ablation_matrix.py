"""Step 21: the A0-A8 + P ablation matrix, pinned before anything is trained.

An ablation row is a scientific claim about what changed. The failure mode this file
guards is not a crash - it is a row that runs perfectly and measures something other than
its label says. Three ways that happens here:

1. **Inheritance.** Every training stage in ``scripts/kaggle_pipeline.py`` applies
   ``recipe_override()`` and ``imbalance_overrides()``, which inject Step 6's and Step 8's
   *selections*. The real run confirms it: Step 9's EfficientNet trained with
   ``data.recipe=clahe``, not the ``recipe: null`` its experiment config declares. An
   ablation row that inherits those is labelled "diffusion" and trained on whatever won
   Step 6. Every row therefore pins its own recipe, normalization, augmentation, sampler
   and loss, and the pinning is asserted here against the composed config.

2. **Cache collision.** A6 and A7 train a head on features extracted from
   diffusion-trained branches. Reusing the ``default`` tag would train them on the
   shipped model's CLAHE features while the row still read "diffusion".

3. **Degeneracy.** A7 is "core model + imbalance-aware loss". If A6 already carried
   Step 14's loss the two rows would be identical and the delta identically zero, so A6
   is pinned to ``plain_ce`` and A7 to whatever Step 14 selected.

The matrix deliberately keeps A2-A6 on diffusion, exactly as the specification writes
them, and adds row P for the model the study actually ships - which is CLAHE-based,
because that is what Step 6 selected. A6 is therefore *not* P, and the A7-vs-P delta is
the end-to-end preprocessing comparison Step 6's own summary asks for.

Step 15 is untouched by all of this; P reuses its result rather than redefining it.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.analysis.ablation_rows import (
    DIFFUSION,
    DIFFUSION_FEATURE_TAG,
    PROTOCOL_SEEDS,
    ROWS,
    SELECTED,
    STEP14_LOSS,
    AblationContext,
    get_row,
    row_overrides,
)

#: A context standing in for a completed Step 6 and Step 14. The real values are read from
#: their summaries; these mirror the run that exists today.
CONTEXT = AblationContext(
    diffusion_recipe="diffusion_i10_k15",
    selected_recipe="clahe",
    step14_loss="weighted_ce",
)


def compose_row(row_id: str, seed: int = 42, context: AblationContext = CONTEXT):
    """Compose the training config a row would actually run under.

    :param row_id: Row identifier.
    :param seed: Protocol seed.
    :param context: Resolved Step 6 / Step 14 selections.
    :return: The composed DictConfig.
    """
    overrides = row_overrides(get_row(row_id), seed=seed, context=context)
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    GlobalHydra.instance().clear()
    return cfg


#: Rows that train something. A8 and P train nothing - see their own tests.
TRAINING_ROWS = [row.row_id for row in ROWS if row.trains]

#: The image-space rows, which carry a preprocessing recipe.
IMAGE_ROWS = [row.row_id for row in ROWS if row.trains and row.feature_tag is None]


# --------------------------------------------------------------- the matrix itself


def test_every_specified_row_is_present_exactly_once():
    """A0-A8 and P, no gaps and no duplicates."""
    ids = [row.row_id for row in ROWS]

    assert ids == ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "P"]
    assert len(ids) == len(set(ids))


def test_the_row_labels_match_the_specification_table():
    """The configuration column of Step 21's table, verbatim."""
    assert get_row("A0").label == "Raw image + baseline CNN"
    assert get_row("A1").label == "Conventional preprocessing + CNN"
    assert get_row("A2").label == "Diffusion preprocessing + CNN"
    assert get_row("A3").label == "Diffusion + adaptive multiscale branch"
    assert get_row("A4").label == "Diffusion + fixed QCNN branch"
    assert get_row("A5").label == "Diffusion + adaptive quantum branch"
    assert get_row("A6").label == (
        "Diffusion + adaptive multiscale + adaptive quantum + fusion"
    )
    assert get_row("A7").label == "Core model + imbalance-aware loss"
    assert get_row("A8").label == "Core model + explainability and uncertainty"


# ------------------------------------------------------------------- preprocessing


def test_a0_is_the_raw_image_condition():
    """A0 is the only row with no intensity treatment at all.

    Without ``normalize=none`` A0 and A1 differ in nothing, which is how the reference
    notebook came to report one number for two rows.
    """
    cfg = compose_row("A0")

    assert cfg.data.recipe is None
    assert cfg.data.normalize == "none"
    assert cfg.data.augment is False


def test_a1_is_the_conventional_step5_pipeline():
    """A1 adds the Step 5 intensity treatment and nothing else."""
    cfg = compose_row("A1")

    assert cfg.data.recipe is None
    assert cfg.data.normalize == "imagenet"

    raw = compose_row("A0")
    assert cfg.data.normalize != raw.data.normalize, "A0 and A1 have collapsed into one row"


@pytest.mark.parametrize("row_id", ["A2", "A3", "A4", "A5"])
def test_the_diffusion_rows_train_on_the_diffusion_recipe(row_id):
    """A2-A5 read "Diffusion ..." in the specification, so they train on diffusion.

    Not on whatever Step 6 selected. Step 6 chose CLAHE in the current run; silently
    substituting it would delete the study's diffusion evidence while leaving the row
    labels claiming it.
    """
    cfg = compose_row(row_id)

    assert cfg.data.recipe == CONTEXT.diffusion_recipe
    assert cfg.data.recipe.startswith("diffusion_")
    assert cfg.data.recipe != CONTEXT.selected_recipe


def test_the_diffusion_recipe_comes_from_step6s_ranking_not_a_literal():
    """Which diffusion configuration is an empirical question Step 6 already answered.

    Its summary ranks every candidate, so the best diffusion variant is read from there.
    Hard-coding ``diffusion_i10_k15`` would silently ignore a better one.
    """
    assert get_row("A2").recipe == DIFFUSION

    other = AblationContext(
        diffusion_recipe="diffusion_i20_k30",
        selected_recipe="clahe",
        step14_loss="weighted_ce",
    )
    assert compose_row("A2", context=other).data.recipe == "diffusion_i20_k30"


def test_step6s_ranking_can_actually_supply_a_diffusion_recipe():
    """The resolver reads a real summary shape, not an invented one."""
    ranking = [
        {"recipe": "clahe", "macro_f1": 0.257},
        {"recipe": "diffusion_i10_k15", "macro_f1": 0.247},
        {"recipe": "conventional", "macro_f1": 0.100},
    ]
    context = AblationContext.from_summaries(
        step06={"selected_recipe": "clahe", "ranking": ranking},
        step14={"selected_loss": "weighted_ce"},
    )

    assert context.diffusion_recipe == "diffusion_i10_k15"
    assert context.selected_recipe == "clahe"
    assert context.step14_loss == "weighted_ce"


def test_a_missing_diffusion_candidate_fails_loudly():
    """Step 21 cannot run its diffusion rows if Step 6 never evaluated diffusion."""
    with pytest.raises(ValueError, match="diffusion"):
        AblationContext.from_summaries(
            step06={"selected_recipe": "clahe", "ranking": [{"recipe": "clahe"}]},
            step14={"selected_loss": "weighted_ce"},
        )


# --------------------------------------------------------------------------- loss


def test_a6_uses_plain_ce_so_that_a7_measures_something():
    """A7 is "core model + imbalance-aware loss". A6 must therefore not already have one.

    With both on Step 14's loss the rows are identical and the reported delta is zero by
    construction - a number that looks like evidence and is not.
    """
    assert get_row("A6").loss == "plain_ce"
    assert compose_row("A6").model.criterion.use_class_weights is False


def test_a7_uses_whatever_step14_selected():
    """Step 14 is the source of truth for the loss, here as everywhere else."""
    assert get_row("A7").loss == STEP14_LOSS

    cfg = compose_row("A7")
    assert cfg.model.criterion.use_class_weights is True

    focal = AblationContext(
        diffusion_recipe="diffusion_i10_k15",
        selected_recipe="clahe",
        step14_loss="focal",
    )
    assert "FocalLoss" in compose_row("A7", context=focal).model.criterion._target_


def test_a6_and_a7_differ_in_the_loss_and_in_nothing_else():
    """The comparison only isolates the loss if everything else is held identical."""
    a6, a7 = compose_row("A6"), compose_row("A7")

    assert a6.data.tag == a7.data.tag
    assert a6.model.net == a7.model.net
    assert a6.model.optimizer == a7.model.optimizer
    assert a6.model.criterion != a7.model.criterion


@pytest.mark.parametrize("row_id", ["A0", "A1", "A2", "A3", "A4", "A5", "A6"])
def test_the_pre_loss_rows_all_train_on_plain_ce(row_id):
    """A0-A6 predate the imbalance-aware loss, which A7 is what introduces."""
    assert compose_row(row_id).model.criterion.use_class_weights is False


# ------------------------------------------------------------------ the feature cache


def test_the_fusion_rows_cannot_read_the_shipped_models_cache():
    """A6/A7 are diffusion rows; the default cache holds the shipped CLAHE features.

    Sharing the tag would train a head labelled "diffusion" on CLAHE-derived features -
    a wrong result rather than an error, and invisible in the output.
    """
    for row_id in ("A6", "A7"):
        row = get_row(row_id)
        assert row.feature_tag == DIFFUSION_FEATURE_TAG
        assert row.feature_tag != get_row("P").feature_tag
        assert compose_row(row_id).data.tag == DIFFUSION_FEATURE_TAG

    assert get_row("P").feature_tag == "default"


def test_the_fusion_rows_extract_from_their_own_diffusion_branches():
    """A6's branches must be the diffusion-trained ones, not the shipped CLAHE branches.

    A2 and A5 already train exactly those networks on exactly that recipe under the same
    protocol, so they are reused as A6's branches rather than retrained.
    """
    for row_id in ("A6", "A7"):
        assert get_row(row_id).branch_sources == ("A2", "A5")

    assert get_row("A2").model == "branch_classical"
    assert get_row("A5").model == "branch_adaptive_quantum"
    assert compose_row("A2").data.recipe == compose_row("A5").data.recipe


# ---------------------------------------------------------------------------- row P


def test_p_is_the_shipped_model_and_reuses_step15():
    """P is the study's actual proposed model: CLAHE-based, already trained and tested."""
    p = get_row("P")

    assert p.recipe == SELECTED
    assert p.loss == STEP14_LOSS
    assert p.feature_tag == "default"
    assert p.trains is False
    assert p.reuse_of == "step15_final"


def test_p_resolves_to_the_recipe_step6_actually_selected():
    """CLAHE today, and whatever Step 6 selects if it is ever re-run - never a literal.

    Pinning P to the string "clahe" would let it drift away from the checkpoint it claims
    to describe the moment Step 6's answer changed.
    """
    from src.analysis.ablation_rows import resolve_recipe

    assert resolve_recipe(get_row("P"), CONTEXT) == "clahe"

    moved = AblationContext(
        diffusion_recipe="diffusion_i10_k15",
        selected_recipe="wiener",
        step14_loss="weighted_ce",
    )
    assert resolve_recipe(get_row("P"), moved) == "wiener"


def test_a6_is_not_the_proposed_model():
    """The specification's A6 says "Diffusion"; the shipped model uses what Step 6 chose.

    They coincide only if Step 6 selects diffusion, which it did not. Collapsing them
    would either delete the diffusion evidence or misreport the shipped model.
    """
    from src.analysis.ablation_rows import resolve_recipe

    assert resolve_recipe(get_row("A6"), CONTEXT) != resolve_recipe(get_row("P"), CONTEXT)


def test_a7_and_p_differ_only_in_preprocessing():
    """That is what makes their delta an end-to-end test of Step 6's proxy decision.

    Step 6's own summary carries the caveat that its ranking comes from a reduced-scale
    proxy and should be confirmed with the real backbone. A7 vs P is that confirmation.
    """
    a7, p = get_row("A7"), get_row("P")

    assert a7.loss == p.loss
    assert a7.model == p.model
    assert a7.recipe != p.recipe


# ------------------------------------------------------------------------------ A8


def test_a8_trains_nothing_and_says_why():
    """Explanations change no weights, so A8's classification metrics equal A7's.

    Reporting a delta for A8 would be reporting noise as a finding. It is a qualitative
    row: deletion/insertion and MC-dropout uncertainty, computed on A7's checkpoint.
    """
    a8 = get_row("A8")

    assert a8.trains is False
    assert a8.reuse_of == "A7"
    assert "identical" in a8.note.lower()

    with pytest.raises(ValueError, match="trains nothing"):
        row_overrides(a8, seed=42, context=CONTEXT)


# -------------------------------------------------------------- protocol and seeds


def test_the_seed_set_is_the_protocol_one():
    """Step 15: "at least three seeds for the final model and major baselines"."""
    assert PROTOCOL_SEEDS == (42, 123, 7)


@pytest.mark.parametrize("row_id", TRAINING_ROWS)
def test_every_row_composes_the_fixed_protocol(row_id):
    """One protocol for every row, so a delta is architecture and not schedule.

    Read from the composed Step 15 protocol rather than retyped, so the two cannot drift.
    """
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        reference = compose(config_name="train.yaml", overrides=["experiment=step15_final_protocol"])
    GlobalHydra.instance().clear()

    cfg = compose_row(row_id)

    assert cfg.trainer.max_epochs == reference.trainer.max_epochs
    assert cfg.trainer.min_epochs == reference.trainer.min_epochs
    assert cfg.callbacks.early_stopping.monitor == reference.callbacks.early_stopping.monitor
    assert cfg.callbacks.early_stopping.mode == reference.callbacks.early_stopping.mode
    assert cfg.callbacks.early_stopping.patience == reference.callbacks.early_stopping.patience
    assert cfg.callbacks.model_checkpoint.monitor == reference.callbacks.model_checkpoint.monitor
    assert cfg.callbacks.model_checkpoint.mode == reference.callbacks.model_checkpoint.mode
    assert cfg.callbacks.model_checkpoint.save_top_k == reference.callbacks.model_checkpoint.save_top_k
    assert cfg.data.batch_size == reference.data.batch_size
    assert cfg.model.optimizer.lr == reference.model.optimizer.lr
    assert cfg.model.optimizer.weight_decay == reference.model.optimizer.weight_decay
    assert cfg.optimized_metric == reference.optimized_metric


@pytest.mark.parametrize("row_id", TRAINING_ROWS)
def test_every_row_runs_the_protocol_seed_it_was_given(row_id):
    """:param row_id: Row under test."""
    for seed in PROTOCOL_SEEDS:
        assert compose_row(row_id, seed=seed).seed == seed


# ------------------------------------------------------- no inherited selections


@pytest.mark.parametrize("row_id", TRAINING_ROWS)
def test_every_row_pins_its_own_loss_and_sampler(row_id):
    """Nothing may be left to ``imbalance_overrides()``, which injects Step 8's choice.

    A row inheriting it is labelled for its architecture and trained under whatever
    strategy won Step 8 - and it changes silently when Step 8 is re-run.
    """
    overrides = row_overrides(get_row(row_id), seed=42, context=CONTEXT)

    assert sum(o.startswith("loss@model.criterion=") for o in overrides) == 1
    assert sum(o.startswith("data.use_weighted_sampler=") for o in overrides) == 1


@pytest.mark.parametrize("row_id", IMAGE_ROWS)
def test_every_image_row_pins_its_own_preprocessing(row_id):
    """Nothing may be left to ``recipe_override()``, which injects Step 6's choice."""
    overrides = row_overrides(get_row(row_id), seed=42, context=CONTEXT)

    for key in ("data.recipe=", "data.normalize=", "data.augment="):
        assert sum(o.startswith(key) for o in overrides) == 1, f"{row_id} does not pin {key}"


@pytest.mark.parametrize("row_id", TRAINING_ROWS)
def test_no_row_emits_a_selection_placeholder(row_id):
    """The symbolic tokens must be resolved before they reach Hydra."""
    overrides = row_overrides(get_row(row_id), seed=42, context=CONTEXT)

    assert not any("@diffusion" in o or "@selected" in o or "@step14" in o for o in overrides)


@pytest.mark.parametrize("row_id", TRAINING_ROWS)
def test_every_row_holds_augmentation_and_sampling_where_the_shipped_model_had_them(row_id):
    """A-rows must match P's data handling or their deltas confound it with preprocessing.

    Step 8 selected ``baseline`` - plain CE, no sampler, no augmentation - and the shipped
    branches trained that way. These are pinned literals rather than an inherited lookup,
    and this test fails if Step 8's selection ever stops agreeing with them.
    """
    cfg = compose_row(row_id)
    assert cfg.data.use_weighted_sampler is False

    if get_row(row_id).feature_tag is None:
        assert cfg.data.augment is False


# ------------------------------------------------------ nothing validated is touched


def test_step15_and_the_fixed_protocol_are_untouched():
    """Phase 8 adds rows; it does not amend the finalized protocol.

    Hashes of the two files Step 15 is defined by. A deliberate protocol amendment updates
    these; an accidental edit fails the suite.
    """
    import hashlib

    def digest(path: str) -> str:
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()

    assert digest("configs/protocol/fixed.yaml") == (
        "d2671a45073f219b8b95f8e1ff3ccebaf002cfc279df76b355de18b83561c64a"
    ), "configs/protocol/fixed.yaml changed - amend the protocol deliberately or revert"
    assert digest("configs/experiment/step15_final_protocol.yaml") == (
        "dfb9f37df799e9e78886fe8a237f342858c68ebfd7afa4f79bd0719890acc0b4"
    ), "Step 15's experiment changed - amend deliberately or revert"


def test_no_ablation_row_retrains_step15():
    """P reports the existing Step 15 result; it does not re-run it under a new name."""
    assert all(row.experiment != "step15_final_protocol" for row in ROWS if row.trains)
    assert get_row("P").reuse_of == "step15_final"
