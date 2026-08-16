"""Step 24: the receptive-field strategy ladder, defined once as data.

Step 24 asks whether *spatially adaptive selection among* several receptive fields earns
its place - against committing to a single fixed receptive field, and against combining the
same fields without any gate:

    fixed 3x3 -> fixed 5x5 -> fixed dilated 3x3 -> multi-scale, ungated -> adaptive

**No new architecture is introduced.** All five conditions already exist as Step 11 arms in
``src/models/components/multiscale.py``; this module selects five of the eight and pins the
settings under which they are compared. Building a sixth implementation of a 3x3 path would
add a way for the ladder to disagree with the module it is meant to describe.

Two things this file exists to prevent.

**Inherited settings.** ``configs/experiment/step11_arm_ablation.yaml`` declares
``augment: true`` and ``use_weighted_sampler: true``, and ``scripts/kaggle_pipeline.py``
then overrides both with whatever Step 8 selected. An arm trained that way is not
comparable with one trained under a different selection, and nothing in the output would
show it. Every condition here pins recipe, normalization, augmentation, sampler and loss
explicitly, and :func:`condition_overrides` emits all five.

**A capacity claim that is not true.** The conditions are *not* parameter-matched, and
saying so is part of the result rather than a caveat to bury. The fixed conditions are
smaller because a single path is smaller. What makes the primary comparison defensible is
the opposite relationship: the ungated control is the *larger* of the two models it is
compared against, so a win for the adaptive condition cannot be attributed to capacity.

The terminology is deliberate. The convolutions themselves are fixed; what adapts is the
per-pixel weighting over their outputs. "Spatially adaptive multi-scale receptive-field
selection" is accurate; "dynamic kernels" would not be.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.analysis.ablation_rows import PROTOCOL_SEEDS

#: Step 15's seeds, reused rather than redeclared - Step 24 is not a new seed policy.
SEEDS: Tuple[int, ...] = PROTOCOL_SEEDS

#: Settings held identical across every condition, so the only thing that varies is the
#: receptive-field strategy. Pinned as literals and emitted explicitly, never inherited.
#:
#: ``plain_ce`` matches the Phase 8 ablation rows, which makes the adaptive condition
#: configuration-identical to row A3 and the two experiments mutually checkable.
PINNED: Dict[str, Any] = {
    "normalize": "imagenet",
    "augment": False,
    "use_weighted_sampler": False,
    "loss": "plain_ce",
}

#: One formal hypothesis. Holm over a family of one is the identity, which is reported
#: explicitly rather than left for a reader to infer.
FAMILY_SIZE = 1

#: Recipes that read the raw tree directly and need no materialised mirror. Mirrors
#: ``IDENTITY_RECIPES`` in ``src/data/components/preprocessing.py``; a confirmation that
#: selects one of these becomes ``data.recipe=null``.
_IDENTITY_RECIPES = ("raw", "conventional")


@dataclass(frozen=True)
class ReceptiveFieldCondition:
    """One rung of the ladder.

    :param condition_id: Stable identifier used in outputs and run directories.
    :param arm: The existing Step 11 arm this condition reuses.
    :param variant: The arm's branch variant, for assertion against the built model.
    :param receptive_field_strategy: What the condition does, in words.
    :param fusion: How its paths are combined. ``n/a`` for the single-path conditions.
    :param adaptive: Whether the combination is input-dependent.
    :param note: Why the condition is in the ladder.
    """

    condition_id: str
    arm: str
    variant: str
    receptive_field_strategy: str
    fusion: str
    adaptive: bool
    note: str = ""


#: The ladder, in the order it is reported.
CONDITIONS: Tuple[ReceptiveFieldCondition, ...] = (
    ReceptiveFieldCondition(
        condition_id="FIXED_3X3",
        arm="arm1_fixed_3x3",
        variant="fixed_3x3",
        receptive_field_strategy="single 3x3 convolution (3x3 effective)",
        fusion="n/a - single path",
        adaptive=False,
        note="Conventional fixed receptive field, and the smallest model in the ladder.",
    ),
    ReceptiveFieldCondition(
        condition_id="FIXED_5X5",
        arm="arm2_fixed_5x5",
        variant="fixed_5x5",
        receptive_field_strategy="single 5x5 convolution (5x5 effective)",
        fusion="n/a - single path",
        adaptive=False,
        note="A wider fixed field, at roughly 2.8x the fine path's parameter cost.",
    ),
    ReceptiveFieldCondition(
        condition_id="FIXED_DILATED_3X3",
        arm="arm3_fixed_dilated",
        variant="fixed_dilated",
        receptive_field_strategy="single 3x3 convolution, dilation 3 (7x7 effective)",
        fusion="n/a - single path",
        adaptive=False,
        note=(
            "The broad field at 3x3 parameter cost. NOT a literal 7x7 convolution - "
            "substituting one would change the condition's capacity and stop it matching "
            "the same path inside the multi-scale conditions."
        ),
    ),
    ReceptiveFieldCondition(
        condition_id="MULTISCALE_NO_GATE",
        arm="arm4_concat_nogate",
        variant="concat_nogate",
        receptive_field_strategy="parallel 3x3 + 5x5 + dilated 3x3",
        fusion="ungated: concatenation followed by a learned 1x1 projection",
        adaptive=False,
        note=(
            "The control for H24: the same three receptive fields, combined without any "
            "gate. Its mixer is learned but input-INDEPENDENT once trained, and it is a "
            "projection rather than an average - so it must not be described as "
            "equal-weight fusion."
        ),
    ),
    ReceptiveFieldCondition(
        condition_id="ADAPTIVE_MULTISCALE",
        arm="arm6_spatial_gate",
        variant="spatial_gate",
        receptive_field_strategy="parallel 3x3 + 5x5 + dilated 3x3",
        fusion=(
            "spatially adaptive: per-pixel softmax over the three paths, fused as a "
            "convex combination"
        ),
        adaptive=True,
        note=(
            "The proposed module. The convolutions are fixed; what adapts is the "
            "per-pixel weighting over their outputs, computed from the input on every "
            "forward pass."
        ),
    ),
)

#: The one formal hypothesis. Both sides carry the identical three receptive fields, so the
#: gate is the only thing that differs - which is what makes this the causal comparison.
PRIMARY_COMPARISON: Dict[str, str] = {
    "id": "H24",
    "condition_a": "ADAPTIVE_MULTISCALE",
    "condition_b": "MULTISCALE_NO_GATE",
    "question": (
        "Does spatially adaptive gating improve performance over ungated multi-scale "
        "fusion of the same three receptive fields?"
    ),
    "rq": "RQ4",
}

#: Descriptive comparisons. Each changes the receptive field AND the parameter budget at
#: once, so none of them can isolate gating; reporting a p-value for them would overclaim.
SUPPORTING_COMPARISONS: Tuple[Dict[str, str], ...] = (
    {
        "id": "S24a_vs_fixed_3x3",
        "condition_a": "ADAPTIVE_MULTISCALE",
        "condition_b": "FIXED_3X3",
        "note": (
            "Adaptive multi-scale against a conventional fine kernel. Descriptive: the "
            "adaptive condition also carries roughly three times the parameters."
        ),
    },
    {
        "id": "S24b_vs_fixed_5x5",
        "condition_a": "ADAPTIVE_MULTISCALE",
        "condition_b": "FIXED_5X5",
        "note": (
            "Adaptive multi-scale against a single wider kernel. Descriptive: capacity "
            "and receptive field both change."
        ),
    },
    {
        "id": "S24c_vs_fixed_dilated",
        "condition_a": "ADAPTIVE_MULTISCALE",
        "condition_b": "FIXED_DILATED_3X3",
        "note": (
            "Adaptive multi-scale against the broad field alone. Descriptive: capacity "
            "and receptive field both change."
        ),
    },
)

_BY_ID: Dict[str, ReceptiveFieldCondition] = {c.condition_id: c for c in CONDITIONS}


@dataclass(frozen=True)
class ReceptiveFieldContext:
    """The preprocessing choice every condition shares.

    Held in a context rather than written into each condition so the ladder cannot end up
    with two recipes in it: one object supplies the recipe to all five, and there is no
    per-condition recipe field that could diverge.

    :param recipe: The materialised recipe name, or ``None`` for the raw tree.
    :param source: Where the recipe came from, carried into the output so a reader can
        tell a confirmed decision from a hand-supplied override.
    """

    recipe: Optional[str]
    source: str = "explicit"

    @classmethod
    def from_confirmation(cls, summary_path: Optional[str]) -> "ReceptiveFieldContext":
        """Resolve the recipe from Step 6's authoritative confirmation.

        Step 24 spends fifteen training runs establishing whether adaptive gating helps.
        Running them on a preprocessing the study never confirmed would make the answer
        apply to a configuration the thesis does not ship - so there is deliberately **no
        fallback to the proxy ranking**. If the confirmation has not happened, this raises
        and Step 24 does not start.

        :param summary_path: Path to ``step06_confirm_summary.json``.
        :return: The context.
        :raises ConfirmationIncomplete: If no authoritative decision exists.
        """
        from src.analysis.preprocessing_confirmation import load_confirmed_recipe

        recipe = load_confirmed_recipe(summary_path)
        return cls(
            recipe=None if recipe in _IDENTITY_RECIPES else recipe,
            source=f"step06_confirm ({summary_path})",
        )


def get_condition(condition_id: str) -> ReceptiveFieldCondition:
    """:param condition_id: Condition identifier.

    :return: That condition.
    :raises KeyError: If the identifier is not in the ladder.
    """
    if condition_id not in _BY_ID:
        raise KeyError(
            f"Unknown condition {condition_id!r}; expected one of {sorted(_BY_ID)}"
        )
    return _BY_ID[condition_id]


def condition_overrides(
    condition: ReceptiveFieldCondition, seed: int, context: ReceptiveFieldContext
) -> List[str]:
    """The Hydra overrides that train one condition at one seed.

    Every controlled setting is emitted explicitly. Nothing is left to the pipeline's
    ``recipe_override()`` or ``imbalance_overrides()``, which would inject Step 6's and
    Step 8's selections and make two conditions incomparable for a reason that has nothing
    to do with receptive fields.

    :param condition: The condition to train.
    :param seed: One of :data:`SEEDS`.
    :param context: The shared preprocessing choice.
    :return: Override strings for ``src/train.py``.
    """
    recipe = context.recipe or "null"

    return [
        "experiment=step24_receptive_field",
        f"model.net.arm={condition.arm}",
        f"seed={seed}",
        f"loss@model.criterion={PINNED['loss']}",
        f"data.recipe={recipe}",
        f"data.normalize={PINNED['normalize']}",
        f"data.augment={str(PINNED['augment']).lower()}",
        f"data.use_weighted_sampler={str(PINNED['use_weighted_sampler']).lower()}",
    ]


def condition_parameters(condition: ReceptiveFieldCondition) -> int:
    """Count a condition's trainable parameters by building it.

    Measured rather than tabulated: a written-down number would survive the architecture
    changing underneath it, and the whole point of recording capacity here is that the
    ladder is *not* parameter-matched.

    :param condition: The condition.
    :return: Total parameter count.
    """
    from src.models.components.multiscale import MultiscaleClassifier

    model = MultiscaleClassifier.from_arm(condition.arm)
    return int(sum(p.numel() for p in model.parameters()))
