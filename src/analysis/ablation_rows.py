"""Step 21: the ablation matrix, defined once as data.

    "|A0|Raw image + baseline CNN| ... |A8|Core model + explainability and uncertainty|
     The ablation study should report the same metrics for every configuration."

Each row is a claim about what changed. Keeping the rows here rather than in nine hand
written config files is what makes the claims checkable: a test can assert that A6 and A7
differ in the loss and in nothing else, which nine YAML files cannot promise.

Three decisions are encoded here that the specification does not settle on its own.

**A2-A6 stay on diffusion, and row P is added.** The specification writes "Diffusion" into
rows A2-A6, but Step 6 selected CLAHE on measurement, so the model the study actually
ships is CLAHE-based. Substituting the selection into A2-A6 would delete the diffusion
evidence while leaving the labels claiming it; substituting diffusion into the shipped
model would misreport what was trained. Both are therefore represented - the specification's
ladder as written, and row P for the shipped model. Their difference is not an
inconvenience: Step 6's own summary warns that its ranking comes from a reduced-scale proxy
and "should be confirmed with the real backbone on the full validation split". A7 against P
is that confirmation, at full scale, on validation-selected checkpoints.

**A6 uses plain CE.** A7 reads "core model + imbalance-aware loss". If A6 already carried
Step 14's loss the two rows would be identical and their delta zero by construction - a
number that looks like evidence and is not. A6 is pinned to ``plain_ce`` so A7 measures the
loss and nothing else. This defines the ablation row; Step 15 is untouched and still trains
with whatever Step 14 selects.

**Nothing is inherited.** Every training stage in ``scripts/kaggle_pipeline.py`` applies
``recipe_override()`` and ``imbalance_overrides()``, which inject Step 6's and Step 8's
selections. That is right for the main study and wrong for an ablation: a row would be
labelled "diffusion" and trained on whatever won Step 6. Rows therefore pin their own
recipe, normalization, augmentation, sampler and loss, and :func:`row_overrides` emits all
of them explicitly.

What *is* resolved at run time is which diffusion configuration to use and which loss
Step 14 chose - both empirical answers that belong to their own steps, read from their
summaries through :class:`AblationContext` rather than hard-coded here.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Recipe placeholder: the best-ranked diffusion candidate from Step 6.
DIFFUSION = "@diffusion"

#: Recipe placeholder: whatever recipe Step 6 selected, which is what the shipped model
#: trained on. Row P uses this so it cannot drift from the checkpoint it describes.
SELECTED = "@selected"

#: Loss placeholder: whatever Step 14 selected.
STEP14_LOSS = "@step14"

#: Recipes that read the raw tree directly, expressed as ``data.recipe=null``.
_IDENTITY_RECIPES = ("raw", "conventional")

#: Feature cache for the diffusion fusion rows. Must differ from the shipped model's
#: ``default`` cache: sharing it would train A6/A7 on CLAHE-derived features while the
#: rows still read "diffusion".
DIFFUSION_FEATURE_TAG = "a6_diffusion"

#: Step 15: "Run at least three seeds for the final model and major baselines."
PROTOCOL_SEEDS: Tuple[int, ...] = (42, 123, 7)

#: Step 8 selected ``baseline`` - plain CE, no sampler, no augmentation - and the shipped
#: branches trained that way. The A-rows match it so their deltas are not confounded with
#: data handling. Pinned as literals rather than looked up, and a test fails if Step 8's
#: selection ever stops agreeing with them.
_SHIPPED_AUGMENT = False
_SHIPPED_SAMPLER = False


@dataclass(frozen=True)
class AblationRow:
    """One row of the Step 21 table.

    :param row_id: ``A0``-``A8`` or ``P``.
    :param label: The specification's configuration text, verbatim.
    :param recipe: Recipe name, or :data:`DIFFUSION` / :data:`SELECTED`.
    :param normalize: Step 5 intensity treatment; ``none`` is A0's raw condition.
    :param augment: Step 7 augmentation on the training split.
    :param use_weighted_sampler: Step 8 balanced sampling on the training loader.
    :param loss: Loss config name, or :data:`STEP14_LOSS`.
    :param model: Model config name.
    :param experiment: Experiment config the row composes.
    :param feature_tag: Feature cache for fusion rows; ``None`` for image-space rows.
    :param branch_sources: Rows whose checkpoints supply this row's frozen branches.
    :param trains: Whether the row trains anything at all.
    :param reuse_of: What the row reports instead, when it trains nothing.
    :param rqs: Research questions the row supplies evidence for.
    :param extra: Additional Hydra overrides specific to the row.
    :param note: Why the row is defined the way it is.
    """

    row_id: str
    label: str
    recipe: Optional[str] = None
    normalize: str = "imagenet"
    augment: bool = _SHIPPED_AUGMENT
    use_weighted_sampler: bool = _SHIPPED_SAMPLER
    loss: str = "plain_ce"
    model: Optional[str] = None
    experiment: Optional[str] = None
    feature_tag: Optional[str] = None
    branch_sources: Tuple[str, ...] = ()
    trains: bool = True
    reuse_of: Optional[str] = None
    rqs: Tuple[str, ...] = ()
    extra: Tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class AblationContext:
    """The empirical answers Step 21 needs from earlier steps.

    :param diffusion_recipe: Best-ranked diffusion candidate from Step 6.
    :param selected_recipe: The recipe Step 6 selected, which the shipped model uses.
    :param step14_loss: The loss Step 14 selected.
    """

    diffusion_recipe: str
    selected_recipe: str
    step14_loss: str

    @classmethod
    def from_summaries(
        cls, step06: Mapping[str, Any], step14: Mapping[str, Any]
    ) -> "AblationContext":
        """Build the context from the two summaries that decide it.

        The diffusion recipe is taken from Step 6's *ranking*, not from its selection:
        Step 6 selects one recipe overall, and rows A2-A6 need the best diffusion one
        whether or not diffusion won. Ranking order is Step 6's own, so "best" means
        whatever it ranked first.

        :param step06: Parsed ``step06_preprocessing_summary.json``.
        :param step14: Parsed ``step14_loss_selection_summary.json``.
        :return: The resolved context.
        :raises ValueError: If Step 6 evaluated no diffusion candidate, or Step 14
            recorded no selection.
        """
        ranking: Sequence[Mapping[str, Any]] = step06.get("ranking") or []
        diffusion = next(
            (str(entry["recipe"]) for entry in ranking if str(entry.get("recipe", "")).startswith("diffusion_")),
            None,
        )
        if diffusion is None:
            raise ValueError(
                "Step 21 rows A2-A6 are defined on diffusion preprocessing, but Step 6's "
                "ranking contains no diffusion candidate. Re-run step06_preprocessing "
                "with diffusion in its sweep before building the ablation matrix."
            )

        selected = step06.get("selected_recipe")
        if not selected:
            raise ValueError("Step 6 recorded no selected_recipe; row P has nothing to describe.")

        loss = step14.get("selected_loss")
        if not loss:
            raise ValueError("Step 14 recorded no selected_loss; row A7 has no loss to apply.")

        return cls(
            diffusion_recipe=diffusion,
            selected_recipe=str(selected),
            step14_loss=str(loss),
        )


#: The Step 21 table. Order is the specification's.
ROWS: Tuple[AblationRow, ...] = (
    AblationRow(
        row_id="A0",
        label="Raw image + baseline CNN",
        recipe="raw",
        # The only row without an intensity treatment. Without this A0 and A1 differ in
        # nothing, which is how the reference notebook reported one number for two rows.
        normalize="none",
        model="branch_classical",
        experiment="step21_ablation_image",
        rqs=("RQ2", "RQ10"),
        note="Floor condition: resize and ToTensor only.",
    ),
    AblationRow(
        row_id="A1",
        label="Conventional preprocessing + CNN",
        recipe="conventional",
        model="branch_classical",
        experiment="step21_ablation_image",
        rqs=("RQ2", "RQ10"),
        note="Adds the Step 5 intensity treatment and nothing else.",
    ),
    AblationRow(
        row_id="A2",
        label="Diffusion preprocessing + CNN",
        recipe=DIFFUSION,
        model="branch_classical",
        experiment="step21_ablation_image",
        rqs=("RQ2", "RQ10"),
        note=(
            "Same network as A0/A1, so the delta is preprocessing alone. Doubles as A6's "
            "classical branch: identical architecture, recipe, protocol, loss and seed."
        ),
    ),
    AblationRow(
        row_id="A3",
        label="Diffusion + adaptive multiscale branch",
        recipe=DIFFUSION,
        model="branch_multiscale",
        experiment="step21_ablation_image",
        extra=("model.net.arm=arm6_spatial_gate",),
        rqs=("RQ4", "RQ10"),
        note="Step 11's per-pixel gated arm, which is the proposed adaptive module.",
    ),
    AblationRow(
        row_id="A4",
        label="Diffusion + fixed QCNN branch",
        recipe=DIFFUSION,
        model="baseline_fixed_qcnn",
        experiment="step21_ablation_image",
        rqs=("RQ4", "RQ8", "RQ10"),
        note="The quantum floor: a fixed circuit, for A5 to be measured against.",
    ),
    AblationRow(
        row_id="A5",
        label="Diffusion + adaptive quantum branch",
        recipe=DIFFUSION,
        model="branch_adaptive_quantum",
        experiment="step21_ablation_image",
        rqs=("RQ4", "RQ8", "RQ10"),
        note=(
            "Supplies both the spatial and the quantum features for A6, exactly as Step 12 "
            "does for the shipped model - the spatial branch lives inside this network."
        ),
    ),
    AblationRow(
        row_id="A6",
        label="Diffusion + adaptive multiscale + adaptive quantum + fusion",
        recipe=DIFFUSION,
        # Plain CE, so that A7 measures the imbalance-aware loss rather than repeating A6.
        loss="plain_ce",
        model="final_classifier",
        experiment="step21_ablation_fusion",
        feature_tag=DIFFUSION_FEATURE_TAG,
        branch_sources=("A2", "A5"),
        rqs=("RQ8", "RQ10"),
        note="The specification's full model. Not the shipped model - see row P.",
    ),
    AblationRow(
        row_id="A7",
        label="Core model + imbalance-aware loss",
        recipe=DIFFUSION,
        loss=STEP14_LOSS,
        model="final_classifier",
        experiment="step21_ablation_fusion",
        feature_tag=DIFFUSION_FEATURE_TAG,
        branch_sources=("A2", "A5"),
        rqs=("RQ6", "RQ10"),
        note="A6 with Step 14's loss. The only difference from A6 is the criterion.",
    ),
    AblationRow(
        row_id="A8",
        label="Core model + explainability and uncertainty",
        recipe=DIFFUSION,
        loss=STEP14_LOSS,
        model="final_classifier",
        feature_tag=DIFFUSION_FEATURE_TAG,
        trains=False,
        reuse_of="A7",
        rqs=("RQ3", "RQ9"),
        note=(
            "Explanations change no weights, so its classification metrics are identical "
            "to A7's by construction. Reported qualitatively - deletion/insertion and "
            "MC-dropout uncertainty on A7's checkpoint - rather than as a delta."
        ),
    ),
    AblationRow(
        row_id="P",
        label="Proposed model as shipped",
        recipe=SELECTED,
        loss=STEP14_LOSS,
        model="final_classifier",
        feature_tag="default",
        trains=False,
        reuse_of="step15_final",
        rqs=("RQ1", "RQ2", "RQ10"),
        note=(
            "The model the study actually ships, trained on the recipe Step 6 selected. "
            "Reuses the validated Step 15 checkpoints and the Step 16 test result; "
            "nothing is retrained. A7 against P is the end-to-end preprocessing "
            "comparison Step 6's proxy caveat asks for."
        ),
    ),
)

#: Row lookup by identifier.
_BY_ID: Dict[str, AblationRow] = {row.row_id: row for row in ROWS}


def get_row(row_id: str) -> AblationRow:
    """:param row_id: Row identifier.

    :return: That row.
    :raises KeyError: If the identifier is not in the matrix.
    """
    if row_id not in _BY_ID:
        raise KeyError(f"Unknown ablation row {row_id!r}; expected one of {sorted(_BY_ID)}")
    return _BY_ID[row_id]


def resolve_recipe(row: AblationRow, context: AblationContext) -> str:
    """:param row: The row.

    :param context: Resolved Step 6 / Step 14 selections.
    :return: The concrete recipe name this row trains on.
    """
    if row.recipe == DIFFUSION:
        return context.diffusion_recipe
    if row.recipe == SELECTED:
        return context.selected_recipe
    return str(row.recipe)


def resolve_loss(row: AblationRow, context: AblationContext) -> str:
    """:param row: The row.

    :param context: Resolved Step 6 / Step 14 selections.
    :return: The concrete loss config name this row trains with.
    """
    return context.step14_loss if row.loss == STEP14_LOSS else row.loss


def row_overrides(row: AblationRow, seed: int, context: AblationContext) -> List[str]:
    """The Hydra overrides that run one row at one seed.

    Every setting the row claims is emitted explicitly. Nothing is left to the pipeline's
    ``recipe_override()`` or ``imbalance_overrides()``, which inject Step 6's and Step 8's
    selections and would silently relabel the row.

    :param row: The row to run.
    :param seed: One of :data:`PROTOCOL_SEEDS`.
    :param context: Resolved Step 6 / Step 14 selections.
    :return: Override strings for ``src/train.py``.
    :raises ValueError: If the row trains nothing.
    """
    if not row.trains:
        raise ValueError(
            f"Row {row.row_id} trains nothing: it reports {row.reuse_of}. {row.note}"
        )

    overrides = [
        f"experiment={row.experiment}",
        f"model={row.model}",
        f"seed={seed}",
        f"loss@model.criterion={resolve_loss(row, context)}",
        f"data.use_weighted_sampler={str(row.use_weighted_sampler).lower()}",
    ]

    if row.feature_tag is not None:
        overrides.append(f"data.tag={row.feature_tag}")
    else:
        recipe = resolve_recipe(row, context)
        overrides += [
            f"data.recipe={'null' if recipe in _IDENTITY_RECIPES else recipe}",
            f"data.normalize={row.normalize}",
            f"data.augment={str(row.augment).lower()}",
        ]

    return [*overrides, *row.extra]
