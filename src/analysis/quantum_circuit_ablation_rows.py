"""Step 25: the quantum-circuit adaptivity ladder, defined once as data.

Step 12 builds a branch that runs **five** quantum circuits on every image and combines
their outputs with a learned per-image softmax. Nothing in the study has ever tested
whether that mixture beats simply committing to one circuit - Phase 8's H2 compares
``branch_adaptive_quantum`` against ``baseline_fixed_qcnn``, which is a different
architecture with a different backbone and a fourteenth of the parameters, so it cannot
isolate the mixture.

Step 25 asks that question directly:

    fixed basic -> fixed deep -> fixed strong -> adaptive mixture

**No new architecture is introduced.** All four conditions are
:class:`~src.models.components.quantum.AdaptiveQuantumClassifier` with different
``circuit_names``. Setting it to a single circuit makes the softmax mathematically inert -
a softmax over one element is identically 1.0 - so the same class expresses both the
mixture and its controls, and the two cannot drift apart.

Three things this file exists to prevent.

**Inherited settings.** ``configs/experiment/step12_adaptive_quantum.yaml`` declares
``augment: true`` and ``use_weighted_sampler: true``, and ``scripts/kaggle_pipeline.py``
then overrides both with whatever Step 8 selected. Two conditions run under different
selections are not comparable for a reason unrelated to circuits. Every condition here pins
recipe, normalization, augmentation, sampler and loss explicitly.

**A capacity claim that is not true.** The conditions are *not* parameter-matched: the
adaptive one carries five circuits' weights. The gap is small - at most 356 parameters,
0.49% - but it runs in the adaptive model's favour and must be reported rather than
assumed away.

**Overstating what is tested.** The Step 11 spatial gate is hard-coded inside the Step 12
branch and is 77.65% of the parameters. It is held identical across all four conditions,
so the comparison is valid - but the claim it licenses is about *circuit-mixture
adaptivity only*, never about the model as a whole.

Terminology is deliberate. All five circuits execute on every forward pass; none is
skipped and no gate is chosen at run time. This is an **adaptive soft mixture of quantum
circuits**, not dynamic circuit selection and not conditional quantum execution.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.analysis.ablation_rows import PROTOCOL_SEEDS

#: Step 15's seeds, reused rather than redeclared - Step 25 is not a new seed policy.
SEEDS: Tuple[int, ...] = PROTOCOL_SEEDS

#: Settings held identical across every condition, so the only thing that varies is which
#: circuits the branch contains. Pinned as literals and emitted explicitly, never inherited
#: from Step 6's or Step 8's running selections.
PINNED: Dict[str, Any] = {
    "normalize": "imagenet",
    "augment": False,
    "use_weighted_sampler": False,
    "loss": "plain_ce",
    "channels": 32,
    "n_qubits": 4,
    "num_classes": 4,
    "hidden_dims": (128, 64),
    "dropout": 0.3,
}

#: Recipes reading the raw tree directly; a confirmation selecting one becomes
#: ``data.recipe=null``. Mirrors ``IDENTITY_RECIPES`` in the preprocessing module.
_IDENTITY_RECIPES = ("raw", "conventional")

#: The circuits the adaptive branch mixes over when ``circuit_names`` is unset. Mirrors
#: ``CIRCUIT_NAMES`` in the quantum module; a test asserts they agree.
ADAPTIVE_CIRCUITS: Tuple[str, ...] = ("fixed", "deep", "strong", "combined", "reupload")

#: The primary family: three comparisons, so Holm genuinely corrects here (unlike Step 24's
#: family of one).
FAMILY_SIZE = 3

#: Selection and reporting happen on VALIDATION. Step 25 is an architecture question, like
#: Steps 6, 8, 13 and 14 - the internal test set stays sealed for Step 16.
EVALUATION_SPLIT = "val"


@dataclass(frozen=True)
class QuantumCondition:
    """One rung of the ladder.

    :param condition_id: Stable identifier used in outputs and run directories.
    :param circuit_names: Circuits the branch contains. ``None`` keeps Step 12's default
        of all five; a single-element list makes the mixture inert.
    :param circuit_description: What the condition's circuit is, in words.
    :param mixture: How the circuit outputs are combined.
    :param adaptive: Whether the combination is input-dependent.
    :param note: Why the condition is in the ladder.
    """

    condition_id: str
    circuit_names: Optional[Tuple[str, ...]]
    circuit_description: str
    mixture: str
    adaptive: bool
    note: str = ""

    @property
    def n_circuits(self) -> int:
        """:return: How many circuits the branch executes per image."""
        return len(self.circuit_names) if self.circuit_names else len(ADAPTIVE_CIRCUITS)


#: The ladder, in the order it is reported.
CONDITIONS: Tuple[QuantumCondition, ...] = (
    QuantumCondition(
        condition_id="FIXED_BASIC",
        circuit_names=("fixed",),
        circuit_description="2 BasicEntangler layers (8 quantum parameters)",
        mixture="none - a single circuit; the softmax is identically 1.0",
        adaptive=False,
        note=(
            "The traditional fixed circuit, and the same circuit Step 9's Baseline 5 uses - "
            "but here inside the Step 12 architecture, so it is actually comparable."
        ),
    ),
    QuantumCondition(
        condition_id="FIXED_DEEP",
        circuit_names=("deep",),
        circuit_description="4 BasicEntangler layers (16 quantum parameters)",
        mixture="none - a single circuit; the softmax is identically 1.0",
        adaptive=False,
        note="Same entanglement pattern as FIXED_BASIC at twice the depth.",
    ),
    QuantumCondition(
        condition_id="FIXED_STRONG",
        circuit_names=("strong",),
        circuit_description="2 StronglyEntangling layers (24 quantum parameters)",
        mixture="none - a single circuit; the softmax is identically 1.0",
        adaptive=False,
        note="Same depth as FIXED_BASIC with a richer entanglement pattern.",
    ),
    QuantumCondition(
        condition_id="ADAPTIVE_QUANTUM",
        circuit_names=None,
        circuit_description=(
            "all five circuits - fixed, deep, strong, combined, reupload "
            "(104 quantum parameters)"
        ),
        mixture=(
            "adaptive soft mixture: a learned per-image softmax over the five circuits' "
            "outputs, conditioned on the classical features"
        ),
        adaptive=True,
        note=(
            "Step 12 exactly as it exists. All five circuits execute on every image; none "
            "is skipped and no gate is chosen at run time."
        ),
    ),
)

#: The primary family. Each compares the mixture against one fixed circuit, holding the
#: spatial branch, projection, fusion, classifier and protocol identical.
PRIMARY_COMPARISONS: Tuple[Dict[str, str], ...] = (
    {
        "id": "H25a_vs_basic",
        "condition_a": "ADAPTIVE_QUANTUM",
        "condition_b": "FIXED_BASIC",
        "question": "Does the mixture beat a single shallow BasicEntangler circuit?",
    },
    {
        "id": "H25b_vs_deep",
        "condition_a": "ADAPTIVE_QUANTUM",
        "condition_b": "FIXED_DEEP",
        "question": "Does the mixture beat a single deeper BasicEntangler circuit?",
    },
    {
        "id": "H25c_vs_strong",
        "condition_a": "ADAPTIVE_QUANTUM",
        "condition_b": "FIXED_STRONG",
        "question": "Does the mixture beat a single StronglyEntangling circuit?",
    },
)

#: Descriptive comparisons among the fixed circuits. They answer whether depth or
#: entanglement alone explains any difference, which is a separate question from whether
#: mixing helps - so they carry no significance claim.
SECONDARY_COMPARISONS: Tuple[Dict[str, str], ...] = (
    {
        "id": "S25a_deep_vs_basic",
        "condition_a": "FIXED_DEEP",
        "condition_b": "FIXED_BASIC",
        "note": "Circuit depth alone, at a fixed entanglement pattern. Descriptive.",
    },
    {
        "id": "S25b_strong_vs_basic",
        "condition_a": "FIXED_STRONG",
        "condition_b": "FIXED_BASIC",
        "note": "Entanglement richness alone, at a fixed depth. Descriptive.",
    },
    {
        "id": "S25c_strong_vs_deep",
        "condition_a": "FIXED_STRONG",
        "condition_b": "FIXED_DEEP",
        "note": "Richer entanglement against greater depth. Descriptive.",
    },
)

_BY_ID: Dict[str, QuantumCondition] = {c.condition_id: c for c in CONDITIONS}


@dataclass(frozen=True)
class QuantumAblationContext:
    """The preprocessing choice every condition shares.

    Held in a context rather than written into each condition so the ladder cannot end up
    with two recipes in it.

    :param recipe: The materialised recipe name, or ``None`` for the raw tree.
    :param source: Where the recipe came from, carried into the output so a reader can tell
        a confirmed decision from a development override.
    """

    recipe: Optional[str]
    source: str = "explicit"

    @classmethod
    def from_confirmation(cls, summary_path: Optional[str]) -> "QuantumAblationContext":
        """Resolve the recipe from Step 6's authoritative confirmation.

        Step 25 follows the same preprocessing policy as the Step 12 experiment it
        ablates, and does not select a recipe of its own. There is deliberately **no
        fallback to the proxy ranking**: twelve training runs on an unconfirmed
        preprocessing would answer a question about a configuration the study does not
        ship.

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


def get_condition(condition_id: str) -> QuantumCondition:
    """:param condition_id: Condition identifier.

    :return: That condition.
    :raises KeyError: If the identifier is not in the ladder.
    """
    if condition_id not in _BY_ID:
        raise KeyError(f"Unknown condition {condition_id!r}; expected one of {sorted(_BY_ID)}")
    return _BY_ID[condition_id]


def circuit_override(condition: QuantumCondition) -> str:
    """The one override that distinguishes a condition.

    :param condition: The condition.
    :return: A Hydra override setting ``model.net.circuit_names``.
    """
    if condition.circuit_names is None:
        return "model.net.circuit_names=null"
    return "model.net.circuit_names=[" + ",".join(condition.circuit_names) + "]"


def condition_overrides(
    condition: QuantumCondition, seed: int, context: QuantumAblationContext
) -> List[str]:
    """The Hydra overrides that train one condition at one seed.

    Every controlled setting is emitted explicitly. Nothing is left to the pipeline's
    ``recipe_override()`` or ``imbalance_overrides()``, which inject Step 6's and Step 8's
    selections and would make two conditions incomparable for a reason that has nothing to
    do with quantum circuits.

    :param condition: The condition to train.
    :param seed: One of :data:`SEEDS`.
    :param context: The shared preprocessing choice.
    :return: Override strings for ``src/train.py``.
    """
    recipe = context.recipe or "null"

    return [
        "experiment=step25_quantum_circuit_ablation",
        circuit_override(condition),
        f"seed={seed}",
        f"loss@model.criterion={PINNED['loss']}",
        f"data.recipe={recipe}",
        f"data.normalize={PINNED['normalize']}",
        f"data.augment={str(PINNED['augment']).lower()}",
        f"data.use_weighted_sampler={str(PINNED['use_weighted_sampler']).lower()}",
    ]


def build_model(condition: QuantumCondition) -> Any:
    """Instantiate a condition's network from the existing Step 12 class.

    One construction path for every condition, so a control cannot quietly acquire a
    different backbone, projection, fusion or head.

    :param condition: The condition.
    :return: The instantiated ``AdaptiveQuantumClassifier``.
    """
    from src.models.components.quantum import AdaptiveQuantumClassifier

    return AdaptiveQuantumClassifier(
        channels=PINNED["channels"],
        num_classes=PINNED["num_classes"],
        n_qubits=PINNED["n_qubits"],
        circuit_names=list(condition.circuit_names) if condition.circuit_names else None,
        hidden_dims=list(PINNED["hidden_dims"]),
        dropout=PINNED["dropout"],
    )


def condition_parameters(condition: QuantumCondition) -> Dict[str, int]:
    """Count a condition's parameters by building it.

    Measured rather than tabulated: a written-down number would survive the architecture
    changing underneath it, and the whole point of recording capacity here is that the
    ladder is *not* parameter-matched.

    :param condition: The condition.
    :return: Per-group counts plus the total.
    """
    model = build_model(condition)

    groups = {"spatial_branch": 0, "reduce": 0, "quantum_experts": 0, "selector": 0,
              "classifier": 0}
    for name, parameter in model.named_parameters():
        if "spatial_branch" in name:
            key = "spatial_branch"
        elif "experts" in name:
            key = "quantum_experts"
        elif "selector" in name:
            key = "selector"
        elif name.startswith("classifier"):
            key = "classifier"
        else:
            key = "reduce"
        groups[key] += parameter.numel()

    groups["total"] = int(sum(p.numel() for p in model.parameters()))
    groups["feature_dim"] = int(model.feature_dim)
    groups["n_circuits"] = len(model.branch.experts)
    return groups
