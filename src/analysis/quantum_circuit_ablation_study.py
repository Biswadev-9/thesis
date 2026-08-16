"""Step 25: does an adaptive mixture of quantum circuits beat one fixed circuit?

    "Does selecting/weighting multiple quantum circuit architectures adaptively for each
     input image improve performance compared with using a single fixed quantum circuit?"

Four conditions - three single fixed circuits and Step 12's five-circuit mixture - all
built from the same class with different ``circuit_names``. Five decisions shape the
analysis.

**Validation, not test.** This is an architecture question, like Steps 6, 8, 13 and 14, so
it is decided on the validation split and the internal test set stays sealed for Step 16.
The evaluation split is asserted to be ``val``, and a test enforces it. Every training run
also writes ``test/*`` into its ``metrics.csv``; none of it is read.

**Three primary comparisons, Holm-corrected as one family.** The mixture against each fixed
circuit. Unlike Step 24's family of one, Holm genuinely bites here: three comparisons at
alpha=0.05 give roughly a one-in-seven chance of a spurious positive uncorrected.

**Fixed-versus-fixed stays descriptive.** Whether depth or entanglement alone explains a
difference is a separate question from whether *mixing* helps, and answering it with
p-values would inflate the family from three to six.

**Capacity is measured, and the gap runs the wrong way.** The conditions are not
parameter-matched: the adaptive model carries five circuits' weights, at most 356
parameters more (0.49%). Small, but in the adaptive model's favour - so it is reported
rather than assumed away, and a positive result must be read with it in view.

**Adaptivity is analysed, not assumed.** The claimed contribution is the per-image
weighting, so the mixture weights are summarised directly: their spread, their entropy,
whether they collapse onto one circuit, and whether they vary between images at all. A
selector that has collapsed to a constant is a fixed circuit with extra parameters, and the
report says so - but weights that *do* vary are not evidence of usefulness either. Only the
paired comparison answers that.

Nothing here can make the adaptive condition win. A null or negative result is a valid
outcome, and given that the quantum experts are 0.14% of the model's parameters, it is a
likely one worth reporting plainly.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.analysis import metric_battery as battery
from src.analysis.base import Analysis
from src.analysis.quantum_circuit_ablation_rows import (
    ADAPTIVE_CIRCUITS,
    CONDITIONS,
    EVALUATION_SPLIT,
    FAMILY_SIZE,
    PINNED,
    PRIMARY_COMPARISONS,
    SECONDARY_COMPARISONS,
    SEEDS,
    QuantumAblationContext,
    QuantumCondition,
    condition_parameters,
)
from src.utils import RankedLogger
from src.utils.checkpoints import find_checkpoint, load_module
from src.utils.statistics import (
    MIN_WILCOXON_PAIRS,
    bootstrap_ci,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap,
    summarise_across_seeds,
    wilcoxon_paired,
)

log = RankedLogger(__name__, rank_zero_only=True)

#: Condition order, fixed so the ladder always reads the same way.
CONDITION_IDS: Tuple[str, ...] = tuple(c.condition_id for c in CONDITIONS)

#: The study's primary metric, on the validation split.
PRIMARY_METRIC = "macro_f1"

#: Overall metrics lifted into the flat table.
TABLE_METRICS = (
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall_sensitivity",
    "weighted_f1",
    "mcc",
)

#: Flat-table columns, pinned so the schema does not follow dict iteration order.
TABLE_COLUMNS: Tuple[str, ...] = (
    "condition",
    "circuit_names",
    "n_circuits",
    "circuit_description",
    "mixture",
    "adaptive",
    "seed",
    "split",
    "total_parameters",
    "quantum_parameters",
    "selector_parameters",
    "spatial_parameters",
    "classifier_parameters",
    "recipe",
    "loss",
    "augment",
    "use_weighted_sampler",
    "n_samples",
    *TABLE_METRICS,
    "min_class_recall",
    "worst_class",
    "statistical_role",
    "checkpoint",
    "provenance",
)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """:param y_true: True labels.

    :param y_pred: Predictions.
    :return: Macro-averaged F1.
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


class QuantumCircuitAblationStudy(Analysis):
    """Evaluate the four circuit conditions on validation and test the mixture hypothesis.

    :param name: Analysis identifier.
    :param run_root: Directory holding the training runs, as
        ``<run_root>/<condition>/seed_<n>``.
    :param recipe: Explicit preprocessing recipe, overriding the confirmation. Mirrors the
        pipeline's ``--recipe`` flag; a development override, used deliberately.
    :param confirmation_summary: Path to Step 6's ``step06_confirm_summary.json``. When no
        explicit recipe is given this decides the recipe, and its absence stops the stage.
    :param seeds: Protocol seeds to look for.
    :param split: Evaluation split. Must be ``val``: the test set stays sealed for Step 16.
    :param alpha: Family-wise error rate for the Holm correction.
    :param confidence_level: Interval level, as a fraction.
    :param n_resamples: Bootstrap resamples; 2000 matches the project convention.
    :param random_seed: Seed for every resampling procedure.
    :param paired_seed: Which training seed the paired tests read.
    :param n_calibration_bins: Bins for the ECE estimate.
    :param accelerator: ``auto``, ``cpu`` or ``gpu``.
    :param strict: Fail when any condition's checkpoints are incomplete.
    """

    def __init__(
        self,
        name: str = "step25_quantum_circuit_ablation",
        run_root: Optional[str] = None,
        recipe: Optional[str] = None,
        confirmation_summary: Optional[str] = None,
        seeds: Sequence[int] = SEEDS,
        split: str = EVALUATION_SPLIT,
        alpha: float = 0.05,
        confidence_level: float = 0.95,
        n_resamples: int = 2000,
        random_seed: int = 42,
        paired_seed: int = 42,
        n_calibration_bins: int = 10,
        accelerator: str = "auto",
        strict: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.run_root = run_root
        self.confirmation_summary = confirmation_summary
        self._explicit_recipe = recipe
        self._recipe_source = "explicit analysis.recipe override" if recipe else None
        self.seeds = tuple(int(s) for s in seeds)
        self.split = split
        self.alpha = alpha
        self.confidence_level = confidence_level
        self.n_resamples = n_resamples
        self.random_seed = random_seed
        self.paired_seed = paired_seed
        self.n_calibration_bins = n_calibration_bins
        self.accelerator = accelerator
        self.strict = strict
        self._predictions: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._weights: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------- recipe

    @property
    def recipe(self) -> Optional[str]:
        """The one preprocessing recipe every condition shares.

        Resolved from Step 6's authoritative confirmation unless an explicit override was
        given. Step 25 follows the Step 12 experiment's preprocessing policy rather than
        selecting one of its own, and there is no fallback to the proxy ranking.

        :return: The recipe, or ``None`` for the raw tree.
        :raises ConfirmationIncomplete: If no confirmation and no override exist.
        """
        return self._context().recipe

    def _context(self) -> QuantumAblationContext:
        """:return: The shared preprocessing context.

        :raises ConfirmationIncomplete: If no authoritative decision is available.
        """
        if self._explicit_recipe is not None:
            return QuantumAblationContext(
                recipe=self._explicit_recipe, source=self._recipe_source or "explicit"
            )
        return QuantumAblationContext.from_confirmation(self.confirmation_summary)

    # -------------------------------------------------------------------- compute

    def compute(self, datamodule: Any = None) -> Dict[str, Any]:
        """Evaluate every condition, then run the corrected family and the descriptives.

        :param datamodule: Unused; each condition supplies its own data.
        :return: The Step 25 summary.
        :raises ValueError: If the evaluation split is not validation.
        :raises FileNotFoundError: If a condition needed by the family has no checkpoints.
        """
        if self.split != EVALUATION_SPLIT:
            raise ValueError(
                f"Step 25 selects among architectures, so it evaluates on "
                f"{EVALUATION_SPLIT!r} only; {self.split!r} was requested. The internal "
                "test set stays sealed until Step 16."
            )

        self._predictions.clear()
        self._weights.clear()
        # Resolved first: if Step 6 has not been confirmed, stop before loading anything.
        context = self._context()

        conditions = {
            condition.condition_id: self._evaluate_condition(condition)
            for condition in CONDITIONS
        }
        self._require_family_sides(conditions)

        capacity = self._capacity()
        primary = [self._compare(spec, formal=True) for spec in PRIMARY_COMPARISONS]
        self._apply_holm(primary)
        secondary = [self._compare(spec, formal=False) for spec in SECONDARY_COMPARISONS]

        table = self._flat_table(conditions, capacity)
        self.save_table(table, "step25_quantum_circuit_ablation.csv")

        summary = {
            "question": (
                "Does adaptively weighting multiple quantum circuit architectures per "
                "image improve performance over a single fixed quantum circuit?"
            ),
            "parameters": {
                "primary_metric": PRIMARY_METRIC,
                "evaluation_split": self.split,
                "alpha": self.alpha,
                "confidence_level": self.confidence_level,
                "n_resamples": self.n_resamples,
                "random_seed": self.random_seed,
                "paired_seed": self.paired_seed,
                "seeds": list(self.seeds),
                "correction": "holm-bonferroni",
                "family_size": FAMILY_SIZE,
                "recipe": context.recipe,
                "recipe_source": context.source,
                "parameter_matched": False,
                "pinned": {k: v for k, v in PINNED.items()},
            },
            "matrix": self._matrix(capacity),
            "conditions": conditions,
            "capacity": capacity,
            "primary": primary,
            "secondary": secondary,
            "adaptive_behaviour": self._adaptive_behaviour(),
            "cost": self._cost(),
            "integrity": self._integrity(conditions),
            "notes": self._notes(capacity),
        }
        self._log(summary)
        return summary

    # ------------------------------------------------------------- per condition

    def run_dir(self, condition: QuantumCondition, seed: int) -> Path:
        """:param condition: The condition.

        :param seed: Protocol seed.
        :return: Where that condition's run for that seed lives.
        """
        root = Path(self.run_root or "logs/train/runs/step25_quantum_circuit_ablation")
        return root / condition.condition_id / f"seed_{seed}"

    def build_datamodule(self, condition: QuantumCondition) -> Any:
        """Instantiate the datamodule a condition is evaluated on.

        Identical for every condition by construction - the ladder varies circuits, and
        nothing else may vary with them.

        :param condition: The condition.
        :return: An un-setup datamodule.
        """
        from src.data.bt_mri_datamodule import BTMRIDataModule

        return BTMRIDataModule(
            recipe=self.recipe,
            normalize=PINNED["normalize"],
            augment=False,  # never on an evaluation split
            use_weighted_sampler=False,
        )

    def predict_condition(
        self, condition: QuantumCondition, seed: int, checkpoint: Path
    ) -> Dict[str, Any]:
        """Run one condition's checkpoint over the validation split.

        Also returns the per-image mixture weights, which are the evidence for the
        adaptivity analysis and are produced by the same forward pass.

        :param condition: The condition.
        :param seed: Protocol seed.
        :param checkpoint: The validation-selected checkpoint.
        :return: ``{"y_true", "y_pred", "y_prob", "class_names", "quantum_weights"}``.
        """
        device = torch.device(
            "cuda" if (self.accelerator != "cpu" and torch.cuda.is_available()) else "cpu"
        )
        datamodule = self.build_datamodule(condition)
        datamodule.prepare_data()
        datamodule.setup()

        model_cfg = {
            "_target_": "src.models.mri_classification_module.MRIClassificationModule",
            "optimizer": {"_target_": "torch.optim.AdamW", "_partial_": True, "lr": 1e-4},
            "net": {
                "_target_": "src.models.components.quantum.AdaptiveQuantumClassifier",
                "channels": PINNED["channels"],
                "num_classes": PINNED["num_classes"],
                "n_qubits": PINNED["n_qubits"],
                "circuit_names": (
                    list(condition.circuit_names) if condition.circuit_names else None
                ),
                "hidden_dims": list(PINNED["hidden_dims"]),
                "dropout": PINNED["dropout"],
            },
            "num_classes": PINNED["num_classes"],
        }
        module = load_module(checkpoint, model_cfg=model_cfg).to(device).eval()

        loader = datamodule.val_dataloader()
        y_true, y_pred, y_prob, weights = [], [], [], []
        with torch.no_grad():
            for images, labels in loader:
                outputs = module.net.extract(images.to(device))
                probabilities = torch.softmax(outputs["logits"], dim=1)
                y_pred.append(outputs["logits"].argmax(dim=1).cpu().numpy())
                y_prob.append(probabilities.cpu().numpy())
                y_true.append(labels.numpy())
                weights.append(outputs["quantum_weights"].cpu().numpy())

        return {
            "y_true": np.concatenate(y_true),
            "y_pred": np.concatenate(y_pred),
            "y_prob": np.concatenate(y_prob),
            "quantum_weights": np.concatenate(weights),
            "class_names": list(datamodule.class_names),
        }

    def _evaluate_condition(self, condition: QuantumCondition) -> Dict[str, Any]:
        """Evaluate one condition at each protocol seed.

        :param condition: The condition.
        :return: Its record.
        :raises FileNotFoundError: In strict mode, if a seed's checkpoint is absent.
        """
        per_seed: Dict[str, Any] = {}
        missing: List[int] = []

        for seed in self.seeds:
            try:
                checkpoint = find_checkpoint(self.run_dir(condition, seed), prefer="best")
            except FileNotFoundError as error:
                if self.strict:
                    raise
                log.warning(f"{condition.condition_id} seed {seed}: {error}")
                missing.append(seed)
                continue

            outputs = self.predict_condition(condition, seed, checkpoint)
            self._predictions[(condition.condition_id, seed)] = outputs
            if condition.adaptive and outputs.get("quantum_weights") is not None:
                self._weights[seed] = np.asarray(outputs["quantum_weights"])

            metrics = battery.full_battery(
                outputs["y_true"], outputs["y_pred"], outputs["y_prob"],
                outputs["class_names"], self.n_calibration_bins,
            )
            metrics["checkpoint"] = str(checkpoint)
            metrics["split"] = self.split
            metrics["selection"] = "val/f1_macro (training-time ModelCheckpoint)"
            metrics["provenance"] = (
                f"computed by step25 on the {self.split} split from the "
                "validation-selected checkpoint; no test metric was read"
            )
            per_seed[str(seed)] = metrics

            np.savez_compressed(
                self.output_dir
                / f"step25_predictions_{condition.condition_id}_seed{seed}.npz",
                y_true=outputs["y_true"],
                y_pred=outputs["y_pred"],
                y_prob=outputs["y_prob"],
                quantum_weights=outputs["quantum_weights"],
            )

        return {
            "condition": condition.condition_id,
            "circuit_names": (
                list(condition.circuit_names) if condition.circuit_names
                else list(ADAPTIVE_CIRCUITS)
            ),
            "n_circuits": condition.n_circuits,
            "circuit_description": condition.circuit_description,
            "mixture": condition.mixture,
            "adaptive": condition.adaptive,
            "note": condition.note,
            "per_seed": per_seed,
            "across_seeds": self._across_seeds(per_seed),
            "missing_seeds": missing,
            "evaluated": bool(per_seed),
        }

    def _require_family_sides(self, conditions: Dict[str, Any]) -> None:
        """Refuse to report a comparison with one side missing.

        :param conditions: Evaluated condition records.
        :raises FileNotFoundError: If either side of a primary comparison is absent.
        """
        for spec in PRIMARY_COMPARISONS:
            for key in ("condition_a", "condition_b"):
                name = spec[key]
                if not conditions.get(name, {}).get("per_seed"):
                    raise FileNotFoundError(
                        f"{spec['id']} compares {spec['condition_a']} against "
                        f"{spec['condition_b']}, and {name} has no evaluated checkpoints. "
                        "A family reported one member short would still be Holm-corrected "
                        "as though it were whole."
                    )

    # ---------------------------------------------------------------- comparisons

    def _aligned(self, name_a: str, name_b: str, label: str) -> Tuple[Dict, Dict]:
        """Fetch two conditions' predictions at the paired seed and verify alignment.

        :param name_a: First condition.
        :param name_b: Second condition.
        :param label: Comparison identifier, for the message.
        :return: The two prediction sets.
        :raises FileNotFoundError: If either side is missing at the paired seed.
        :raises ValueError: If the label vectors differ.
        """
        try:
            a = self._predictions[(name_a, self.paired_seed)]
            b = self._predictions[(name_b, self.paired_seed)]
        except KeyError as error:
            raise FileNotFoundError(
                f"{label}: no predictions for {error.args[0]} at seed {self.paired_seed}."
            ) from error

        if a["y_true"].shape != b["y_true"].shape or not np.array_equal(
            a["y_true"], b["y_true"]
        ):
            raise ValueError(
                f"{label}: predictions for {name_a} and {name_b} at seed "
                f"{self.paired_seed} are not aligned - their label vectors differ, so a "
                "paired test on them would be meaningless."
            )
        return a, b

    def _compare(self, spec: Dict[str, str], formal: bool) -> Dict[str, Any]:
        """Compare two conditions on the paired seed.

        :param spec: Comparison specification.
        :param formal: Whether this is a member of the corrected family.
        :return: The comparison record.
        """
        name_a, name_b = spec["condition_a"], spec["condition_b"]
        a, b = self._aligned(name_a, name_b, spec["id"])
        y_true = a["y_true"]

        bootstrap = paired_bootstrap(
            y_true, a["y_pred"], b["y_pred"], metric_fn=_macro_f1,
            n_resamples=self.n_resamples,
            confidence=self.confidence_level * 100.0, seed=self.random_seed,
        )
        mcnemar = mcnemar_test(y_true, a["y_pred"], b["y_pred"]) if formal else None

        record: Dict[str, Any] = {
            "comparison": spec["id"],
            "family": "primary" if formal else "secondary",
            "statistical_status": "formal" if formal else "descriptive",
            "question": spec.get("question"),
            "condition_a": name_a,
            "condition_b": name_b,
            "metric": PRIMARY_METRIC,
            "split": self.split,
            "observed_delta": bootstrap["observed_delta"],
            "ci_low": bootstrap["ci_lower"],
            "ci_high": bootstrap["ci_upper"],
            "confidence_level": self.confidence_level,
            "n_samples": int(len(y_true)),
            "paired_seed": self.paired_seed,
            "effect_direction": _direction(bootstrap["observed_delta"], name_a, name_b),
            "condition_a_ci": self._condition_ci(name_a),
            "condition_b_ci": self._condition_ci(name_b),
            "seeds": self._seed_spread(name_a, name_b),
            "provenance": {
                "source_stage": "step25_quantum_circuit_ablation",
                "condition_a_predictions": (
                    f"step25_predictions_{name_a}_seed{self.paired_seed}.npz"
                ),
                "condition_b_predictions": (
                    f"step25_predictions_{name_b}_seed{self.paired_seed}.npz"
                ),
                "split": self.split,
                "note": (
                    "Computed by this stage from validation-selected checkpoints on the "
                    "validation split. No test metric was read."
                ),
            },
        }

        if formal:
            record.update({
                "mcnemar": mcnemar,
                "mcnemar_method": mcnemar["test"],
                "mcnemar_p_value": mcnemar["p_value"],
                "raw_p_value": mcnemar["p_value"],
                "p_value_source": "mcnemar",
                "adjusted_p_value": None,  # filled by _apply_holm
                "significant": False,
                "correction": "holm-bonferroni",
                "family_size": FAMILY_SIZE,
                "interpretation": (
                    "The p-value tests discordant errors (McNemar); the effect size and "
                    "interval are macro-F1 from the paired bootstrap. They answer "
                    "different questions and should be quoted together."
                ),
            })
        else:
            record.update({
                "mcnemar": None,
                "mcnemar_method": None,
                "mcnemar_p_value": None,
                "raw_p_value": None,
                "p_value_source": None,
                "adjusted_p_value": None,
                # Not False: there is no claim, rather than a claim of no effect.
                "significant": None,
                "correction": None,
                "note": spec.get("note", "Descriptive; no significance claim."),
            })
        return record

    def _apply_holm(self, primary: List[Dict[str, Any]]) -> None:
        """Correct the family in place.

        :param primary: The primary comparison records.
        """
        adjusted = holm_bonferroni(
            {r["comparison"]: r["raw_p_value"] for r in primary}, alpha=self.alpha
        )
        for record in primary:
            entry = adjusted[record["comparison"]]
            record["adjusted_p_value"] = entry["adjusted_p_value"]
            record["holm_rank"] = entry["rank"]
            record["significant"] = entry["significant"]
            if entry.get("note"):
                record["correction_note"] = entry["note"]

    def _condition_ci(self, name: str) -> Optional[Dict[str, float]]:
        """:param name: Condition identifier.

        :return: Bootstrap interval for that condition's macro-F1 on its own.
        """
        outputs = self._predictions.get((name, self.paired_seed))
        if outputs is None:
            return None
        return bootstrap_ci(
            outputs["y_true"], outputs["y_pred"], metric_fn=_macro_f1,
            n_resamples=self.n_resamples,
            confidence=self.confidence_level * 100.0, seed=self.random_seed,
        )

    def _seed_spread(self, name_a: str, name_b: str) -> Dict[str, Any]:
        """Per-seed deltas, described rather than tested.

        :param name_a: First condition.
        :param name_b: Second condition.
        :return: Per-seed deltas and the refused Wilcoxon.
        """
        deltas: Dict[str, float] = {}
        values_a: List[float] = []
        values_b: List[float] = []
        missing: List[int] = []

        for seed in self.seeds:
            a = self._predictions.get((name_a, seed))
            b = self._predictions.get((name_b, seed))
            if a is None or b is None or not np.array_equal(a["y_true"], b["y_true"]):
                missing.append(seed)
                continue
            score_a = _macro_f1(a["y_true"], a["y_pred"])
            score_b = _macro_f1(b["y_true"], b["y_pred"])
            values_a.append(score_a)
            values_b.append(score_b)
            deltas[str(seed)] = score_a - score_b

        spread = (
            summarise_across_seeds(list(deltas.values()))
            if deltas
            else {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        )
        return {
            "role": "descriptive",
            "per_seed_delta": deltas,
            "delta_across_seeds": spread,
            "missing_seeds": missing,
            "wilcoxon": (
                wilcoxon_paired(values_a, values_b, label="training seeds")
                if values_a
                else {"n_pairs": 0, "statistic": None, "p_value": None,
                      "note": "No paired seed measurements were available."}
            ),
            "note": (
                f"Seed spread is descriptive. A two-sided Wilcoxon needs at least "
                f"{MIN_WILCOXON_PAIRS} pairs to reach p < 0.05 at any effect size."
            ),
        }

    # ------------------------------------------------------- adaptivity analysis

    def _adaptive_behaviour(self) -> Dict[str, Any]:
        """Summarise what the learned selector actually did.

        The claimed contribution is per-image weighting, so the weights are inspected
        directly. Two failure modes matter and neither shows up in accuracy alone: a
        selector that **collapsed** onto one circuit is a fixed circuit with extra
        parameters, and one that stayed **uniform** is an unweighted average.

        Weights that vary are still not evidence of usefulness - only the paired
        comparison answers that - and the note says so.

        :return: Per-circuit statistics and the diagnosis.
        """
        if not self._weights:
            return {
                "available": False,
                "note": (
                    "No mixture weights were collected; the adaptive condition was not "
                    "evaluated."
                ),
            }

        circuits = list(ADAPTIVE_CIRCUITS)
        per_seed: Dict[str, Any] = {}
        for seed, weights in sorted(self._weights.items()):
            array = np.asarray(weights, dtype=float)
            uniform = 1.0 / array.shape[1]
            # Normalised entropy: 1.0 is perfectly uniform, 0.0 is fully collapsed.
            with np.errstate(divide="ignore", invalid="ignore"):
                entropy = -np.sum(array * np.log(np.clip(array, 1e-12, None)), axis=1)
            normalised = float(np.mean(entropy) / np.log(array.shape[1]))

            per_seed[str(seed)] = {
                "n_images": int(array.shape[0]),
                "per_circuit": {
                    circuits[i]: {
                        "mean": float(array[:, i].mean()),
                        "std": float(array[:, i].std(ddof=1)) if array.shape[0] > 1 else 0.0,
                        "min": float(array[:, i].min()),
                        "max": float(array[:, i].max()),
                    }
                    for i in range(array.shape[1])
                },
                "mean_normalised_entropy": normalised,
                "max_across_image_std": float(array.std(axis=0).max()),
                "dominant_circuit": circuits[int(array.mean(axis=0).argmax())],
                "max_mean_weight": float(array.mean(axis=0).max()),
                "uniform_weight": uniform,
                "varies_between_images": bool(array.std(axis=0).max() > 1e-4),
            }

        return {
            "available": True,
            "circuits": circuits,
            "per_seed": per_seed,
            "note": (
                "Weights that differ between images show the selector is input-dependent; "
                "they do NOT show the mixture helps. Only the paired comparison answers "
                "that. A selector collapsed onto one circuit is a fixed circuit with extra "
                "parameters; one that stayed uniform is an unweighted average."
            ),
        }

    def _cost(self) -> Dict[str, Any]:
        """Circuits executed per image, which is where the mixture's cost lives.

        :return: Per-condition circuit-evaluation counts.
        """
        return {
            "circuits_executed_per_image": {
                c.condition_id: c.n_circuits for c in CONDITIONS
            },
            "note": (
                "All circuits execute on every forward pass. The adaptive condition does "
                "NOT skip circuits and performs no conditional execution, so it costs "
                "roughly five simulator evaluations per image against one for each fixed "
                "condition."
            ),
        }

    # ------------------------------------------------------------------ capacity

    def _capacity(self) -> Dict[str, Any]:
        """Parameter counts for every condition, measured by building the models.

        :return: Per-condition capacity records.
        """
        counts = {c.condition_id: condition_parameters(c) for c in CONDITIONS}
        reference = counts["ADAPTIVE_QUANTUM"]["total"]

        return {
            name: {
                **groups,
                "delta_vs_adaptive": groups["total"] - reference,
                "percent_vs_adaptive": (
                    round(100.0 * (groups["total"] - reference) / reference, 4)
                    if reference else None
                ),
            }
            for name, groups in counts.items()
        }

    # -------------------------------------------------------------- aggregation

    def _across_seeds(self, per_seed: Dict[str, Any]) -> Dict[str, Any]:
        """Mean, std and 95% interval per metric across the seeds present.

        :param per_seed: Per-seed metric records.
        :return: Across-seed summary.
        """
        if not per_seed:
            return {}

        ordered = sorted(per_seed, key=int)
        summary: Dict[str, Any] = {
            metric: battery.summarise_seeds(
                [per_seed[s].get("overall", {}).get(metric) for s in ordered]
            )
            for metric in TABLE_METRICS
        }
        classes = [e["class_name"] for e in per_seed[ordered[0]].get("per_class", [])]
        for key, field in (
            ("per_class_recall", "recall_sensitivity"),
            ("per_class_precision", "precision"),
            ("per_class_f1", "f1"),
        ):
            summary[key] = {
                name: battery.summarise_seeds(
                    [_class_metric(per_seed[s], name, field) for s in ordered]
                )
                for name in classes
            }
        return summary

    def _matrix(self, capacity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The ladder as a table, independent of whether anything was evaluated.

        :param capacity: Parameter counts.
        :return: One record per condition.
        """
        return [
            {
                "condition": c.condition_id,
                "circuit_names": (
                    list(c.circuit_names) if c.circuit_names else list(ADAPTIVE_CIRCUITS)
                ),
                "n_circuits": c.n_circuits,
                "circuit_description": c.circuit_description,
                "mixture": c.mixture,
                "adaptive": c.adaptive,
                "total_parameters": capacity[c.condition_id]["total"],
                "quantum_parameters": capacity[c.condition_id]["quantum_experts"],
                "delta_vs_adaptive": capacity[c.condition_id]["delta_vs_adaptive"],
                "statistical_role": "primary (H25)" if True else "secondary",
                "note": c.note,
            }
            for c in CONDITIONS
        ]

    def _flat_table(
        self, conditions: Dict[str, Any], capacity: Dict[str, Any]
    ) -> pd.DataFrame:
        """One line per (condition, seed), with the schema pinned.

        :param conditions: Evaluated condition records.
        :param capacity: Parameter counts.
        :return: The table.
        """
        try:
            recipe = self.recipe
        except Exception:  # pragma: no cover - table is written after resolution
            recipe = None

        records: List[Dict[str, Any]] = []
        for condition in CONDITIONS:
            record = conditions.get(condition.condition_id, {})
            counts = capacity[condition.condition_id]
            for seed in sorted(record.get("per_seed", {}), key=int):
                metrics = record["per_seed"][seed]
                overall = metrics.get("overall", {})
                worst = _worst_class(metrics)
                records.append({
                    "condition": condition.condition_id,
                    "circuit_names": ",".join(
                        condition.circuit_names or ADAPTIVE_CIRCUITS
                    ),
                    "n_circuits": condition.n_circuits,
                    "circuit_description": condition.circuit_description,
                    "mixture": condition.mixture,
                    "adaptive": condition.adaptive,
                    "seed": int(seed),
                    "split": self.split,
                    "total_parameters": counts["total"],
                    "quantum_parameters": counts["quantum_experts"],
                    "selector_parameters": counts["selector"],
                    "spatial_parameters": counts["spatial_branch"],
                    "classifier_parameters": counts["classifier"],
                    "recipe": recipe,
                    "loss": PINNED["loss"],
                    "augment": PINNED["augment"],
                    "use_weighted_sampler": PINNED["use_weighted_sampler"],
                    "n_samples": metrics.get("n_samples"),
                    **{m: overall.get(m) for m in TABLE_METRICS},
                    "min_class_recall": worst[1],
                    "worst_class": worst[0],
                    "statistical_role": (
                        "adaptive (H25 subject)" if condition.adaptive else "fixed control"
                    ),
                    "checkpoint": metrics.get("checkpoint"),
                    "provenance": metrics.get("provenance"),
                })

        return pd.DataFrame(records, columns=list(TABLE_COLUMNS))

    def _integrity(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """What is incomplete, so a partial ladder cannot pass for a full one.

        :param conditions: Evaluated condition records.
        :return: Integrity record.
        """
        incomplete = {
            name: record["missing_seeds"]
            for name, record in conditions.items()
            if record.get("missing_seeds")
        }
        return {
            "expected_conditions": list(CONDITION_IDS),
            "evaluated_conditions": [n for n, r in conditions.items() if r.get("evaluated")],
            "incomplete_conditions": incomplete,
            "complete": (
                not incomplete and all(r.get("evaluated") for r in conditions.values())
            ),
        }

    def _notes(self, capacity: Dict[str, Any]) -> Dict[str, str]:
        """:param capacity: Parameter counts.

        :return: Standing caveats a reader needs before quoting the table.
        """
        adaptive = capacity["ADAPTIVE_QUANTUM"]
        return {
            "terminology": (
                "All circuits execute on every image. The adaptive condition applies a "
                "learned per-image softmax over their outputs. This is an adaptive soft "
                "mixture of quantum circuits - not dynamic circuit selection, not "
                "conditional quantum execution, and not attention."
            ),
            "scope": (
                "The Step 11 spatial gate is inside the Step 12 branch and is held "
                f"identical across all four conditions ({adaptive['spatial_branch']:,} "
                "parameters, the large majority of the model). This experiment therefore "
                "measures circuit-mixture adaptivity ONLY. It does not show that the model "
                "as a whole is more powerful because of adaptivity, and must not be "
                "reported as though it did."
            ),
            "capacity": (
                "The conditions are NOT parameter-matched. The adaptive condition carries "
                f"{adaptive['quantum_experts']} quantum parameters against "
                f"{capacity['FIXED_BASIC']['quantum_experts']}-"
                f"{capacity['FIXED_STRONG']['quantum_experts']} for the fixed ones, and is "
                "the larger model overall by at most 356 parameters (0.49%). The gap runs "
                "in the adaptive condition's favour and is reported rather than assumed "
                "away."
            ),
            "quantum_share": (
                "Quantum parameters are a very small fraction of the model. A null result "
                "here is unsurprising and remains a valid, reportable outcome."
            ),
            "split": (
                f"Everything is computed on the {self.split} split. This is an architecture "
                "question, like Steps 6, 8, 13 and 14, so the internal test set stays "
                "sealed for Step 16 and no test metric was read."
            ),
            "family": (
                "Three primary comparisons - the mixture against each fixed circuit - form "
                "one Holm-corrected family. The fixed-versus-fixed comparisons are "
                "descriptive and carry no p-value; promoting them would double the family."
            ),
        }

    @staticmethod
    def _log(summary: Dict[str, Any]) -> None:
        """:param summary: The computed summary."""
        parameters = summary["parameters"]
        log.info("=== Step 25: quantum circuit adaptivity ===")
        log.info(
            f"  split={parameters['evaluation_split']}  metric={parameters['primary_metric']}  "
            f"recipe={parameters['recipe']}  family={parameters['family_size']}"
        )
        for record in summary["conditions"].values():
            macro = (record.get("across_seeds", {}).get("macro_f1") or {})
            counts = summary["capacity"][record["condition"]]
            if macro.get("mean") is None:
                log.info(f"  {record['condition']:<18} not evaluated")
                continue
            log.info(
                f"  {record['condition']:<18} circuits={record['n_circuits']} "
                f"params={counts['total']:,} (q={counts['quantum_experts']})  "
                f"macro-F1 {macro['mean']:.4f} +- {macro['std'] or 0.0:.4f}"
            )

        for record in summary["primary"]:
            raw, adj = record["raw_p_value"], record["adjusted_p_value"]
            log.info(
                f"  {record['comparison']}: delta {record['observed_delta']:+.4f} "
                f"[{record['ci_low']:+.4f}, {record['ci_high']:+.4f}]  "
                f"p_raw={'n/a' if raw is None else f'{raw:.4f}'} "
                f"p_holm={'n/a' if adj is None else f'{adj:.4f}'}  "
                f"{'SIGNIFICANT' if record['significant'] else 'not significant'}"
            )

        behaviour = summary["adaptive_behaviour"]
        if behaviour.get("available"):
            for seed, record in behaviour["per_seed"].items():
                log.info(
                    f"  selector seed {seed}: entropy={record['mean_normalised_entropy']:.3f} "
                    f"(1.0=uniform) dominant={record['dominant_circuit']} "
                    f"({record['max_mean_weight']:.3f})  "
                    f"varies_between_images={record['varies_between_images']}"
                )

        if not summary["integrity"]["complete"]:
            log.warning(f"Ladder INCOMPLETE: {summary['integrity']['incomplete_conditions']}")


def _class_metric(metrics: Dict[str, Any], class_name: str, key: str) -> Optional[float]:
    """:param metrics: One seed's metric record.

    :param class_name: Class to look up.
    :param key: Which per-class metric.
    :return: That value, or ``None``.
    """
    for entry in metrics.get("per_class", []):
        if entry.get("class_name") == class_name:
            return entry.get(key)
    return None


def _worst_class(metrics: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """:param metrics: One seed's metric record.

    :return: ``(class_name, recall)`` for the lowest class-wise recall.
    """
    entries = [
        e for e in metrics.get("per_class", []) if e.get("recall_sensitivity") is not None
    ]
    if not entries:
        return (None, None)
    worst = min(entries, key=lambda e: e["recall_sensitivity"])
    return (worst["class_name"], float(worst["recall_sensitivity"]))


def _direction(delta: float, name_a: str, name_b: str) -> str:
    """:param delta: Observed difference.

    :param name_a: First condition.
    :param name_b: Second condition.
    :return: Which condition the effect favours.
    """
    if delta > 0:
        return f"favours {name_a}"
    if delta < 0:
        return f"favours {name_b}"
    return "no difference"
