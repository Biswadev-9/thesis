"""Step 24: evaluate the receptive-field strategy ladder and test the gating hypothesis.

    "Does spatially adaptive selection among multiple receptive fields improve MRI
     brain-tumour classification compared with conventional fixed-receptive-field
     convolution and ungated multi-scale feature extraction?"

The stage evaluates five conditions - three single fixed kernels, one ungated multi-scale
model and the proposed adaptive one - and answers that question with **one** formal
hypothesis and three descriptive comparisons. Four decisions shape it.

**One formal hypothesis, in its own family.** H24 is ADAPTIVE_MULTISCALE against
MULTISCALE_NO_GATE, because those two carry the identical three receptive fields and differ
only in whether a gate selects among them. The three fixed-kernel comparisons change the
receptive field *and* the parameter budget together, so they cannot isolate gating and are
reported descriptively. This family is separate from Phase 8's registered H1-H4: appending
to that family would turn four Holm-corrected tests into five and silently weaken every one
of them.

**Holm over a family of one is the identity.** It is still applied and still reported, with
a note saying so, because an "adjusted p-value" that silently equals the raw one is worse
than one that explains why.

**Capacity is measured, not assumed.** The conditions are not parameter-matched. What makes
H24 defensible is that the *control* is the larger model: on the shipped configuration the
ungated arm carries more parameters than the adaptive one, so a win for adaptivity cannot
be attributed to extra capacity. That relationship is computed and recorded here rather
than asserted in prose.

**The test set informs nothing.** Checkpoints come from each run's own ``val/f1_macro``
callback and predictions are recomputed here; the ``test/*`` columns ``trainer.test()``
writes into every run's ``metrics.csv`` are never read.

Nothing in this module can make the adaptive condition win. The verdict is derived from the
predictions, and a negative or null result is a valid outcome of the experiment.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.analysis import metric_battery as battery
from src.analysis.base import Analysis
from src.analysis.receptive_field_rows import (
    CONDITIONS,
    FAMILY_SIZE,
    PINNED,
    PRIMARY_COMPARISON,
    SEEDS,
    SUPPORTING_COMPARISONS,
    ReceptiveFieldCondition,
    ReceptiveFieldContext,
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

#: Condition order, fixed so the ladder always reads bottom to top.
CONDITION_IDS: Tuple[str, ...] = tuple(c.condition_id for c in CONDITIONS)

#: The study's established primary metric, as Steps 21 and 23 use. Not accuracy.
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
    "receptive_field_strategy",
    "fusion",
    "adaptive",
    "arm",
    "seed",
    "total_parameters",
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


class ReceptiveFieldStudy(Analysis):
    """Evaluate the five receptive-field conditions and test H24.

    :param name: Analysis identifier.
    :param run_root: Directory holding the training runs, laid out as
        ``<run_root>/<condition>/seed_<n>``.
    :param recipe: Explicit preprocessing recipe, overriding the confirmation. Mirrors
        the pipeline's ``--recipe`` flag; use it only deliberately.
    :param confirmation_summary: Path to Step 6's ``step06_confirm_summary.json``. When no
        explicit recipe is given this decides the recipe, and its absence stops the stage
        rather than letting an unconfirmed preprocessing through.
    :param seeds: Protocol seeds to look for.
    :param alpha: Significance level for the single formal hypothesis.
    :param confidence_level: Interval level, as a fraction.
    :param n_resamples: Bootstrap resamples; 2000 matches the project convention.
    :param random_seed: Seed for every resampling procedure.
    :param paired_seed: Which training seed the paired test reads.
    :param n_calibration_bins: Bins for the ECE estimate.
    :param accelerator: ``auto``, ``cpu`` or ``gpu``.
    :param strict: Fail when any condition's checkpoints are incomplete.
    """

    def __init__(
        self,
        name: str = "step24_receptive_field",
        run_root: Optional[str] = None,
        recipe: Optional[str] = None,
        confirmation_summary: Optional[str] = None,
        seeds: Sequence[int] = SEEDS,
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
        # Resolved lazily: an explicit recipe is an operator override, otherwise the
        # authoritative Step 6 confirmation decides. There is no proxy fallback.
        self._explicit_recipe = recipe
        self._recipe_source = "explicit analysis.recipe override" if recipe else None
        self.seeds = tuple(int(s) for s in seeds)
        self.alpha = alpha
        self.confidence_level = confidence_level
        self.n_resamples = n_resamples
        self.random_seed = random_seed
        self.paired_seed = paired_seed
        self.n_calibration_bins = n_calibration_bins
        self.accelerator = accelerator
        self.strict = strict
        self._predictions: Dict[Tuple[str, int], Dict[str, Any]] = {}

    # ------------------------------------------------------------------- recipe

    @property
    def recipe(self) -> Optional[str]:
        """The one preprocessing recipe every condition shares.

        Resolved from Step 6's authoritative confirmation unless an explicit override was
        given. **There is no fallback to the proxy ranking**: Step 24 spends fifteen
        training runs deciding whether adaptive gating helps, and running them on an
        unconfirmed preprocessing would make the answer apply to a configuration the study
        does not ship.

        :return: The recipe, or ``None`` for the raw tree.
        :raises ConfirmationIncomplete: If no confirmation and no override exist.
        """
        return self._context().recipe

    def _context(self) -> ReceptiveFieldContext:
        """:return: The shared preprocessing context.

        :raises ConfirmationIncomplete: If no authoritative decision is available.
        """
        if self._explicit_recipe is not None:
            return ReceptiveFieldContext(
                recipe=self._explicit_recipe, source=self._recipe_source or "explicit"
            )
        return ReceptiveFieldContext.from_confirmation(self.confirmation_summary)

    # -------------------------------------------------------------------- compute

    def compute(self, datamodule: Any = None) -> Dict[str, Any]:
        """Evaluate every condition, then run the one formal and three descriptive tests.

        :param datamodule: Unused; each condition supplies its own data.
        :return: The Step 24 summary.
        :raises FileNotFoundError: If a condition needed by H24 has no checkpoints.
        """
        self._predictions.clear()
        # Resolved first: if Step 6 has not been confirmed, stop here rather than after
        # loading five conditions' checkpoints.
        context = self._context()

        conditions = {
            condition.condition_id: self._evaluate_condition(condition)
            for condition in CONDITIONS
        }
        self._require_primary_sides(conditions)

        capacity = self._capacity()
        primary = self._primary_comparison(capacity)
        supporting = [self._supporting_comparison(spec) for spec in SUPPORTING_COMPARISONS]

        table = self._flat_table(conditions)
        self.save_table(table, "step24_receptive_field.csv")

        summary = {
            "question": (
                "Does spatially adaptive selection among multiple receptive fields improve "
                "classification over fixed-receptive-field convolution and over ungated "
                "multi-scale fusion?"
            ),
            "parameters": {
                "primary_metric": PRIMARY_METRIC,
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
                "pinned": dict(PINNED),
            },
            "matrix": self._matrix(capacity),
            "conditions": conditions,
            "capacity": capacity,
            "primary": primary,
            "supporting": supporting,
            "integrity": self._integrity(conditions),
            "notes": self._notes(capacity),
        }
        self._log(summary)
        return summary

    # ------------------------------------------------------------- per condition

    def run_dir(self, condition: ReceptiveFieldCondition, seed: int) -> Path:
        """:param condition: The condition.

        :param seed: Protocol seed.
        :return: Where that condition's run for that seed lives.
        """
        root = Path(self.run_root or "logs/train/runs/step24_receptive_field")
        return root / condition.condition_id / f"seed_{seed}"

    def build_datamodule(self, condition: ReceptiveFieldCondition) -> Any:
        """Instantiate the datamodule a condition is evaluated on.

        Identical for every condition by construction - the ladder varies receptive
        fields, and nothing else may vary with them.

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
        self, condition: ReceptiveFieldCondition, seed: int, checkpoint: Path
    ) -> Dict[str, Any]:
        """Run one condition's checkpoint over the internal test split.

        :param condition: The condition.
        :param seed: Protocol seed.
        :param checkpoint: The validation-selected checkpoint.
        :return: ``{"y_true", "y_pred", "y_prob", "class_names"}``.
        """
        from src.models.full_pipeline import predict

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
                "_target_": "src.models.components.multiscale.MultiscaleClassifier.from_arm",
                "arm": condition.arm,
                "num_classes": 4,
            },
            "num_classes": 4,
        }
        module = load_module(checkpoint, model_cfg=model_cfg).to(device)

        outputs = predict(module, datamodule.test_dataloader(), device)
        outputs["class_names"] = list(datamodule.class_names)
        return outputs

    def _evaluate_condition(self, condition: ReceptiveFieldCondition) -> Dict[str, Any]:
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

            metrics = battery.full_battery(
                outputs["y_true"], outputs["y_pred"], outputs["y_prob"],
                outputs["class_names"], self.n_calibration_bins,
            )
            metrics["checkpoint"] = str(checkpoint)
            metrics["selection"] = "val/f1_macro (training-time ModelCheckpoint)"
            metrics["provenance"] = (
                "computed by step24 from the validation-selected checkpoint; no training "
                "metrics were read"
            )
            per_seed[str(seed)] = metrics

            np.savez_compressed(
                self.output_dir
                / f"step24_predictions_{condition.condition_id}_seed{seed}.npz",
                y_true=outputs["y_true"],
                y_pred=outputs["y_pred"],
                y_prob=outputs["y_prob"],
            )

        return {
            "condition": condition.condition_id,
            "arm": condition.arm,
            "receptive_field_strategy": condition.receptive_field_strategy,
            "fusion": condition.fusion,
            "adaptive": condition.adaptive,
            "note": condition.note,
            "per_seed": per_seed,
            "across_seeds": self._across_seeds(per_seed),
            "missing_seeds": missing,
            "evaluated": bool(per_seed),
        }

    def _require_primary_sides(self, conditions: Dict[str, Any]) -> None:
        """Refuse to report H24 with one side missing.

        :param conditions: Evaluated condition records.
        :raises FileNotFoundError: If either side of H24 was not evaluated.
        """
        for key in ("condition_a", "condition_b"):
            name = PRIMARY_COMPARISON[key]
            if not conditions.get(name, {}).get("per_seed"):
                raise FileNotFoundError(
                    f"H24 compares {PRIMARY_COMPARISON['condition_a']} against "
                    f"{PRIMARY_COMPARISON['condition_b']}, and {name} has no evaluated "
                    "checkpoints. Train it before running Step 24; a one-sided family "
                    "would be reported as though it were whole."
                )

    # ---------------------------------------------------------------- comparisons

    def _aligned(self, name_a: str, name_b: str, label: str) -> Tuple[Dict, Dict]:
        """Fetch two conditions' predictions at the paired seed and verify alignment.

        A paired test over misaligned samples returns a confident wrong answer rather than
        an error, so the label vectors are compared element-wise.

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
                f"{label}: no predictions for {error.args[0]} - it was not evaluated at "
                f"seed {self.paired_seed}."
            ) from error

        if a["y_true"].shape != b["y_true"].shape or not np.array_equal(
            a["y_true"], b["y_true"]
        ):
            raise ValueError(
                f"{label}: predictions for {name_a} and {name_b} at seed "
                f"{self.paired_seed} are not aligned - their label vectors differ, so the "
                "samples are not in the same order. A paired test on these would be "
                "meaningless."
            )
        return a, b

    def _primary_comparison(self, capacity: Dict[str, Any]) -> Dict[str, Any]:
        """The one formal hypothesis: adaptive gating against ungated fusion.

        :param capacity: Parameter counts, for the fairness note.
        :return: The comparison record.
        """
        name_a = PRIMARY_COMPARISON["condition_a"]
        name_b = PRIMARY_COMPARISON["condition_b"]
        a, b = self._aligned(name_a, name_b, PRIMARY_COMPARISON["id"])
        y_true = a["y_true"]

        bootstrap = paired_bootstrap(
            y_true, a["y_pred"], b["y_pred"], metric_fn=_macro_f1,
            n_resamples=self.n_resamples,
            confidence=self.confidence_level * 100.0, seed=self.random_seed,
        )
        mcnemar = mcnemar_test(y_true, a["y_pred"], b["y_pred"])

        adjusted = holm_bonferroni(
            {PRIMARY_COMPARISON["id"]: mcnemar["p_value"]}, alpha=self.alpha
        )[PRIMARY_COMPARISON["id"]]

        larger = max(capacity, key=lambda k: capacity[k]["total_parameters"])
        capacity_note = (
            f"{name_b} carries "
            f"{capacity[name_b]['total_parameters'] - capacity[name_a]['total_parameters']:+,} "
            f"parameters relative to {name_a}. The control is the "
            f"{'larger' if capacity[name_b]['total_parameters'] > capacity[name_a]['total_parameters'] else 'smaller'}"
            " of the two, which is what determines whether a win here could be explained "
            "by capacity rather than by gating."
        )

        return {
            "comparison": PRIMARY_COMPARISON["id"],
            "family": "primary",
            "statistical_status": "formal",
            "rq": PRIMARY_COMPARISON["rq"],
            "question": PRIMARY_COMPARISON["question"],
            "condition_a": name_a,
            "condition_b": name_b,
            "metric": PRIMARY_METRIC,
            "observed_delta": bootstrap["observed_delta"],
            "ci_low": bootstrap["ci_lower"],
            "ci_high": bootstrap["ci_upper"],
            "confidence_level": self.confidence_level,
            "n_test_samples": int(len(y_true)),
            "paired_seed": self.paired_seed,
            "effect_direction": _direction(bootstrap["observed_delta"], name_a, name_b),
            "mcnemar": mcnemar,
            "mcnemar_method": mcnemar["test"],
            "mcnemar_p_value": mcnemar["p_value"],
            "raw_p_value": mcnemar["p_value"],
            "p_value_source": "mcnemar",
            "adjusted_p_value": adjusted["adjusted_p_value"],
            "significant": adjusted["significant"],
            "correction": "holm-bonferroni",
            "family_size": FAMILY_SIZE,
            "correction_note": (
                "Holm over a family of one leaves the p-value unchanged; the adjusted "
                "value is reported so it cannot be mistaken for a correction that "
                "happened. This family is separate from Phase 8's registered H1-H4."
            ),
            "interpretation": (
                "The p-value tests discordant errors (McNemar); the effect size and "
                "interval are macro-F1 from the paired bootstrap. They answer different "
                "questions and should be quoted together."
            ),
            "capacity_note": capacity_note,
            "largest_condition": larger,
            "condition_a_ci": self._condition_ci(name_a),
            "condition_b_ci": self._condition_ci(name_b),
            "seeds": self._seed_spread(name_a, name_b),
            "provenance": self._provenance(name_a, name_b),
        }

    def _supporting_comparison(self, spec: Dict[str, str]) -> Dict[str, Any]:
        """A descriptive comparison: effect size and interval, no significance claim.

        :param spec: Comparison specification.
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

        return {
            "comparison": spec["id"],
            "family": "supporting",
            "statistical_status": "descriptive",
            "condition_a": name_a,
            "condition_b": name_b,
            "metric": PRIMARY_METRIC,
            "observed_delta": bootstrap["observed_delta"],
            "ci_low": bootstrap["ci_lower"],
            "ci_high": bootstrap["ci_upper"],
            "confidence_level": self.confidence_level,
            "n_test_samples": int(len(y_true)),
            "paired_seed": self.paired_seed,
            "effect_direction": _direction(bootstrap["observed_delta"], name_a, name_b),
            "mcnemar": None,
            "mcnemar_method": None,
            "mcnemar_p_value": None,
            "raw_p_value": None,
            "p_value_source": None,
            "adjusted_p_value": None,
            # Not False: there is no claim, rather than a claim of no effect.
            "significant": None,
            "correction": None,
            "note": spec["note"],
            "seeds": self._seed_spread(name_a, name_b),
            "provenance": self._provenance(name_a, name_b),
        }

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

        Three seeds cannot support a significance claim: a two-sided Wilcoxon over three
        pairs has a floor of p=0.25. The powered paired test is the bootstrap over test
        samples.

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

    def _provenance(self, name_a: str, name_b: str) -> Dict[str, str]:
        """:param name_a: First condition.

        :param name_b: Second condition.
        :return: Where the compared numbers came from.
        """
        return {
            "source_stage": "step24_receptive_field",
            "condition_a_predictions": (
                f"step24_predictions_{name_a}_seed{self.paired_seed}.npz"
            ),
            "condition_b_predictions": (
                f"step24_predictions_{name_b}_seed{self.paired_seed}.npz"
            ),
            "run_root": str(self.run_root),
            "note": (
                "Predictions were computed by this stage from validation-selected "
                "checkpoints. No training metrics.csv was read."
            ),
        }

    # ------------------------------------------------------------------ capacity

    def _capacity(self) -> Dict[str, Any]:
        """Parameter counts for every condition, measured by building the arms.

        The ladder is deliberately not parameter-matched, and pretending otherwise would
        misstate what the comparisons can support.

        :return: Per-condition capacity records.
        """
        counts = {c.condition_id: condition_parameters(c) for c in CONDITIONS}
        reference = counts[PRIMARY_COMPARISON["condition_a"]]

        return {
            condition_id: {
                "total_parameters": total,
                "delta_vs_adaptive": total - reference,
                "ratio_vs_adaptive": round(total / reference, 4) if reference else None,
            }
            for condition_id, total in counts.items()
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
        summary["per_class_recall"] = {
            name: battery.summarise_seeds(
                [_class_metric(per_seed[s], name, "recall_sensitivity") for s in ordered]
            )
            for name in classes
        }
        summary["per_class_precision"] = {
            name: battery.summarise_seeds(
                [_class_metric(per_seed[s], name, "precision") for s in ordered]
            )
            for name in classes
        }
        summary["per_class_f1"] = {
            name: battery.summarise_seeds(
                [_class_metric(per_seed[s], name, "f1") for s in ordered]
            )
            for name in classes
        }
        return summary

    def _matrix(self, capacity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The ladder as a table, independent of whether anything was evaluated.

        :param capacity: Parameter counts.
        :return: One record per condition.
        """
        primary_ids = {PRIMARY_COMPARISON["condition_a"], PRIMARY_COMPARISON["condition_b"]}
        return [
            {
                "condition": c.condition_id,
                "receptive_field_strategy": c.receptive_field_strategy,
                "fusion": c.fusion,
                "adaptive": c.adaptive,
                "arm": c.arm,
                "total_parameters": capacity[c.condition_id]["total_parameters"],
                "delta_vs_adaptive": capacity[c.condition_id]["delta_vs_adaptive"],
                "statistical_role": (
                    "primary (H24)" if c.condition_id in primary_ids else "supporting"
                ),
                "note": c.note,
            }
            for c in CONDITIONS
        ]

    def _flat_table(self, conditions: Dict[str, Any]) -> pd.DataFrame:
        """One line per (condition, seed), with the schema pinned.

        :param conditions: Evaluated condition records.
        :return: The table.
        """
        capacity = self._capacity()
        primary_ids = {PRIMARY_COMPARISON["condition_a"], PRIMARY_COMPARISON["condition_b"]}
        records: List[Dict[str, Any]] = []

        for condition in CONDITIONS:
            record = conditions.get(condition.condition_id, {})
            for seed in sorted(record.get("per_seed", {}), key=int):
                metrics = record["per_seed"][seed]
                overall = metrics.get("overall", {})
                worst = _worst_class(metrics)
                records.append(
                    {
                        "condition": condition.condition_id,
                        "receptive_field_strategy": condition.receptive_field_strategy,
                        "fusion": condition.fusion,
                        "adaptive": condition.adaptive,
                        "arm": condition.arm,
                        "seed": int(seed),
                        "total_parameters": capacity[condition.condition_id][
                            "total_parameters"
                        ],
                        "recipe": self.recipe,
                        "loss": PINNED["loss"],
                        "augment": PINNED["augment"],
                        "use_weighted_sampler": PINNED["use_weighted_sampler"],
                        "n_samples": metrics.get("n_samples"),
                        **{m: overall.get(m) for m in TABLE_METRICS},
                        "min_class_recall": worst[1],
                        "worst_class": worst[0],
                        "statistical_role": (
                            "primary (H24)"
                            if condition.condition_id in primary_ids
                            else "supporting"
                        ),
                        "checkpoint": metrics.get("checkpoint"),
                        "provenance": metrics.get("provenance"),
                    }
                )

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
            "evaluated_conditions": [
                n for n, r in conditions.items() if r.get("evaluated")
            ],
            "incomplete_conditions": incomplete,
            "complete": (
                not incomplete
                and all(r.get("evaluated") for r in conditions.values())
            ),
        }

    def _notes(self, capacity: Dict[str, Any]) -> Dict[str, str]:
        """:param capacity: Parameter counts.

        :return: Standing caveats a reader needs before quoting the table.
        """
        adaptive = PRIMARY_COMPARISON["condition_a"]
        ungated = PRIMARY_COMPARISON["condition_b"]
        return {
            "terminology": (
                "The convolutions are fixed in every condition. What adapts in "
                f"{adaptive} is the per-pixel softmax weighting over the three paths' "
                "outputs, computed from the input on every forward pass. This is "
                "spatially adaptive receptive-field selection, not a dynamic kernel and "
                "not query-key attention."
            ),
            "ungated_control": (
                f"{ungated} fuses by concatenation followed by a learned 1x1 projection. "
                "The mixer is learned but input-independent once trained; it is not "
                "equal-weight averaging and must not be described as such."
            ),
            "capacity": (
                "The five conditions are NOT parameter-matched. The fixed conditions are "
                "smaller because a single path is smaller, which is why their comparisons "
                "are descriptive. For the formal hypothesis the relationship runs the "
                f"other way: {ungated} has "
                f"{capacity[ungated]['total_parameters']:,} parameters against "
                f"{capacity[adaptive]['total_parameters']:,} for {adaptive}."
            ),
            "family": (
                "One formal hypothesis, in its own family of one, entirely separate from "
                "Phase 8's registered H1-H4. The three fixed-kernel comparisons are "
                "descriptive and carry no p-value."
            ),
            "test_set": (
                "Every metric was computed by this stage from a checkpoint the training "
                "run selected on val/f1_macro. No test metric influenced any checkpoint "
                "or condition choice, and no training metrics.csv was read."
            ),
            "outcome": (
                "The verdict is derived from the predictions. A result showing the "
                "adaptive condition equal to or worse than the ungated control is a valid "
                "outcome of this experiment, not a failure of it."
            ),
        }

    @staticmethod
    def _log(summary: Dict[str, Any]) -> None:
        """:param summary: The computed summary."""
        log.info("=== Step 24: receptive-field strategy ladder ===")
        log.info(f"  {'condition':<22} {'params':>9}  macro-F1 (mean +- std)")

        for record in summary["conditions"].values():
            macro = (record.get("across_seeds", {}).get("macro_f1") or {})
            params = summary["capacity"][record["condition"]]["total_parameters"]
            if macro.get("mean") is None:
                log.info(f"  {record['condition']:<22} {params:>9,}  not evaluated")
                continue
            log.info(
                f"  {record['condition']:<22} {params:>9,}  "
                f"{macro['mean']:.4f} +- {macro['std'] or 0.0:.4f} (n={macro['n']})"
            )

        primary = summary["primary"]
        raw = primary["raw_p_value"]
        log.info(
            f"  H24 {primary['condition_a']} vs {primary['condition_b']}: "
            f"delta macro-F1 {primary['observed_delta']:+.4f} "
            f"[{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}]  "
            f"p={'n/a' if raw is None else f'{raw:.4f}'}  "
            f"{'SIGNIFICANT' if primary['significant'] else 'not significant'}"
        )
        log.info(f"  {primary['capacity_note']}")

        for comparison in summary["supporting"]:
            log.info(
                f"  {comparison['comparison']} (descriptive): delta "
                f"{comparison['observed_delta']:+.4f} "
                f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]"
            )

        if not summary["integrity"]["complete"]:
            log.warning(
                f"Ladder is INCOMPLETE: {summary['integrity']['incomplete_conditions']}"
            )


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
