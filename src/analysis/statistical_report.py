"""Step 23: consolidated statistical reporting over the Step 21 ablation.

    "The final paper should include uncertainty estimates. Single-run results are weak for
     Q1-level reporting. ... Report p-values only when the experimental design supports
     paired testing. Do not overstate minor improvements unless they are statistically and
     clinically meaningful."

This stage builds nothing and trains nothing. It reads the predictions Step 21 saved and
turns them into claims - which means every way it can be wrong is a way of claiming more
than the data supports. Four decisions follow from that.

**The primary family is pre-registered and small.** Four comparisons, each isolating one
factor, declared in the config before any of them is computed:

    H1  A2 vs A1   diffusion vs conventional preprocessing   RQ2
    H2  A5 vs A4   adaptive quantum vs fixed QCNN            RQ4
    H3  A7 vs A6   imbalance-aware loss                      RQ6
    H4  A6 vs A3   quantum + fusion over multiscale alone    RQ8

Testing every row against A6 instead would be eight hypotheses chosen after seeing the
table, and Holm over a family assembled that way controls nothing. Anything outside these
four is reported descriptively, with an interval and no significance claim.

**p-values are corrected, and labelled.** Every comparison carries both its raw and its
Holm-adjusted p, and the verdict comes from the adjusted one. A p-value that does not say
which it is invites being quoted as the other.

**Pairing is verified, not assumed.** H4 pairs A6, whose predictions come through the
feature cache, with A3, whose come through the image loader. Both preserve test order
today - but a paired test over misaligned samples returns a confident wrong answer rather
than an error, so the label vectors are compared element-wise before anything is computed.

**Three seeds describe; they do not test.** A two-sided Wilcoxon over three pairs has a
floor of p=0.25 and cannot reach significance at any effect size. Seed spread is reported
as mean and standard deviation, and the powered paired test is the bootstrap over the ~1000
test samples, which is where the resolution actually is.

Row P is single-seed by Step 16's design and row A8's metrics are A7's by construction;
neither is given a significance claim it cannot support.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from src.analysis.base import Analysis
from src.utils import RankedLogger
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

#: The pre-registered primary hypothesis family. One multiplicity family, Holm-corrected.
#: Declared here as well as in the config so a config that drifts is detectable.
PRIMARY_FAMILY: Tuple[Dict[str, str], ...] = (
    {
        "id": "H1",
        "row_a": "A2",
        "row_b": "A1",
        "rq": "RQ2",
        "question": "Does diffusion preprocessing beat the conventional Step 5 pipeline?",
    },
    {
        "id": "H2",
        "row_a": "A5",
        "row_b": "A4",
        "rq": "RQ4",
        "question": "Does the adaptive quantum branch beat a fixed QCNN?",
    },
    {
        "id": "H3",
        "row_a": "A7",
        "row_b": "A6",
        "rq": "RQ6",
        "question": "Does the imbalance-aware loss Step 14 selected help?",
    },
    {
        "id": "H4",
        "row_a": "A6",
        "row_b": "A3",
        "rq": "RQ8",
        "question": "Does adding the quantum branch and fusion beat the multiscale branch alone?",
    },
)

#: Descriptive comparisons. Reported with an interval, never with a significance claim -
#: they are not members of the corrected family.
SECONDARY_COMPARISONS: Tuple[Dict[str, str], ...] = (
    {
        "id": "A1_vs_A0",
        "row_a": "A1",
        "row_b": "A0",
        "note": "Intensity treatment alone. Descriptive: not a pre-registered hypothesis.",
    },
    {
        "id": "A3_vs_A2",
        "row_a": "A3",
        "row_b": "A2",
        "note": (
            "Adaptive multiscale against the CNN. Confounded - the backbone changes as well "
            "as the module - so descriptive only."
        ),
    },
    {
        "id": "A7_vs_P",
        "row_a": "A7",
        "row_b": "P",
        "note": (
            "End-to-end preprocessing comparison: A7 is diffusion, P is what Step 6 selected. "
            "P is single-seed (Step 16 evaluated one checkpoint by design), so this carries no "
            "seed-level claim and is reported descriptively."
        ),
    },
)

#: The specification's primary effect metric. Not accuracy.
PRIMARY_METRIC = "macro_f1"

#: Flat table columns, pinned so the schema does not follow dict iteration order.
REPORT_COLUMNS: Tuple[str, ...] = (
    "comparison",
    "family",
    "rq",
    "row_a",
    "row_b",
    "metric",
    "observed_delta",
    "ci_low",
    "ci_high",
    "raw_p_value",
    "adjusted_p_value",
    "correction",
    "significant",
    "effect_direction",
    "mcnemar_method",
    "mcnemar_p_value",
    "n_discordant",
    "n_test_samples",
    "paired_seed",
    "n_seeds",
    "delta_seed_mean",
    "delta_seed_std",
    "provenance_row_a",
    "provenance_row_b",
)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """:param y_true: True labels.

    :param y_pred: Predictions.
    :return: Macro-averaged F1.
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


class StatisticalReport(Analysis):
    """Turn Step 21's saved predictions into corrected, provenance-carrying claims.

    :param name: Analysis identifier.
    :param ablation_dir: Step 21's output directory, holding its summary and the per-row
        prediction files.
    :param primary_comparisons: The pre-registered family. Must match
        :data:`PRIMARY_FAMILY`; a mismatch is refused rather than silently corrected over.
    :param secondary_comparisons: Descriptive comparisons, reported without p-values.
    :param alpha: Family-wise error rate for the Holm correction.
    :param confidence_level: Interval level, as a fraction.
    :param n_resamples: Bootstrap resamples. 2000 matches Step 20's convention.
    :param random_seed: Seed for every resampling procedure, so the report is reproducible.
    :param paired_seed: Which training seed's predictions the paired tests use. Fixed
        rather than best-of-three, matching how every other stage picks a checkpoint.
    :param minimum_wilcoxon_pairs: Below this many pairs, Wilcoxon is refused.
    :param strict: Fail if any row's seeds are incomplete, rather than reporting the gap.
    """

    def __init__(
        self,
        name: str = "step23_statistics",
        ablation_dir: Optional[str] = None,
        primary_comparisons: Optional[Sequence[Any]] = None,
        secondary_comparisons: Optional[Sequence[Any]] = None,
        alpha: float = 0.05,
        confidence_level: float = 0.95,
        n_resamples: int = 2000,
        random_seed: int = 42,
        paired_seed: int = 42,
        minimum_wilcoxon_pairs: int = MIN_WILCOXON_PAIRS,
        step16_predictions: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.ablation_dir = ablation_dir
        self.primary_comparisons = [
            dict(entry) for entry in (primary_comparisons or PRIMARY_FAMILY)
        ]
        self.secondary_comparisons = [
            dict(entry) for entry in (secondary_comparisons or SECONDARY_COMPARISONS)
        ]
        self.alpha = alpha
        self.confidence_level = confidence_level
        self.n_resamples = n_resamples
        self.random_seed = random_seed
        self.paired_seed = paired_seed
        self.minimum_wilcoxon_pairs = minimum_wilcoxon_pairs
        self.step16_predictions = step16_predictions
        self.strict = strict
        # Loading and bootstrapping dominate the runtime, and both are pure functions of
        # their inputs. Cached per instance so a seven-comparison report reads each file
        # once and bootstraps each row once.
        self._prediction_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
        self._ci_cache: Dict[Tuple[str, int], Dict[str, float]] = {}
        self._summary_cache: Optional[Dict[str, Any]] = None

    # --------------------------------------------------------------------- inputs

    def _directory(self) -> Path:
        """:return: The Step 21 output directory.

        :raises FileNotFoundError: If it is unset or absent.
        """
        if not self.ablation_dir:
            raise FileNotFoundError(
                "Step 23 reads Step 21's artefacts; analysis.ablation_dir was not set."
            )
        directory = Path(self.ablation_dir)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Step 23 needs Step 21's output directory; {directory} does not exist. "
                "Run analysis=step21_ablation first."
            )
        return directory

    def _summary(self) -> Dict[str, Any]:
        """:return: Step 21's summary.

        :raises FileNotFoundError: If the summary is absent.
        :raises ValueError: If it is malformed or carries no rows.
        """
        if self._summary_cache is not None:
            return self._summary_cache

        path = self._directory() / "step21_ablation_summary.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Step 23 needs Step 21's summary; {path} does not exist."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Step 21's summary at {path} is not valid JSON: {error}") from error

        if not payload.get("rows"):
            raise ValueError(
                f"Step 21's summary at {path} contains no rows; there is nothing to report."
            )
        self._summary_cache = payload
        return payload

    def _prediction_path(self, row_id: str, seed: int) -> Path:
        """:param row_id: Row identifier.

        :param seed: Training seed.
        :return: Where Step 21 saved that row's predictions.
        """
        return self._directory() / f"step21_predictions_{row_id}_seed{seed}.npz"

    def _predictions(self, row_id: str, seed: int) -> Dict[str, np.ndarray]:
        """Load one row's saved test predictions at one seed.

        :param row_id: Row identifier.
        :param seed: Training seed.
        :return: ``{"y_true", "y_pred", "y_prob"}``.
        :raises FileNotFoundError: If Step 21 saved no predictions for this row and seed.
        """
        cached = self._prediction_cache.get((row_id, seed))
        if cached is not None:
            return cached

        path = self._prediction_path(row_id, seed)
        if not path.is_file():
            # Row P is the known case: Step 21 reports it from Step 16's summary rather
            # than re-evaluating it, so no prediction file is written. See
            # `step16_predictions` for the opt-in that restores a paired comparison.
            raise FileNotFoundError(
                f"Step 23 needs Step 21's predictions for row {row_id} at seed {seed}; "
                f"{path} does not exist. Silently skipping it would leave the primary "
                "family a member short while still correcting as if it were complete."
            )
        payload = np.load(path)
        loaded = {key: payload[key] for key in ("y_true", "y_pred", "y_prob")}
        self._prediction_cache[(row_id, seed)] = loaded
        return loaded

    def _shipped_predictions(self) -> Optional[Dict[str, np.ndarray]]:
        """Row P's predictions, if the caller has opted into supplying them.

        Step 21 does not write a prediction file for P: it reports the shipped model from
        Step 16's summary rather than re-evaluating it, which is what keeps the once-only
        test budget intact. Step 16 *did* save its own ``test_predictions.npz``, so a
        paired A7-vs-P comparison is possible without any new evaluation - but reading it
        means consuming an artefact from outside Step 21, so it is off unless configured.

        With it unset, A7-vs-P degrades to a metrics-only descriptive comparison: a
        difference of point estimates, no interval, no p-value.

        :return: P's predictions, or ``None`` if not configured or absent.
        """
        if not self.step16_predictions:
            return None
        path = Path(self.step16_predictions)
        if not path.is_file():
            log.warning(
                f"analysis.step16_predictions={path} does not exist; A7-vs-P will be "
                "reported as a metrics-only descriptive comparison."
            )
            return None
        payload = np.load(path)
        return {key: payload[key] for key in ("y_true", "y_pred", "y_prob")}

    # -------------------------------------------------------------------- compute

    def compute(self, datamodule: Any = None) -> Dict[str, Any]:
        """Run the corrected family, the descriptive comparisons and the row summaries.

        :param datamodule: Unused; this stage reads artefacts, not data.
        :return: The statistical report.
        :raises ValueError: If the configured family is not the pre-registered one.
        """
        summary = self._summary()
        self._check_family()

        primary = [self._compare(spec, formal=True) for spec in self.primary_comparisons]
        self._apply_holm(primary)

        secondary = [
            self._compare(spec, formal=False)
            for spec in self.secondary_comparisons
            if self._rows_available(spec, summary)
        ]

        rows = {row_id: self._row_record(row_id, record)
                for row_id, record in summary["rows"].items()}

        table = self._flat_table(primary, secondary)
        self.save_table(table, "step23_comparisons.csv")

        report = {
            "parameters": {
                "alpha": self.alpha,
                "confidence_level": self.confidence_level,
                "n_resamples": self.n_resamples,
                "random_seed": self.random_seed,
                "paired_seed": self.paired_seed,
                "minimum_wilcoxon_pairs": self.minimum_wilcoxon_pairs,
                "primary_metric": PRIMARY_METRIC,
                "correction": "holm-bonferroni",
                "family_size": len(self.primary_comparisons),
            },
            "primary": primary,
            "secondary": secondary,
            "rows": rows,
            "integrity": self._integrity(primary, rows),
            "notes": self._notes(),
        }
        self._log(report)
        return report

    def _check_family(self) -> None:
        """Refuse a family that is not the pre-registered one.

        The family size scales every adjusted p-value, so adding a member after seeing the
        table changes every verdict - and the correction would then control nothing.

        :raises ValueError: If the configured family differs from :data:`PRIMARY_FAMILY`.
        """
        configured = [(c["id"], c["row_a"], c["row_b"]) for c in self.primary_comparisons]
        expected = [(c["id"], c["row_a"], c["row_b"]) for c in PRIMARY_FAMILY]

        if configured != expected:
            raise ValueError(
                "The primary family must be the pre-registered one.\n"
                f"  configured: {configured}\n"
                f"  registered: {expected}\n"
                "Adding or reordering members after seeing results invalidates the Holm "
                "correction. Report the extra comparison as secondary, or amend "
                "PRIMARY_FAMILY deliberately and say so in the write-up."
            )

    def _rows_available(self, spec: Dict[str, str], summary: Dict[str, Any]) -> bool:
        """:param spec: Comparison specification.

        :param summary: Step 21's summary.
        :return: Whether both rows were evaluated.
        """
        rows = summary.get("rows", {})
        for key in ("row_a", "row_b"):
            record = rows.get(spec[key], {})
            if not record.get("per_seed"):
                log.warning(f"{spec['id']}: row {spec[key]} was not evaluated; skipping.")
                return False
        return True

    # ----------------------------------------------------------------- comparison

    def _compare(self, spec: Dict[str, str], formal: bool) -> Dict[str, Any]:
        """Compare two rows on the paired seed, with the seed spread alongside.

        :param spec: Comparison specification.
        :param formal: Whether this comparison is a member of the corrected family. Only
            members carry p-values and significance verdicts.
        :return: The comparison record.
        """
        row_a, row_b = spec["row_a"], spec["row_b"]
        seed = self._paired_seed_for(row_a, row_b)

        a = self._predictions(row_a, seed)
        try:
            b = self._predictions(row_b, seed)
        except FileNotFoundError:
            if formal:
                # A missing member would leave the family short while Holm still corrected
                # as though it were whole, so this is fatal for a primary hypothesis.
                raise
            # Row P is the known case: Step 21 reports it from Step 16's summary rather
            # than re-evaluating it, so there is nothing to pair with. Degrade to a
            # difference of point estimates rather than fabricate an interval.
            return self._metrics_only_comparison(spec, seed)

        self._aligned_labels(a, b, spec, seed)
        return self._paired_record(spec, seed, a, b, formal=formal)

    def _paired_record(
        self,
        spec: Dict[str, str],
        seed: int,
        a: Dict[str, np.ndarray],
        b: Dict[str, np.ndarray],
        formal: bool,
        paired_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a comparison from two aligned prediction sets.

        :param spec: Comparison specification.
        :param seed: The seed both rows were read at.
        :param a: First row's predictions.
        :param b: Second row's predictions.
        :param formal: Whether this is a member of the corrected family.
        :param paired_note: Extra provenance, when the pairing came from elsewhere.
        :return: The comparison record.
        """
        row_a, row_b = spec["row_a"], spec["row_b"]
        y_true = a["y_true"]

        bootstrap = paired_bootstrap(
            y_true,
            a["y_pred"],
            b["y_pred"],
            metric_fn=_macro_f1,
            n_resamples=self.n_resamples,
            confidence=self.confidence_level * 100.0,
            seed=self.random_seed,
        )
        mcnemar = mcnemar_test(y_true, a["y_pred"], b["y_pred"]) if formal else None
        seeds = self._seed_spread(row_a, row_b)

        record: Dict[str, Any] = {
            "comparison": spec["id"],
            "family": "primary" if formal else "secondary",
            "rq": spec.get("rq"),
            "question": spec.get("question"),
            "row_a": row_a,
            "row_b": row_b,
            "metric": PRIMARY_METRIC,
            "observed_delta": bootstrap["observed_delta"],
            "ci_low": bootstrap["ci_lower"],
            "ci_high": bootstrap["ci_upper"],
            "confidence_level": self.confidence_level,
            "n_test_samples": int(len(y_true)),
            "paired_seed": seed,
            "effect_direction": _direction(bootstrap["observed_delta"], row_a, row_b),
            "per_class_recall_delta": self._recall_delta(y_true, a["y_pred"], b["y_pred"]),
            "row_a_ci": self._row_ci(row_a, seed, y_true, a["y_pred"]),
            "row_b_ci": self._row_ci(row_b, seed, y_true, b["y_pred"]),
            "seeds": seeds,
            "provenance": {
                "source": "step21_ablation",
                "row_a_predictions": f"step21_predictions_{row_a}_seed{seed}.npz",
                "row_b_predictions": f"step21_predictions_{row_b}_seed{seed}.npz",
                "ablation_dir": str(self.ablation_dir),
                "note": paired_note
                or (
                    "Predictions were produced by Step 21 from validation-selected "
                    "checkpoints. No training metrics were read here."
                ),
            },
        }

        if formal:
            record.update(
                {
                    "mcnemar": mcnemar,
                    "mcnemar_method": mcnemar["test"],
                    "mcnemar_p_value": mcnemar["p_value"],
                    "raw_p_value": mcnemar["p_value"],
                    "p_value_source": "mcnemar",
                    "adjusted_p_value": None,  # filled by _apply_holm
                    "significant": False,
                    "correction": "holm-bonferroni",
                    "family_size": len(self.primary_comparisons),
                    "interpretation": (
                        "The p-value tests discordant errors (McNemar); the effect size and "
                        "interval are macro-F1 from the paired bootstrap. They answer "
                        "different questions and should be quoted together."
                    ),
                }
            )
        else:
            record.update(
                {
                    "mcnemar": None,
                    "mcnemar_method": None,
                    "mcnemar_p_value": None,
                    "raw_p_value": None,
                    "p_value_source": None,
                    "adjusted_p_value": None,
                    # Not False: there is no claim, rather than a claim of no effect.
                    "significant": None,
                    "correction": None,
                    "note": spec.get("note", "Descriptive comparison; no significance claim."),
                }
            )

        return record

    def _metrics_only_comparison(self, spec: Dict[str, str], seed: int) -> Dict[str, Any]:
        """A descriptive comparison for which one row has no saved predictions.

        Step 21 writes no prediction file for row P: it reports the shipped model from
        Step 16's summary rather than re-evaluating it, which is what keeps the once-only
        test budget intact. Without predictions there is nothing to resample, so the delta
        is a difference of point estimates and the interval is honestly absent - not
        imputed, not widened, not quietly dropped.

        Setting ``analysis.step16_predictions`` to Step 16's own ``test_predictions.npz``
        upgrades this to a paired descriptive comparison without any new evaluation.

        :param spec: Comparison specification.
        :param seed: The seed that was sought.
        :return: The comparison record, with nulls where an interval would be.
        """
        row_a, row_b = spec["row_a"], spec["row_b"]
        shipped = self._shipped_predictions()

        if shipped is not None:
            a = self._predictions(row_a, seed)
            if np.array_equal(a["y_true"], shipped["y_true"]):
                return self._paired_record(spec, seed, a, shipped, formal=False, paired_note=(
                    "Paired against Step 16's saved predictions for the shipped model. No "
                    "re-evaluation was performed; the once-only test budget is untouched."
                ))
            log.warning(
                f"{spec['id']}: Step 16's predictions are not aligned with {row_a}'s; "
                "falling back to a metrics-only comparison."
            )

        score_a = self._row_point_estimate(row_a, seed)
        score_b = self._row_point_estimate(row_b, seed)
        delta = None if (score_a is None or score_b is None) else score_a - score_b

        return {
            "comparison": spec["id"],
            "family": "secondary",
            "rq": spec.get("rq"),
            "question": spec.get("question"),
            "row_a": row_a,
            "row_b": row_b,
            "metric": PRIMARY_METRIC,
            "observed_delta": delta,
            "ci_low": None,
            "ci_high": None,
            "confidence_level": self.confidence_level,
            "n_test_samples": None,
            "paired_seed": seed,
            "effect_direction": "unknown" if delta is None else _direction(delta, row_a, row_b),
            "per_class_recall_delta": {},
            "row_a_ci": None,
            "row_b_ci": None,
            "seeds": {
                "role": "descriptive",
                "per_seed_delta": {},
                "delta_across_seeds": {"n": 0, "mean": None, "std": None},
                "missing_seeds": [],
                "wilcoxon": {"n_pairs": 0, "statistic": None, "p_value": None,
                             "note": "No paired predictions were available."},
                "note": "Metrics-only comparison; no seed-level pairing is possible.",
            },
            "mcnemar": None,
            "mcnemar_method": None,
            "mcnemar_p_value": None,
            "raw_p_value": None,
            "p_value_source": None,
            "adjusted_p_value": None,
            "significant": None,
            "correction": None,
            "estimation": "metrics-only",
            "note": (
                f"{spec.get('note', '')} No prediction file exists for row {row_b}: Step 21 "
                "reports it from Step 16's summary rather than re-evaluating it. The delta "
                "is a difference of point estimates and no interval is reported. Set "
                "analysis.step16_predictions to Step 16's test_predictions.npz to obtain a "
                "paired descriptive interval without any new evaluation."
            ).strip(),
            "provenance": {
                "source": "step21_ablation",
                "row_a_predictions": f"step21_predictions_{row_a}_seed{seed}.npz",
                "row_b_predictions": f"step21_ablation_summary.json::rows.{row_b}",
                "ablation_dir": str(self.ablation_dir),
                "note": (
                    "Row A from Step 21's saved predictions; row B from Step 21's summary, "
                    "which carries Step 16's reused metrics. Nothing was retrained."
                ),
            },
        }

    def _row_point_estimate(self, row_id: str, seed: int) -> Optional[float]:
        """The row's macro-F1 as Step 21 recorded it.

        :param row_id: Row identifier.
        :param seed: Preferred seed; the row's only seed is used when it has just one.
        :return: The metric, or ``None`` if absent.
        """
        record = self._summary()["rows"].get(row_id, {})
        per_seed = record.get("per_seed", {})
        if not per_seed:
            return None
        key = str(seed) if str(seed) in per_seed else sorted(per_seed, key=int)[0]
        return per_seed[key].get("overall", {}).get(PRIMARY_METRIC)

    def _paired_seed_for(self, row_a: str, row_b: str) -> int:
        """The seed both rows are compared at.

        Fixed rather than chosen, matching how every other stage picks its checkpoint. Row
        P has only the one Step 16 evaluated, so a comparison involving it uses that.

        :param row_a: First row.
        :param row_b: Second row.
        :return: The seed.
        """
        return self.paired_seed

    def _aligned_labels(
        self, a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], spec: Dict[str, str], seed: int
    ) -> np.ndarray:
        """Verify the two rows' predictions describe the same samples in the same order.

        H4 pairs a feature-space row with an image-space row. Both preserve test order
        today, so this passes - but a paired test over misaligned samples returns a
        confident wrong p-value rather than an error, which is exactly the failure that
        would survive review.

        :param a: First row's predictions.
        :param b: Second row's predictions.
        :param spec: Comparison specification, for the message.
        :param seed: The seed, for the message.
        :return: The shared label vector.
        :raises ValueError: If the label vectors differ.
        """
        if a["y_true"].shape != b["y_true"].shape or not np.array_equal(a["y_true"], b["y_true"]):
            raise ValueError(
                f"{spec['id']}: predictions for {spec['row_a']} and {spec['row_b']} at seed "
                f"{seed} are not aligned - their label vectors differ, so the samples are "
                "not in the same order. A paired test on these would be meaningless. Check "
                "that both rows were evaluated on the same test split with shuffle=False."
            )
        return a["y_true"]

    def _recall_delta(
        self, y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
    ) -> Dict[str, float]:
        """Class-wise recall difference, which Step 21 names a primary metric.

        :param y_true: True labels.
        :param pred_a: First model's predictions.
        :param pred_b: Second model's predictions.
        :return: Per-class delta, keyed by class index as a string.
        """
        labels = sorted(set(np.unique(y_true).tolist()))
        recall_a = recall_score(y_true, pred_a, average=None, labels=labels, zero_division=0)
        recall_b = recall_score(y_true, pred_b, average=None, labels=labels, zero_division=0)

        names = self._class_names(len(labels))
        return {names[index]: float(recall_a[index] - recall_b[index]) for index in range(len(labels))}

    def _class_names(self, count: int) -> List[str]:
        """Class names from Step 21's summary, falling back to indices.

        :param count: How many classes.
        :return: Names ordered by label index.
        """
        try:
            rows = self._summary()["rows"]
            for record in rows.values():
                for metrics in record.get("per_seed", {}).values():
                    names = [entry["class_name"] for entry in metrics.get("per_class", [])]
                    if len(names) == count:
                        return names
        except (FileNotFoundError, ValueError, KeyError):
            pass
        return [str(index) for index in range(count)]

    def _row_ci(
        self, row_id: str, seed: int, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """:param row_id: Row identifier, for the cache key.

        :param seed: Training seed, for the cache key.
        :param y_true: True labels.
        :param y_pred: Predictions.
        :return: Bootstrap interval for this row's macro-F1 on its own.
        """
        key = (row_id, seed)
        if key in self._ci_cache:
            return self._ci_cache[key]

        self._ci_cache[key] = bootstrap_ci(
            y_true,
            y_pred,
            metric_fn=_macro_f1,
            n_resamples=self.n_resamples,
            confidence=self.confidence_level * 100.0,
            seed=self.random_seed,
        )
        return self._ci_cache[key]

    def _seed_spread(self, row_a: str, row_b: str) -> Dict[str, Any]:
        """Per-seed deltas, described rather than tested.

        Step 23 permits Wilcoxon "for repeated fold/seed comparisons", but three seeds
        cannot reach significance at any effect size, so the refusal is recorded in place
        of a p-value.

        :param row_a: First row.
        :param row_b: Second row.
        :return: Per-seed deltas, their summary, and the refused Wilcoxon.
        """
        summary = self._summary()
        seeds = [int(s) for s in summary.get("seeds", [])] or [self.paired_seed]

        deltas: Dict[str, float] = {}
        values_a: List[float] = []
        values_b: List[float] = []
        missing: List[int] = []

        for seed in seeds:
            try:
                a = self._predictions(row_a, seed)
                b = self._predictions(row_b, seed)
            except FileNotFoundError:
                # Recorded, never dropped: a two-seed spread presented as three overstates.
                missing.append(seed)
                continue
            if not np.array_equal(a["y_true"], b["y_true"]):
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
            "wilcoxon": wilcoxon_paired(values_a, values_b, label="training seeds")
            if values_a
            else {"n_pairs": 0, "statistic": None, "p_value": None,
                  "note": "No paired seed measurements were available."},
            "note": (
                f"Seed spread is descriptive. A two-sided Wilcoxon needs at least "
                f"{self.minimum_wilcoxon_pairs} pairs to reach p < 0.05 at any effect size; "
                "the powered paired test here is the bootstrap over test samples."
            ),
        }

    # --------------------------------------------------------------- corrections

    def _apply_holm(self, primary: List[Dict[str, Any]]) -> None:
        """Correct the family in place.

        :param primary: The primary comparison records.
        """
        adjusted = holm_bonferroni(
            {record["comparison"]: record["raw_p_value"] for record in primary},
            alpha=self.alpha,
        )
        for record in primary:
            entry = adjusted[record["comparison"]]
            record["adjusted_p_value"] = entry["adjusted_p_value"]
            record["holm_rank"] = entry["rank"]
            record["significant"] = entry["significant"]
            if entry.get("note"):
                record["correction_note"] = entry["note"]

    # ------------------------------------------------------------------ per row

    def _row_record(self, row_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Summarise one ablation row's seed coverage and testability.

        :param row_id: Row identifier.
        :param record: Step 21's record for the row.
        :return: The row summary.
        """
        per_seed = record.get("per_seed", {})
        scores = [
            metrics.get("overall", {}).get(PRIMARY_METRIC)
            for metrics in per_seed.values()
        ]
        clean = [value for value in scores if value is not None]

        summary: Dict[str, Any] = {
            "row_id": row_id,
            "n_seeds": len(per_seed),
            "missing_seeds": record.get("missing_seeds", []),
            "single_seed": len(per_seed) == 1,
            # No spread is invented from one point: None means "not estimable", not zero.
            "seed_spread": summarise_across_seeds(clean) if len(clean) > 1 else None,
            "provenance": {
                "source": "step21_ablation",
                "per_seed": {
                    seed: metrics.get("provenance") for seed, metrics in per_seed.items()
                },
            },
            "excluded_from_testing": False,
            "note": "",
        }

        if record.get("mirrors"):
            summary["mirrors"] = record["mirrors"]
            summary["excluded_from_testing"] = True
            summary["note"] = (
                f"Metrics are {record['mirrors']}'s by construction - explanations change no "
                "weights - so no classification comparison is meaningful. Its statistical "
                "contribution is the explainability and uncertainty output of Step 19."
            )
        elif summary["single_seed"]:
            summary["excluded_from_testing"] = True
            summary["note"] = (
                "Single-seed: Step 16 evaluated one checkpoint by design. Its spread is not "
                "estimable and is not directly comparable with a three-seed row's, so "
                "comparisons involving it are descriptive."
            )

        return summary

    # ------------------------------------------------------------------- output

    def _flat_table(
        self, primary: Sequence[Dict[str, Any]], secondary: Sequence[Dict[str, Any]]
    ) -> pd.DataFrame:
        """One line per comparison, with the schema pinned.

        :param primary: Corrected comparisons.
        :param secondary: Descriptive comparisons.
        :return: The table, columns in :data:`REPORT_COLUMNS` order.
        """
        records = []
        for record in [*primary, *secondary]:
            spread = record.get("seeds", {}).get("delta_across_seeds", {})
            mcnemar = record.get("mcnemar") or {}
            records.append(
                {
                    "comparison": record["comparison"],
                    "family": record["family"],
                    "rq": record.get("rq"),
                    "row_a": record["row_a"],
                    "row_b": record["row_b"],
                    "metric": record["metric"],
                    "observed_delta": record["observed_delta"],
                    "ci_low": record["ci_low"],
                    "ci_high": record["ci_high"],
                    "raw_p_value": record["raw_p_value"],
                    "adjusted_p_value": record["adjusted_p_value"],
                    "correction": record.get("correction"),
                    "significant": record["significant"],
                    "effect_direction": record["effect_direction"],
                    "mcnemar_method": record["mcnemar_method"],
                    "mcnemar_p_value": record["mcnemar_p_value"],
                    "n_discordant": mcnemar.get("n_discordant"),
                    "n_test_samples": record["n_test_samples"],
                    "paired_seed": record["paired_seed"],
                    "n_seeds": spread.get("n"),
                    "delta_seed_mean": spread.get("mean"),
                    "delta_seed_std": spread.get("std"),
                    "provenance_row_a": record["provenance"]["row_a_predictions"],
                    "provenance_row_b": record["provenance"]["row_b_predictions"],
                }
            )

        return pd.DataFrame(records, columns=list(REPORT_COLUMNS))

    def _integrity(
        self, primary: Sequence[Dict[str, Any]], rows: Dict[str, Any]
    ) -> Dict[str, Any]:
        """What is incomplete, so a partial report cannot pass for a full one.

        :param primary: Corrected comparisons.
        :param rows: Row summaries.
        :return: Integrity record.
        """
        incomplete = {
            record["comparison"]: record["seeds"]["missing_seeds"]
            for record in primary
            if record["seeds"]["missing_seeds"]
        }
        return {
            "complete": not incomplete,
            "incomplete_comparisons": incomplete,
            "single_seed_rows": [rid for rid, r in rows.items() if r["single_seed"]],
            "rows_excluded_from_testing": [
                rid for rid, r in rows.items() if r["excluded_from_testing"]
            ],
        }

    def _notes(self) -> Dict[str, str]:
        """:return: Standing caveats a reader needs before quoting any number."""
        return {
            "family": (
                "Four pre-registered primary comparisons form one Holm-corrected family at "
                f"alpha={self.alpha}. Everything else is descriptive and carries no "
                "significance claim; treating those as hypotheses would invalidate the "
                "correction."
            ),
            "p_values": (
                "Every primary comparison reports both a raw and a Holm-adjusted p-value, "
                "and the verdict uses the adjusted one. The p-value is McNemar's, over "
                "discordant errors; the effect size and interval are macro-F1."
            ),
            "seeds": (
                "Three seeds describe variability and do not test it. The powered paired "
                "test is the bootstrap over test samples."
            ),
            "row_p": (
                "Row P is single-seed by Step 16's design. No variance is imputed for it, "
                "and A7-vs-P is descriptive."
            ),
            "row_a8": (
                "Row A8's classification metrics are A7's by construction and it receives no "
                "performance test. Its statistical contribution is Step 19's explainability "
                "and uncertainty output."
            ),
            "data": (
                "Every number here derives from predictions Step 21 saved from "
                "validation-selected checkpoints. No training metrics.csv was read, no model "
                "was retrained, and no checkpoint or split was touched."
            ),
        }

    @staticmethod
    def _log(report: Dict[str, Any]) -> None:
        """:param report: The computed report."""
        parameters = report["parameters"]
        log.info("=== Step 23: statistical report ===")
        log.info(
            f"  Holm-corrected family of {parameters['family_size']} at "
            f"alpha={parameters['alpha']}, {parameters['n_resamples']} resamples, "
            f"seed {parameters['random_seed']}"
        )

        for record in report["primary"]:
            adjusted = record["adjusted_p_value"]
            adjusted_text = "n/a" if adjusted is None else f"{adjusted:.4f}"
            raw = record["raw_p_value"]
            raw_text = "n/a" if raw is None else f"{raw:.4f}"
            verdict = "SIGNIFICANT" if record["significant"] else "not significant"
            log.info(
                f"  {record['comparison']} {record['row_a']} vs {record['row_b']}: "
                f"delta macro-F1 {_fmt(record['observed_delta'])} "
                f"{_fmt_interval(record['ci_low'], record['ci_high'])}  "
                f"p_raw={raw_text} p_holm={adjusted_text}  {verdict}"
            )

        for record in report["secondary"]:
            log.info(
                f"  {record['comparison']} (descriptive): delta "
                f"{_fmt(record['observed_delta'])} "
                f"{_fmt_interval(record['ci_low'], record['ci_high'])}  no significance claim"
            )

        if not report["integrity"]["complete"]:
            log.warning(
                f"Report is INCOMPLETE: {report['integrity']['incomplete_comparisons']}"
            )


def _fmt(value: Optional[float]) -> str:
    """:param value: A signed quantity, or ``None`` when it could not be estimated.

    :return: A fixed-width rendering that does not pretend a missing value is zero.
    """
    return "n/a" if value is None else f"{value:+.4f}"


def _fmt_interval(low: Optional[float], high: Optional[float]) -> str:
    """:param low: Lower bound, or ``None``.

    :param high: Upper bound, or ``None``.
    :return: The interval, or a marker saying none was estimable.
    """
    if low is None or high is None:
        return "[no interval]"
    return f"[{low:+.4f}, {high:+.4f}]"


def _direction(delta: float, row_a: str, row_b: str) -> str:
    """:param delta: Observed difference.

    :param row_a: First row.
    :param row_b: Second row.
    :return: Which row the effect favours.
    """
    if delta > 0:
        return f"favours {row_a}"
    if delta < 0:
        return f"favours {row_b}"
    return "no difference"
