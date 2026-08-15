"""Statistical tests for paired model comparison (Steps 20 and 23).

Step 20 asks for "statistical testing across folds or seeds ... paired bootstrap, Wilcoxon
signed-rank test, McNemar test, or confidence intervals", and Step 23 requires uncertainty
estimates on every headline metric.

The tests here are **paired** wherever the design allows it. Two models evaluated on the
same test set are not independent samples: they see identical images and make correlated
errors, so an unpaired comparison throws away exactly the information that makes small
differences detectable.

One lesson is carried over from the reference notebook. It attempted a Wilcoxon signed-rank
test over **four** per-class F1 values, which cannot reach significance at any effect size -
the minimum two-sided p-value for n=4 is 0.125. :func:`wilcoxon_paired` refuses samples too
small to produce a meaningful result rather than reporting one.
"""

from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

#: Below this many pairs, a two-sided Wilcoxon cannot reach p < 0.05 whatever the effect.
MIN_WILCOXON_PAIRS = 6


def mcnemar_test(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, exact_threshold: int = 25
) -> Dict[str, object]:
    """McNemar's test on the errors two models make over the same samples.

    Only the *discordant* pairs carry information - the cases where exactly one model is
    right. Cases where both agree, however numerous, say nothing about which is better.

    :param y_true: True labels.
    :param pred_a: First model's predictions.
    :param pred_b: Second model's predictions.
    :param exact_threshold: Below this many discordant pairs, use the exact binomial test
        rather than the chi-square approximation, which is unreliable for small counts.
    :return: Contingency counts, statistic, p-value and which model the discordance favours.
    """
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    both_correct = int(np.sum(correct_a & correct_b))
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    discordant = only_a + only_b
    if discordant == 0:
        return {
            "both_correct": both_correct,
            "only_a_correct": only_a,
            "only_b_correct": only_b,
            "both_wrong": both_wrong,
            "n_discordant": 0,
            "statistic": None,
            "p_value": None,
            "test": "none",
            "note": "The models made identical predictions; there is nothing to compare.",
        }

    if discordant < exact_threshold:
        p_value = float(stats.binomtest(only_a, discordant, 0.5).pvalue)
        statistic = float(min(only_a, only_b))
        test = "exact binomial"
    else:
        # Chi-square with Edwards' continuity correction.
        statistic = float((abs(only_a - only_b) - 1) ** 2 / discordant)
        p_value = float(stats.chi2.sf(statistic, df=1))
        test = "chi-square with continuity correction"

    return {
        "both_correct": both_correct,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_wrong": both_wrong,
        "n_discordant": discordant,
        "statistic": statistic,
        "p_value": p_value,
        "test": test,
        "favours": "a" if only_a > only_b else "b" if only_b > only_a else "neither",
    }


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    confidence: float = 95.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap the difference between two models over the same samples.

    Each resample draws sample *indices* and scores both models on that same draw, so the
    correlation between them is preserved. This is the test the reference notebook needed
    and did not have: its Wilcoxon over four per-class values had no power, whereas a
    paired bootstrap over ~1000 test samples resolves differences of a fraction of a point.

    :param y_true: True labels.
    :param pred_a: First model's predictions.
    :param pred_b: Second model's predictions.
    :param metric_fn: Metric taking ``(y_true, y_pred)``.
    :param n_resamples: Bootstrap resamples.
    :param confidence: Confidence level for the interval, in percent.
    :param seed: Random seed.
    :return: Observed delta, its confidence interval, and the fraction of resamples
        favouring model A.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    deltas = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        draw = rng.integers(0, n, size=n)
        deltas[index] = metric_fn(y_true[draw], pred_a[draw]) - metric_fn(y_true[draw], pred_b[draw])

    tail = (100.0 - confidence) / 2.0
    observed = float(metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b))

    return {
        "observed_delta": observed,
        "ci_lower": float(np.percentile(deltas, tail)),
        "ci_upper": float(np.percentile(deltas, 100.0 - tail)),
        "fraction_favouring_a": float(np.mean(deltas > 0)),
        "n_resamples": n_resamples,
        # An interval spanning zero means the data cannot distinguish the two models.
        "significant": bool(
            np.percentile(deltas, tail) > 0 or np.percentile(deltas, 100.0 - tail) < 0
        ),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    confidence: float = 95.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap confidence interval for a single metric.

    Step 23: "Use 95% confidence intervals for main metrics."

    :param y_true: True labels.
    :param y_pred: Predictions.
    :param metric_fn: Metric taking ``(y_true, y_pred)``.
    :param n_resamples: Bootstrap resamples.
    :param confidence: Confidence level, in percent.
    :param seed: Random seed.
    :return: Point estimate and interval bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    scores = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        draw = rng.integers(0, n, size=n)
        scores[index] = metric_fn(y_true[draw], y_pred[draw])

    tail = (100.0 - confidence) / 2.0
    return {
        "point_estimate": float(metric_fn(y_true, y_pred)),
        "ci_lower": float(np.percentile(scores, tail)),
        "ci_upper": float(np.percentile(scores, 100.0 - tail)),
        "confidence": confidence,
        "n_resamples": n_resamples,
    }


def wilcoxon_paired(
    values_a: Sequence[float], values_b: Sequence[float], label: str = "paired values"
) -> Dict[str, object]:
    """Wilcoxon signed-rank test, refusing samples too small to be meaningful.

    A two-sided Wilcoxon over n pairs cannot produce a p-value below ``2 / 2**n``; at n=4
    the floor is 0.125, so significance is unreachable regardless of the effect. The
    reference notebook ran exactly that test over four per-class F1 scores. Rather than
    report an uninformative p-value, this returns a note explaining why the design cannot
    answer the question.

    :param values_a: First model's paired measurements, e.g. one per seed.
    :param values_b: Second model's, in the same order.
    :param label: What the pairs represent, used in the explanatory note.
    :return: Statistic and p-value, or a note if the sample is too small.
    """
    first = np.asarray(list(values_a), dtype=float)
    second = np.asarray(list(values_b), dtype=float)

    if first.shape != second.shape:
        raise ValueError(f"Paired inputs must match in length: {first.shape} vs {second.shape}")

    n = len(first)
    if n < MIN_WILCOXON_PAIRS:
        return {
            "n_pairs": n,
            "statistic": None,
            "p_value": None,
            "note": (
                f"Only {n} {label} - a two-sided Wilcoxon signed-rank test needs at least "
                f"{MIN_WILCOXON_PAIRS} pairs to reach p < 0.05 at any effect size "
                f"(the floor here is {2 / 2 ** n:.3f}). Reporting mean and standard "
                "deviation instead; use a paired bootstrap over samples for a powered test."
            ),
        }

    if np.allclose(first, second):
        return {
            "n_pairs": n,
            "statistic": None,
            "p_value": None,
            "note": "All pairs are identical; the test is undefined.",
        }

    statistic, p_value = stats.wilcoxon(first, second)
    return {
        "n_pairs": n,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "median_difference": float(np.median(first - second)),
    }


def holm_bonferroni(
    p_values: Mapping[str, Optional[float]], alpha: float = 0.05
) -> Dict[str, Dict[str, object]]:
    """Holm step-down correction over a family of hypotheses.

    Four comparisons each judged at alpha=0.05 give roughly a one-in-five chance of at
    least one false positive. Holm controls that family-wise rate while being uniformly
    more powerful than Bonferroni: p-values are sorted ascending and the i-th is scaled by
    the number of hypotheses still untested, ``(n - i)``, rather than by ``n`` throughout.

    The adjusted values are made monotone non-decreasing, which is what makes the step-down
    a valid procedure - without it a later hypothesis could be reported as more significant
    than an earlier one it is conditioned on.

    :param p_values: Comparison id to raw p-value. ``None`` marks a comparison that could
        not be tested; it is reported as such and excluded from the family size rather than
        counted as a hypothesis that happened not to reach significance.
    :param alpha: Family-wise error rate.
    :return: Per comparison: raw p, adjusted p, rank, and the corrected decision.
    """
    testable = {key: float(value) for key, value in p_values.items() if value is not None}
    n = len(testable)

    ordered = sorted(testable.items(), key=lambda item: item[1])
    adjusted: Dict[str, float] = {}
    running = 0.0
    for index, (key, raw) in enumerate(ordered):
        # Monotonicity: an adjusted p can never fall below the one before it.
        running = max(running, min(1.0, (n - index) * raw))
        adjusted[key] = running

    results: Dict[str, Dict[str, object]] = {}
    ranks = {key: index + 1 for index, (key, _) in enumerate(ordered)}

    for key, raw in p_values.items():
        if raw is None:
            results[key] = {
                "raw_p_value": None,
                "adjusted_p_value": None,
                "rank": None,
                "significant": False,
                "note": "No p-value was available; excluded from the correction family.",
            }
            continue
        results[key] = {
            "raw_p_value": float(raw),
            "adjusted_p_value": float(adjusted[key]),
            "rank": ranks[key],
            "significant": bool(adjusted[key] <= alpha),
        }

    return results


def summarise_across_seeds(values: Sequence[float]) -> Dict[str, float]:
    """Mean, standard deviation and range across seeds.

    Step 23: "Report mean and standard deviation across folds or random seeds."

    :param values: One measurement per seed.
    :return: Summary statistics, with the sample standard deviation.
    """
    array = np.asarray(list(values), dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        # ddof=1: these are a sample of seeds, not the population of them.
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }
