"""Metrics the specification names that scikit-learn does not provide directly.

Step 16 asks for specificity and for calibration "using expected calibration error or
Brier score", and Step 14's loss selection uses calibration as a tie-break. All three are
implemented here so the same definitions serve every step that reports them.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[float, pd.DataFrame]:
    """Expected calibration error over equal-width confidence bins.

    Measures the average gap between a model's confidence and its actual accuracy. A model
    that says "0.99" and is right 99 % of the time scores 0; one that says "0.99" and is
    right 80 % of the time scores about 0.19.

    This matters here because the reference notebook found a confidently *wrong*
    prediction - true Glioma, predicted Meningioma at probability 1.000 - which is exactly
    the failure a headline accuracy figure hides.

    :param y_true: Integer labels, ``(N,)``.
    :param y_prob: Predicted probabilities, ``(N, K)``.
    :param n_bins: Number of confidence bins.
    :return: ``(ece, per_bin_table)``.
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows: List[Dict[str, float]] = []

    for lower, upper in zip(edges[:-1], edges[1:]):
        # Left-open intervals so a confidence of exactly 1.0 lands in the final bin.
        in_bin = (confidences > lower) & (confidences <= upper)
        proportion = float(in_bin.mean())
        if proportion == 0.0:
            continue

        accuracy = float(correct[in_bin].mean())
        confidence = float(confidences[in_bin].mean())
        ece += abs(confidence - accuracy) * proportion

        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "n_samples": int(in_bin.sum()),
                "mean_confidence": confidence,
                "accuracy": accuracy,
                "gap": confidence - accuracy,
            }
        )

    return float(ece), pd.DataFrame(rows)


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    """Multiclass Brier score - mean squared error against the one-hot label.

    Unlike ECE this is a proper scoring rule: it penalises both miscalibration and
    inaccuracy, so the two are worth reporting together.

    :param y_true: Integer labels, ``(N,)``.
    :param y_prob: Predicted probabilities, ``(N, K)``.
    :param num_classes: Number of classes.
    :return: The score; lower is better, 0 is perfect.
    """
    one_hot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((one_hot - y_prob) ** 2, axis=1)))


def specificity_per_class(
    confusion: np.ndarray, class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """One-vs-rest specificity for each class.

    Step 16 lists specificity alongside sensitivity. Sensitivity (recall) comes from
    scikit-learn; specificity - true negatives over all actual negatives - does not, for
    the multiclass case.

    :param confusion: Confusion matrix, ``(K, K)``, rows true and columns predicted.
    :param class_names: Class names ordered by label index.
    :return: Mapping of class name to specificity.
    """
    total = confusion.sum()
    names = class_names or [f"class_{i}" for i in range(len(confusion))]

    results: Dict[str, float] = {}
    for index, name in enumerate(names):
        true_positive = confusion[index, index]
        false_negative = confusion[index, :].sum() - true_positive
        false_positive = confusion[:, index].sum() - true_positive
        true_negative = total - true_positive - false_negative - false_positive

        denominator = true_negative + false_positive
        results[name] = float(true_negative / denominator) if denominator > 0 else 0.0

    return results
