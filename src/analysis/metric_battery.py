"""The Step 16 metric battery, as a pure function.

    Step 16: "accuracy, balanced accuracy, macro precision/recall/F1, weighted F1,
    sensitivity, specificity, MCC, one-vs-rest AUC, the class-wise table with support, a
    confusion matrix ... and calibration via ECE and Brier score."

Step 21 reports "the same metrics for every configuration", and *same* has to mean
computed by the same code. Two implementations that agree today drift the first time one
of them changes a zero-division policy or an averaging mode, and the ablation table would
then compare numbers that are not comparable - silently, because both would still look
like macro-F1.

So the arithmetic lives here, and both :class:`~src.analysis.internal_test.InternalTest`
and the Step 21 ablation call it. This module deliberately does no file I/O and holds no
state: callers own their own artefacts and their own output directories.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import RankedLogger
from src.utils.metrics import (
    expected_calibration_error,
    multiclass_brier_score,
    specificity_per_class,
)

log = RankedLogger(__name__, rank_zero_only=True)


def overall_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str]
) -> Dict[str, Optional[float]]:
    """The headline metrics Step 16 lists.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param y_prob: Predicted probabilities, ``(n, n_classes)``.
    :param class_names: Class names ordered by label index.
    :return: Metric mapping; ``auc_ovr_macro`` is ``None`` when undefined.
    """
    labels = list(range(len(class_names)))
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = specificity_per_class(confusion, list(class_names))

    metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        # Macro recall is sensitivity; named both ways because Step 16 lists both.
        "macro_recall_sensitivity": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_specificity": float(np.mean(list(specificity.values()))),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    try:
        one_hot = np.eye(len(class_names))[y_true]
        metrics["auc_ovr_macro"] = float(
            roc_auc_score(one_hot, y_prob, average="macro", multi_class="ovr")
        )
    except ValueError as error:
        # Undefined when a class is absent from the split, which should not happen on a
        # stratified test set but must not crash the report if it does.
        log.warning(f"AUC could not be computed: {error}")
        metrics["auc_ovr_macro"] = None

    return metrics


def per_class_table(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]
) -> List[Dict[str, Any]]:
    """Class-wise precision, recall, F1, specificity and support.

    Class-wise recall is one of Step 21's two primary metrics, so it is a first-class
    output rather than something to be read back off a confusion matrix.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param class_names: Class names ordered by label index.
    :return: One record per class, in ``class_names`` order.
    """
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    specificity = specificity_per_class(
        confusion_matrix(y_true, y_pred, labels=labels), list(class_names)
    )

    return [
        {
            "class_name": name,
            "precision": float(report[name]["precision"]),
            "recall_sensitivity": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "specificity": float(specificity[name]),
            "support": int(report[name]["support"]),
        }
        for name in class_names
    ]


def confusion_summary(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], top_k: int = 6
) -> Dict[str, Any]:
    """Confusion matrix plus its most-confused pairs.

    Step 16 asks not just for the matrix but to "analyze which tumor classes are
    confused", so the worst pairs are extracted rather than left for the reader to spot.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param class_names: Class names ordered by label index.
    :param top_k: How many confusion pairs to rank.
    :return: Matrix, class names and ranked confusion pairs.
    """
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    pairs = [
        {"true": class_names[i], "predicted": class_names[j], "count": int(matrix[i, j])}
        for i in range(len(class_names))
        for j in range(len(class_names))
        if i != j and matrix[i, j] > 0
    ]
    pairs.sort(key=lambda row: -row["count"])

    return {
        "matrix": matrix.tolist(),
        "class_names": list(class_names),
        "top_confusions": pairs[:top_k],
    }


def calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str], n_bins: int = 10
) -> Dict[str, Any]:
    """Expected calibration error, Brier score, and the confidently-wrong count.

    :param y_true: True labels.
    :param y_prob: Predicted probabilities.
    :param class_names: Class names ordered by label index.
    :param n_bins: Bins for the ECE estimate.
    :return: Calibration metrics and the reliability bins.
    """
    ece, bins = expected_calibration_error(y_true, y_prob, n_bins)

    confident_wrong = int(
        np.sum((y_prob.max(axis=1) > 0.99) & (y_prob.argmax(axis=1) != y_true))
    )

    return {
        "expected_calibration_error": ece,
        "brier_score": multiclass_brier_score(y_true, y_prob, len(class_names)),
        # The specific pathology the notebook found: near-certain and wrong.
        "confidently_wrong_over_0.99": confident_wrong,
        "bins": bins,
    }


def full_battery(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    n_calibration_bins: int = 10,
) -> Dict[str, Any]:
    """Every Step 16 metric for one set of predictions.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param y_prob: Predicted probabilities.
    :param class_names: Class names ordered by label index.
    :param n_calibration_bins: Bins for the ECE estimate.
    :return: ``{"n_samples", "overall", "per_class", "confusion", "calibration"}``. The
        calibration entry carries no ``bins`` frame, which is an artefact rather than a
        metric; callers wanting it should use :func:`calibration_metrics` directly.
    """
    calibration = calibration_metrics(y_true, y_prob, class_names, n_calibration_bins)
    calibration.pop("bins", None)

    return {
        "n_samples": int(len(y_true)),
        "overall": overall_metrics(y_true, y_pred, y_prob, class_names),
        "per_class": per_class_table(y_true, y_pred, class_names),
        "confusion": confusion_summary(y_true, y_pred, class_names),
        "calibration": calibration,
    }


def summarise_seeds(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Mean, standard deviation and a 95% interval across seeds.

    Step 23: "Report mean and standard deviation across folds or random seeds. Use 95%
    confidence intervals for main metrics."

    With three seeds the interval is a normal-approximation ``mean +- 1.96 * sem`` and is
    *descriptive*: three points cannot support a significance claim, and the paired
    bootstrap over test predictions is what Step 23 uses for that. ``n`` is reported so a
    reader can see how thin the estimate is.

    :param values: One value per seed; ``None`` entries are dropped.
    :return: ``{"mean", "std", "ci_low", "ci_high", "n"}``, all ``None`` when empty.
    """
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None, "n": 0}

    array = np.asarray(clean, dtype=float)
    mean = float(array.mean())
    # Sample standard deviation: these are draws from the seed distribution, not a census.
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0

    if array.size > 1:
        half = 1.96 * std / np.sqrt(array.size)
        ci_low, ci_high = mean - half, mean + half
    else:
        ci_low = ci_high = None

    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(array.size),
    }


def bins_frame(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """:param y_true: True labels.

    :param y_prob: Predicted probabilities.
    :param n_bins: Bins for the ECE estimate.
    :return: The reliability-bin table, for callers that write it out.
    """
    return expected_calibration_error(y_true, y_prob, n_bins)[1]
