"""Step 16: internal testing - the study's headline result.

    "After model selection using the validation set, evaluate the final model **once** on
     the internal test set. The test set should remain unseen during training and
     hyperparameter tuning."

That word "once" is the whole point of the step, and it is the constraint the reference
notebook broke: it read test metrics while choosing the Step 14 loss, then reported that
same test set as the final result. Every selection decision in this repository is made on
validation, and this analysis writes a lock file recording that the test set has been
consumed. Re-running against the same checkpoint requires ``force=true``, which exists so
the violation has to be deliberate rather than accidental.

The metric battery is exactly what Step 16 lists: accuracy, balanced accuracy, macro
precision/recall/F1, weighted F1, sensitivity, specificity, MCC, one-vs-rest AUC, the
class-wise table with support, a confusion matrix with its worst pairs identified, and
calibration via ECE and Brier score.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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

from src.analysis.base import Analysis
from src.models.full_pipeline import load_full_pipeline, predict
from src.utils import RankedLogger
from src.utils.metrics import (
    expected_calibration_error,
    multiclass_brier_score,
    specificity_per_class,
)

log = RankedLogger(__name__, rank_zero_only=True)

LOCK_FILENAME = "test_evaluated.lock"


class InternalTest(Analysis):
    """Evaluate the final model once on the held-out internal test set.

    :param name: Analysis identifier.
    :param classical_ckpt: Step 10 checkpoint or run directory.
    :param quantum_ckpt: Step 12 checkpoint or run directory.
    :param fusion_ckpt: Step 15 checkpoint or run directory. The lock file is written
        beside it.
    :param classical_model: Hydra config for the Step 10 module.
    :param quantum_model: Hydra config for the Step 12 module.
    :param fusion_model: Hydra config for the fusion module.
    :param force: Re-evaluate a checkpoint that has already been tested. Every use of this
        flag weakens the "evaluated once" claim and is recorded in the summary.
    :param n_calibration_bins: Bins for the ECE estimate.
    :param accelerator: ``auto``, ``cpu`` or ``gpu``.
    """

    def __init__(
        self,
        name: str = "step16_internal",
        classical_ckpt: Optional[str] = None,
        quantum_ckpt: Optional[str] = None,
        fusion_ckpt: Optional[str] = None,
        classical_model: Optional[Any] = None,
        quantum_model: Optional[Any] = None,
        fusion_model: Optional[Any] = None,
        force: bool = False,
        n_calibration_bins: int = 10,
        accelerator: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self.classical_ckpt = classical_ckpt
        self.quantum_ckpt = quantum_ckpt
        self.fusion_ckpt = fusion_ckpt
        self.classical_model = classical_model
        self.quantum_model = quantum_model
        self.fusion_model = fusion_model
        self.force = force
        self.n_calibration_bins = n_calibration_bins
        self.accelerator = accelerator

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Run the one-time evaluation.

        :param datamodule: Datamodule supplying the internal test split.
        :return: The full Step 16 metric battery.
        :raises RuntimeError: If this checkpoint has already been evaluated and ``force``
            is not set.
        """
        lock_path = self._check_lock()

        device = torch.device(
            "cuda" if (self.accelerator != "cpu" and torch.cuda.is_available()) else "cpu"
        )
        pipeline = load_full_pipeline(
            classical_ckpt=self.classical_ckpt,
            quantum_ckpt=self.quantum_ckpt,
            fusion_ckpt=self.fusion_ckpt,
            classical_model=self.classical_model,
            quantum_model=self.quantum_model,
            fusion_model=self.fusion_model,
            device=device,
        )

        datamodule.prepare_data()
        datamodule.setup()
        class_names = datamodule.class_names

        log.info("Evaluating on the internal test set - this consumes the once-only budget")
        outputs = predict(pipeline, datamodule.test_dataloader(), device)
        y_true, y_pred, y_prob = outputs["y_true"], outputs["y_pred"], outputs["y_prob"]

        summary: Dict[str, Any] = {
            "checkpoint": str(self.fusion_ckpt),
            "n_test_samples": int(len(y_true)),
            "overall": self._overall_metrics(y_true, y_pred, y_prob, class_names),
            "per_class": self._per_class(y_true, y_pred, class_names),
            "confusion": self._confusion(y_true, y_pred, class_names),
            "calibration": self._calibration(y_true, y_prob, class_names),
            "forced_reevaluation": self.force,
        }

        np.savez_compressed(
            self.output_dir / "test_predictions.npz",
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )
        self._write_lock(lock_path, summary)
        self._log(summary)
        return summary

    # ---------------------------------------------------------------- lock file

    def _lock_path(self) -> Optional[Path]:
        """:return: Where the lock lives - beside the evaluated checkpoint."""
        if not self.fusion_ckpt:
            return None
        candidate = Path(self.fusion_ckpt)
        directory = candidate if candidate.is_dir() else candidate.parent.parent
        return directory / LOCK_FILENAME

    def _check_lock(self) -> Optional[Path]:
        """Refuse a second evaluation of the same checkpoint.

        :return: The lock path, or ``None`` if it cannot be determined.
        :raises RuntimeError: If the checkpoint was already evaluated and ``force`` is off.
        """
        lock_path = self._lock_path()
        if lock_path is None or not lock_path.is_file():
            return lock_path

        if not self.force:
            raise RuntimeError(
                f"This checkpoint has already been evaluated on the internal test set "
                f"(see {lock_path}). Step 16 requires the test set to be used once.\n\n"
                "If the model changed, evaluate the NEW checkpoint instead. If you really "
                "intend to re-read the test set with this checkpoint, pass "
                "`analysis.force=true` - the summary will record that the result is no "
                "longer a single-use estimate."
            )

        log.warning(
            f"Re-evaluating a checkpoint already tested (lock at {lock_path}). "
            "The 'evaluated once' claim no longer holds for this result."
        )
        return lock_path

    def _write_lock(self, lock_path: Optional[Path], summary: Dict[str, Any]) -> None:
        """Record that this checkpoint has consumed the test set.

        :param lock_path: Where to write.
        :param summary: Result summary, for provenance.
        """
        if lock_path is None:
            return

        from datetime import datetime

        history = []
        if lock_path.is_file():
            try:
                history = json.loads(lock_path.read_text(encoding="utf-8")).get("evaluations", [])
            except (OSError, json.JSONDecodeError):
                history = []

        history.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "output_dir": str(self.output_dir),
                "macro_f1": summary["overall"]["macro_f1"],
                "forced": self.force,
            }
        )
        lock_path.write_text(
            json.dumps({"evaluations": history}, indent=2), encoding="utf-8"
        )
        log.info(f"Test-set lock written to {lock_path}")

    # ------------------------------------------------------------------ metrics

    def _overall_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: List[str]
    ) -> Dict[str, float]:
        """Compute the headline metrics Step 16 lists.

        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :param y_prob: Predicted probabilities.
        :param class_names: Class names ordered by label index.
        :return: Metric mapping.
        """
        labels = list(range(len(class_names)))
        confusion = confusion_matrix(y_true, y_pred, labels=labels)
        specificity = specificity_per_class(confusion, class_names)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            # Macro recall is sensitivity; named both ways because Step 16 lists both.
            "macro_recall_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
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
            # Undefined when a class is absent from the split, which should not happen on
            # a stratified test set but must not crash the report if it does.
            log.warning(f"AUC could not be computed: {error}")
            metrics["auc_ovr_macro"] = None

        return metrics

    def _per_class(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Class-wise precision, recall, F1, support and specificity.

        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :param class_names: Class names ordered by label index.
        :return: One record per class.
        """
        labels = list(range(len(class_names)))
        report = classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
        )
        specificity = specificity_per_class(confusion_matrix(y_true, y_pred, labels=labels), class_names)

        rows = []
        for name in class_names:
            entry = report[name]
            rows.append(
                {
                    "class_name": name,
                    "precision": float(entry["precision"]),
                    "recall_sensitivity": float(entry["recall"]),
                    "f1": float(entry["f1-score"]),
                    "specificity": float(specificity[name]),
                    "support": int(entry["support"]),
                }
            )

        self.save_table(pd.DataFrame(rows), "step16_per_class.csv")
        return rows

    def _confusion(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
    ) -> Dict[str, Any]:
        """Confusion matrix plus its most-confused pairs.

        Step 16 asks not just for the matrix but to "analyze which tumor classes are
        confused", so the worst pairs are extracted explicitly rather than left for the
        reader to spot.

        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :param class_names: Class names ordered by label index.
        :return: Matrix and ranked confusion pairs.
        """
        labels = list(range(len(class_names)))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)

        frame = pd.DataFrame(matrix, index=class_names, columns=class_names)
        self.save_table(frame, "step16_confusion_matrix.csv", index=True)
        self._plot_confusion(matrix, class_names)

        pairs = [
            {"true": class_names[i], "predicted": class_names[j], "count": int(matrix[i, j])}
            for i in range(len(class_names))
            for j in range(len(class_names))
            if i != j and matrix[i, j] > 0
        ]
        pairs.sort(key=lambda row: -row["count"])

        return {"matrix": matrix.tolist(), "class_names": class_names, "top_confusions": pairs[:6]}

    def _calibration(
        self, y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]
    ) -> Dict[str, Any]:
        """Expected calibration error and Brier score.

        :param y_true: True labels.
        :param y_prob: Predicted probabilities.
        :param class_names: Class names ordered by label index.
        :return: Calibration metrics plus the count of confidently-wrong predictions.
        """
        ece, bins = expected_calibration_error(y_true, y_prob, self.n_calibration_bins)
        self.save_table(bins, "step16_calibration_bins.csv")

        confident_wrong = int(
            np.sum((y_prob.max(axis=1) > 0.99) & (y_prob.argmax(axis=1) != y_true))
        )

        return {
            "expected_calibration_error": ece,
            "brier_score": multiclass_brier_score(y_true, y_prob, len(class_names)),
            # The specific pathology the notebook found: near-certain and wrong.
            "confidently_wrong_over_0.99": confident_wrong,
        }

    # ------------------------------------------------------------------ output

    def _plot_confusion(self, matrix: np.ndarray, class_names: List[str]) -> None:
        """Draw the confusion matrix.

        :param matrix: Confusion matrix.
        :param class_names: Class names ordered by label index.
        """
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        image = axis.imshow(matrix, cmap="Blues")

        axis.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
        axis.set_yticks(range(len(class_names)), class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title("Step 16: internal test set")

        threshold = matrix.max() / 2 if matrix.max() else 0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                axis.text(
                    j,
                    i,
                    int(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > threshold else "black",
                )

        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(self.figure_path("step16_confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _log(summary: Dict[str, Any]) -> None:
        """:param summary: Computed summary, logged in readable form."""
        log.info("=== Step 16: internal test results ===")
        for key, value in summary["overall"].items():
            log.info(f"  {key:<28} {value:.4f}" if value is not None else f"  {key:<28} n/a")

        log.info("Calibration:")
        for key, value in summary["calibration"].items():
            log.info(f"  {key:<28} {value}")

        if summary["confusion"]["top_confusions"]:
            log.info("Most confused pairs:")
            for pair in summary["confusion"]["top_confusions"][:3]:
                log.info(f"  {pair['true']} -> {pair['predicted']}: {pair['count']}")
