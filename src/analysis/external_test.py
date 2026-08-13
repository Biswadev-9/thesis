"""Step 17: cross-dataset validation on the Figshare set.

    "Cross-dataset validation is necessary because benchmark accuracy alone does not prove
     clinical robustness."

Figshare carries only glioma, meningioma and pituitary - there is no non-tumour class - so
Step 17's second clause applies: "evaluate a three-class external task using the same
trained feature extractor". No fine-tuning happens here; the model is used exactly as
Step 15 left it.

**Both a restricted and an unrestricted score are reported.** Restricting the argmax to the
three present classes is the standard way to handle a missing class, and it is what the
reference notebook did - but it is also a *favourable* choice, because it silently forgives
every case where the model would have answered "No-tumor". Reporting only the restricted
figure would overstate how well the model transfers. The unrestricted score counts those as
errors, and the gap between the two is itself informative: a large gap means the model
frequently reaches for the class that does not exist here.
"""

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
)

from src.analysis.base import Analysis
from src.data.components.split_builder import CLASS_MAP
from src.models.full_pipeline import load_full_pipeline, predict
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ExternalTest(Analysis):
    """Evaluate the final model on an external dataset without retraining.

    :param name: Analysis identifier.
    :param classical_ckpt: Step 10 checkpoint or run directory.
    :param quantum_ckpt: Step 12 checkpoint or run directory.
    :param fusion_ckpt: Step 15 checkpoint or run directory.
    :param classical_model: Hydra config for the Step 10 module.
    :param quantum_model: Hydra config for the Step 12 module.
    :param fusion_model: Hydra config for the fusion module.
    :param internal_summary: Path to the Step 16 summary JSON. When given, the
        internal-to-external performance drop is computed on the same three classes.
    :param accelerator: ``auto``, ``cpu`` or ``gpu``.
    """

    def __init__(
        self,
        name: str = "step17_external",
        classical_ckpt: Optional[str] = None,
        quantum_ckpt: Optional[str] = None,
        fusion_ckpt: Optional[str] = None,
        classical_model: Optional[Any] = None,
        quantum_model: Optional[Any] = None,
        fusion_model: Optional[Any] = None,
        internal_summary: Optional[str] = None,
        accelerator: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self.classical_ckpt = classical_ckpt
        self.quantum_ckpt = quantum_ckpt
        self.fusion_ckpt = fusion_ckpt
        self.classical_model = classical_model
        self.quantum_model = quantum_model
        self.fusion_model = fusion_model
        self.internal_summary = internal_summary
        self.accelerator = accelerator

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Evaluate externally and report the transfer gap.

        :param datamodule: A ``FigshareDataModule``.
        :return: Restricted and unrestricted metrics, plus the drop from internal.
        """
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

        present = datamodule.present_classes
        present_indices = sorted(CLASS_MAP[name] for name in present)
        class_names = datamodule.class_names

        log.info(f"External evaluation over {present} ({len(datamodule.data_external)} scans)")
        outputs = predict(pipeline, datamodule.test_dataloader(), device)
        y_true, y_prob = outputs["y_true"], outputs["y_prob"]

        unrestricted = outputs["y_pred"]
        # Restricted: argmax over the present classes only, mapped back to model indices.
        restricted = np.array(
            [present_indices[i] for i in y_prob[:, present_indices].argmax(axis=1)]
        )

        summary: Dict[str, Any] = {
            "n_external_samples": int(len(y_true)),
            "present_classes": present,
            "restricted": self._metrics(y_true, restricted, present_indices, class_names),
            "unrestricted": self._metrics(y_true, unrestricted, present_indices, class_names),
            "predicted_absent_class_count": int(np.sum(~np.isin(unrestricted, present_indices))),
        }
        summary["restriction_effect_macro_f1"] = (
            summary["restricted"]["macro_f1"] - summary["unrestricted"]["macro_f1"]
        )

        self._per_class_table(y_true, restricted, present_indices, class_names)
        self._plot_confusion(y_true, restricted, present_indices, present)

        drop = self._performance_drop(summary["restricted"]["macro_f1"], present)
        if drop is not None:
            summary["performance_drop"] = drop

        summary["domain_shift_notes"] = [
            "Figshare scans are 16-bit with varying intensity ranges and are min-max "
            "normalised per image before the shared pipeline; the internal set is 8-bit "
            "and natively 224x224.",
            "Acquisition differs in scanner, sequence weighting and slice orientation, "
            "none of which is controlled for.",
            "Figshare provides one label per slice and multiple slices per patient (PID); "
            "slice-level metrics therefore treat correlated slices as independent.",
            "No non-tumour class exists externally, so the restricted score forgives every "
            "case the model would have answered No-tumor - see "
            "'predicted_absent_class_count'.",
        ]
        self._log(summary)
        return summary

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _metrics(
        y_true: np.ndarray, y_pred: np.ndarray, labels: List[int], class_names: List[str]
    ) -> Dict[str, float]:
        """Compute metrics over the present classes only.

        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :param labels: Model indices of the present classes.
        :param class_names: All class names, ordered by index.
        :return: Metric mapping.
        """
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(
                precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            ),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }

    def _per_class_table(
        self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[int], class_names: List[str]
    ) -> None:
        """Write the class-wise report for the present classes.

        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :param labels: Model indices of the present classes.
        :param class_names: All class names, ordered by index.
        """
        names = [class_names[i] for i in labels]
        report = classification_report(
            y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0
        )
        rows = [
            {
                "class_name": name,
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in names
        ]
        self.save_table(pd.DataFrame(rows), "step17_per_class.csv")

    def _performance_drop(self, external_macro_f1: float, present: List[str]) -> Optional[Dict[str, float]]:
        """Compare against the internal result on the same three classes.

        Step 17 asks for "the performance drop from internal testing to external testing".
        The comparison uses the internal per-class F1 restricted to the classes Figshare
        actually contains - comparing a three-class score against a four-class one would
        conflate the domain shift with the missing class.

        :param external_macro_f1: External restricted macro-F1.
        :param present: Class names present externally.
        :return: The comparison, or ``None`` if no internal summary was supplied.
        """
        if not self.internal_summary:
            log.warning(
                "analysis.internal_summary not set - the internal-to-external drop cannot "
                "be computed. Point it at the Step 16 summary JSON."
            )
            return None

        import json
        from pathlib import Path

        path = Path(self.internal_summary)
        if not path.is_file():
            log.warning(f"Internal summary not found at {path}; skipping the drop calculation")
            return None

        internal = json.loads(path.read_text(encoding="utf-8"))
        per_class = {row["class_name"]: row for row in internal.get("per_class", [])}

        scores = [per_class[name]["f1"] for name in present if name in per_class]
        if not scores:
            return None

        internal_macro_f1 = float(np.mean(scores))
        return {
            "internal_macro_f1_same_classes": internal_macro_f1,
            "external_macro_f1": float(external_macro_f1),
            "drop": float(internal_macro_f1 - external_macro_f1),
        }

    def _plot_confusion(
        self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[int], present: List[str]
    ) -> None:
        """Draw the external confusion matrix.

        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :param labels: Model indices of the present classes.
        :param present: Names of the present classes.
        """
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        names = [name for _, name in sorted(zip(labels, present))]

        figure, axis = plt.subplots(figsize=(5.5, 4.8))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(names)), names, rotation=30, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title("Step 17: external (Figshare) test set")

        threshold = matrix.max() / 2 if matrix.max() else 0
        for i in range(len(names)):
            for j in range(len(names)):
                axis.text(
                    j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black",
                )

        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(self.figure_path("step17_confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _log(summary: Dict[str, Any]) -> None:
        """:param summary: Computed summary, logged in readable form."""
        log.info("=== Step 17: external validation ===")
        log.info(f"  restricted macro-F1   {summary['restricted']['macro_f1']:.4f}")
        log.info(f"  unrestricted macro-F1 {summary['unrestricted']['macro_f1']:.4f}")
        log.info(
            f"  predicted the absent class {summary['predicted_absent_class_count']} times "
            f"of {summary['n_external_samples']}"
        )
        if "performance_drop" in summary:
            log.info(f"  drop from internal    {summary['performance_drop']['drop']:+.4f}")
