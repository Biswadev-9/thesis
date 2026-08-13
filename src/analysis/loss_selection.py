"""Step 14: choose the final loss function on validation results only.

The specification is narrow about this:

    "Loss: class-weighted cross-entropy or focal loss based on validation results."

Two constraints follow, and the reference notebook broke both.

**The candidate set is restricted.** Only weighted cross-entropy and focal loss may be
*selected*. Plain cross-entropy is trained and reported as a reference row - it is useful
to know how much the imbalance handling buys - but it cannot win. The notebook selected
plain CE.

**Selection uses validation only.** The notebook found a three-way tie on validation
(0.9897 for all three) and broke it by comparing *test* macro-F1, then reported that same
test set as the final result. That inflates the reported performance and violates Step 16's
requirement that the test set stay unseen.

The tie-break here is fixed in advance and never consults test data:

1. validation macro-F1
2. validation balanced accuracy
3. lower validation expected calibration error

The third is a deliberate choice rather than a coin toss. When two losses classify equally
well, the one whose confidences are better calibrated is the more useful model - and the
notebook's own spot-check turned up a confidently wrong prediction (true Glioma, predicted
Meningioma at probability 1.000), which is exactly what poor calibration looks like.
"""

import time
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.base import Analysis
from src.analysis.imbalance_study import build_loss
from src.analysis.sweep_utils import run_feature_trial
from src.data.bt_mri_feature_datamodule import BTMRIFeatureDataModule
from src.models.components.fusion import FusedFeatureClassifier
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

#: Losses the specification permits as the final choice.
SELECTABLE_LOSSES = ("weighted_ce", "focal")

#: Losses trained for comparison but not eligible to be selected.
REFERENCE_LOSSES = ("plain_ce",)

#: Tie-break order, applied in sequence. ``higher_is_better`` flips the comparison.
TIE_BREAK = (
    ("macro_f1", True),
    ("balanced_accuracy", True),
    ("ece", False),
)


class LossSelection(Analysis):
    """Train the final classifier under each candidate loss and select on validation.

    :param name: Analysis identifier.
    :param tag: Feature cache to read.
    :param candidates: Losses to train. Defaults to the selectable set plus the reference.
    :param epochs: Training epochs per candidate.
    :param lr: Learning rate.
    :param dropout: Dropout in the final classifier.
    :param focal_gamma: Focusing parameter for the focal candidate.
    :param seed: Seed shared by every candidate.
    :param accelerator: Lightning accelerator.
    """

    def __init__(
        self,
        name: str = "step14_loss_selection",
        tag: str = "default",
        candidates: Optional[List[str]] = None,
        epochs: int = 25,
        lr: float = 1e-3,
        dropout: float = 0.4,
        focal_gamma: float = 2.0,
        seed: int = 42,
        accelerator: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self.tag = tag
        self.candidates = list(candidates or list(SELECTABLE_LOSSES) + list(REFERENCE_LOSSES))
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.focal_gamma = focal_gamma
        self.seed = seed
        self.accelerator = accelerator

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Train the candidates and apply the selection rule.

        :param datamodule: Datamodule supplying the data directory.
        :return: Summary naming the selected loss and the rule that chose it.
        """
        data_dir = datamodule.hparams.data_dir if datamodule is not None else "data/"

        reference = BTMRIFeatureDataModule(data_dir=data_dir, tag=self.tag)
        reference.prepare_data()
        reference.setup()
        dims = (reference.classical_dim, reference.spatial_dim, reference.quantum_dim)

        rows: List[Dict[str, Any]] = []
        started = time.perf_counter()

        for position, candidate in enumerate(self.candidates, start=1):
            elapsed = ""
            if position > 1:
                per_item = (time.perf_counter() - started) / (position - 1)
                elapsed = f"  (~{per_item * (len(self.candidates) - position + 1):.0f}s remaining)"
            log.info(f"[{position}/{len(self.candidates)}] loss={candidate}{elapsed}")

            metrics = run_feature_trial(
                datamodule=BTMRIFeatureDataModule(data_dir=data_dir, tag=self.tag),
                net=FusedFeatureClassifier(
                    classical_dim=dims[0],
                    spatial_dim=dims[1],
                    quantum_dim=dims[2],
                    num_classes=4,
                    dropout=self.dropout,
                ),
                criterion=build_loss(candidate, gamma=self.focal_gamma),
                epochs=self.epochs,
                seed=self.seed,
                lr=self.lr,
                accelerator=self.accelerator,
            )
            rows.append(
                {
                    "loss": candidate,
                    "selectable": candidate in SELECTABLE_LOSSES,
                    **metrics,
                }
            )
            log.info(
                f"    val macro-F1 {metrics['macro_f1']:.4f} | "
                f"balanced acc {metrics['balanced_accuracy']:.4f} | "
                f"ECE {metrics['ece']:.4f}"
            )
            self.save_table(pd.DataFrame(rows), "step14_loss_ablation.csv", quiet=True)

        results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
        self.save_table(results, "step14_loss_ablation.csv")
        self._plot(results)
        log.info("\n" + results.to_string(index=False))

        selected, rationale = self._select(results)
        log.info(f"Selected loss: {selected} - {rationale}")

        return {
            "tag": self.tag,
            "selected_loss": selected,
            "selection_rationale": rationale,
            "selectable_candidates": list(SELECTABLE_LOSSES),
            "reference_candidates": list(REFERENCE_LOSSES),
            "tie_break_order": [name for name, _ in TIE_BREAK],
            "results": results.to_dict("records"),
            "protocol": {"epochs": self.epochs, "lr": self.lr, "seed": self.seed},
            "next_step": (
                f"python src/train.py -m experiment=step15_final_protocol seed=42,123,7 "
                f"loss@model.criterion={selected}"
            ),
            "caveat": (
                "Selection used validation metrics only. The internal test set has not "
                "been read; Step 16 evaluates it exactly once."
            ),
        }

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _select(results: pd.DataFrame) -> tuple:
        """Apply the fixed tie-break over the selectable candidates.

        :param results: Results table.
        :return: ``(selected_loss, rationale)``.
        :raises ValueError: If no selectable candidate was trained.
        """
        eligible = results[results["selectable"]]
        if eligible.empty:
            raise ValueError(
                "No selectable loss was trained. Step 14 permits only class-weighted "
                f"cross-entropy or focal loss as the final choice: {SELECTABLE_LOSSES}"
            )

        ranked = eligible.sort_values(
            by=[name for name, _ in TIE_BREAK],
            ascending=[not higher for _, higher in TIE_BREAK],
        ).reset_index(drop=True)

        winner = ranked.iloc[0]
        selected = str(winner["loss"])

        if len(ranked) == 1:
            return selected, "only selectable candidate trained"

        runner_up = ranked.iloc[1]
        for metric, higher_is_better in TIE_BREAK:
            difference = float(winner[metric]) - float(runner_up[metric])
            if abs(difference) > 1e-6:
                direction = "higher" if higher_is_better else "lower"
                return selected, (
                    f"{direction} validation {metric} than {runner_up['loss']} "
                    f"({winner[metric]:.4f} vs {runner_up[metric]:.4f}); selected on "
                    "validation only"
                )

        return selected, (
            "tied with the other candidate on every validation criterion; the first in "
            "the specification's order was taken. Test metrics were not consulted."
        )

    def _plot(self, results: pd.DataFrame) -> None:
        """Plot macro-F1 and calibration side by side.

        Calibration is plotted because it is the tie-break, and because a loss can match
        another on accuracy while being markedly less honest about its confidence.

        :param results: Results table.
        """
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        order = results.sort_values("macro_f1")
        colours = ["#4A5A8C" if s else "#9AA4B2" for s in order["selectable"]]

        axes[0].barh(range(len(order)), order["macro_f1"], color=colours)
        axes[0].set_yticks(range(len(order)))
        axes[0].set_yticklabels(order["loss"])
        axes[0].set_xlabel("Validation macro-F1")
        axes[0].set_title("Step 14: loss candidates\n(grey = reference only, not selectable)")

        axes[1].barh(range(len(order)), order["ece"], color=colours)
        axes[1].set_yticks(range(len(order)))
        axes[1].set_yticklabels(order["loss"])
        axes[1].set_xlabel("Validation ECE (lower is better)")
        axes[1].set_title("Calibration — the tie-break")

        figure.tight_layout()
        figure.savefig(self.figure_path("step14_loss_selection.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)
