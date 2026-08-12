"""Step 8: choose the class-imbalance strategy on evidence.

The specification requires four strategies to be compared - class weighting, focal loss,
balanced sampler and augmentation - judged on macro-F1, balanced accuracy and class-wise
recall, because *"Accuracy alone is not sufficient"*. It also constrains how they may be
combined: *"Use more than one strategy only when ablation confirms benefit."*

That constraint is enforced here rather than assumed. A combined arm (sampler + weighted
cross-entropy) runs alongside the individual arms, and the selection rule only prefers it
when it actually beats both of its components. In the reference notebook the combined arm
was *worse* than either component - it collapsed one class's recall - and the single
sampler was correctly kept; this study reproduces that comparison rather than inheriting
its conclusion.

The focal-loss arm uses the corrected implementation. The notebook's version modulated
against ``p_t ** w_t`` instead of ``p_t`` whenever class weighting was active, which is
precisely this arm's configuration. See docs/DEVIATIONS.md (F6).
"""

import time
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.base import Analysis
from src.analysis.sweep_utils import run_proxy_trial
from src.data.bt_mri_proxy_datamodule import BTMRIProxyDataModule
from src.data.components.preprocessing import build_recipe, is_identity_recipe
from src.models.components.backbones import SmallCNN
from src.models.components.losses import CrossEntropyLoss, FocalLoss, LegacyFocalLoss
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

#: How each strategy configures the trial. ``loss`` names a factory below; ``sampler``
#: and ``augment`` toggle the corresponding datamodule behaviour.
STRATEGIES: Dict[str, Dict[str, Any]] = {
    "baseline": {"loss": "plain_ce", "sampler": False, "augment": False},
    "class_weighting": {"loss": "weighted_ce", "sampler": False, "augment": False},
    "focal_loss": {"loss": "focal", "sampler": False, "augment": False},
    "weighted_sampler": {"loss": "plain_ce", "sampler": True, "augment": False},
    "augmentation": {"loss": "plain_ce", "sampler": False, "augment": True},
    "combined_sampler_weighting": {"loss": "weighted_ce", "sampler": True, "augment": False},
}

#: Arms that combine strategies, mapped to the individual arms they must beat.
COMBINED_ARMS: Dict[str, List[str]] = {
    "combined_sampler_weighting": ["weighted_sampler", "class_weighting"],
}


def build_loss(name: str, gamma: float = 2.0) -> Any:
    """Construct a loss for one strategy arm.

    :param name: ``plain_ce``, ``weighted_ce``, ``focal`` or ``focal_legacy``.
    :param gamma: Focusing parameter for the focal variants.
    :return: A loss module.
    :raises ValueError: If the name is unknown.
    """
    if name == "plain_ce":
        return CrossEntropyLoss(use_class_weights=False)
    if name == "weighted_ce":
        return CrossEntropyLoss(use_class_weights=True)
    if name == "focal":
        return FocalLoss(gamma=gamma, use_class_weights=True)
    if name == "focal_legacy":
        return LegacyFocalLoss(gamma=gamma, use_class_weights=True)
    raise ValueError(f"Unknown loss {name!r}")


class ImbalanceStudy(Analysis):
    """Ablate class-imbalance strategies on the proxy subset.

    :param name: Analysis identifier.
    :param strategies: Strategy names to evaluate; defaults to all of :data:`STRATEGIES`.
    :param recipe: Preprocessing recipe applied during the study. Defaults to
        ``conventional`` so imbalance handling is measured independently of Step 6.
    :param epochs: Training epochs per arm.
    :param per_class_train: Training images per class in the proxy subset.
    :param per_class_val: Validation images per class in the proxy subset.
    :param image_size: Proxy input size.
    :param lr: Learning rate.
    :param focal_gamma: Focusing parameter for the focal arm.
    :param compare_legacy_focal: Also run the notebook's focal formulation, quantifying
        what F6 changed.
    :param seed: Seed shared by every arm.
    :param accelerator: Lightning accelerator.
    """

    def __init__(
        self,
        name: str = "step08_imbalance",
        strategies: Optional[List[str]] = None,
        recipe: str = "conventional",
        epochs: int = 5,
        per_class_train: int = 200,
        per_class_val: int = 75,
        image_size: int = 128,
        lr: float = 1e-3,
        focal_gamma: float = 2.0,
        compare_legacy_focal: bool = True,
        seed: int = 42,
        accelerator: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self.strategies = list(strategies or STRATEGIES.keys())
        self.recipe = recipe
        self.epochs = epochs
        self.per_class_train = per_class_train
        self.per_class_val = per_class_val
        self.image_size = image_size
        self.lr = lr
        self.focal_gamma = focal_gamma
        self.compare_legacy_focal = compare_legacy_focal
        self.seed = seed
        self.accelerator = accelerator

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Run the ablation.

        :param datamodule: Datamodule locating the dataset.
        :return: Summary including the selected strategy.
        """
        datamodule.prepare_data()

        arms = list(self.strategies)
        if self.compare_legacy_focal and "focal_loss" in arms:
            arms.append("focal_loss_legacy")

        log.info(f"Step 8: evaluating {len(arms)} imbalance strategies")
        rows: List[Dict[str, Any]] = []
        started = time.perf_counter()

        for position, arm in enumerate(arms, start=1):
            config = (
                {**STRATEGIES["focal_loss"], "loss": "focal_legacy"}
                if arm == "focal_loss_legacy"
                else STRATEGIES[arm]
            )
            log.info(f"[{position}/{len(arms)}] {arm}{self._eta(started, position, len(arms))}")

            metrics = run_proxy_trial(
                datamodule=self._build_datamodule(datamodule, config),
                net=SmallCNN(num_classes=len(datamodule.class_names)),
                criterion=build_loss(config["loss"], gamma=self.focal_gamma),
                epochs=self.epochs,
                seed=self.seed,
                lr=self.lr,
                accelerator=self.accelerator,
            )
            rows.append({"strategy": arm, **config, **metrics})
            log.info(
                f"    macro-F1 {metrics['macro_f1']:.4f} | "
                f"balanced acc {metrics['balanced_accuracy']:.4f} | "
                f"worst-class recall {metrics['min_class_recall']:.4f}"
            )
            # Rewrite after every arm so an interrupt keeps the completed ones.
            self.save_table(pd.DataFrame(rows), "step08_imbalance_comparison.csv", quiet=True)

        results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
        self.save_table(results, "step08_imbalance_comparison.csv")
        self._plot(results, datamodule.class_names)
        log.info("\n" + results.to_string(index=False))

        selected, rationale = self._select(results)
        log.info(f"Selected imbalance strategy: {selected} - {rationale}")

        summary: Dict[str, Any] = {
            "selected_strategy": selected,
            "selection_rationale": rationale,
            "selected_macro_f1": float(
                results.loc[results["strategy"] == selected, "macro_f1"].iloc[0]
            ),
            "selected_min_class_recall": float(
                results.loc[results["strategy"] == selected, "min_class_recall"].iloc[0]
            ),
            "n_strategies": len(results),
            "recipe": self.recipe,
            "protocol": {
                "epochs": self.epochs,
                "image_size": self.image_size,
                "per_class_train": self.per_class_train,
                "per_class_val": self.per_class_val,
                "lr": self.lr,
                "seed": self.seed,
                "model": "SmallCNN proxy",
            },
            "ranking": results[
                ["strategy", "macro_f1", "balanced_accuracy", "min_class_recall"]
            ].to_dict("records"),
        }

        focal_delta = self._focal_correction_delta(results)
        if focal_delta is not None:
            summary["focal_correction_delta_macro_f1"] = focal_delta
            log.info(
                "Corrected focal loss differs from the notebook's formulation by "
                f"{focal_delta:+.4f} macro-F1 (see docs/DEVIATIONS.md F6)"
            )

        summary["caveat"] = (
            "Ranking comes from a reduced-scale proxy (SmallCNN, "
            f"{self.image_size}px, {self.per_class_train} images/class), matching the "
            "reference protocol. It selects a strategy; it is not reportable performance."
        )
        return summary

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _eta(started: float, position: int, total: int) -> str:
        """Format a remaining-time estimate for the progress log.

        A sweep of this length on CPU looks stalled without one; the estimate is what tells
        a reader the run is progressing rather than hung.

        :param started: ``time.perf_counter()`` at the start of the sweep.
        :param position: 1-based index of the candidate about to run.
        :param total: Total candidates.
        :return: A short suffix such as ``"  (~6m remaining)"``, empty for the first.
        """
        if position <= 1:
            return ""
        per_item = (time.perf_counter() - started) / (position - 1)
        remaining = per_item * (total - position + 1)
        if remaining < 90:
            return f"  (~{remaining:.0f}s remaining)"
        return f"  (~{remaining / 60:.0f}m remaining)"


    def _build_datamodule(self, source: Any, config: Dict[str, Any]) -> BTMRIProxyDataModule:
        """Build a proxy datamodule configured for one arm.

        :param source: Datamodule supplying the dataset location.
        :param config: Strategy configuration from :data:`STRATEGIES`.
        :return: The configured proxy datamodule.
        """
        return BTMRIProxyDataModule(
            data_dir=source.hparams.data_dir,
            raw_subdir=source.hparams.raw_subdir,
            split_subpath=source.hparams.split_subpath,
            preprocess=None if is_identity_recipe(self.recipe) else build_recipe(self.recipe),
            per_class_train=self.per_class_train,
            per_class_val=self.per_class_val,
            image_size=self.image_size,
            augment=config["augment"],
            use_weighted_sampler=config["sampler"],
            seed=self.seed,
        )

    @staticmethod
    def _select(results: pd.DataFrame) -> tuple:
        """Apply the specification's rule on combining strategies.

        Step 8 permits stacking strategies "only when ablation confirms benefit". A
        combined arm must therefore beat every component it is built from on **both**
        macro-F1 and worst-class recall.

        Requiring both matters. Leading on macro-F1 alone is not evidence of benefit,
        because a combined arm trivially leads whenever it is ranked first - and the
        reference notebook's combined arm did well on aggregate while collapsing
        Meningioma recall to 0.373. Worst-class recall is what catches that, and it is
        exactly why the specification names class-wise recall as a judging criterion.

        :param results: Results table with ``strategy``, ``macro_f1`` and
            ``min_class_recall`` columns.
        :return: ``(selected_strategy, rationale)``.
        """
        ranked = results.sort_values("macro_f1", ascending=False).reset_index(drop=True)
        leader = str(ranked.iloc[0]["strategy"])

        if leader not in COMBINED_ARMS:
            return leader, "highest validation macro-F1 among all tested strategies"

        indexed = results.set_index("strategy")
        components = [c for c in COMBINED_ARMS[leader] if c in indexed.index]

        leader_f1 = float(indexed.loc[leader, "macro_f1"])
        leader_recall = float(indexed.loc[leader, "min_class_recall"])

        confirmed = all(
            leader_f1 > float(indexed.loc[component, "macro_f1"])
            and leader_recall >= float(indexed.loc[component, "min_class_recall"])
            for component in components
        )
        if confirmed:
            return leader, (
                "combined strategy beat every component on both macro-F1 and worst-class "
                "recall, so combining is confirmed beneficial"
            )

        individual = ranked[~ranked["strategy"].isin(COMBINED_ARMS)]
        fallback = str(individual.iloc[0]["strategy"])
        return fallback, (
            f"{leader} led on macro-F1 but did not beat all of its components "
            f"({components}) on both macro-F1 and worst-class recall; the specification "
            "permits combining only when ablation confirms benefit, so the best "
            "individual strategy is used"
        )

    @staticmethod
    def _focal_correction_delta(results: pd.DataFrame) -> Optional[float]:
        """Macro-F1 difference between the corrected and legacy focal formulations.

        :param results: Results table.
        :return: ``corrected - legacy``, or ``None`` if the legacy arm did not run.
        """
        scores = dict(zip(results["strategy"], results["macro_f1"]))
        if "focal_loss" in scores and "focal_loss_legacy" in scores:
            return float(scores["focal_loss"] - scores["focal_loss_legacy"])
        return None

    def _plot(self, results: pd.DataFrame, class_names: List[str]) -> None:
        """Plot macro-F1 per strategy and per-class recall.

        Per-class recall is plotted because the specification names it as a judging
        criterion: a strategy can lift macro-F1 while collapsing one class entirely, and
        only the per-class view shows that.

        :param results: Ranked results table.
        :param class_names: Class names ordered by label index.
        """
        figure, axes = plt.subplots(1, 2, figsize=(14, 5))

        order = results.sort_values("macro_f1")
        axes[0].barh(range(len(order)), order["macro_f1"], color="#4A5A8C")
        axes[0].set_yticks(range(len(order)))
        axes[0].set_yticklabels(order["strategy"], fontsize=9)
        axes[0].set_xlabel("Validation macro-F1 (proxy)")
        axes[0].set_title("Step 8: imbalance strategies")

        recall_columns = [f"recall_{name}" for name in class_names if f"recall_{name}" in results]
        for _, row in results.iterrows():
            axes[1].plot(
                range(len(recall_columns)),
                [row[column] for column in recall_columns],
                marker="o",
                label=row["strategy"],
                alpha=0.85,
            )
        axes[1].set_xticks(range(len(recall_columns)))
        axes[1].set_xticklabels([c.replace("recall_", "") for c in recall_columns], rotation=20)
        axes[1].set_ylabel("Recall")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_title("Class-wise recall (watch for collapse)")
        axes[1].legend(fontsize=7)

        figure.tight_layout()
        figure.savefig(self.figure_path("step08_imbalance.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)
