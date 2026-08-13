"""Step 13: choose the fusion strategy and measure each branch's contribution.

Step 13 sets both the order and the burden of proof:

    "Simple concatenation should be used as the first fusion baseline. Then add
     attention-based or gated fusion only if it improves validation performance."

and requires the outcome to be explained, not just scored:

    "Report branch contribution through ablation and learned fusion weights. This helps
     explain whether the quantum branch, diffusion preprocessing, or adaptive kernels
     contribute meaningfully."

Both halves are implemented. The strategy comparison trains all three heads on identical
cached features, and concatenation is only displaced if a more complex variant actually
beats it on validation. The contribution ablation then re-trains the winner with each
branch zeroed at input - identical architecture, identical parameter count, identical
protocol - so the measured drop is attributable to that branch's *information* rather
than to a smaller model.

Everything here reads the validation split only. Step 16 owns the test set.
"""

import time
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.analysis.base import Analysis
from src.analysis.sweep_utils import collect_feature_predictions, run_feature_trial
from src.data.bt_mri_feature_datamodule import BTMRIFeatureDataModule
from src.models.components.fusion import BRANCH_NAMES, FUSION_STRATEGIES
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class FusionStudy(Analysis):
    """Compare fusion strategies, then ablate the winner's branches.

    :param name: Analysis identifier.
    :param tag: Feature cache to read.
    :param strategies: Strategy names from :data:`FUSION_STRATEGIES`.
    :param epochs: Training epochs per configuration.
    :param lr: Learning rate.
    :param proj_dim: Shared projection width.
    :param dropout: Dropout in the fusion heads.
    :param run_branch_ablation: Also re-train the winner with each branch zeroed.
    :param improvement_threshold: Validation macro-F1 a more complex strategy must exceed
        concatenation by before it is preferred. Guards against adopting attention or
        gating on noise.
    :param seed: Seed shared by every configuration.
    :param accelerator: Lightning accelerator.
    """

    def __init__(
        self,
        name: str = "step13_fusion",
        tag: str = "default",
        strategies: Optional[List[str]] = None,
        epochs: int = 20,
        lr: float = 1e-3,
        proj_dim: int = 64,
        dropout: float = 0.3,
        run_branch_ablation: bool = True,
        improvement_threshold: float = 0.0,
        seed: int = 42,
        accelerator: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self.tag = tag
        self.strategies = list(strategies or FUSION_STRATEGIES.keys())
        self.epochs = epochs
        self.lr = lr
        self.proj_dim = proj_dim
        self.dropout = dropout
        self.run_branch_ablation = run_branch_ablation
        self.improvement_threshold = improvement_threshold
        self.seed = seed
        self.accelerator = accelerator

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Run the strategy comparison and the contribution ablation.

        :param datamodule: Ignored; the study builds its own feature datamodules so each
            configuration starts from a clean one.
        :return: Summary naming the selected strategy and the per-branch contributions.
        """
        reference = self._datamodule()
        reference.prepare_data()
        reference.setup()
        dims = (reference.classical_dim, reference.spatial_dim, reference.quantum_dim)
        log.info(f"Feature widths: classical={dims[0]}, spatial={dims[1]}, quantum={dims[2]}")

        results = self._compare_strategies(dims)
        selected, rationale = self._select(results)
        log.info(f"Selected fusion strategy: {selected} - {rationale}")

        summary: Dict[str, Any] = {
            "tag": self.tag,
            "feature_dims": dict(zip(BRANCH_NAMES, dims)),
            "selected_strategy": selected,
            "selection_rationale": rationale,
            "selection_metric": "validation macro-F1",
            "strategies": results.to_dict("records"),
            "protocol": {"epochs": self.epochs, "lr": self.lr, "seed": self.seed},
        }

        if self.run_branch_ablation:
            ablation = self._ablate_branches(selected, dims)
            summary["branch_ablation"] = ablation.to_dict("records")
            summary["branch_contributions"] = self._contributions(ablation)
            log.info("\n" + ablation.to_string(index=False))

        weights = self._branch_weights(dims, reference)
        if weights is not None:
            summary["mean_branch_weights"] = weights
            log.info(f"Mean learned branch weights (gated fusion): {weights}")

        summary["caveat"] = (
            "All figures are validation-split only. The internal test set is not touched "
            "until Step 16."
        )
        return summary

    # ---------------------------------------------------------------- internals

    def _datamodule(self, zero_branches: Optional[List[str]] = None) -> BTMRIFeatureDataModule:
        """Build a feature datamodule, optionally with branches zeroed.

        :param zero_branches: Branch names to zero.
        :return: The datamodule.
        """
        return BTMRIFeatureDataModule(
            data_dir=self.data_dir,
            tag=self.tag,
            zero_branches=zero_branches,
        )

    def _build(self, strategy: str, dims: tuple) -> torch.nn.Module:
        """Construct a fusion head sized to the cached features.

        :param strategy: Key of :data:`FUSION_STRATEGIES`.
        :param dims: ``(classical_dim, spatial_dim, quantum_dim)``.
        :return: The head.
        :raises ValueError: If the strategy is unknown.
        """
        if strategy not in FUSION_STRATEGIES:
            raise ValueError(
                f"Unknown fusion strategy {strategy!r}. Available: {sorted(FUSION_STRATEGIES)}"
            )
        return FUSION_STRATEGIES[strategy](
            classical_dim=dims[0],
            spatial_dim=dims[1],
            quantum_dim=dims[2],
            proj_dim=self.proj_dim,
            num_classes=4,
            dropout=self.dropout,
        )

    def _compare_strategies(self, dims: tuple) -> pd.DataFrame:
        """Train each strategy on identical features.

        :param dims: Branch widths.
        :return: Results table sorted by validation macro-F1.
        """
        rows: List[Dict[str, Any]] = []
        started = time.perf_counter()

        for position, strategy in enumerate(self.strategies, start=1):
            elapsed = ""
            if position > 1:
                per_item = (time.perf_counter() - started) / (position - 1)
                elapsed = f"  (~{per_item * (len(self.strategies) - position + 1):.0f}s remaining)"
            log.info(f"[{position}/{len(self.strategies)}] fusion={strategy}{elapsed}")

            metrics = run_feature_trial(
                datamodule=self._datamodule(),
                net=self._build(strategy, dims),
                epochs=self.epochs,
                seed=self.seed,
                lr=self.lr,
                accelerator=self.accelerator,
            )
            rows.append({"strategy": strategy, **metrics})
            log.info(
                f"    val macro-F1 {metrics['macro_f1']:.4f} | "
                f"balanced acc {metrics['balanced_accuracy']:.4f} | "
                f"ECE {metrics['ece']:.4f} | params {metrics['trainable_parameters']:,}"
            )
            self.save_table(pd.DataFrame(rows), "step13_fusion_comparison.csv", quiet=True)

        results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
        self.save_table(results, "step13_fusion_comparison.csv")
        self._plot(results)
        log.info("\n" + results.to_string(index=False))
        return results

    def _select(self, results: pd.DataFrame) -> tuple:
        """Apply Step 13's rule: concatenation unless something clearly beats it.

        :param results: Strategy comparison table.
        :return: ``(selected_strategy, rationale)``.
        """
        scores = dict(zip(results["strategy"], results["macro_f1"]))
        leader = str(results.iloc[0]["strategy"])

        if "concat" not in scores:
            return leader, "highest validation macro-F1 (concatenation baseline not run)"

        if leader == "concat":
            return "concat", "concatenation was best; no more complex strategy improved on it"

        margin = scores[leader] - scores["concat"]
        if margin > self.improvement_threshold:
            return leader, (
                f"{leader} beat the concatenation baseline by {margin:+.4f} validation "
                f"macro-F1, exceeding the {self.improvement_threshold} threshold"
            )

        return "concat", (
            f"{leader} led by only {margin:+.4f} validation macro-F1, within the "
            f"{self.improvement_threshold} threshold; the specification adds attention or "
            "gating only when it improves validation performance, so the simpler "
            "concatenation baseline is kept"
        )

    def _ablate_branches(self, strategy: str, dims: tuple) -> pd.DataFrame:
        """Re-train the winner with each branch zeroed in turn.

        :param strategy: Selected fusion strategy.
        :param dims: Branch widths.
        :return: Ablation table.
        """
        conditions: Dict[str, Optional[List[str]]] = {"full": None}
        conditions.update({f"no_{branch}": [branch] for branch in BRANCH_NAMES})

        rows: List[Dict[str, Any]] = []
        for name, zeroed in conditions.items():
            log.info(f"Branch ablation: {name}")
            metrics = run_feature_trial(
                datamodule=self._datamodule(zero_branches=zeroed),
                net=self._build(strategy, dims),
                epochs=self.epochs,
                seed=self.seed,
                lr=self.lr,
                accelerator=self.accelerator,
            )
            rows.append({"condition": name, "zeroed": zeroed or [], **metrics})
            log.info(f"    val macro-F1 {metrics['macro_f1']:.4f}")
            self.save_table(pd.DataFrame(rows), "step13_branch_ablation.csv", quiet=True)

        ablation = pd.DataFrame(rows)
        self.save_table(ablation, "step13_branch_ablation.csv")
        return ablation

    @staticmethod
    def _contributions(ablation: pd.DataFrame) -> Dict[str, float]:
        """Macro-F1 lost when each branch is removed.

        A near-zero or negative value means the branch carries no information the others
        do not already supply - which is a finding, not a failure, and Step 20 revisits it
        for the quantum branch specifically.

        :param ablation: Ablation table.
        :return: Mapping of branch name to macro-F1 drop.
        """
        indexed = ablation.set_index("condition")["macro_f1"]
        if "full" not in indexed:
            return {}

        return {
            branch: float(indexed["full"] - indexed[f"no_{branch}"])
            for branch in BRANCH_NAMES
            if f"no_{branch}" in indexed
        }

    def _branch_weights(self, dims: tuple, datamodule: Any) -> Optional[Dict[str, float]]:
        """Train gated fusion and report its mean learned branch weights.

        Step 13 asks for the learned fusion weights to be reported. Only gated fusion has
        per-branch weights; SE attention's are per channel, so they cannot be read this
        way.

        :param dims: Branch widths.
        :param datamodule: A prepared feature datamodule, used for its loaders.
        :return: Mean weight per branch, or ``None`` if gated fusion was not run.
        """
        if "gated" not in self.strategies:
            return None

        from src.models.feature_fusion_module import FeatureFusionModule
        from src.analysis.sweep_utils import _optimizer_factory

        module = FeatureFusionModule(
            net=self._build("gated", dims),
            optimizer=_optimizer_factory(self.lr, 1e-4),
            num_classes=datamodule.num_classes,
            class_names=datamodule.class_names,
        )

        # Weights from an untrained gate would be meaningless, so reuse a short fit.
        from lightning import Trainer

        trainer = Trainer(
            max_epochs=self.epochs,
            accelerator=self.accelerator,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model=module, datamodule=self._datamodule())

        collected: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        module.eval()
        with torch.no_grad():
            for classical, spatial, quantum, batch_labels in datamodule.val_dataloader():
                outputs = module.net.extract(
                    classical.to(module.device), spatial.to(module.device), quantum.to(module.device)
                )
                collected.append(outputs["branch_weights"].cpu().numpy())
                labels.append(batch_labels.numpy())

        weights = np.concatenate(collected)
        frame = pd.DataFrame(weights, columns=[f"w_{name}" for name in BRANCH_NAMES])
        frame["class_name"] = [datamodule.class_names[i] for i in np.concatenate(labels)]
        self.save_table(frame, "step13_gated_branch_weights.csv")

        return {name: float(weights[:, index].mean()) for index, name in enumerate(BRANCH_NAMES)}

    def _plot(self, results: pd.DataFrame) -> None:
        """Plot validation macro-F1 per strategy.

        :param results: Strategy comparison table.
        """
        figure, axis = plt.subplots(figsize=(7, 4))
        order = results.sort_values("macro_f1")
        colours = ["#B26A16" if s == "concat" else "#4A5A8C" for s in order["strategy"]]

        axis.barh(range(len(order)), order["macro_f1"], color=colours)
        axis.set_yticks(range(len(order)))
        axis.set_yticklabels(order["strategy"])
        axis.set_xlabel("Validation macro-F1")
        axis.set_title("Step 13: fusion strategies (concatenation = baseline)")

        figure.tight_layout()
        figure.savefig(self.figure_path("step13_fusion.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)

    @property
    def data_dir(self) -> str:
        """:return: Project data directory, inferred from the run's output directory."""
        return self._data_dir

    def run(self, datamodule: Any = None, output_dir: Any = None) -> Dict[str, Any]:
        """Capture the data directory from the datamodule before running.

        :param datamodule: Datamodule supplying ``data_dir``.
        :param output_dir: Directory for artefacts.
        :return: Summary metrics.
        """
        self._data_dir = (
            datamodule.hparams.data_dir if datamodule is not None else "data/"
        )
        return super().run(datamodule=datamodule, output_dir=output_dir)
