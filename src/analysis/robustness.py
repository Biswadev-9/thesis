"""Step 18: robustness under controlled image degradation.

    "Robustness testing checks whether the model is stable under realistic image
     degradation. ... Compare robustness of the proposed model against strong CNN and
     Transformer baselines. Report whether diffusion preprocessing improves robustness
     under noisy inputs."

Three questions, answered together:

1. **How far does each model fall?** Every model is measured on the clean test split
   first, then on each degradation, and reported as an absolute score *and* as a drop from
   its own clean baseline. The drop is what matters: a model that starts lower can still
   be more robust, and reporting only absolute scores would hide that.
2. **Is the proposed model more robust than the baselines?** The CNN and Transformer
   baselines run through the identical degraded loaders.
3. **Does diffusion preprocessing help under noise?** Configure the same model twice, once
   with ``preprocess: null`` and once with the diffusion recipe. The filter is applied
   *after* the degradation, so it actually sees the noise - see ``DegradedDataset``.

**On the shipped "Challenging Datasets".** The archive includes pre-degraded blurred,
noisy and motion-artefact directories, which look like a ready-made robustness set. They
are deliberately not used: their filenames (``bilateral_glioma (1).jpg``) do not
correspond to the primary set's (``BT-MRI GL Train (1).jpg``), and there are 3,354 of them
against 7,023 originals, so it is impossible to tell which are degraded copies of
*training* images. Evaluating on them could silently score the model on its own training
data. The synthetic degradations here are applied to the held-out test split only, where
provenance is known.
"""

import time
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.analysis.base import Analysis
from src.data.components.degradations import DEGRADATION_SWEEP
from src.data.components.preprocessing import build_recipe, is_identity_recipe
from src.data.eval_datamodules import DegradedTestDataModule
from src.models.full_pipeline import load_full_pipeline, predict
from src.utils import RankedLogger
from src.utils.checkpoints import find_checkpoint, load_module

log = RankedLogger(__name__, rank_zero_only=True)


class RobustnessStudy(Analysis):
    """Sweep degradations across the proposed model and the baselines.

    :param name: Analysis identifier.
    :param models: Mapping of display name to a spec. Each spec needs ``kind``
        (``pipeline`` or ``simple``); ``pipeline`` specs need the three checkpoints and
        configs, ``simple`` specs need ``ckpt`` and ``model_cfg``. An optional
        ``preprocess`` names a Step 6 recipe applied after degradation.
    :param categories: Degradation categories to sweep; defaults to all of them.
    :param image_size: Input size; must match training.
    :param normalize: Step 5 intensity treatment; must match training.
    :param batch_size: Evaluation batch size.
    :param limit_batches: Evaluate at most this many batches per condition, for smoke runs.
    :param accelerator: ``auto``, ``cpu`` or ``gpu``.
    """

    def __init__(
        self,
        name: str = "step18_robustness",
        models: Optional[Dict[str, Dict[str, Any]]] = None,
        categories: Optional[List[str]] = None,
        image_size: int = 224,
        normalize: str = "imagenet",
        batch_size: int = 32,
        limit_batches: Optional[int] = None,
        accelerator: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self.models = dict(models or {})
        self.categories = list(categories or DEGRADATION_SWEEP.keys())
        self.image_size = image_size
        self.normalize = normalize
        self.batch_size = batch_size
        self.limit_batches = limit_batches
        self.accelerator = accelerator

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Run the degradation sweep.

        :param datamodule: Datamodule supplying the data directory and class names.
        :return: Summary with per-condition scores, drops, and the diffusion verdict.
        :raises ValueError: If no models were configured.
        """
        if not self.models:
            raise ValueError(
                "analysis.models is empty. Configure at least the proposed model and one "
                "baseline - Step 18 requires a comparison against CNN and Transformer "
                "baselines."
            )

        data_dir = datamodule.hparams.data_dir
        class_names = datamodule.class_names
        device = torch.device(
            "cuda" if (self.accelerator != "cpu" and torch.cuda.is_available()) else "cpu"
        )

        loaded = {name: self._load(spec, device) for name, spec in self.models.items()}
        conditions = self._conditions()
        log.info(
            f"Step 18: {len(loaded)} models x {len(conditions)} conditions "
            f"= {len(loaded) * len(conditions)} evaluations"
        )

        rows: List[Dict[str, Any]] = []
        started = time.perf_counter()
        total = len(loaded) * len(conditions)
        done = 0

        for category, severity, degradation in conditions:
            for model_name, model in loaded.items():
                done += 1
                eta = ""
                if done > 1:
                    per_item = (time.perf_counter() - started) / (done - 1)
                    eta = f"  (~{per_item * (total - done + 1) / 60:.0f}m remaining)"
                log.info(f"[{done}/{total}] {model_name} | {category} | {severity}{eta}")

                score = self._evaluate(
                    model=model,
                    spec=self.models[model_name],
                    data_dir=data_dir,
                    degradation=degradation,
                    device=device,
                )
                rows.append(
                    {
                        "model": model_name,
                        "category": category,
                        "severity": severity,
                        "macro_f1": score,
                    }
                )
                log.info(f"    macro-F1 {score:.4f}")
                self.save_table(pd.DataFrame(rows), "step18_robustness.csv", quiet=True)

        results = self._add_drops(pd.DataFrame(rows))
        self.save_table(results, "step18_robustness.csv")
        self._plot(results)
        log.info("\n" + results.to_string(index=False))

        return {
            "models": list(loaded),
            "categories": self.categories,
            "clean_scores": self._clean_scores(results),
            "mean_drop_by_model": self._mean_drops(results),
            "most_damaging": self._most_damaging(results),
            "diffusion_verdict": self._diffusion_verdict(results),
            "results": results.to_dict("records"),
            "excluded_data_note": (
                "The archive's pre-degraded 'Challenging Datasets' were not used: their "
                "filenames do not correspond to the primary set, so degraded copies of "
                "training images cannot be excluded."
            ),
        }

    # ---------------------------------------------------------------- internals

    def _conditions(self) -> List[tuple]:
        """:return: ``(category, severity, degradation)`` triples, clean first."""
        conditions = []
        for category in self.categories:
            for severity, degradation in DEGRADATION_SWEEP[category]:
                conditions.append((category, severity, degradation))
        return conditions

    def _load(self, spec: Dict[str, Any], device: torch.device) -> torch.nn.Module:
        """Load one model from its spec.

        :param spec: Model spec.
        :param device: Device to place it on.
        :return: A module taking images and returning logits.
        :raises ValueError: If ``kind`` is unknown.
        """
        kind = spec.get("kind", "simple")

        if kind == "pipeline":
            return load_full_pipeline(
                classical_ckpt=spec["classical_ckpt"],
                quantum_ckpt=spec["quantum_ckpt"],
                fusion_ckpt=spec["fusion_ckpt"],
                classical_model=spec["classical_model"],
                quantum_model=spec["quantum_model"],
                fusion_model=spec["fusion_model"],
                device=device,
            )

        if kind == "simple":
            from pathlib import Path

            candidate = Path(spec["ckpt"])
            checkpoint = find_checkpoint(candidate) if candidate.is_dir() else candidate
            module = load_module(checkpoint, model_cfg=spec["model_cfg"])
            return module.net.to(device).eval()

        raise ValueError(f"Unknown model kind {kind!r}; expected 'pipeline' or 'simple'")

    def _evaluate(
        self,
        model: torch.nn.Module,
        spec: Dict[str, Any],
        data_dir: str,
        degradation: Any,
        device: torch.device,
    ) -> float:
        """Score one model under one condition.

        :param model: Loaded model.
        :param spec: Its spec, which may name a preprocessing recipe.
        :param data_dir: Project data directory.
        :param degradation: Degradation, or ``None`` for clean.
        :param device: Device to run on.
        :return: Macro-F1 on the degraded test split.
        """
        recipe = spec.get("preprocess")
        preprocess = (
            build_recipe(recipe) if recipe and not is_identity_recipe(recipe) else None
        )

        datamodule = DegradedTestDataModule(
            data_dir=data_dir,
            degradation=degradation,
            preprocess=preprocess,
            image_size=self.image_size,
            normalize=self.normalize,
            batch_size=self.batch_size,
        )
        datamodule.setup()

        loader = datamodule.test_dataloader()
        if self.limit_batches:
            from itertools import islice

            loader = list(islice(loader, self.limit_batches))

        outputs = predict(model, loader, device)
        return float(
            f1_score(outputs["y_true"], outputs["y_pred"], average="macro", zero_division=0)
        )

    @staticmethod
    def _add_drops(results: pd.DataFrame) -> pd.DataFrame:
        """Add each row's drop from that model's own clean score.

        The drop is the robustness measure. Absolute scores conflate robustness with
        baseline accuracy: a weaker model that degrades gently is more robust, and only
        the drop shows it.

        :param results: Raw results.
        :return: Results with a ``drop_from_clean`` column.
        """
        clean = (
            results[results["category"] == "clean"].set_index("model")["macro_f1"].to_dict()
        )
        results = results.copy()
        results["drop_from_clean"] = [
            clean.get(model, np.nan) - score
            for model, score in zip(results["model"], results["macro_f1"])
        ]
        return results

    @staticmethod
    def _clean_scores(results: pd.DataFrame) -> Dict[str, float]:
        """:param results: Results table.
        :return: Each model's clean macro-F1.
        """
        clean = results[results["category"] == "clean"]
        return {row["model"]: float(row["macro_f1"]) for _, row in clean.iterrows()}

    @staticmethod
    def _mean_drops(results: pd.DataFrame) -> Dict[str, float]:
        """:param results: Results table.
        :return: Each model's mean drop across all degraded conditions.
        """
        degraded = results[results["category"] != "clean"]
        return {
            str(model): float(group["drop_from_clean"].mean())
            for model, group in degraded.groupby("model")
        }

    @staticmethod
    def _most_damaging(results: pd.DataFrame) -> Dict[str, Any]:
        """:param results: Results table.
        :return: The degradation category that hurts most, averaged over models.
        """
        degraded = results[results["category"] != "clean"]
        if degraded.empty:
            return {}
        by_category = degraded.groupby("category")["drop_from_clean"].mean().sort_values()
        return {
            "worst_category": str(by_category.index[-1]),
            "worst_mean_drop": float(by_category.iloc[-1]),
            "by_category": {str(k): float(v) for k, v in by_category.items()},
        }

    def _diffusion_verdict(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Answer Step 18's diffusion question, if the configuration allows it.

        Requires two entries differing only in ``preprocess``. Without that pair the
        question is unanswerable, and saying so is better than implying an answer.

        :param results: Results table.
        :return: Per-noise-level comparison, or a note explaining why it was skipped.
        """
        with_diffusion = [
            name for name, spec in self.models.items() if spec.get("preprocess")
        ]
        without = [name for name, spec in self.models.items() if not spec.get("preprocess")]

        if not with_diffusion or not without:
            return {
                "answered": False,
                "reason": (
                    "Needs two model entries differing only in 'preprocess' - one with a "
                    "diffusion recipe and one without. Configure both to answer whether "
                    "diffusion preprocessing improves robustness under noise."
                ),
            }

        treated, control = with_diffusion[0], without[0]
        noise = results[results["category"] == "gaussian_noise"]

        comparison = []
        for severity in noise["severity"].unique():
            subset = noise[noise["severity"] == severity].set_index("model")["macro_f1"]
            if treated in subset and control in subset:
                comparison.append(
                    {
                        "severity": str(severity),
                        "with_diffusion": float(subset[treated]),
                        "without_diffusion": float(subset[control]),
                        "delta": float(subset[treated] - subset[control]),
                    }
                )

        helps_at = [row["severity"] for row in comparison if row["delta"] > 0]
        return {
            "answered": True,
            "with_diffusion_model": treated,
            "without_diffusion_model": control,
            "per_noise_level": comparison,
            "helps_at_severities": helps_at,
            "verdict": (
                "diffusion improved robustness at every noise level tested"
                if len(helps_at) == len(comparison) and comparison
                else "diffusion improved robustness at some noise levels only"
                if helps_at
                else "diffusion did not improve robustness at any noise level tested"
            ),
        }

    def _plot(self, results: pd.DataFrame) -> None:
        """Plot macro-F1 against severity, one panel per degradation category.

        :param results: Results table.
        """
        categories = [c for c in self.categories if c != "clean"]
        if not categories:
            return

        figure, axes = plt.subplots(
            1, len(categories), figsize=(4 * len(categories), 4), sharey=True
        )
        axes = np.atleast_1d(axes)

        for axis, category in zip(axes, categories):
            subset = results[results["category"] == category]
            order = [s for s, _ in DEGRADATION_SWEEP[category]]

            for model in subset["model"].unique():
                model_rows = subset[subset["model"] == model].set_index("severity")
                values = [model_rows.loc[s, "macro_f1"] for s in order if s in model_rows.index]
                axis.plot(order[: len(values)], values, marker="o", label=model)

            axis.set_title(category)
            axis.set_xlabel("severity")
            axis.tick_params(axis="x", rotation=30)

        axes[0].set_ylabel("macro-F1")
        axes[0].legend(fontsize=8)
        figure.suptitle("Step 18: robustness under degradation")
        figure.tight_layout()
        figure.savefig(self.figure_path("step18_robustness.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)
