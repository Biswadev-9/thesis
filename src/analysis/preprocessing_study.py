"""Step 6: select the preprocessing module on evidence.

The specification is prescriptive about how this choice is made:

- sweep diffusion iterations, starting with 5, 10, 15 and 20, and tune kappa;
- keep lambda small, not above 0.25;
- compare against no preprocessing, Wiener filtering, adaptive gamma correction, CLAHE
  and logarithmic transformation;
- *"Do not choose preprocessing based only on visual appearance. Select it using
  validation performance and boundary/texture preservation checks."*

Both criteria are reported: validation macro-F1 from a trained proxy model, and a Sobel
edge-preservation score. Macro-F1 decides the ranking, with the edge score reported
alongside so a candidate that wins by blurring the image away is visible as such.

The reference notebook ran this sweep and selected diffusion - then never applied it to
real training. The winner here is written to the run directory as a materialisation
instruction, and Phase 3 onward trains on the cached mirror it produces. See
docs/DEVIATIONS.md (F4, F5).
"""

import time
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.analysis.base import Analysis
from src.analysis.sweep_utils import run_proxy_trial
from src.data.bt_mri_proxy_datamodule import BTMRIProxyDataModule
from src.data.components.preprocessing import (
    build_recipe,
    default_diffusion_grid,
    edge_preservation_score,
    is_identity_recipe,
)
from src.models.components.backbones import SmallCNN
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PreprocessingStudy(Analysis):
    """Rank preprocessing candidates by validation macro-F1 and edge preservation.

    :param name: Analysis identifier.
    :param diffusion_iterations: Iteration counts to sweep.
    :param diffusion_kappas: Edge thresholds to sweep.
    :param comparators: Non-diffusion candidates required by the specification.
    :param epochs: Training epochs per candidate.
    :param per_class_train: Training images per class in the proxy subset.
    :param per_class_val: Validation images per class in the proxy subset.
    :param image_size: Proxy input size.
    :param lr: Learning rate for the proxy model.
    :param edge_sample_size: Images used to estimate the edge-preservation score.
    :param seed: Seed shared by every candidate.
    :param accelerator: Lightning accelerator.
    """

    def __init__(
        self,
        name: str = "step06_preprocessing",
        diffusion_iterations: Optional[List[int]] = None,
        diffusion_kappas: Optional[List[float]] = None,
        comparators: Optional[List[str]] = None,
        epochs: int = 5,
        per_class_train: int = 200,
        per_class_val: int = 75,
        image_size: int = 128,
        lr: float = 1e-3,
        edge_sample_size: int = 20,
        seed: int = 42,
        accelerator: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self.diffusion_iterations = list(diffusion_iterations or [5, 10, 15, 20])
        self.diffusion_kappas = [float(k) for k in (diffusion_kappas or [15.0, 30.0])]
        self.comparators = list(comparators or ["conventional", "wiener", "gamma", "clahe", "log"])
        self.epochs = epochs
        self.per_class_train = per_class_train
        self.per_class_val = per_class_val
        self.image_size = image_size
        self.lr = lr
        self.edge_sample_size = edge_sample_size
        self.seed = seed
        self.accelerator = accelerator

    def candidate_recipes(self) -> List[str]:
        """Assemble the full candidate list.

        :return: Recipe names, comparators first so the no-preprocessing reference leads.
        """
        diffusion = list(
            default_diffusion_grid(
                iterations=tuple(self.diffusion_iterations), kappas=tuple(self.diffusion_kappas)
            )
        )
        return self.comparators + diffusion

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Run the sweep.

        :param datamodule: Any datamodule carrying ``data_dir``/``raw_subdir``/
            ``split_subpath`` hyperparameters; used only to locate the dataset.
        :return: Summary including the selected recipe.
        """
        datamodule.prepare_data()
        recipes = self.candidate_recipes()
        log.info(f"Step 6: evaluating {len(recipes)} preprocessing candidates")

        edge_images = self._sample_images(datamodule)
        rows: List[Dict[str, Any]] = []
        started = time.perf_counter()

        for position, recipe in enumerate(recipes, start=1):
            log.info(f"[{position}/{len(recipes)}] {recipe}{self._eta(started, position, len(recipes))}")
            filter_fn = build_recipe(recipe)

            proxy = BTMRIProxyDataModule(
                data_dir=datamodule.hparams.data_dir,
                raw_subdir=datamodule.hparams.raw_subdir,
                split_subpath=datamodule.hparams.split_subpath,
                preprocess=None if is_identity_recipe(recipe) else filter_fn,
                per_class_train=self.per_class_train,
                per_class_val=self.per_class_val,
                image_size=self.image_size,
                seed=self.seed,
            )

            metrics = run_proxy_trial(
                datamodule=proxy,
                net=SmallCNN(num_classes=proxy.num_classes),
                epochs=self.epochs,
                seed=self.seed,
                lr=self.lr,
                accelerator=self.accelerator,
            )

            edge_score = self._edge_score(edge_images, filter_fn, recipe)
            rows.append(
                {
                    "recipe": recipe,
                    "family": self._family(recipe),
                    **metrics,
                    "edge_preservation": edge_score,
                }
            )
            log.info(
                f"    macro-F1 {metrics['macro_f1']:.4f} | "
                f"balanced acc {metrics['balanced_accuracy']:.4f} | "
                f"edge preservation {edge_score:.4f}"
            )
            # Rewrite after every candidate: this sweep takes minutes on CPU, and an
            # interrupt should cost the current candidate rather than all of them.
            self.save_table(pd.DataFrame(rows), "step06_preprocessing_comparison.csv", quiet=True)

        results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
        self.save_table(results, "step06_preprocessing_comparison.csv")
        self._plot(results)

        selected = results.iloc[0]
        reference = results[results["recipe"] == "conventional"]
        baseline_f1 = float(reference["macro_f1"].iloc[0]) if not reference.empty else None

        log.info("\n" + results.to_string(index=False))
        log.info(f"Selected preprocessing: {selected['recipe']}")

        summary: Dict[str, Any] = {
            "selected_recipe": str(selected["recipe"]),
            "selected_macro_f1": float(selected["macro_f1"]),
            "selected_balanced_accuracy": float(selected["balanced_accuracy"]),
            "selected_edge_preservation": float(selected["edge_preservation"]),
            "conventional_macro_f1": baseline_f1,
            "improvement_over_conventional": (
                float(selected["macro_f1"]) - baseline_f1 if baseline_f1 is not None else None
            ),
            "n_candidates": len(results),
            "selection_metric": "validation macro-F1 (proxy model)",
            "protocol": {
                "epochs": self.epochs,
                "image_size": self.image_size,
                "per_class_train": self.per_class_train,
                "per_class_val": self.per_class_val,
                "lr": self.lr,
                "seed": self.seed,
                "model": "SmallCNN proxy",
            },
            "ranking": results[["recipe", "macro_f1", "edge_preservation"]].to_dict("records"),
        }

        if not is_identity_recipe(str(selected["recipe"])):
            summary["next_step"] = (
                f"python src/prepare_dataset.py recipe={selected['recipe']}  "
                "# materialise, then train with data.recipe=<same>"
            )
            log.info(f"Next: {summary['next_step']}")

        summary["caveat"] = (
            "Ranking comes from a reduced-scale proxy (SmallCNN, "
            f"{self.image_size}px, {self.per_class_train} images/class). Confirm the top "
            "candidates with the real backbone on the full validation split before "
            "committing - see docs/DEVIATIONS.md (F5)."
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


    @staticmethod
    def _family(recipe: str) -> str:
        """:param recipe: Recipe name.
        :return: Coarse grouping for the results table.
        """
        if recipe.startswith("diffusion"):
            return "anisotropic_diffusion"
        return "none" if is_identity_recipe(recipe) else recipe

    def _sample_images(self, datamodule: Any) -> List[Image.Image]:
        """Draw images for the edge-preservation estimate.

        Training rows only: the score is a property of the filter, and reading validation
        or test images here would be an unnecessary peek.

        :param datamodule: Datamodule locating the dataset.
        :return: Loaded PIL images.
        """
        table = datamodule.load_split_table("train")
        take = min(self.edge_sample_size, len(table))
        picks = table.sample(n=take, random_state=self.seed)

        images: List[Image.Image] = []
        for rel_path in picks["rel_path"]:
            with Image.open(datamodule.image_root / rel_path) as image:
                images.append(image.convert("RGB").copy())
        return images

    def _edge_score(self, images: List[Image.Image], filter_fn: Any, recipe: str) -> float:
        """Mean Sobel edge-map correlation between original and filtered images.

        :param images: Sample images.
        :param filter_fn: The candidate filter.
        :param recipe: Recipe name, used to short-circuit identity recipes.
        :return: Mean correlation; exactly 1.0 for a filter that changes nothing.
        """
        if is_identity_recipe(recipe):
            return 1.0
        scores = [edge_preservation_score(image, filter_fn(image)) for image in images]
        return float(np.mean(scores)) if scores else 0.0

    def _plot(self, results: pd.DataFrame) -> None:
        """Plot the two selection criteria against each other.

        :param results: Ranked results table.
        """
        figure, axes = plt.subplots(1, 2, figsize=(14, 5))

        order = results.sort_values("macro_f1")
        colours = ["#B26A16" if f == "anisotropic_diffusion" else "#4A5A8C" for f in order["family"]]
        axes[0].barh(range(len(order)), order["macro_f1"], color=colours)
        axes[0].set_yticks(range(len(order)))
        axes[0].set_yticklabels(order["recipe"], fontsize=8)
        axes[0].set_xlabel("Validation macro-F1 (proxy)")
        axes[0].set_title("Step 6: preprocessing candidates")

        for family, group in results.groupby("family"):
            axes[1].scatter(
                group["edge_preservation"], group["macro_f1"], label=family, s=45, alpha=0.8
            )
        axes[1].set_xlabel("Edge preservation (Sobel correlation)")
        axes[1].set_ylabel("Validation macro-F1")
        axes[1].set_title("Performance vs boundary preservation")
        axes[1].legend(fontsize=8)

        figure.tight_layout()
        figure.savefig(self.figure_path("step06_preprocessing.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)
