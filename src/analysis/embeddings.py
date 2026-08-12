"""Step 10: feature embedding export and separability analysis.

Step 10 closes with a requirement that is easy to skip and hard to reconstruct later:

    "Save the intermediate feature embeddings for later feature separability analysis
     using t-SNE or UMAP."

The embeddings are also reused by Step 20, which compares separability with and without
the quantum branch. Exporting them once, from the trained branch, means neither step has
to re-run a backbone.

Separability is quantified with a silhouette score computed in the **original feature
space**, not on the 2-D projection. t-SNE and UMAP do not preserve global distances, so a
silhouette taken on their output measures the projection rather than the features.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from src.analysis.base import Analysis
from src.utils import RankedLogger
from src.utils.checkpoints import find_checkpoint, load_module

log = RankedLogger(__name__, rank_zero_only=True)


class EmbeddingAnalysis(Analysis):
    """Export branch embeddings and measure how separable the classes are.

    :param name: Analysis identifier.
    :param ckpt_path: Checkpoint, or a run directory containing one.
    :param model_cfg: Hydra config used to rebuild the model before loading weights.
    :param splits: Splits to export.
    :param projection: ``tsne``, ``umap`` or ``none``.
    :param max_points: Cap on points sent to the projection, which scales poorly.
    :param perplexity: t-SNE perplexity.
    :param seed: Seed for sampling and projection.
    :param accelerator: ``auto``, ``cpu`` or ``gpu``.
    """

    def __init__(
        self,
        name: str = "step10_embeddings",
        ckpt_path: Optional[str] = None,
        model_cfg: Optional[Any] = None,
        splits: Optional[List[str]] = None,
        projection: str = "tsne",
        max_points: int = 1500,
        perplexity: float = 30.0,
        seed: int = 42,
        accelerator: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self.ckpt_path = ckpt_path
        self.model_cfg = model_cfg
        self.splits = list(splits or ["train", "val", "test"])
        self.projection = projection
        self.max_points = max_points
        self.perplexity = perplexity
        self.seed = seed
        self.accelerator = accelerator

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Export embeddings and analyse separability.

        :param datamodule: Datamodule supplying the splits.
        :return: Summary including the silhouette score per split.
        :raises ValueError: If no checkpoint was configured.
        """
        if self.ckpt_path is None:
            raise ValueError(
                "analysis.ckpt_path is required - point it at a Step 10 run directory or "
                "checkpoint, e.g. logs/train/runs/<timestamp>"
            )

        checkpoint = Path(self.ckpt_path)
        if checkpoint.is_dir():
            checkpoint = find_checkpoint(checkpoint)

        device = self._device()
        module = load_module(checkpoint, model_cfg=self.model_cfg).to(device)

        datamodule.prepare_data()
        datamodule.setup()

        loaders = {
            "train": datamodule.train_dataloader,
            "val": datamodule.val_dataloader,
            "test": datamodule.test_dataloader,
        }

        exported: Dict[str, Any] = {}
        summary: Dict[str, Any] = {"checkpoint": str(checkpoint), "splits": {}}

        for split in self.splits:
            features, labels = self._extract(module, loaders[split](), device)
            exported[f"{split}_features"] = features
            exported[f"{split}_labels"] = labels

            score = self._silhouette(features, labels)
            summary["splits"][split] = {
                "n_samples": int(len(labels)),
                "feature_dim": int(features.shape[1]),
                "silhouette": score,
            }
            log.info(f"{split}: {features.shape} embeddings, silhouette {score}")

        destination = self.output_dir / "embeddings.npz"
        np.savez_compressed(destination, **exported)
        log.info(f"[{self.name}] wrote {destination.name}")

        if self.projection != "none" and "test" in self.splits:
            self._plot(exported["test_features"], exported["test_labels"], datamodule.class_names)

        summary["embeddings_file"] = str(destination)
        summary["projection"] = self.projection
        summary["note"] = (
            "Silhouette is computed in the original feature space, not on the 2-D "
            "projection - t-SNE and UMAP do not preserve global distances."
        )
        return summary

    # ---------------------------------------------------------------- internals

    def _device(self) -> torch.device:
        """:return: The device to run extraction on."""
        if self.accelerator == "cpu":
            return torch.device("cpu")
        if self.accelerator == "gpu" or torch.cuda.is_available():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    @torch.no_grad()
    def _extract(self, module: Any, loader: Any, device: torch.device) -> tuple:
        """Run the branch over a loader and collect its pre-classifier features.

        :param module: Trained module.
        :param loader: Dataloader.
        :param device: Device to run on.
        :return: ``(features, labels)`` as numpy arrays.
        """
        module.eval()
        features: List[np.ndarray] = []
        labels: List[np.ndarray] = []

        for images, targets in loader:
            outputs = module.net.extract(images.to(device))
            features.append(outputs["features"].cpu().numpy())
            labels.append(targets.numpy())

        return np.concatenate(features), np.concatenate(labels)

    def _silhouette(self, features: np.ndarray, labels: np.ndarray) -> Optional[float]:
        """:param features: Embeddings.
        :param labels: Class labels.
        :return: Silhouette score, or ``None`` if undefined for this data.
        """
        if len(np.unique(labels)) < 2 or len(labels) < 3:
            return None
        return float(silhouette_score(features, labels))

    def _plot(self, features: np.ndarray, labels: np.ndarray, class_names: List[str]) -> None:
        """Project the embeddings to 2-D and plot them by class.

        :param features: Embeddings.
        :param labels: Class labels.
        :param class_names: Class names ordered by label index.
        """
        if len(labels) > self.max_points:
            index = np.random.RandomState(self.seed).choice(
                len(labels), self.max_points, replace=False
            )
            features, labels = features[index], labels[index]

        projected = self._project(features)
        if projected is None:
            return

        figure, axis = plt.subplots(figsize=(7.5, 6.5))
        for class_index, class_name in enumerate(class_names):
            mask = labels == class_index
            if mask.any():
                axis.scatter(
                    projected[mask, 0], projected[mask, 1], s=12, alpha=0.65, label=class_name
                )

        axis.set_title(f"Step 10: classical branch embeddings ({self.projection.upper()})")
        axis.set_xlabel("dimension 1")
        axis.set_ylabel("dimension 2")
        axis.legend(fontsize=9)
        figure.tight_layout()
        figure.savefig(
            self.figure_path(f"step10_embeddings_{self.projection}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)

    def _project(self, features: np.ndarray) -> Optional[np.ndarray]:
        """:param features: Embeddings.
        :return: 2-D projection, or ``None`` if the projector is unavailable.
        """
        if self.projection == "umap":
            try:
                import umap
            except ImportError:
                log.warning("umap-learn is not installed; falling back to t-SNE")
            else:
                return umap.UMAP(n_components=2, random_state=self.seed).fit_transform(features)

        # Perplexity must stay below the sample count or sklearn raises.
        perplexity = min(self.perplexity, max(5.0, (len(features) - 1) / 3.0))
        return TSNE(n_components=2, random_state=self.seed, perplexity=perplexity).fit_transform(
            features
        )
