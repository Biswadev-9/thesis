"""Step 11: report the learned scale weights and test them against tumour morphology.

Step 11 does not stop at accuracy:

    "Report the learned scale weights for example images to show whether the model adapts
     to tumor morphology."

This analysis does both halves. It visualises the per-pixel gate weights over example
images from each class, and it tests quantitatively whether the weighting varies with
lesion extent - Spearman correlation between the proxy tumour area of an image and the
mean gate weight *inside* that region, computed per class as well as pooled.

Three limitations belong in the write-up, and are recorded in the summary so they cannot
be lost:

1. **The tumour mask is a proxy.** This dataset ships no segmentation masks, so an Otsu
   threshold over in-brain intensity stands in for the lesion. It assumes tumours are
   relatively hyperintense - reasonable for these sequences, but not a segmentation.
2. **The three weights are not independent.** They are a softmax, so they sum to 1 at
   every pixel. One path rising forces the others down; a negative correlation on one path
   is partly a mathematical consequence of positive correlations on the others, not
   separate evidence.
3. **Pooling across classes can manufacture a correlation** if classes differ in both
   lesion size and gate behaviour. Correlations are therefore reported per class as well,
   and only a direction that holds *within* classes should be claimed.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from skimage.filters import threshold_otsu

from src.analysis.base import Analysis
from src.data.components.transforms import build_transform
from src.models.components.multiscale import PATH_LABELS
from src.utils import RankedLogger
from src.utils.checkpoints import find_checkpoint, load_module

log = RankedLogger(__name__, rank_zero_only=True)

#: No-tumor is excluded from the correlation: there is no lesion whose extent could
#: correlate with anything, so including it would test a different question.
TUMOUR_CLASSES = ("Glioma", "Meningioma", "Pituitary")


class GateMorphologyAnalysis(Analysis):
    """Visualise gate weights and correlate them with proxy tumour extent.

    :param name: Analysis identifier.
    :param ckpt_path: Checkpoint, or a run directory containing one. Must be a
        spatial-gate arm - the analysis needs per-pixel weights.
    :param model_cfg: Hydra config used to rebuild the model before loading weights.
    :param samples_per_class: Images per class for the correlation.
    :param examples_per_class: Images per class shown in the figure.
    :param background_threshold: Intensity at or below which a pixel is background.
    :param seed: Sampling seed.
    :param accelerator: ``auto``, ``cpu`` or ``gpu``.
    """

    def __init__(
        self,
        name: str = "step11_gate_morphology",
        ckpt_path: Optional[str] = None,
        model_cfg: Optional[Any] = None,
        samples_per_class: int = 40,
        examples_per_class: int = 1,
        background_threshold: int = 10,
        seed: int = 42,
        accelerator: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self.ckpt_path = ckpt_path
        self.model_cfg = model_cfg
        self.samples_per_class = samples_per_class
        self.examples_per_class = examples_per_class
        self.background_threshold = background_threshold
        self.seed = seed
        self.accelerator = accelerator

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Run the analysis.

        :param datamodule: Datamodule locating the dataset.
        :return: Summary with pooled and per-class correlations.
        :raises ValueError: If no checkpoint is configured, or the model has no spatial gate.
        """
        if self.ckpt_path is None:
            raise ValueError(
                "analysis.ckpt_path is required - point it at the Step 11 spatial-gate "
                "(arm6) run directory or checkpoint"
            )

        checkpoint = Path(self.ckpt_path)
        if checkpoint.is_dir():
            checkpoint = find_checkpoint(checkpoint)

        device = torch.device("cuda" if (self.accelerator != "cpu" and torch.cuda.is_available()) else "cpu")
        module = load_module(checkpoint, model_cfg=self.model_cfg).to(device).eval()

        datamodule.prepare_data()
        table = datamodule.load_split_table("test")
        image_root = Path(datamodule.image_root)
        transform = build_transform(
            image_size=datamodule.hparams.image_size,
            normalize=datamodule.hparams.normalize,
            augment=False,
        )

        rows = self._collect(module, table, image_root, transform, device)
        if not rows:
            raise ValueError("No samples produced gate maps - is this a spatial-gate arm?")

        measurements = pd.DataFrame(rows)
        self.save_table(measurements, "step11_gate_morphology.csv")

        summary: Dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "n_samples": int(len(measurements)),
            "pooled": self._correlations(measurements),
            "per_class": {
                class_name: self._correlations(measurements[measurements["class_name"] == class_name])
                for class_name in TUMOUR_CLASSES
                if (measurements["class_name"] == class_name).any()
            },
            "mean_weights": {
                label: float(measurements[f"weight_{index}"].mean())
                for index, label in enumerate(PATH_LABELS)
            },
            "limitations": [
                "Tumour region is an Otsu intensity proxy, not a segmentation mask; this "
                "dataset ships none.",
                "The three path weights are a softmax and sum to 1, so they are not "
                "independent - a negative correlation on one path is partly the "
                "arithmetic consequence of positive correlations on the others.",
                "Pooled correlations can be driven by between-class differences; only "
                "directions that also hold within each class should be claimed.",
            ],
        }

        self._plot_examples(module, table, image_root, transform, device, datamodule)
        self._log_summary(summary)
        return summary

    # ---------------------------------------------------------------- internals

    @torch.no_grad()
    def _gate_maps(self, module: Any, image: Image.Image, transform: Any, device: torch.device):
        """Extract gate weights for one image, upsampled to the image's own resolution.

        :param module: Trained module.
        :param image: Source image.
        :param transform: Evaluation transform.
        :param device: Device to run on.
        :return: Weight maps ``(3, H, W)``, or ``None`` if the model exposes none.
        """
        tensor = transform(image).unsqueeze(0).to(device)
        outputs = module.net.extract(tensor)

        maps = outputs.get("gate_maps")
        if maps is None or maps.ndim != 4:
            return None

        # The gate acts at stem resolution (input/4); upsample so weights can be indexed
        # by the tumour proxy mask, which is computed at full resolution.
        resized = F.interpolate(
            maps, size=(image.height, image.width), mode="bilinear", align_corners=False
        )
        return resized.squeeze(0).cpu().numpy()

    def _tumour_proxy(self, image: Image.Image) -> np.ndarray:
        """Otsu-threshold the in-brain region as a stand-in for the lesion.

        :param image: Source image.
        :return: Boolean mask of hyperintense in-brain pixels.
        """
        grey = np.asarray(image.convert("L")).astype(np.float32)
        in_brain = grey > self.background_threshold
        if not in_brain.any():
            return np.zeros_like(grey, dtype=bool)
        return (grey > threshold_otsu(grey[in_brain])) & in_brain

    def _collect(
        self, module: Any, table: pd.DataFrame, image_root: Path, transform: Any, device: torch.device
    ) -> List[Dict[str, Any]]:
        """Measure gate weights inside the proxy lesion for a sample of test images.

        :param module: Trained module.
        :param table: Test split table.
        :param image_root: Directory relative paths resolve against.
        :param transform: Evaluation transform.
        :param device: Device to run on.
        :return: One record per usable image.
        """
        rows: List[Dict[str, Any]] = []

        for class_name in TUMOUR_CLASSES:
            pool = table[table["class_name"] == class_name]
            take = min(self.samples_per_class, len(pool))
            if take == 0:
                continue

            for rel_path in pool.sample(n=take, random_state=self.seed)["rel_path"]:
                with Image.open(image_root / rel_path) as handle:
                    image = handle.convert("RGB").copy()

                maps = self._gate_maps(module, image, transform, device)
                if maps is None:
                    continue

                mask = self._tumour_proxy(image)
                if not mask.any():
                    continue

                record = {
                    "class_name": class_name,
                    "rel_path": rel_path,
                    "tumour_area_fraction": float(mask.sum() / mask.size),
                }
                for index in range(maps.shape[0]):
                    record[f"weight_{index}"] = float(maps[index][mask].mean())
                rows.append(record)

        return rows

    @staticmethod
    def _correlations(measurements: pd.DataFrame) -> Dict[str, Any]:
        """Spearman correlation of proxy lesion extent against each path's weight.

        :param measurements: Collected measurements.
        :return: Per-path ``rho``, ``p_value`` and ``n``.
        """
        results: Dict[str, Any] = {}
        if len(measurements) < 5:
            return {"note": f"too few samples (n={len(measurements)}) for a correlation"}

        for index, label in enumerate(PATH_LABELS):
            rho, p_value = spearmanr(
                measurements["tumour_area_fraction"], measurements[f"weight_{index}"]
            )
            results[label] = {
                "rho": float(rho),
                "p_value": float(p_value),
                "n": int(len(measurements)),
            }
        return results

    def _plot_examples(
        self,
        module: Any,
        table: pd.DataFrame,
        image_root: Path,
        transform: Any,
        device: torch.device,
        datamodule: Any,
    ) -> None:
        """Show the gate's weight maps over one example image per class.

        This is the figure Step 11 asks for directly.

        :param module: Trained module.
        :param table: Test split table.
        :param image_root: Directory relative paths resolve against.
        :param transform: Evaluation transform.
        :param device: Device to run on.
        :param datamodule: Datamodule supplying class names.
        """
        examples: List[tuple] = []
        for class_name in datamodule.class_names:
            pool = table[table["class_name"] == class_name]
            take = min(self.examples_per_class, len(pool))
            for rel_path in pool.sample(n=take, random_state=self.seed)["rel_path"]:
                examples.append((class_name, rel_path))

        if not examples:
            return

        figure, axes = plt.subplots(len(examples), 4, figsize=(13, 3.2 * len(examples)))
        axes = np.atleast_2d(axes)
        image_handle = None

        for row, (class_name, rel_path) in enumerate(examples):
            with Image.open(image_root / rel_path) as handle:
                image = handle.convert("RGB").copy()

            maps = self._gate_maps(module, image, transform, device)
            axes[row, 0].imshow(image)
            axes[row, 0].set_title(f"{class_name}\noriginal", fontsize=9)
            axes[row, 0].axis("off")

            for path_index in range(3):
                axis = axes[row, path_index + 1]
                axis.axis("off")
                if maps is None:
                    continue
                # Shared 0-1 scale: the weights are a softmax over three paths, so a
                # per-panel scale would make every path look equally dominant.
                image_handle = axis.imshow(maps[path_index], cmap="inferno", vmin=0, vmax=1)
                axis.set_title(PATH_LABELS[path_index], fontsize=9)

        if image_handle is not None:
            figure.colorbar(
                image_handle, ax=axes.ravel().tolist(), shrink=0.6, label="gate weight"
            )
        figure.suptitle("Step 11: learned per-pixel scale weights", fontsize=13)
        figure.savefig(
            self.figure_path("step11_gate_weight_maps.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(figure)

    @staticmethod
    def _log_summary(summary: Dict[str, Any]) -> None:
        """:param summary: Computed summary, logged in readable form."""
        log.info("Pooled Spearman correlation (proxy tumour area vs gate weight):")
        for label, values in summary["pooled"].items():
            if isinstance(values, dict) and "rho" in values:
                log.info(
                    f"  {label:<20} rho={values['rho']:+.3f} p={values['p_value']:.4f} "
                    f"(n={values['n']})"
                )
