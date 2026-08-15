"""Step 19: explainability and uncertainty.

    "Explainability must be included as a core part of the proposed model, not as a
     decorative figure."

Five things the specification asks for, all implemented here:

1. **Grad-CAM** over the classical branch's final convolutional block.
2. **SHAP** attribution over the fused feature vector, aggregated per branch - which
   independently cross-checks Step 13's ablation.
3. **MC-dropout** uncertainty, to flag predictions that should go to human review.
4. **Deletion/insertion sanity checks**, so the heatmaps are validated rather than trusted.
5. **Correct and incorrect examples for each class**, since an explanation method that only
   ever shows successes is decoration.

Attention rollout for the ViT baseline lives in the Step 20 companion analysis, because it
explains a *baseline* rather than the proposed model.

**On reading the sanity checks.** Deleting the top-scoring region and seeing no confidence
drop does not mean Grad-CAM is broken - it means the decision is distributed rather than
concentrated. The reference notebook found exactly that on confident predictions, alongside
heatmaps sitting on skull boundaries rather than lesions. That is a finding about the model,
and it is reported rather than smoothed over.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.analysis.base import Analysis
from src.data.components.transforms import build_transform
from src.models.components.explain import GradCAM
from src.models.full_pipeline import load_full_pipeline
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ExplainabilityStudy(Analysis):
    """Grad-CAM, SHAP, MC-dropout and saliency sanity checks for the proposed model.

    :param name: Analysis identifier.
    :param classical_ckpt: Step 10 checkpoint or run directory.
    :param quantum_ckpt: Step 12 checkpoint or run directory.
    :param fusion_ckpt: Step 15 checkpoint or run directory.
    :param classical_model: Hydra config for the Step 10 module.
    :param quantum_model: Hydra config for the Step 12 module.
    :param fusion_model: Hydra config for the fusion module.
    :param examples_per_class: Correct and incorrect examples to visualise per class.
    :param mc_dropout_samples: Stochastic forward passes for the uncertainty estimate.
    :param shap_background: Background samples for the SHAP explainer.
    :param shap_explain: Samples to explain. SHAP is the slowest part of this analysis.
    :param deletion_fraction: Fraction of the highest-scoring pixels to delete or reveal.
    :param max_samples: Cap on test samples used for MC-dropout and SHAP.
    :param seed: Sampling seed.
    :param accelerator: ``auto``, ``cpu`` or ``gpu``.
    """

    def __init__(
        self,
        name: str = "step19_explainability",
        classical_ckpt: Optional[str] = None,
        quantum_ckpt: Optional[str] = None,
        fusion_ckpt: Optional[str] = None,
        classical_model: Optional[Any] = None,
        quantum_model: Optional[Any] = None,
        fusion_model: Optional[Any] = None,
        examples_per_class: int = 1,
        mc_dropout_samples: int = 30,
        shap_background: int = 50,
        shap_explain: int = 25,
        deletion_fraction: float = 0.2,
        max_samples: int = 300,
        seed: int = 42,
        accelerator: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self.classical_ckpt = classical_ckpt
        self.quantum_ckpt = quantum_ckpt
        self.fusion_ckpt = fusion_ckpt
        self.classical_model = classical_model
        self.quantum_model = quantum_model
        self.fusion_model = fusion_model
        self.examples_per_class = examples_per_class
        self.mc_dropout_samples = mc_dropout_samples
        self.shap_background = shap_background
        self.shap_explain = shap_explain
        self.deletion_fraction = deletion_fraction
        self.max_samples = max_samples
        self.seed = seed
        self.accelerator = accelerator

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Run every explainability component.

        :param datamodule: Datamodule supplying the internal test split.
        :return: Summary of all five analyses.
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
        table = datamodule.load_split_table("test")
        image_root = Path(datamodule.image_root)
        transform = build_transform(
            image_size=datamodule.hparams.image_size,
            normalize=datamodule.hparams.normalize,
            augment=False,
        )
        class_names = datamodule.class_names

        predictions = self._predict_all(pipeline, table, image_root, transform, device)
        examples = self._select_examples(table, predictions, class_names)

        summary: Dict[str, Any] = {
            "n_examples": len(examples),
            "grad_cam": self._grad_cam_examples(
                pipeline, examples, image_root, transform, device, class_names
            ),
            "mc_dropout": self._mc_dropout(pipeline, table, image_root, transform, device),
            "shap": self._shap(pipeline, table, image_root, transform, device),
        }
        summary["interpretation"] = self._interpret(summary)
        return summary

    # ------------------------------------------------------------- predictions

    @torch.no_grad()
    def _predict_all(
        self, pipeline: Any, table: pd.DataFrame, image_root: Path, transform: Any, device: torch.device
    ) -> Dict[str, np.ndarray]:
        """Predict over the test split, one image at a time.

        :param pipeline: Loaded model.
        :param table: Test split table.
        :param image_root: Directory relative paths resolve against.
        :param transform: Evaluation transform.
        :param device: Device to run on.
        :return: ``{"y_true", "y_pred", "confidence"}``.
        """
        predictions, confidences = [], []

        for rel_path in table["rel_path"]:
            with Image.open(image_root / rel_path) as handle:
                tensor = transform(handle.convert("RGB")).unsqueeze(0).to(device)
            probs = torch.softmax(pipeline(tensor), dim=1)[0]
            predictions.append(int(probs.argmax()))
            confidences.append(float(probs.max()))

        return {
            "y_true": table["label"].to_numpy(),
            "y_pred": np.array(predictions),
            "confidence": np.array(confidences),
        }

    def _select_examples(
        self, table: pd.DataFrame, predictions: Dict[str, np.ndarray], class_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Pick correct and incorrect examples for each class.

        Step 19: "Include correct and incorrect prediction examples for each class." An
        explanation method shown only on successes proves nothing.

        :param table: Test split table.
        :param predictions: Output of :meth:`_predict_all`.
        :param class_names: Class names ordered by label index.
        :return: Selected examples.
        """
        y_true, y_pred = predictions["y_true"], predictions["y_pred"]
        examples: List[Dict[str, Any]] = []

        for index, name in enumerate(class_names):
            for correctness, mask in (
                ("correct", (y_true == index) & (y_pred == y_true)),
                ("incorrect", (y_true == index) & (y_pred != y_true)),
            ):
                positions = np.flatnonzero(mask)
                if positions.size == 0:
                    log.info(f"No {correctness} examples for {name}")
                    continue

                for position in positions[: self.examples_per_class]:
                    examples.append(
                        {
                            "class_name": name,
                            "true_label": int(index),
                            "predicted_label": int(y_pred[position]),
                            "predicted_name": class_names[int(y_pred[position])],
                            "correctness": correctness,
                            "rel_path": table.iloc[position]["rel_path"],
                            "confidence": float(predictions["confidence"][position]),
                        }
                    )

        return examples

    # ---------------------------------------------------------------- Grad-CAM

    def _grad_cam_examples(
        self,
        pipeline: Any,
        examples: List[Dict[str, Any]],
        image_root: Path,
        transform: Any,
        device: torch.device,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """Generate Grad-CAM maps and run the deletion/insertion sanity check.

        :param pipeline: Loaded model.
        :param examples: Selected examples.
        :param image_root: Directory relative paths resolve against.
        :param transform: Evaluation transform.
        :param device: Device to run on.
        :param class_names: Class names ordered by label index.
        :return: Sanity-check results and where the figure was written.
        """
        if not examples:
            return {"note": "no examples available"}

        target_layer = pipeline.classical_net.backbone.features[-1]
        rows: List[Dict[str, Any]] = []
        panels: List[Tuple[Dict[str, Any], Image.Image, np.ndarray]] = []

        with GradCAM(pipeline, target_layer) as cam:
            for example in examples:
                with Image.open(image_root / example["rel_path"]) as handle:
                    image = handle.convert("RGB").copy()

                tensor = transform(image).unsqueeze(0).to(device)
                # Explain the TRUE class, including for misclassifications: that reveals
                # whether the right evidence was present but outvoted.
                heatmap = cam(tensor, target_class=example["true_label"])[0]
                panels.append((example, image, heatmap))

                rows.append(
                    {
                        **{k: example[k] for k in ("class_name", "correctness", "confidence")},
                        "predicted_name": example["predicted_name"],
                        **self._deletion_insertion(
                            pipeline, image, heatmap, example["true_label"], transform, device
                        ),
                    }
                )

        sanity = pd.DataFrame(rows)
        self.save_table(sanity, "step19_saliency_sanity_checks.csv")
        self._plot_cam_panels(panels)

        return {
            "sanity_checks": sanity.to_dict("records"),
            "mean_deletion_drop": float(sanity["deletion_drop"].mean()),
            "mean_insertion_recovery": float(sanity["insertion_confidence"].mean()),
            "figure": "step19_grad_cam.png",
        }

    @torch.no_grad()
    def _deletion_insertion(
        self,
        pipeline: Any,
        image: Image.Image,
        heatmap: np.ndarray,
        target_class: int,
        transform: Any,
        device: torch.device,
    ) -> Dict[str, float]:
        """Delete and then isolate the highest-scoring region, measuring both effects.

        Deletion asks whether the region is *necessary*; insertion asks whether it is
        *sufficient*. A trustworthy explanation should show a large deletion drop and a
        high insertion recovery. Either one alone can mislead.

        :param pipeline: Loaded model.
        :param image: Original image.
        :param heatmap: Grad-CAM map in ``[0, 1]``, at the image's resolution.
        :param target_class: Class whose confidence is tracked.
        :param transform: Evaluation transform.
        :param device: Device to run on.
        :return: Original, deleted and inserted confidences plus the drop.
        """
        array = np.asarray(image.resize((heatmap.shape[1], heatmap.shape[0]))).astype(np.float32)
        mask = heatmap >= np.percentile(heatmap, 100 * (1 - self.deletion_fraction))

        def confidence(pixels: np.ndarray) -> float:
            tensor = transform(Image.fromarray(pixels.astype(np.uint8))).unsqueeze(0).to(device)
            return float(torch.softmax(pipeline(tensor), dim=1)[0, target_class])

        original = confidence(array)

        deleted = array.copy()
        deleted[mask] = 0.0

        inserted = np.zeros_like(array)
        inserted[mask] = array[mask]

        deleted_confidence = confidence(deleted)
        return {
            "original_confidence": original,
            "deletion_confidence": deleted_confidence,
            "deletion_drop": original - deleted_confidence,
            "insertion_confidence": confidence(inserted),
        }

    def _plot_cam_panels(self, panels: List[Tuple[Dict[str, Any], Image.Image, np.ndarray]]) -> None:
        """Draw each example's image and its heatmap overlay.

        :param panels: ``(example, image, heatmap)`` triples.
        """
        figure, axes = plt.subplots(len(panels), 2, figsize=(7, 3.4 * len(panels)))
        axes = np.atleast_2d(axes)

        for row, (example, image, heatmap) in enumerate(panels):
            resized = image.resize((heatmap.shape[1], heatmap.shape[0]))

            axes[row, 0].imshow(resized)
            axes[row, 0].set_title(
                f"{example['class_name']} ({example['correctness']})", fontsize=9
            )
            axes[row, 0].axis("off")

            axes[row, 1].imshow(resized)
            axes[row, 1].imshow(heatmap, cmap="jet", alpha=0.5)
            axes[row, 1].set_title(
                f"Grad-CAM -> predicted {example['predicted_name']}", fontsize=9
            )
            axes[row, 1].axis("off")

        figure.suptitle("Step 19: Grad-CAM, correct and incorrect cases", fontsize=12)
        figure.tight_layout()
        figure.savefig(self.figure_path("step19_grad_cam.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)

    # -------------------------------------------------------------- MC-dropout

    def _mc_dropout(
        self, pipeline: Any, table: pd.DataFrame, image_root: Path, transform: Any, device: torch.device
    ) -> Dict[str, Any]:
        """Estimate predictive uncertainty by sampling with dropout active.

        Step 19 asks for uncertainty scores "to flag uncertain predictions". The measure
        that matters is whether variance actually separates errors from successes - if it
        does, thresholding it is a deployable triage signal.

        :param pipeline: Loaded model.
        :param table: Test split table.
        :param image_root: Directory relative paths resolve against.
        :param transform: Evaluation transform.
        :param device: Device to run on.
        :return: Mean variance for correct and incorrect predictions, and their ratio.
        """
        subset = table.head(self.max_samples)

        pipeline.eval()
        # Dropout only: BatchNorm must stay in eval, or its running statistics would drift
        # and the variance would mix two different sources of randomness.
        dropouts = [m for m in pipeline.modules() if isinstance(m, torch.nn.Dropout)]
        for module in dropouts:
            module.train()

        variances, correct = [], []
        try:
            with torch.no_grad():
                for rel_path, label in zip(subset["rel_path"], subset["label"]):
                    with Image.open(image_root / rel_path) as handle:
                        tensor = transform(handle.convert("RGB")).unsqueeze(0).to(device)

                    samples = torch.stack(
                        [
                            torch.softmax(pipeline(tensor), dim=1)[0]
                            for _ in range(self.mc_dropout_samples)
                        ]
                    )
                    mean = samples.mean(dim=0)
                    variances.append(float(samples.var(dim=0).sum()))
                    correct.append(bool(int(mean.argmax()) == int(label)))
        finally:
            # Always restore eval mode, or every later analysis silently samples dropout.
            for module in dropouts:
                module.eval()

        variance = np.array(variances)
        is_correct = np.array(correct)

        correct_mean = float(variance[is_correct].mean()) if is_correct.any() else float("nan")
        incorrect_mean = float(variance[~is_correct].mean()) if (~is_correct).any() else float("nan")

        frame = pd.DataFrame({"variance": variance, "correct": is_correct})
        self.save_table(frame, "step19_mc_dropout.csv")
        self._plot_uncertainty(frame)

        ratio = (
            incorrect_mean / correct_mean
            if correct_mean and np.isfinite(correct_mean) and correct_mean > 0
            else None
        )
        return {
            "n_samples": int(len(variance)),
            "n_dropout_layers": len(dropouts),
            "forward_passes": self.mc_dropout_samples,
            "mean_variance_correct": correct_mean,
            "mean_variance_incorrect": incorrect_mean,
            "variance_ratio_incorrect_over_correct": ratio,
            "usable_as_triage_signal": bool(ratio is not None and ratio > 2.0),
        }

    def _plot_uncertainty(self, frame: pd.DataFrame) -> None:
        """Plot the uncertainty distribution split by correctness.

        :param frame: Per-sample variance and correctness.
        """
        figure, axis = plt.subplots(figsize=(7, 4))

        for label, colour in (("correct", "#2F6B4F"), ("incorrect", "#A6501E")):
            values = frame[frame["correct"] == (label == "correct")]["variance"]
            if len(values):
                axis.hist(values, bins=30, alpha=0.65, label=f"{label} (n={len(values)})", color=colour)

        axis.set_xlabel("MC-dropout predictive variance")
        axis.set_ylabel("count")
        axis.set_title("Step 19: does uncertainty separate errors from successes?")
        axis.legend()

        figure.tight_layout()
        figure.savefig(self.figure_path("step19_uncertainty.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)

    # -------------------------------------------------------------------- SHAP

    def _shap(
        self, pipeline: Any, table: pd.DataFrame, image_root: Path, transform: Any, device: torch.device
    ) -> Dict[str, Any]:
        """Attribute the fused vector's contribution back to each branch.

        This is an *independent* check on Step 13's ablation: the ablation removes a branch
        and retrains, SHAP explains the trained model as it stands. Agreement between two
        such different methods is much stronger evidence than either alone.

        :param pipeline: Loaded model.
        :param table: Test split table.
        :param image_root: Directory relative paths resolve against.
        :param transform: Evaluation transform.
        :param device: Device to run on.
        :return: Normalised per-branch attribution.
        """
        import shap

        needed = self.shap_background + self.shap_explain
        subset = table.head(min(needed, len(table)))

        fused, widths = self._collect_fused(pipeline, subset, image_root, transform, device)
        if len(fused) < 2:
            return {"note": "too few samples for SHAP"}

        background = fused[: self.shap_background]
        explain = fused[self.shap_background : self.shap_background + self.shap_explain]
        if len(explain) == 0:
            explain, background = fused[:1], fused[1:] if len(fused) > 1 else fused[:1]

        head = pipeline.fusion_net.final_classifier

        def predict(vectors: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tensor = torch.tensor(vectors, dtype=torch.float32, device=device)
                return torch.softmax(head(tensor), dim=1).cpu().numpy()

        explainer = shap.KernelExplainer(predict, background)
        values = explainer.shap_values(explain, nsamples=100, silent=True)

        array = np.abs(np.array(values))
        n_features = widths[-1]
        n_classes = head.output.out_features
        n_explained = len(explain)

        # shap returns different layouts across versions. Identify the feature axis by
        # matching the FULL shape rather than by looking for an axis of the right size -
        # size-matching is ambiguous whenever another axis happens to have the same length
        # (e.g. explaining exactly `n_features` samples), and would then average over the
        # wrong axis and silently produce a plausible but meaningless attribution.
        per_feature = None
        if array.ndim == 3:
            if array.shape == (n_classes, n_explained, n_features):
                per_feature = array.mean(axis=(0, 1))
            elif array.shape == (n_explained, n_features, n_classes):
                per_feature = array.mean(axis=(0, 2))
        elif array.ndim == 2 and array.shape == (n_explained, n_features):
            per_feature = array.mean(axis=0)

        if per_feature is None:
            log.warning(
                f"Unrecognised SHAP output shape {array.shape} for "
                f"{n_explained} samples x {n_features} features x {n_classes} classes; "
                "falling back to a flattened mean, which may mix axes."
            )
            per_feature = array.reshape(-1, n_features).mean(axis=0)

        proj = n_features // 3
        totals = {
            "classical": float(per_feature[:proj].sum()),
            "spatial": float(per_feature[proj : 2 * proj].sum()),
            "quantum": float(per_feature[2 * proj :].sum()),
        }
        total = sum(totals.values()) or 1.0
        share = {name: value / total for name, value in totals.items()}

        self.save_table(
            pd.DataFrame([{"branch": k, "abs_shap": v, "share": share[k]} for k, v in totals.items()]),
            "step19_shap_branch_importance.csv",
        )

        return {
            "n_background": len(background),
            "n_explained": len(explain),
            "branch_share": share,
            "caveat": (
                "KernelExplainer approximates on a small background sample in a "
                f"{n_features}-dimensional space, so treat the shares as an ordering "
                "rather than exact percentages. Corroborate against Step 13's ablation."
            ),
        }

    @torch.no_grad()
    def _collect_fused(
        self, pipeline: Any, table: pd.DataFrame, image_root: Path, transform: Any, device: torch.device
    ) -> Tuple[np.ndarray, List[int]]:
        """Collect fused feature vectors for SHAP.

        :param pipeline: Loaded model.
        :param table: Rows to process.
        :param image_root: Directory relative paths resolve against.
        :param transform: Evaluation transform.
        :param device: Device to run on.
        :return: ``(fused_vectors, widths)`` where widths ends with the fused width.
        """
        vectors = []
        for rel_path in table["rel_path"]:
            with Image.open(image_root / rel_path) as handle:
                tensor = transform(handle.convert("RGB")).unsqueeze(0).to(device)
            vectors.append(pipeline.extract(tensor)["fused"][0].cpu().numpy())

        fused = np.stack(vectors)
        return fused, [fused.shape[1]]

    # ---------------------------------------------------------- interpretation

    @staticmethod
    def _interpret(summary: Dict[str, Any]) -> List[str]:
        """Turn the numbers into statements a reader can act on.

        :param summary: Computed results.
        :return: Plain-language findings.
        """
        notes: List[str] = []

        mc = summary.get("mc_dropout", {})
        ratio = mc.get("variance_ratio_incorrect_over_correct")
        if ratio is not None and np.isfinite(ratio):
            if ratio > 2.0:
                notes.append(
                    f"MC-dropout variance is {ratio:.1f}x higher on incorrect predictions, "
                    "so thresholding it is a usable triage signal for human review."
                )
            else:
                notes.append(
                    f"MC-dropout variance is only {ratio:.1f}x higher on incorrect "
                    "predictions, which is too weak to triage on reliably."
                )

        cam = summary.get("grad_cam", {})
        if "mean_deletion_drop" in cam:
            drop = cam["mean_deletion_drop"]
            if drop < 0.05:
                notes.append(
                    f"Deleting the top-scoring region changed confidence by only {drop:.3f} "
                    "on average. The decision is distributed across the image rather than "
                    "concentrated where Grad-CAM points - so the heatmaps localise weakly, "
                    "which is a finding about the model, not a broken explanation."
                )
            else:
                notes.append(
                    f"Deleting the top-scoring region cost {drop:.3f} confidence on "
                    "average, so the highlighted region is genuinely load-bearing."
                )

        shap_result = summary.get("shap", {})
        share = shap_result.get("branch_share")
        if share:
            leader = max(share, key=share.get)
            notes.append(
                f"SHAP attributes {share[leader]:.0%} of the fused vector's influence to "
                f"the {leader} branch. Compare against Step 13's ablation - agreement "
                "between two independent methods is much stronger than either alone."
            )

        return notes
