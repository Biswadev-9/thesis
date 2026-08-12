"""Step 4: initial data audit and quality control.

The specification requires, before any model development:

- image dimensions, grayscale/RGB status, **bit depth** and intensity ranges;
- a class-distribution plot and the imbalance ratio;
- inspection of representative images from each class;
- checks for corrupted, duplicated, unreadable and inconsistently labelled files;
- all exclusions and corrections recorded in a dataset audit table.

Two gaps in the reference notebook are closed here. It never recorded bit depth, and its
audit table was hand-written prose whose duplicate counts (726 duplicates across 363
groups) do not reconcile with the split sizes it went on to produce - train 4617 / val
990 / test 990 implies roughly 426 rows removed, not 726. Every figure below is computed
from the data at run time, so the table cannot drift from the split it describes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: analyses run under Hydra, never in a GUI session
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.analysis.base import Analysis
from src.data.components.cropping import BrainBoundingBoxCrop, validate_crop_preserves_foreground
from src.data.components.split_builder import (
    CLASS_MAP,
    audit_images,
    class_distribution,
    imbalance_ratio,
    verify_no_leakage,
)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DataAuditAnalysis(Analysis):
    """Build the Step 4 dataset audit table, distribution plot and sample montage.

    :param name: Analysis identifier.
    :param sample_images_per_class: Representative images to show per class.
    :param validate_cropping: Also report whether the Step 5 background crop would
        preserve bright tissue, so the crop can be enabled on evidence rather than hope.
    :param crop_validation_samples: Training images to test the crop against.
    :param seed: Sampling seed for the montage and crop validation.
    """

    def __init__(
        self,
        name: str = "step04_audit",
        sample_images_per_class: int = 5,
        validate_cropping: bool = True,
        crop_validation_samples: int = 60,
        seed: int = 42,
    ) -> None:
        super().__init__(name=name)
        self.sample_images_per_class = sample_images_per_class
        self.validate_cropping = validate_cropping
        self.crop_validation_samples = crop_validation_samples
        self.seed = seed

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Run the audit.

        :param datamodule: A ``BTMRIDataModule``; its ``prepare_data`` builds the split
            if needed.
        :return: Audit summary.
        """
        datamodule.prepare_data()
        df = datamodule.load_split_table()
        image_root = Path(datamodule.image_root)

        verify_no_leakage(df)

        per_image, corrupted = audit_images(df, image_root)
        self.save_table(per_image, "image_audit.csv")

        distribution = class_distribution(df)
        self.save_table(distribution, "class_distribution.csv")
        self._plot_class_distribution(distribution)
        self._plot_sample_images(df, image_root)

        crop_report = self._validate_cropping(df, image_root) if self.validate_cropping else None

        audit_table = self._build_audit_table(df, per_image, corrupted, crop_report)
        self.save_table(audit_table, "dataset_audit_table.csv")
        log.info("\n" + audit_table.to_string(index=False))

        summary: Dict[str, Any] = {
            "n_images": int(len(df)),
            "n_readable": int(len(per_image)),
            "n_corrupted": len(corrupted),
            "corrupted_files": corrupted[:20],
            "split_sizes": {
                split: int((df["split"] == split).sum()) for split in ("train", "val", "test")
            },
            "train_imbalance_ratio": imbalance_ratio(df, "train"),
            "unique_sizes": sorted(
                {f"{w}x{h}" for w, h in zip(per_image["width"], per_image["height"])}
            )[:20],
            "modes": per_image["mode"].value_counts().to_dict() if len(per_image) else {},
            "bits_per_channel": (
                per_image["bits_per_channel"].value_counts().to_dict() if len(per_image) else {}
            ),
            "intensity_min": float(per_image["min_value"].min()) if len(per_image) else None,
            "intensity_max": float(per_image["max_value"].max()) if len(per_image) else None,
            "leak_free": True,
        }
        if crop_report is not None:
            summary["background_crop_validation"] = crop_report
        return summary

    # --------------------------------------------------------------- figures

    def _plot_class_distribution(self, distribution: pd.DataFrame) -> None:
        """Plot per-split class counts.

        :param distribution: Long-form table from ``class_distribution``.
        """
        splits = ["train", "val", "test"]
        present = [s for s in splits if s in set(distribution["split"])]
        if not present:
            return

        class_names = sorted(CLASS_MAP, key=CLASS_MAP.get)
        figure, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 3.8), sharey=True)
        axes = [axes] if len(present) == 1 else list(axes)

        for axis, split in zip(axes, present):
            subset = distribution[distribution["split"] == split].set_index("class_name")
            counts = [int(subset.loc[name, "count"]) for name in class_names]
            axis.bar(range(len(class_names)), counts, color="#4A5A8C")
            axis.set_xticks(range(len(class_names)))
            axis.set_xticklabels(class_names, rotation=30, ha="right")
            ratio = max(counts) / min(counts) if min(counts) else float("inf")
            axis.set_title(f"{split}  (n={sum(counts)}, imbalance {ratio:.2f})")
            for index, count in enumerate(counts):
                axis.text(index, count, str(count), ha="center", va="bottom", fontsize=8)

        axes[0].set_ylabel("Images")
        figure.suptitle("Step 4: class distribution per split")
        figure.tight_layout()
        figure.savefig(self.figure_path("class_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)

    def _plot_sample_images(self, df: pd.DataFrame, image_root: Path) -> None:
        """Show representative images from each class.

        :param df: Split table.
        :param image_root: Directory relative paths resolve against.
        """
        class_names = sorted(CLASS_MAP, key=CLASS_MAP.get)
        columns = self.sample_images_per_class
        figure, axes = plt.subplots(
            len(class_names), columns, figsize=(2.2 * columns, 2.4 * len(class_names))
        )
        axes = axes.reshape(len(class_names), columns)

        for row, class_name in enumerate(class_names):
            pool = df[df["class_name"] == class_name]
            take = min(columns, len(pool))
            picks = pool.sample(n=take, random_state=self.seed) if take else pool

            for column in range(columns):
                axis = axes[row, column]
                axis.axis("off")
                if column >= take:
                    continue
                with Image.open(image_root / picks.iloc[column]["rel_path"]) as image:
                    axis.imshow(image, cmap="gray")
                if column == 0:
                    axis.set_title(class_name, fontsize=10, loc="left")

        figure.suptitle("Step 4: representative images per class")
        figure.tight_layout()
        figure.savefig(self.figure_path("sample_images.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)

    # ------------------------------------------------------------ validation

    def _validate_cropping(self, df: pd.DataFrame, image_root: Path) -> Dict[str, Any]:
        """Test whether background cropping would discard bright tissue.

        :param df: Split table.
        :param image_root: Directory relative paths resolve against.
        :return: Report from ``validate_crop_preserves_foreground``.
        """
        train_rows = df[df["split"] == "train"]
        take = min(self.crop_validation_samples, len(train_rows))
        if take == 0:
            return {"passed": False, "reason": "no training rows"}

        picks = train_rows.sample(n=take, random_state=self.seed)
        images: List[Image.Image] = []
        for rel_path in picks["rel_path"]:
            with Image.open(image_root / rel_path) as image:
                images.append(image.convert("L").copy())

        report = validate_crop_preserves_foreground(images, BrainBoundingBoxCrop())
        log.info(
            f"Background-crop validation: passed={report['passed']}, "
            f"max foreground lost={report['max_foreground_lost']:.5f}"
        )
        return report

    # ----------------------------------------------------------- audit table

    def _build_audit_table(
        self,
        df: pd.DataFrame,
        per_image: pd.DataFrame,
        corrupted: List[Dict[str, str]],
        crop_report: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Assemble the audit table the specification asks for.

        Every value is derived from the data, never transcribed.

        :param df: Split table.
        :param per_image: Per-image audit table.
        :param corrupted: Unreadable files.
        :param crop_report: Crop validation report, if run.
        :return: Check / result / action-taken table.
        """
        sizes = {f"{w}x{h}" for w, h in zip(per_image["width"], per_image["height"])}
        modes = per_image["mode"].value_counts().to_dict()
        depths = per_image["bits_per_channel"].value_counts().to_dict()
        split_counts = {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")}

        # Duplicate accounting is reconstructed from hashes still present in the split.
        # Because deduplication happens before splitting, the split itself must contain
        # no repeated hash - which is what makes it leak-free.
        residual_duplicates = int(df.duplicated(subset="md5", keep=False).sum())

        rows: List[Dict[str, Any]] = [
            {
                "Check": "Images in final split",
                "Result": f"{len(df)} unique images",
                "Action Taken": "Basis for the 70/15/15 stratified split",
            },
            {
                "Check": "Corrupted / unreadable files",
                "Result": f"{len(corrupted)} of {len(df)}",
                "Action Taken": (
                    "None found - no exclusions needed"
                    if not corrupted
                    else "Excluded; see image_audit.csv for the list"
                ),
            },
            {
                "Check": "Image dimension consistency",
                "Result": (
                    f"All images {next(iter(sizes))}"
                    if len(sizes) == 1
                    else f"{len(sizes)} distinct sizes, e.g. {sorted(sizes)[:3]}"
                ),
                "Action Taken": "Resized to a fixed square input in the Step 5 pipeline",
            },
            {
                "Check": "Colour mode consistency",
                "Result": ", ".join(f"{mode}: {count}" for mode, count in modes.items()),
                "Action Taken": "Converted to 3-channel RGB for pretrained backbones",
            },
            {
                "Check": "Bit depth",
                "Result": ", ".join(f"{bits}-bit/channel: {count}" for bits, count in depths.items()),
                "Action Taken": "Recorded; no re-quantisation applied",
            },
            {
                "Check": "Intensity range",
                "Result": (
                    f"{per_image['min_value'].min():.0f}-{per_image['max_value'].max():.0f}"
                    if len(per_image)
                    else "n/a"
                ),
                "Action Taken": "Normalised in the Step 5 pipeline",
            },
            {
                "Check": "Exact duplicate images (MD5) within the final split",
                "Result": f"{residual_duplicates} rows",
                "Action Taken": "Duplicates removed BEFORE splitting, so none remain",
            },
            {
                "Check": "Duplicate groups spanning multiple splits (leakage)",
                "Result": "0 groups",
                "Action Taken": "Prevented by deduplicating before splitting; asserted at build time",
            },
            {
                "Check": "Class distribution and imbalance",
                "Result": f"train imbalance ratio {imbalance_ratio(df, 'train'):.2f}",
                "Action Taken": "Handled at training time per Step 8 (weighted sampler / loss)",
            },
            {
                "Check": "Final split sizes",
                "Result": (
                    f"Train: {split_counts['train']} | Val: {split_counts['val']} "
                    f"| Test: {split_counts['test']}"
                ),
                "Action Taken": "Stratified by class, seed recorded in the split report",
            },
        ]

        if crop_report is not None:
            rows.append(
                {
                    "Check": "Background cropping preserves bright tissue",
                    "Result": (
                        f"passed={crop_report.get('passed')}, "
                        f"max foreground lost={crop_report.get('max_foreground_lost', 0):.5f}"
                    ),
                    "Action Taken": (
                        "Crop is available but disabled by default; enable only on this evidence"
                    ),
                }
            )

        return pd.DataFrame.from_records(rows)
