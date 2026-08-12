"""`LightningDataModule` for the four-class brain-tumour MRI dataset.

Covers specification Steps 3, 5 and 7:

- Step 3: builds the leak-free 70/15/15 stratified split in ``prepare_data``, before any
  augmentation exists.
- Step 5: one shared geometry and intensity pipeline for every model.
- Step 7: augmentation on the training split only.

Class weights are exposed as a property so the `LightningModule` can pull them without
recomputing - see ``MRIClassificationModule.setup``. They are derived from the training
split alone.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.datasets import BrainTumorDataset
from src.data.components.sampling import compute_class_weights, make_weighted_sampler
from src.data.components.split_builder import (
    CLASS_MAP,
    build_split,
    load_split,
    locate_dataset_root,
)
from src.data.components.transforms import build_transform, describe_pipeline
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class BTMRIDataModule(LightningDataModule):
    """Brain-tumour MRI datamodule reading a single leak-free split table.

    :param data_dir: Project data directory.
    :param raw_subdir: Raw dataset tree, relative to ``data_dir``, holding the
        ``Training``/``Testing`` class folders.
    :param split_subpath: Split CSV location, relative to ``data_dir``.
    :param recipe: Preprocessing recipe directory under ``data_dir/processed``. ``None``
        reads the raw images. Recipes are materialised in Phase 2 (Step 6); until then
        only ``None`` is valid.
    :param image_size: Square input edge. See ``transforms`` for why this is 224.
    :param normalize: Step 5 intensity treatment - ``imagenet``, ``zscore``, ``minmax``
        or ``none``. ``none`` gives ablation row A0 its "raw image" condition.
    :param augment: Apply Step 7 augmentation to the training split.
    :param crop_background: Apply the validated Step 5 background crop.
    :param use_weighted_sampler: Step 8's balanced sampler on the training loader only.
    :param batch_size: Global batch size, divided across devices in ``setup``.
    :param num_workers: Dataloader workers. Defaults to 0 because Windows spawns rather
        than forks worker processes.
    :param pin_memory: Pin host memory for faster host-to-device copies.
    :param seed: Seed for the stratified split.
    :param val_frac: Validation fraction of the pooled dataset.
    :param test_frac: Internal-test fraction of the pooled dataset.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        raw_subdir: str = "raw/bt_mri",
        split_subpath: str = "splits/dataset_split.csv",
        recipe: Optional[str] = None,
        image_size: int = 224,
        normalize: str = "imagenet",
        augment: bool = True,
        crop_background: bool = False,
        use_weighted_sampler: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
    ) -> None:
        super().__init__()

        # Stores every init arg on self.hparams and into checkpoints.
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self._class_weights: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ paths

    @property
    def split_csv(self) -> Path:
        """:return: Absolute path to the split table."""
        return Path(self.hparams.data_dir) / self.hparams.split_subpath

    @property
    def raw_dir(self) -> Path:
        """:return: Absolute path to the raw dataset tree.

        Resolved with :func:`locate_dataset_root`, because the published archive unpacks
        into a doubly-nested folder. Paths in the split table are relative to this.
        """
        return locate_dataset_root(Path(self.hparams.data_dir) / self.hparams.raw_subdir)

    @property
    def image_root(self) -> Path:
        """:return: Directory images are loaded from - a recipe mirror, or the raw tree.

        Recipe directories mirror the raw tree's relative layout exactly, so the same
        split table addresses both.
        """
        if self.hparams.recipe is None:
            return self.raw_dir
        return Path(self.hparams.data_dir) / "processed" / self.hparams.recipe

    # ------------------------------------------------------------ properties

    @property
    def num_classes(self) -> int:
        """:return: Number of target classes (4, fixed by Step 1)."""
        return len(CLASS_MAP)

    @property
    def class_names(self) -> List[str]:
        """:return: Class names ordered by label index."""
        return sorted(CLASS_MAP, key=CLASS_MAP.get)

    @property
    def class_weights(self) -> Optional[torch.Tensor]:
        """:return: Inverse-frequency weights from the training split, or ``None``
        before ``setup`` has run.
        """
        return self._class_weights

    def pipeline_description(self) -> Dict[str, Any]:
        """Serialise the preprocessing and augmentation settings for run artefacts.

        Step 7 requires the exact augmentation ranges to be reported in the method
        section; this returns them in writable form.

        :return: JSON-serialisable description of train and eval pipelines.
        """
        return {
            "recipe": self.hparams.recipe,
            "train": describe_pipeline(
                self.hparams.image_size,
                self.hparams.normalize,
                self.hparams.augment,
                self.hparams.crop_background,
            ),
            "eval": describe_pipeline(
                self.hparams.image_size,
                self.hparams.normalize,
                augment=False,
                crop_background=self.hparams.crop_background,
            ),
            "use_weighted_sampler": self.hparams.use_weighted_sampler,
        }

    # ----------------------------------------------------------- lightning API

    def prepare_data(self) -> None:
        """Build the split table if it does not exist yet.

        Runs on a single process. Deliberately assigns no state - the split is written to
        disk and read back in ``setup`` on every process.
        """
        if self.split_csv.is_file():
            log.info(f"Split table already present at {self.split_csv}")
            return

        if not self.raw_dir.is_dir():
            raise FileNotFoundError(
                f"Raw dataset not found at {self.raw_dir}. Download it first - see "
                "scripts/download_data - or point data.raw_subdir at an existing tree "
                f"containing {list(('Training', 'Testing'))} class folders."
            )

        log.info(f"Building leak-free split from {self.raw_dir}")
        _, report = build_split(
            raw_dir=self.raw_dir,
            out_csv=self.split_csv,
            seed=self.hparams.seed,
            val_frac=self.hparams.val_frac,
            test_frac=self.hparams.test_frac,
        )
        log.info(
            f"Split built: {report['split_sizes']} "
            f"({report['rows_removed']} duplicate rows removed before splitting, "
            f"train imbalance ratio {report['train_imbalance_ratio']:.2f})"
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Construct datasets and derive class weights.

        :param stage: ``fit``, ``validate``, ``test`` or ``predict``.
        :raises RuntimeError: If the batch size is not divisible across devices.
        """
        if self.trainer is not None:
            world_size = self.trainer.world_size
            if self.hparams.batch_size % world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the "
                    f"number of devices ({world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // world_size

        if self.data_train and self.data_val and self.data_test:
            return

        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f"Image root not found: {self.image_root}. "
                + (
                    "Download the dataset into this location (see scripts/download_data)."
                    if self.hparams.recipe is None
                    else f"Materialise recipe {self.hparams.recipe!r} with "
                    "`python src/prepare_dataset.py` first."
                )
            )

        train_transform = build_transform(
            image_size=self.hparams.image_size,
            normalize=self.hparams.normalize,
            augment=self.hparams.augment,
            crop_background=self.hparams.crop_background,
        )
        # Validation and test never see augmentation - Step 7 is training-only.
        eval_transform = build_transform(
            image_size=self.hparams.image_size,
            normalize=self.hparams.normalize,
            augment=False,
            crop_background=self.hparams.crop_background,
        )

        common = {"split_csv": self.split_csv, "image_root": self.image_root}
        self.data_train = BrainTumorDataset(**common, split="train", transform=train_transform)
        self.data_val = BrainTumorDataset(**common, split="val", transform=eval_transform)
        self.data_test = BrainTumorDataset(**common, split="test", transform=eval_transform)

        self._class_weights = compute_class_weights(
            self.data_train.labels.tolist(), self.num_classes
        )
        log.info(
            f"Datasets ready - train {len(self.data_train)}, "
            f"val {len(self.data_val)}, test {len(self.data_test)}"
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """:return: Training loader, balanced by weighted sampler when enabled.

        The sampler and ``shuffle`` are mutually exclusive in PyTorch; the sampler
        already draws in random order.
        """
        sampler = None
        if self.hparams.use_weighted_sampler:
            sampler = make_weighted_sampler(self.data_train.labels.tolist(), self.num_classes)

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=sampler,
            shuffle=sampler is None,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """:return: Validation loader - unshuffled, unaugmented, unsampled."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """:return: Internal-test loader - unshuffled, unaugmented, unsampled."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def load_split_table(self, split: Optional[str] = None):
        """Read the split table directly, for analyses that work on rows not tensors.

        :param split: ``train``, ``val``, ``test`` or ``None`` for all rows.
        :return: The split table.
        """
        return load_split(self.split_csv, split=split)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook; nothing to release.

        :param stage: Stage being torn down.
        """

    def state_dict(self) -> Dict[Any, Any]:
        """:return: Datamodule state to checkpoint - the split file identifies the data."""
        return {"split_csv": str(self.split_csv)}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """:param state_dict: State produced by :meth:`state_dict`."""


if __name__ == "__main__":
    _ = BTMRIDataModule()
