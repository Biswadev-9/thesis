"""Reduced-scale datamodule for the Step 6 and Step 8 selection studies.

Both studies evaluate many configurations - eleven preprocessing candidates, six
imbalance strategies - to choose *one* setting for the real pipeline. Running each on the
full 224 px training split would cost more than the pipeline it configures, so both run
on a balanced subset at reduced resolution.

Three properties keep the comparison honest:

- the subset is **stratified and fixed by seed**, so every candidate sees identical data;
- the validation subset is drawn from the real validation split, never from training;
- preprocessing is applied **on the fly** here, because a selection sweep touches each
  candidate once and caching eleven full mirrors to disk would be wasteful. Only the
  winning recipe gets materialised, by ``src/prepare_dataset.py``.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.datasets import BrainTumorDataset, SubsetDataset
from src.data.components.sampling import (
    compute_class_weights,
    make_weighted_sampler,
    stratified_subset_indices,
)
from src.data.components.split_builder import CLASS_MAP, locate_dataset_root
from src.data.components.transforms import build_transform
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class BTMRIProxyDataModule(LightningDataModule):
    """Balanced subset of the MRI dataset for configuration sweeps.

    :param data_dir: Project data directory.
    :param raw_subdir: Raw dataset tree, relative to ``data_dir``.
    :param split_subpath: Split CSV location, relative to ``data_dir``.
    :param preprocess: Step 6 filter applied before the transform pipeline, or ``None``.
    :param per_class_train: Training images per class in the subset.
    :param per_class_val: Validation images per class in the subset.
    :param image_size: Square input edge; reduced from the pipeline's 224 for speed.
    :param normalize: Step 5 intensity treatment.
    :param augment: Apply Step 7 augmentation to the training subset. One of the
        strategies Step 8 ablates.
    :param use_weighted_sampler: Balanced sampling on the training loader. Another Step 8
        strategy.
    :param batch_size: Batch size.
    :param num_workers: Dataloader workers; 0 on Windows.
    :param pin_memory: Pin host memory.
    :param seed: Subset sampling seed. Fixed across candidates so they see the same data.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        raw_subdir: str = "raw/bt_mri",
        split_subpath: str = "splits/dataset_split.csv",
        preprocess: Optional[Callable] = None,
        per_class_train: int = 200,
        per_class_val: int = 75,
        image_size: int = 128,
        normalize: str = "imagenet",
        augment: bool = False,
        use_weighted_sampler: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()

        # `preprocess` is a callable, so it stays out of hparams and the checkpoint.
        self.save_hyperparameters(logger=False, ignore=["preprocess"])

        self.preprocess = preprocess
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self._class_weights: Optional[torch.Tensor] = None
        self._train_labels: List[int] = []

    @property
    def split_csv(self) -> Path:
        """:return: Absolute path to the split table."""
        return Path(self.hparams.data_dir) / self.hparams.split_subpath

    @property
    def image_root(self) -> Path:
        """:return: Directory images are loaded from.

        Always the raw tree: proxy sweeps apply preprocessing on the fly rather than
        reading a materialised mirror. Resolved the same way as the main datamodule so
        the shared split table's relative paths line up.
        """
        return locate_dataset_root(Path(self.hparams.data_dir) / self.hparams.raw_subdir)

    @property
    def num_classes(self) -> int:
        """:return: Number of target classes."""
        return len(CLASS_MAP)

    @property
    def class_names(self) -> List[str]:
        """:return: Class names ordered by label index."""
        return sorted(CLASS_MAP, key=CLASS_MAP.get)

    @property
    def class_weights(self) -> Optional[torch.Tensor]:
        """:return: Inverse-frequency weights from the training subset, or ``None``
        before ``setup``.
        """
        return self._class_weights

    @property
    def train_labels(self) -> List[int]:
        """:return: Labels of the training subset, in dataset order."""
        return self._train_labels

    def prepare_data(self) -> None:
        """Assert the split table exists.

        The proxy datamodule never builds the split: it must sample from exactly the
        same split the real pipeline uses, so building one here would risk divergence.

        :raises FileNotFoundError: If the split table is missing.
        """
        if not self.split_csv.is_file():
            raise FileNotFoundError(
                f"Split table not found at {self.split_csv}. Build it first with "
                "`python src/analyze.py analysis=step04_audit`."
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Draw the stratified subsets and derive class weights.

        :param stage: ``fit``, ``validate``, ``test`` or ``predict``.
        """
        if self.data_train and self.data_val:
            return

        train_transform = build_transform(
            image_size=self.hparams.image_size,
            normalize=self.hparams.normalize,
            augment=self.hparams.augment,
        )
        eval_transform = build_transform(
            image_size=self.hparams.image_size,
            normalize=self.hparams.normalize,
            augment=False,
        )

        common = {
            "split_csv": self.split_csv,
            "image_root": self.image_root,
            "preprocess": self.preprocess,
        }
        full_train = BrainTumorDataset(**common, split="train", transform=train_transform)
        full_val = BrainTumorDataset(**common, split="val", transform=eval_transform)

        train_indices = stratified_subset_indices(
            full_train.df, self.hparams.per_class_train, seed=self.hparams.seed
        )
        val_indices = stratified_subset_indices(
            full_val.df, self.hparams.per_class_val, seed=self.hparams.seed
        )

        self.data_train = SubsetDataset(full_train, train_indices)
        self.data_val = SubsetDataset(full_val, val_indices)

        self._train_labels = full_train.df.iloc[train_indices]["label"].tolist()
        self._class_weights = compute_class_weights(self._train_labels, self.num_classes)

        log.info(
            f"Proxy subsets ready - train {len(self.data_train)}, val {len(self.data_val)} "
            f"at {self.hparams.image_size}px, preprocess={self.preprocess!r}"
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """:return: Training loader over the balanced subset."""
        sampler = None
        if self.hparams.use_weighted_sampler:
            sampler = make_weighted_sampler(self._train_labels, self.num_classes)

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=sampler,
            shuffle=sampler is None,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """:return: Validation loader - unshuffled, unaugmented, unsampled."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """:return: The validation loader.

        Selection studies must never touch the internal test set - Step 16 requires it to
        stay unseen until the final model is evaluated once.
        """
        return self.val_dataloader()


if __name__ == "__main__":
    _ = BTMRIProxyDataModule()
