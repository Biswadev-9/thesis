"""`LightningDataModule` over the cached tri-branch features (Steps 13-15, 20).

Reads what ``src/extract_features.py`` wrote. Because the branches are frozen, their
outputs are fixed, so the fusion stages train over tensors instead of images - seconds per
epoch instead of minutes, with no quantum simulator in the loop.

The ``zero_branches`` option is how branch-contribution ablation is expressed. Replacing a
branch's features with zeros keeps the architecture, the parameter count and the training
protocol identical, so the measured drop is attributable to the branch's *information*
rather than to a smaller model. Step 13 requires exactly this, and Step 20 reuses it for
the no-quantum comparison.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.datasets import PreExtractedFeatureDataset
from src.data.components.sampling import compute_class_weights, make_weighted_sampler
from src.data.components.split_builder import CLASS_MAP
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class BTMRIFeatureDataModule(LightningDataModule):
    """Serves cached tri-branch features.

    :param data_dir: Project data directory.
    :param tag: Feature cache name under ``data_dir/features``.
    :param zero_branches: Branches to replace with zeros - any of ``classical``,
        ``spatial``, ``quantum``. Used for the Step 13 contribution ablation.
    :param batch_size: Batch size.
    :param num_workers: Dataloader workers. Features are already in memory, so 0 is
        usually fastest.
    :param pin_memory: Pin host memory.
    :param use_weighted_sampler: Balanced sampling on the training loader.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        tag: str = "default",
        zero_branches: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        use_weighted_sampler: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self._class_weights: Optional[torch.Tensor] = None
        self._dims: Dict[str, int] = {}
        self._train_labels: List[int] = []

    # ------------------------------------------------------------------ paths

    @property
    def feature_dir(self) -> Path:
        """:return: Directory holding this cache's split files."""
        return Path(self.hparams.data_dir) / "features" / self.hparams.tag

    # ------------------------------------------------------------- properties

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
        """:return: Inverse-frequency weights from the cached training split."""
        return self._class_weights

    @property
    def classical_dim(self) -> int:
        """:return: Classical branch width, read from the cache."""
        return self._dims.get("classical", 0)

    @property
    def spatial_dim(self) -> int:
        """:return: Spatial-gate branch width, read from the cache."""
        return self._dims.get("spatial", 0)

    @property
    def quantum_dim(self) -> int:
        """:return: Quantum branch width, read from the cache."""
        return self._dims.get("quantum", 0)

    # ----------------------------------------------------------- lightning API

    def prepare_data(self) -> None:
        """Check the cache exists.

        :raises FileNotFoundError: If the cache or any split file is missing.
        """
        if not self.feature_dir.is_dir():
            raise FileNotFoundError(
                f"Feature cache not found at {self.feature_dir}. Build it with "
                "`python src/extract_features.py classical_ckpt=<...> quantum_ckpt=<...> "
                f"tag={self.hparams.tag}`."
            )

        missing = [
            split for split in ("train", "val", "test") if not (self.feature_dir / f"{split}.pt").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Feature cache at {self.feature_dir} is missing splits: {missing}. "
                "Re-run src/extract_features.py."
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load the cached tensors and build the datasets.

        :param stage: ``fit``, ``validate``, ``test`` or ``predict``.
        """
        if self.data_train and self.data_val and self.data_test:
            return

        zeroed = list(self.hparams.zero_branches or [])
        datasets = {}

        for split in ("train", "val", "test"):
            cached = torch.load(self.feature_dir / f"{split}.pt", map_location="cpu")
            datasets[split] = PreExtractedFeatureDataset(
                classical=cached["classical"],
                spatial=cached["spatial"],
                quantum=cached["quantum"],
                labels=cached["labels"],
                zero_branches=zeroed,
            )
            if split == "train":
                self._dims = {
                    "classical": int(cached["classical"].shape[1]),
                    "spatial": int(cached["spatial"].shape[1]),
                    "quantum": int(cached["quantum"].shape[1]),
                }
                self._train_labels = cached["labels"].tolist()

        self.data_train, self.data_val, self.data_test = (
            datasets["train"],
            datasets["val"],
            datasets["test"],
        )
        self._class_weights = compute_class_weights(self._train_labels, self.num_classes)

        log.info(
            f"Feature cache '{self.hparams.tag}' loaded - train {len(self.data_train)}, "
            f"val {len(self.data_val)}, test {len(self.data_test)}; dims "
            f"classical={self.classical_dim}, spatial={self.spatial_dim}, "
            f"quantum={self.quantum_dim}"
            + (f"; zeroed={zeroed}" if zeroed else "")
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """:return: Training loader over cached features."""
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
        """:return: Validation loader over cached features."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """:return: Internal-test loader over cached features."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = BTMRIFeatureDataModule()
