"""Datamodules for the evaluation stages (Steps 17 and 18).

Both serve *evaluation only*. Neither has a training split, neither augments, and neither
resamples - anything else would change what is being measured.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional

from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.components.datasets import BrainTumorDataset
from src.data.components.external import EXTERNAL_CLASSES, FigshareDataset, scan_figshare
from src.data.components.split_builder import CLASS_MAP, locate_dataset_root
from src.data.components.transforms import build_transform
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class FigshareDataModule(LightningDataModule):
    """External validation set for Step 17.

    :param data_dir: Project data directory.
    :param raw_subdir: Figshare tree, relative to ``data_dir``.
    :param image_size: Must match what the model was trained at.
    :param normalize: Must match the internal pipeline, or the comparison is confounded.
    :param batch_size: Batch size.
    :param num_workers: Dataloader workers.
    :param pin_memory: Pin host memory.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        raw_subdir: str = "raw/figshare",
        image_size: int = 224,
        normalize: str = "imagenet",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_external: Optional[Dataset] = None

    @property
    def root(self) -> Path:
        """:return: Directory holding the Figshare release."""
        return Path(self.hparams.data_dir) / self.hparams.raw_subdir

    @property
    def num_classes(self) -> int:
        """:return: The model's class count - four, even though only three are present."""
        return len(CLASS_MAP)

    @property
    def class_names(self) -> List[str]:
        """:return: The model's class names, ordered by label index."""
        return sorted(CLASS_MAP, key=CLASS_MAP.get)

    @property
    def present_classes(self) -> List[str]:
        """:return: Classes actually present externally; No-tumor is absent."""
        return list(EXTERNAL_CLASSES)

    def prepare_data(self) -> None:
        """Check the external dataset is present.

        :raises FileNotFoundError: If it is missing.
        """
        scan_figshare(self.root)

    def setup(self, stage: Optional[str] = None) -> None:
        """Index the scans and build the dataset.

        :param stage: Unused; this datamodule is evaluation-only.
        """
        if self.data_external is not None:
            return

        # The same transform the internal test split uses. Any difference here would be
        # measured as domain shift when it is really a pipeline difference.
        transform = build_transform(
            image_size=self.hparams.image_size,
            normalize=self.hparams.normalize,
            augment=False,
        )
        self.data_external = FigshareDataset(scan_figshare(self.root), transform=transform)
        log.info(f"External dataset ready: {len(self.data_external)} scans")

    def _loader(self) -> DataLoader[Any]:
        """:return: An unshuffled evaluation loader."""
        return DataLoader(
            dataset=self.data_external,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """:return: The external evaluation loader."""
        return self._loader()

    def predict_dataloader(self) -> DataLoader[Any]:
        """:return: The external evaluation loader."""
        return self._loader()


class DegradedDataset(Dataset):
    """The internal test split, degraded and then preprocessed.

    **The order is raw image → degrade → preprocess → Step 5 transform**, and it is the
    whole point of the class. Step 18 asks whether diffusion preprocessing "improves
    robustness under noisy inputs", which only means something if the filter sees the
    noise. Reading from a pre-materialised recipe mirror and degrading afterwards would
    corrupt an already-denoised image and answer the opposite question.

    Applying the degradation before preprocessing also matches deployment: a noisy scan
    arrives, the pipeline denoises it, the model classifies it.

    :param base: The underlying test dataset, with no transform of its own.
    :param degradation: Degradation to apply, or ``None`` for the clean condition.
    :param preprocess: Step 6 filter applied *after* degradation, or ``None``.
    :param transform: The Step 5 evaluation pipeline, applied last.
    """

    def __init__(
        self,
        base: BrainTumorDataset,
        degradation: Optional[Callable],
        preprocess: Optional[Callable],
        transform: Callable,
    ) -> None:
        self.base = base
        self.degradation = degradation
        self.preprocess = preprocess
        self.transform = transform

    def __len__(self) -> int:
        """:return: Number of images."""
        return len(self.base)

    def __getitem__(self, index: int):
        """:param index: Image index.
        :return: ``(degraded_image, label)``.
        """
        row = self.base.df.iloc[index]
        image = Image.open(self.base.image_root / row["rel_path"]).convert("RGB")

        if self.degradation is not None:
            # The index is passed so stochastic degradations corrupt each image
            # identically across the models being compared.
            image = self.degradation(image, index)

        if self.preprocess is not None:
            image = self.preprocess(image)

        return self.transform(image), int(row["label"])


class DegradedTestDataModule(LightningDataModule):
    """Internal test split under a controlled degradation, for Step 18.

    :param data_dir: Project data directory.
    :param raw_subdir: Raw dataset tree, relative to ``data_dir``.
    :param split_subpath: Split CSV location, relative to ``data_dir``.
    :param degradation: Degradation to apply, or ``None`` for the clean baseline.
    :param preprocess: Step 6 filter applied *after* degradation. This is how Step 18's
        "does diffusion improve robustness under noise" question is asked - the filter
        must see the noise, so it runs on the degraded image rather than being read from
        a pre-filtered mirror.
    :param image_size: Square input edge.
    :param normalize: Step 5 intensity treatment.
    :param batch_size: Batch size.
    :param num_workers: Dataloader workers.
    :param pin_memory: Pin host memory.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        raw_subdir: str = "raw/bt_mri",
        split_subpath: str = "splits/dataset_split.csv",
        degradation: Optional[Callable] = None,
        preprocess: Optional[Callable] = None,
        image_size: int = 224,
        normalize: str = "imagenet",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["degradation", "preprocess"])
        self.degradation = degradation
        self.preprocess = preprocess
        self.data_test: Optional[Dataset] = None

    @property
    def image_root(self) -> Path:
        """:return: The raw dataset tree.

        Always raw: any Step 6 filter is applied on the fly *after* degradation, so a
        pre-materialised mirror would be the wrong input.
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

    def prepare_data(self) -> None:
        """Nothing to build; the split and any recipe already exist."""

    def setup(self, stage: Optional[str] = None) -> None:
        """Build the degraded test dataset.

        :param stage: Unused; this datamodule is evaluation-only.
        """
        if self.data_test is not None:
            return

        base = BrainTumorDataset(
            split_csv=Path(self.hparams.data_dir) / self.hparams.split_subpath,
            image_root=self.image_root,
            split="test",
            transform=None,
        )
        transform = build_transform(
            image_size=self.hparams.image_size,
            normalize=self.hparams.normalize,
            augment=False,
        )
        self.data_test = DegradedDataset(base, self.degradation, self.preprocess, transform)

    def test_dataloader(self) -> DataLoader[Any]:
        """:return: The degraded test loader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
