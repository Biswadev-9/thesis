"""Dataset classes backed by the split table.

Every dataset reads the single split CSV produced by
:func:`src.data.components.split_builder.build_split` and resolves image paths against
an ``image_root``. Keeping paths relative in the table and absolute only at load time is
what lets the same split file move between machines - the reference notebook stored
absolute Colab paths and had to rebuild the split whenever the runtime was recycled.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.components.split_builder import load_split


class BrainTumorDataset(Dataset):
    """Four-class brain-tumour MRI slices for one split.

    :param split_csv: Split table written by ``build_split``.
    :param image_root: Directory the table's ``rel_path`` entries resolve against.
    :param split: ``train``, ``val`` or ``test``.
    :param transform: Transform applied to the PIL image.
    :param preprocess: Optional image-space preprocessing applied *before* ``transform``
        (Step 6). Leave ``None`` when reading a pre-materialised recipe directory, since
        the preprocessing is already baked into the files on disk.
    """

    def __init__(
        self,
        split_csv: Path,
        image_root: Path,
        split: str,
        transform: Optional[Callable] = None,
        preprocess: Optional[Callable] = None,
    ) -> None:
        self.df = load_split(split_csv, split=split)
        self.image_root = Path(image_root)
        self.split = split
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self) -> int:
        """:return: Number of images in this split."""
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """:param index: Row index within the split.
        :return: ``(image, label)`` where image is transformed if a transform is set.
        """
        row = self.df.iloc[index]
        image = Image.open(self.image_root / row["rel_path"]).convert("RGB")

        if self.preprocess is not None:
            image = self.preprocess(image)
        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["label"])

    @property
    def labels(self) -> pd.Series:
        """:return: Integer labels in dataset order, for samplers and class weights."""
        return self.df["label"]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(split={self.split!r}, n={len(self)})"


class SubsetDataset(Dataset):
    """A stratified subset of another dataset, for the Step 6 and Step 8 proxy studies.

    Those studies sweep many configurations, so they run on a small balanced subset at
    reduced resolution rather than the full training split.

    :param dataset: Dataset to subset.
    :param indices: Row indices to keep.
    """

    def __init__(self, dataset: Dataset, indices: list) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        """:return: Number of retained rows."""
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """:param index: Index into the subset.
        :return: The underlying dataset's item.
        """
        return self.dataset[self.indices[index]]


class PreExtractedFeatureDataset(Dataset):
    """Frozen tri-branch features for the fusion stages (Steps 13-15, 20).

    The classical, spatial-gate and quantum branches are frozen once trained, so their
    outputs are extracted once and cached. Fusion heads then train over tensors instead
    of re-running three backbones - and, critically, without re-running the CPU-bound
    quantum simulator every epoch.

    :param classical: Classical branch features, shape ``(N, 1280)``.
    :param spatial: Spatial-gate branch features, shape ``(N, 32)``.
    :param quantum: Quantum branch features, shape ``(N, n_qubits)``.
    :param labels: Integer labels, shape ``(N,)``.
    :param zero_branches: Names among ``{"classical", "spatial", "quantum"}`` to replace
        with zeros. This is how the branch-contribution ablation is expressed without
        changing the architecture.
    """

    def __init__(
        self,
        classical: torch.Tensor,
        spatial: torch.Tensor,
        quantum: torch.Tensor,
        labels: torch.Tensor,
        zero_branches: Optional[list] = None,
    ) -> None:
        lengths = {len(classical), len(spatial), len(quantum), len(labels)}
        if len(lengths) != 1:
            raise ValueError(f"Feature tensors disagree on length: {lengths}")

        zero_branches = zero_branches or []
        self.classical = torch.zeros_like(classical) if "classical" in zero_branches else classical
        self.spatial = torch.zeros_like(spatial) if "spatial" in zero_branches else spatial
        self.quantum = torch.zeros_like(quantum) if "quantum" in zero_branches else quantum
        self.labels = labels
        self.zero_branches = zero_branches

    def __len__(self) -> int:
        """:return: Number of cached samples."""
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """:param index: Sample index.
        :return: ``(classical, spatial, quantum, label)``.
        """
        return self.classical[index], self.spatial[index], self.quantum[index], self.labels[index]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self)}, zeroed={self.zero_branches})"
