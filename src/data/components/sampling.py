"""Class-imbalance handling for the training loader (Step 8).

The specification prescribes several strategies and requires them to be *ablated* rather
than stacked: *"Use more than one strategy only when ablation confirms benefit."* This
module supplies the two that operate on the data side - class weights (consumed by the
weighted-CE and focal losses) and the weighted sampler.

Note the specification's constraint that a weighted sampler applies to the **training
loader only**. Applying it to validation or test would resample the evaluation set and
make the reported metrics meaningless.
"""

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalised so the mean weight is 1.

    Uses the standard balanced form ``N / (K * count_k)``, which leaves a perfectly
    balanced dataset with all-ones weights and therefore keeps the loss scale comparable
    across strategies in the Step 8 ablation.

    :param labels: Integer labels from the **training split only**. Deriving weights
        from validation or test data would leak label distribution.
    :param num_classes: Number of classes ``K``.
    :return: Float tensor of shape ``(num_classes,)``.
    :raises ValueError: If ``labels`` is empty.
    """
    label_array = np.asarray(list(labels), dtype=np.int64)
    if label_array.size == 0:
        raise ValueError("Cannot compute class weights from an empty label sequence.")

    counts = np.bincount(label_array, minlength=num_classes).astype(np.float64)
    # A class absent from the training split would divide by zero; treat it as a single
    # sample so the weight is large but finite, and let the audit surface the gap.
    counts[counts == 0] = 1.0

    weights = label_array.size / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(labels: Sequence[int], num_classes: int) -> WeightedRandomSampler:
    """Build a sampler that draws each class with equal probability.

    Each sample's draw weight is the reciprocal of its class count, so an epoch contains
    roughly equal numbers of every class while remaining the same length as the split.

    :param labels: Integer labels from the training split, in dataset order.
    :param num_classes: Number of classes.
    :return: Sampler with replacement, sized to the split.
    :raises ValueError: If ``labels`` is empty.
    """
    label_array = np.asarray(list(labels), dtype=np.int64)
    if label_array.size == 0:
        raise ValueError("Cannot build a weighted sampler from an empty label sequence.")

    counts = np.bincount(label_array, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    sample_weights = 1.0 / counts[label_array]

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(label_array.size),
        replacement=True,
    )


def stratified_subset_indices(
    df: pd.DataFrame,
    per_class: int,
    seed: int = 42,
    label_column: str = "label",
) -> list:
    """Draw up to ``per_class`` rows from each class.

    Used by the Step 6 preprocessing sweep and the Step 8 imbalance ablation, both of
    which evaluate many configurations and so run on a balanced subset for tractability.

    :param df: Table for one split.
    :param per_class: Maximum rows per class.
    :param seed: Sampling seed.
    :param label_column: Column holding integer labels.
    :return: Positional indices into ``df``.
    """
    positions: list = []
    for _, group in df.groupby(label_column, sort=True):
        take = min(per_class, len(group))
        sampled = group.sample(n=take, random_state=seed)
        positions.extend(df.index.get_indexer(sampled.index).tolist())
    return sorted(positions)
