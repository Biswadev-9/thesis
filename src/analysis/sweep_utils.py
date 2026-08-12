"""Shared machinery for the Step 6 and Step 8 selection studies.

Both studies answer the same shape of question: train one small model per candidate
configuration on a fixed balanced subset, then rank the candidates by validation
performance. The reference notebook wrote a separate hand-rolled training loop for each
study, which is how their protocols drifted apart - different optimisers, different
epoch counts, different metric handling.

One trial function keeps every candidate on identical footing, and reuses the project's
own `LightningModule` so the metrics are computed by the same code that will report the
final results.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from lightning import LightningDataModule, Trainer, seed_everything
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from torch import nn

from src.models.mri_classification_module import MRIClassificationModule
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _optimizer_factory(lr: float, weight_decay: float) -> Callable:
    """Build a partially-applied AdamW matching the Step 15 protocol.

    :param lr: Learning rate.
    :param weight_decay: Weight decay.
    :return: Callable taking ``params`` and returning an optimizer.
    """

    def make(params: Any) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    return make


def _scheduler_factory(epochs: int) -> Callable:
    """Build a partially-applied cosine-annealing scheduler.

    :param epochs: Cycle length, matching the trial's epoch budget.
    :return: Callable taking ``optimizer`` and returning a scheduler.
    """

    def make(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    return make


@torch.no_grad()
def collect_predictions(
    module: MRIClassificationModule, dataloader: Any, device: torch.device
) -> Dict[str, np.ndarray]:
    """Run a trained module over a loader and gather predictions.

    :param module: Trained module.
    :param dataloader: Loader to evaluate.
    :param device: Device to run on.
    :return: ``{"y_true", "y_pred", "y_prob"}`` as numpy arrays.
    """
    module = module.to(device).eval()

    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []

    for images, labels in dataloader:
        logits = module(images.to(device))
        probs = torch.softmax(logits, dim=1)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        probabilities.append(probs.cpu().numpy())
        targets.append(labels.numpy())

    return {
        "y_true": np.concatenate(targets),
        "y_pred": np.concatenate(predictions),
        "y_prob": np.concatenate(probabilities),
    }


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """Compute the metrics Step 8 names as the basis for judging imbalance handling.

    The specification is explicit that "Accuracy alone is not sufficient", requiring
    macro-F1, balanced accuracy and class-wise recall.

    :param y_true: Ground-truth labels.
    :param y_pred: Predicted labels.
    :param class_names: Class names ordered by label index.
    :return: Metric mapping, including one ``recall_<class>`` entry per class.
    """
    labels = list(range(len(class_names)))
    per_class = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    summary: Dict[str, float] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "min_class_recall": float(np.min(per_class)),
    }
    summary.update(
        {f"recall_{name}": float(value) for name, value in zip(class_names, per_class)}
    )
    return summary


def run_proxy_trial(
    datamodule: LightningDataModule,
    net: nn.Module,
    criterion: Optional[nn.Module] = None,
    epochs: int = 5,
    seed: int = 42,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    accelerator: str = "auto",
    devices: int = 1,
    deterministic: bool = False,
) -> Dict[str, float]:
    """Train one candidate on the proxy subset and score it on validation.

    All candidates in a study share this function, so any difference in the reported
    numbers comes from the candidate rather than from its training protocol.

    :param datamodule: Configured proxy datamodule.
    :param net: Freshly constructed network; must not be reused across trials.
    :param criterion: Loss module, or ``None`` for unweighted cross-entropy.
    :param epochs: Training epochs.
    :param seed: Seed applied to torch, numpy and python before construction.
    :param lr: Learning rate.
    :param weight_decay: Weight decay.
    :param accelerator: Lightning accelerator.
    :param devices: Device count.
    :param deterministic: Request deterministic kernels; slower but reproducible.
    :return: Validation metrics, plus ``best_val_f1`` tracked during training.
    """
    seed_everything(seed, workers=True)

    module = MRIClassificationModule(
        net=net,
        optimizer=_optimizer_factory(lr, weight_decay),
        scheduler=_scheduler_factory(epochs),
        criterion=criterion,
        num_classes=datamodule.num_classes,
        class_names=datamodule.class_names,
    )

    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=deterministic,
    )
    trainer.fit(model=module, datamodule=datamodule)

    best_val_f1 = float(module.val_f1_best.compute())

    outputs = collect_predictions(module, datamodule.val_dataloader(), module.device)
    metrics = classification_summary(
        outputs["y_true"], outputs["y_pred"], datamodule.class_names
    )
    metrics["best_val_f1"] = best_val_f1

    # Free the graph before the next candidate; sweeps build many models per process.
    del module, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics
