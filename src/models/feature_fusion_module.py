"""`LightningModule` for the feature-space fusion stages (Steps 13-15).

The image-space counterpart, ``MRIClassificationModule``, takes ``(image, label)``
batches. This one takes ``(classical, spatial, quantum, label)`` from the cached feature
datamodule. Everything else - the metric set, the macro-F1 selection metric, the class
weight injection - is deliberately identical, so a fusion head's numbers are directly
comparable with a baseline's.

It serves Step 13's three fusion strategies, Step 14's loss ablation and final classifier,
Step 15's multi-seed protocol runs, and Step 20's no-quantum control.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

FeatureBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class FeatureFusionModule(LightningModule):
    """Trains a fusion head over frozen tri-branch features.

    :param net: A ``FusionNet`` taking ``(classical, spatial, quantum)``.
    :param optimizer: Partially-instantiated optimizer.
    :param scheduler: Partially-instantiated LR scheduler, or ``None``.
    :param criterion: Loss module. If it exposes ``set_class_weights``, weights from the
        datamodule's training split are injected during ``setup``.
    :param num_classes: Number of target classes.
    :param class_names: Class names ordered by label index.
    :param compile: Compile the network before fitting.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        criterion: Optional[torch.nn.Module] = None,
        num_classes: int = 4,
        class_names: Optional[List[str]] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()

        # Coerce before save_hyperparameters, which snapshots this frame's locals.
        class_names = (
            list(class_names) if class_names else [f"class_{i}" for i in range(num_classes)]
        )
        # Same exclusions as MRIClassificationModule: only plain data reaches the
        # checkpoint, so it stays loadable under torch >= 2.6.
        self.save_hyperparameters(
            logger=False, ignore=["net", "criterion", "optimizer", "scheduler"]
        )

        self.net = net
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss()
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.num_classes = num_classes
        self.class_names = class_names

        macro = {"num_classes": num_classes, "average": "macro"}

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self.train_f1 = MulticlassF1Score(**macro)
        self.val_f1 = MulticlassF1Score(**macro)
        self.test_f1 = MulticlassF1Score(**macro)

        self.val_bal_acc = MulticlassRecall(**macro)
        self.test_bal_acc = MulticlassRecall(**macro)
        self.test_precision = MulticlassPrecision(**macro)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_f1_best = MaxMetric()

    def forward(
        self, classical: torch.Tensor, spatial: torch.Tensor, quantum: torch.Tensor
    ) -> torch.Tensor:
        """:param classical: Classical branch features.
        :param spatial: Spatial-gate branch features.
        :param quantum: Quantum branch features.
        :return: Class logits.
        """
        return self.net(classical, spatial, quantum)

    def setup(self, stage: str) -> None:
        """Inject class weights and optionally compile.

        :param stage: ``fit``, ``validate``, ``test`` or ``predict``.
        """
        if hasattr(self.criterion, "set_class_weights"):
            weights = getattr(self.trainer.datamodule, "class_weights", None)
            if weights is not None:
                self.criterion.set_class_weights(weights)

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_train_start(self) -> None:
        """Clear validation metrics polluted by Lightning's sanity-check pass."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_bal_acc.reset()
        self.val_f1_best.reset()

    def model_step(self, batch: FeatureBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one batch through the fusion head and the loss.

        :param batch: ``(classical, spatial, quantum, labels)``.
        :return: ``(loss, predictions, targets)``.
        """
        classical, spatial, quantum, labels = batch
        logits = self.forward(classical, spatial, quantum)
        loss = self.criterion(logits, labels)
        return loss, torch.argmax(logits, dim=1), labels

    def training_step(self, batch: FeatureBatch, batch_idx: int) -> torch.Tensor:
        """:param batch: ``(classical, spatial, quantum, labels)``.
        :param batch_idx: Batch index.
        :return: Loss to backpropagate.
        """
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train/f1_macro", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: FeatureBatch, batch_idx: int) -> None:
        """:param batch: ``(classical, spatial, quantum, labels)``.
        :param batch_idx: Batch index.
        """
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_bal_acc(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val/f1_macro", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/bal_acc", self.val_bal_acc, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Track the best validation macro-F1 seen so far."""
        self.val_f1_best(self.val_f1.compute())
        self.log("val/f1_macro_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: FeatureBatch, batch_idx: int) -> None:
        """:param batch: ``(classical, spatial, quantum, labels)``.
        :param batch_idx: Batch index.
        """
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_bal_acc(preds, targets)
        self.test_precision(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1_macro", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/bal_acc", self.test_bal_acc, on_step=False, on_epoch=True)
        self.log("test/precision_macro", self.test_precision, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """:return: Lightning optimizer configuration, monitoring ``val/f1_macro``."""
        optimizer = self.optimizer_factory(params=self.trainer.model.parameters())

        if self.scheduler_factory is None:
            return {"optimizer": optimizer}

        scheduler = self.scheduler_factory(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/f1_macro",
                "interval": "epoch",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    _ = FeatureFusionModule(None, None)
