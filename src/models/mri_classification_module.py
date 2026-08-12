"""`LightningModule` for image-space MRI classification.

One module serves every image-space model in the study: all seven Step 9 baselines, the
Step 10 classical branch, the eight Step 11 multiscale arms, the Step 12 quantum branch
and the Step 20 fixed-quantum control. Architectures differ only in the injected ``net``,
so switching model is a config change rather than new code.

Metric choice follows Step 15: *"Save the best model using validation macro-F1 or
balanced accuracy, not only validation accuracy."* Macro-F1 is the selection metric
everywhere; ``val/f1_macro_best`` is tracked so multi-seed runs can be aggregated without
re-reading logs.
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


class MRIClassificationModule(LightningModule):
    """Classification module wrapping any network that follows the ``FeatureNet`` contract.

    :param net: Network exposing ``forward(x) -> logits``. Nets deriving from
        ``FeatureNet`` additionally expose ``extract`` for the explainability stages.
    :param optimizer: Partially-instantiated optimizer, bound to parameters here.
    :param scheduler: Partially-instantiated LR scheduler, or ``None``.
    :param criterion: Loss module. If it exposes ``set_class_weights``, weights from the
        datamodule's training split are injected during ``setup``.
    :param num_classes: Number of target classes.
    :param class_names: Class names ordered by label index, used for per-class logging.
    :param compile: Compile the network with ``torch.compile`` before fitting.
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

        # Coerce before save_hyperparameters: it snapshots this frame's locals, so a
        # value converted afterwards would still be stored in its original omegaconf
        # form and would break `weights_only` checkpoint loading.
        class_names = (
            list(class_names) if class_names else [f"class_{i}" for i in range(num_classes)]
        )

        # Only plain data reaches `self.hparams`, and therefore the checkpoint.
        #
        # `net` and `criterion` are modules with their own state, so saving them would
        # duplicate weights. `optimizer` and `scheduler` arrive from Hydra as
        # `functools.partial` objects: pickling those into the checkpoint makes it
        # unloadable under torch >= 2.6, whose `torch.load` defaults to
        # `weights_only=True` and refuses arbitrary callables. Checkpoints in this
        # project are always reloaded by reconstructing the model from its config and
        # then loading the state dict, so nothing is lost by keeping them out.
        self.save_hyperparameters(
            logger=False, ignore=["net", "criterion", "optimizer", "scheduler"]
        )

        self.net = net
        self.criterion = criterion if criterion is not None else torch.nn.CrossEntropyLoss()
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.num_classes = num_classes
        self.class_names = class_names

        metric_kwargs = {"num_classes": num_classes, "average": "macro"}

        # Separate metric instances per stage: torchmetrics accumulates internal state,
        # so sharing one object across stages would blend train and val statistics.
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self.train_f1 = MulticlassF1Score(**metric_kwargs)
        self.val_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_f1 = MulticlassF1Score(**metric_kwargs)

        # Macro-averaged recall is balanced accuracy - the specification's alternative
        # selection metric, reported alongside macro-F1 throughout.
        self.val_bal_acc = MulticlassRecall(**metric_kwargs)
        self.test_bal_acc = MulticlassRecall(**metric_kwargs)
        self.test_precision = MulticlassPrecision(**metric_kwargs)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_f1_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Image batch, ``(B, C, H, W)``.
        :return: Class logits, ``(B, num_classes)``.
        """
        return self.net(x)

    def setup(self, stage: str) -> None:
        """Inject class weights and optionally compile, before fitting begins.

        Class weights are read from the datamodule rather than the config because they
        depend on the realised training split, which only exists after ``prepare_data``.

        :param stage: ``fit``, ``validate``, ``test`` or ``predict``.
        """
        if hasattr(self.criterion, "set_class_weights"):
            weights = getattr(self.trainer.datamodule, "class_weights", None)
            if weights is not None:
                self.criterion.set_class_weights(weights)
                if getattr(self.criterion, "use_class_weights", False):
                    log.info(f"Class weights installed on {type(self.criterion).__name__}")

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_train_start(self) -> None:
        """Clear validation metrics polluted by Lightning's sanity-check pass."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_bal_acc.reset()
        self.val_f1_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one batch through the network and the loss.

        :param batch: ``(images, labels)``.
        :return: ``(loss, predictions, targets)``.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return loss, torch.argmax(logits, dim=1), y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """:param batch: ``(images, labels)``.
        :param batch_idx: Index of the batch within the epoch.
        :return: Loss to backpropagate.
        """
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/f1_macro", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """:param batch: ``(images, labels)``.
        :param batch_idx: Index of the batch within the epoch.
        """
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_bal_acc(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/f1_macro", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/bal_acc", self.val_bal_acc, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        """Track the best validation macro-F1 seen so far.

        Logged via ``.compute()`` rather than as a metric object: Lightning resets metric
        objects each epoch, which would discard the running maximum.
        """
        self.val_f1_best(self.val_f1.compute())
        self.log("val/f1_macro_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """:param batch: ``(images, labels)``.
        :param batch_idx: Index of the batch within the epoch.
        """
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_bal_acc(preds, targets)
        self.test_precision(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/f1_macro", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/bal_acc", self.test_bal_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/precision_macro", self.test_precision, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Bind the configured optimizer and scheduler.

        :return: Lightning optimizer configuration. The scheduler monitors
            ``val/f1_macro`` so plateau-style schedulers follow the same signal used for
            model selection.
        """
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
    _ = MRIClassificationModule(None, None, None)
