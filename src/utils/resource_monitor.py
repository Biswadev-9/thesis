"""Training-time and memory accounting (Step 20).

Step 20 requires the efficiency comparison to report "trainable parameters, inference
time, training time, memory usage, and performance metrics". The reference notebook
reported only parameter counts and inference time, so training cost and memory were never
measured - and by the time Step 20 ran, the models had long since finished training.

Measuring during training is the only way to capture it without retraining everything, so
this callback attaches to every run from Step 9 onward and writes its numbers into the run
directory alongside the checkpoints.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning import Callback, LightningModule, Trainer

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ResourceMonitor(Callback):
    """Record wall-clock training time, per-epoch timing and peak memory.

    :param filename: Artefact name written into the run directory.
    :param log_epoch_time: Log each epoch's duration as a metric, so the logger captures
        the trend as well as the total.
    """

    def __init__(
        self, filename: str = "resource_usage.json", log_epoch_time: bool = True
    ) -> None:
        super().__init__()
        self.filename = filename
        self.log_epoch_time = log_epoch_time

        self._fit_started: Optional[float] = None
        self._epoch_started: Optional[float] = None
        self.epoch_times: List[float] = []
        self.summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------- hooks

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start the clock and reset CUDA's peak-memory counter.

        :param trainer: The trainer.
        :param pl_module: The module being fitted.
        """
        self._fit_started = time.perf_counter()
        self.epoch_times = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """:param trainer: The trainer.
        :param pl_module: The module being fitted.
        """
        self._epoch_started = time.perf_counter()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record the epoch's duration.

        :param trainer: The trainer.
        :param pl_module: The module being fitted.
        """
        if self._epoch_started is None:
            return

        elapsed = time.perf_counter() - self._epoch_started
        self.epoch_times.append(elapsed)
        if self.log_epoch_time:
            pl_module.log("train/epoch_time_sec", elapsed, on_step=False, on_epoch=True)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Assemble and persist the resource summary.

        :param trainer: The trainer.
        :param pl_module: The module that was fitted.
        """
        total = time.perf_counter() - self._fit_started if self._fit_started else 0.0
        parameters = self._count_parameters(pl_module)

        self.summary = {
            "training_time_sec": round(total, 3),
            "epochs_completed": len(self.epoch_times),
            "mean_epoch_time_sec": (
                round(sum(self.epoch_times) / len(self.epoch_times), 3)
                if self.epoch_times
                else None
            ),
            "epoch_times_sec": [round(value, 3) for value in self.epoch_times],
            **parameters,
            **self._memory_usage(),
            "accelerator": trainer.accelerator.__class__.__name__,
            "devices": trainer.num_devices,
        }

        destination = self._output_dir(trainer)
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            with open(destination / self.filename, "w", encoding="utf-8") as handle:
                json.dump(self.summary, handle, indent=2)

        log.info(
            f"Training finished in {self.summary['training_time_sec']}s over "
            f"{self.summary['epochs_completed']} epochs "
            f"({parameters['trainable_parameters']:,} trainable parameters)"
        )

    # --------------------------------------------------------------- internals

    @staticmethod
    def _count_parameters(pl_module: LightningModule) -> Dict[str, int]:
        """:param pl_module: The module.
        :return: Total and trainable parameter counts.

        Both matter: the transfer baselines freeze most of their backbone, so total
        parameters describes model size while trainable describes what the run actually
        optimised.
        """
        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

    @staticmethod
    def _memory_usage() -> Dict[str, Optional[float]]:
        """:return: Peak memory in MiB, or ``None`` on CPU where no equivalent counter
        exists.
        """
        if not torch.cuda.is_available():
            return {"peak_memory_mib": None, "memory_device": "cpu"}

        peak = torch.cuda.max_memory_allocated() / (1024**2)
        return {"peak_memory_mib": round(peak, 2), "memory_device": torch.cuda.get_device_name(0)}

    @staticmethod
    def _output_dir(trainer: Trainer) -> Optional[Path]:
        """:param trainer: The trainer.
        :return: Directory for the artefact, or ``None`` if none can be determined.

        ``default_root_dir`` is preferred over ``log_dir`` because it is the Hydra run
        directory - the same level as ``checkpoints/``. ``log_dir`` points inside the
        attached logger's own tree (``csv/version_0/``), which would scatter the files
        Step 20 has to collect across runs.
        """
        for candidate in (trainer.default_root_dir, trainer.log_dir):
            if candidate:
                return Path(candidate)
        return None
