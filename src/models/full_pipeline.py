"""The complete proposed model: raw image to class logits (Steps 16-20).

Steps 13-15 train the fusion head over *cached* features, which is fast but only works for
images that were in the cache. Steps 16 onward need the whole path from a raw image:

- Step 17 evaluates on an external dataset that was never cached;
- Step 18 evaluates on degraded copies of the test images;
- Step 19 needs gradients to flow back from the logits to the input pixels for Grad-CAM.

This module assembles the trained branches and fusion head into one `nn.Module` that does
exactly that.

**No ``torch.no_grad`` anywhere in the forward path.** The branches are frozen by
``requires_grad = False``, which already prevents their weights updating. Wrapping the
forward in ``no_grad`` as well would additionally block gradients from reaching the input,
silently breaking Grad-CAM in Phase 7 - the failure would look like a uniformly blank
saliency map rather than an error.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch import nn

from src.utils.pylogger import RankedLogger
from src.utils.checkpoints import find_checkpoint, freeze, load_module

log = RankedLogger(__name__, rank_zero_only=True)


class FullPipeline(nn.Module):
    """Frozen tri-branch feature extraction followed by the trained fusion head.

    :param classical_net: The Step 10 network (not the `LightningModule`).
    :param quantum_net: The Step 12 network, which also supplies the spatial features.
    :param fusion_net: The Step 14/15 fusion head.
    :param freeze_branches: Freeze the two branches. Left enabled for evaluation; the
        branches were trained in their own steps and must not drift here.
    """

    def __init__(
        self,
        classical_net: nn.Module,
        quantum_net: nn.Module,
        fusion_net: nn.Module,
        freeze_branches: bool = True,
    ) -> None:
        super().__init__()
        self.classical_net = freeze(classical_net) if freeze_branches else classical_net
        self.quantum_net = freeze(quantum_net) if freeze_branches else quantum_net
        self.fusion_net = fusion_net

    def extract(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the whole pipeline and expose every intermediate the analyses need.

        :param images: Image batch, ``(B, 3, H, W)``.
        :return: ``{"logits", "fused", "classical", "spatial", "quantum",
            "quantum_weights", "gate_maps"}``.
        """
        classical = self.classical_net.extract(images)["features"]
        quantum_outputs = self.quantum_net.branch(images)

        spatial = quantum_outputs["classical_features"]
        quantum = quantum_outputs["quantum_features"]

        fused = self.fusion_net.extract(classical, spatial, quantum)

        return {
            "logits": fused["logits"],
            "fused": fused["fused"],
            "classical": classical,
            "spatial": spatial,
            "quantum": quantum,
            "quantum_weights": quantum_outputs["quantum_weights"],
            "gate_maps": quantum_outputs["gate_maps"],
            **({"branch_weights": fused["branch_weights"]} if "branch_weights" in fused else {}),
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """:param images: Image batch, ``(B, 3, H, W)``.
        :return: Class logits, ``(B, num_classes)``.
        """
        return self.extract(images)["logits"]


def _resolve(path: Optional[str], role: str) -> Path:
    """Resolve a checkpoint path, accepting a run directory.

    :param path: Checkpoint file or run directory.
    :param role: Human-readable name for the error message.
    :return: Path to a checkpoint file.
    :raises ValueError: If ``path`` is missing.
    """
    if not path:
        raise ValueError(f"{role} checkpoint is required but was not configured.")
    candidate = Path(path)
    return find_checkpoint(candidate) if candidate.is_dir() else candidate


def load_full_pipeline(
    classical_ckpt: str,
    quantum_ckpt: str,
    fusion_ckpt: str,
    classical_model: DictConfig,
    quantum_model: DictConfig,
    fusion_model: DictConfig,
    device: Optional[torch.device] = None,
) -> FullPipeline:
    """Rebuild the complete trained model from three checkpoints.

    Checkpoints in this project carry weights only, so each module is reconstructed from
    its config and then loaded - see ``src/utils/checkpoints.py``.

    :param classical_ckpt: Step 10 checkpoint or run directory.
    :param quantum_ckpt: Step 12 checkpoint or run directory.
    :param fusion_ckpt: Step 15 checkpoint or run directory.
    :param classical_model: Hydra config for the Step 10 module.
    :param quantum_model: Hydra config for the Step 12 module.
    :param fusion_model: Hydra config for the fusion module.
    :param device: Device to place the pipeline on.
    :return: The assembled pipeline, in eval mode.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classical = load_module(_resolve(classical_ckpt, "Step 10 classical branch"), model_cfg=classical_model)
    quantum = load_module(_resolve(quantum_ckpt, "Step 12 adaptive quantum branch"), model_cfg=quantum_model)
    fusion = load_module(_resolve(fusion_ckpt, "Step 15 final model"), model_cfg=fusion_model)

    pipeline = FullPipeline(
        classical_net=classical.net,
        quantum_net=quantum.net,
        fusion_net=fusion.net,
    ).to(device)

    log.info(
        f"Full pipeline assembled on {device} "
        f"({sum(p.numel() for p in pipeline.parameters()):,} parameters)"
    )
    return pipeline.eval()


@torch.no_grad()
def predict(pipeline: nn.Module, loader: Any, device: torch.device) -> Dict[str, Any]:
    """Run any image-space model over a loader and collect predictions.

    Accepts both the assembled pipeline and a plain baseline network, so Step 18 can
    compare the proposed model against the CNN and Transformer baselines through one code
    path.

    :param pipeline: Model taking images and returning logits.
    :param loader: Dataloader yielding ``(images, labels)``.
    :param device: Device to run on.
    :return: ``{"y_true", "y_pred", "y_prob"}`` as numpy arrays.
    """
    import numpy as np

    pipeline = pipeline.to(device).eval()

    predictions, targets, probabilities = [], [], []
    for images, labels in loader:
        logits = pipeline(images.to(device))
        if isinstance(logits, dict):
            logits = logits["logits"]
        probs = torch.softmax(logits, dim=1)

        predictions.append(logits.argmax(dim=1).cpu().numpy())
        probabilities.append(probs.cpu().numpy())
        targets.append(labels.numpy())

    return {
        "y_true": np.concatenate(targets),
        "y_pred": np.concatenate(predictions),
        "y_prob": np.concatenate(probabilities),
    }
