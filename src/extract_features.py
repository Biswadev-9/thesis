import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# See src/train.py for what setup_root does.
# ------------------------------------------------------------------------------------ #

from src.utils import RankedLogger, extras, task_wrapper
from src.utils.checkpoints import find_checkpoint, freeze, load_module

log = RankedLogger(__name__, rank_zero_only=True)


class TriBranchExtractor(nn.Module):
    """Runs the three frozen branches and returns their features together.

    **The spatial features come from inside the Step 12 model, not from the Step 11
    checkpoint.** The Step 12 adaptive-quantum branch contains its own spatial-gate
    branch, trained jointly with the quantum mixture, and it is that copy whose features
    are fused. The separately trained Step 11 checkpoint is used only for the arm ablation
    and the gate-morphology analysis.

    This is inherited from the reference notebook and is preserved deliberately: swapping
    in the Step 11 weights would change every downstream result. It is easy to
    misread as a bug, so it is stated here and asserted in the tests.

    :param classical: Trained Step 10 module.
    :param adaptive_quantum: Trained Step 12 module.
    """

    def __init__(self, classical: nn.Module, adaptive_quantum: nn.Module) -> None:
        super().__init__()
        self.classical = freeze(classical)
        self.adaptive_quantum = freeze(adaptive_quantum)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """:param images: Image batch, ``(B, 3, H, W)``.
        :return: ``{"classical", "spatial", "quantum", "quantum_weights"}``.
        """
        classical_features = self.classical.net.extract(images)["features"]
        quantum_outputs = self.adaptive_quantum.net.branch(images)

        return {
            "classical": classical_features,
            "spatial": quantum_outputs["classical_features"],
            "quantum": quantum_outputs["quantum_features"],
            "quantum_weights": quantum_outputs["quantum_weights"],
        }


def _resolve(path: Optional[str], role: str) -> Path:
    """Resolve a checkpoint path, accepting a run directory.

    :param path: Checkpoint file or run directory.
    :param role: Human-readable name, used in the error message.
    :return: Path to a checkpoint file.
    :raises ValueError: If ``path`` is missing.
    """
    if not path:
        raise ValueError(
            f"{role} checkpoint is required. Point it at the run directory or checkpoint "
            f"from the corresponding training step."
        )
    candidate = Path(path)
    return find_checkpoint(candidate) if candidate.is_dir() else candidate


@task_wrapper
def extract(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Caches the frozen tri-branch features for every split.

    Steps 13, 14, 15 and 20 all train many small heads over the *same* frozen branch
    outputs. Recomputing those outputs per epoch would mean re-running EfficientNet-B0 and
    five quantum circuits every time - and the quantum branch alone is the slowest
    component in the study. Extracting once turns the fusion stages from hours into
    seconds and removes the CPU simulator from the training loop entirely.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with the manifest and a dict with all instantiated objects.
    """
    device = torch.device(
        "cuda" if (cfg.get("accelerator") != "cpu" and torch.cuda.is_available()) else "cpu"
    )

    classical_ckpt = _resolve(cfg.get("classical_ckpt"), "Step 10 classical branch")
    quantum_ckpt = _resolve(cfg.get("quantum_ckpt"), "Step 12 adaptive quantum branch")

    log.info(f"Loading classical branch from {classical_ckpt}")
    classical = load_module(classical_ckpt, model_cfg=cfg.classical_model).to(device)

    log.info(f"Loading adaptive quantum branch from {quantum_ckpt}")
    adaptive_quantum = load_module(quantum_ckpt, model_cfg=cfg.quantum_model).to(device)

    extractor = TriBranchExtractor(classical, adaptive_quantum).to(device).eval()

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    output_dir = Path(cfg.paths.data_dir) / "features" / cfg.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "train": datamodule.train_dataloader,
        "val": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }

    manifest: Dict[str, Any] = {
        "tag": cfg.tag,
        "classical_ckpt": str(classical_ckpt),
        "quantum_ckpt": str(quantum_ckpt),
        "recipe": datamodule.hparams.recipe,
        "image_size": datamodule.hparams.image_size,
        "normalize": datamodule.hparams.normalize,
        "device": str(device),
        "splits": {},
    }

    for split in cfg.get("splits", ["train", "val", "test"]):
        started = time.perf_counter()
        cached = _extract_split(extractor, loaders[split](), device)
        torch.save(cached, output_dir / f"{split}.pt")

        manifest["splits"][split] = {
            "n_samples": int(len(cached["labels"])),
            "classical_dim": int(cached["classical"].shape[1]),
            "spatial_dim": int(cached["spatial"].shape[1]),
            "quantum_dim": int(cached["quantum"].shape[1]),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        log.info(
            f"{split}: {len(cached['labels'])} samples -> "
            f"classical {tuple(cached['classical'].shape)}, "
            f"spatial {tuple(cached['spatial'].shape)}, "
            f"quantum {tuple(cached['quantum'].shape)} "
            f"({manifest['splits'][split]['elapsed_seconds']}s)"
        )

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    log.info(f"Feature cache ready at {output_dir}")
    log.info(f"Train a fusion head with: python src/train.py experiment=step13_fusion data.tag={cfg.tag}")

    return manifest, {"cfg": cfg, "datamodule": datamodule, "extractor": extractor}


@torch.no_grad()
def _extract_split(extractor: TriBranchExtractor, loader: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    """Run the extractor over one split.

    :param extractor: The frozen tri-branch extractor.
    :param loader: Dataloader for the split.
    :param device: Device to run on.
    :return: Stacked features and labels, on CPU.
    """
    buffers: Dict[str, list] = {
        "classical": [],
        "spatial": [],
        "quantum": [],
        "quantum_weights": [],
        "labels": [],
    }

    for images, labels in loader:
        outputs = extractor(images.to(device))
        for key, value in outputs.items():
            buffers[key].append(value.cpu())
        buffers["labels"].append(labels)

    return {key: torch.cat(value) for key, value in buffers.items()}


@hydra.main(version_base="1.3", config_path="../configs", config_name="extract_features.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for feature extraction.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    extract(cfg)


if __name__ == "__main__":
    main()
