"""Loading trained branches back out of checkpoints.

Steps 13 onward compose the study's final model from three separately trained branches,
so reloading a trained module is a routine operation from here on.

Checkpoints in this project deliberately carry **weights only** - the network, loss,
optimizer and scheduler are excluded from ``save_hyperparameters`` so the file stays
loadable under torch >= 2.6 (see ``src/utils/serialization.py``). The consequence is that
``LightningModule.load_from_checkpoint`` cannot rebuild a model on its own: the
architecture must be reconstructed from its config first, then the weights loaded into it.
These helpers make that the easy path.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def find_checkpoint(run_dir: Union[str, Path], prefer: str = "best") -> Path:
    """Locate a checkpoint inside a training run directory.

    :param run_dir: A Hydra run directory, or the ``checkpoints/`` folder itself.
    :param prefer: ``best`` picks the highest-scoring epoch checkpoint, ``last`` picks
        ``last.ckpt``.
    :return: Path to the chosen checkpoint.
    :raises FileNotFoundError: If no checkpoint is present.
    """
    run_dir = Path(run_dir)
    directory = run_dir if run_dir.name == "checkpoints" else run_dir / "checkpoints"

    if not directory.is_dir():
        raise FileNotFoundError(f"No checkpoints directory under {run_dir}")

    last = directory / "last.ckpt"
    if prefer == "last":
        if last.is_file():
            return last
        raise FileNotFoundError(f"No last.ckpt in {directory}")

    # The checkpoint callback names epoch files `epoch_XXX.ckpt` and keeps only the best,
    # so any epoch file present is the best one; fall back to last.ckpt otherwise.
    epochs = sorted(p for p in directory.glob("epoch_*.ckpt"))
    if epochs:
        return epochs[-1]
    if last.is_file():
        log.warning(f"No epoch checkpoint in {directory}; falling back to last.ckpt")
        return last

    raise FileNotFoundError(f"No checkpoint files in {directory}")


def load_state_dict(ckpt_path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    """Read a checkpoint's ``state_dict``.

    :param ckpt_path: Path to a Lightning checkpoint.
    :param map_location: Device to map tensors onto.
    :return: The state dict.
    :raises FileNotFoundError: If the checkpoint does not exist.
    :raises KeyError: If the file carries no ``state_dict``.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=map_location)
    if "state_dict" not in checkpoint:
        raise KeyError(f"{ckpt_path} contains no 'state_dict' - is it a Lightning checkpoint?")
    return checkpoint["state_dict"]


def load_module(
    ckpt_path: Union[str, Path],
    model_cfg: Optional[DictConfig] = None,
    module: Optional[torch.nn.Module] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> torch.nn.Module:
    """Rebuild a `LightningModule` and load trained weights into it.

    :param ckpt_path: Path to the checkpoint.
    :param model_cfg: Hydra config for the module, used when ``module`` is not supplied.
    :param module: An already-constructed module to load weights into.
    :param map_location: Device to map tensors onto.
    :param strict: Require the checkpoint's keys to match the module exactly. Leave
        enabled: a silent partial load produces a model that runs and is wrong.
    :return: The module with weights loaded, in eval mode.
    :raises ValueError: If neither ``model_cfg`` nor ``module`` is given.
    """
    if module is None:
        if model_cfg is None:
            raise ValueError("Provide either `model_cfg` or `module`")
        import hydra

        module = hydra.utils.instantiate(model_cfg)

    missing, unexpected = module.load_state_dict(
        load_state_dict(ckpt_path, map_location=map_location), strict=strict
    )
    if missing or unexpected:
        log.warning(f"Loaded with missing={list(missing)}, unexpected={list(unexpected)}")

    log.info(f"Loaded weights from {ckpt_path}")
    return module.eval()


def freeze(module: torch.nn.Module) -> torch.nn.Module:
    """Freeze a module and put it in eval mode.

    Both halves matter. ``requires_grad = False`` stops the weights updating; ``eval()``
    stops BatchNorm updating its running statistics and disables dropout. Freezing without
    the second gives a "frozen" branch whose outputs still drift between epochs, which
    would quietly corrupt the cached features Steps 13-15 are built on.

    :param module: Module to freeze.
    :return: The same module, frozen and in eval mode.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module.eval()
