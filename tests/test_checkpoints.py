"""Tests for reloading trained branches (needed from Step 13 onward)."""

import pytest
import torch

from src.models.components.losses import CrossEntropyLoss, FocalLoss
from src.models.components.multiscale import MultiscaleClassifier
from src.models.mri_classification_module import MRIClassificationModule
from src.utils.checkpoints import find_checkpoint, freeze, load_module, load_state_dict


def build_module(**kwargs) -> MRIClassificationModule:
    """Build a small classification module for round-trip tests.

    :param kwargs: Forwarded to the net.
    :return: The module.
    """
    return MRIClassificationModule(
        net=MultiscaleClassifier(variant="spatial_gate", num_classes=4, channels=8, **kwargs),
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        criterion=CrossEntropyLoss(use_class_weights=True),
        num_classes=4,
    )


def write_checkpoint(module: torch.nn.Module, path) -> None:
    """Write a minimal Lightning-shaped checkpoint.

    :param module: Module whose weights to save.
    :param path: Destination file.
    """
    torch.save({"state_dict": module.state_dict()}, path)


def test_checkpoints_carry_no_derived_class_weights(tmp_path):
    """Class weights come from the training split, so they must not be checkpointed.

    Persisting them makes a checkpoint unloadable into a freshly built module - whose
    buffer is still None and has no matching key - which is exactly the failure that
    breaks every branch reload in Steps 13 onward.
    """
    module = build_module()
    module.criterion.set_class_weights(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert module.criterion.class_weights is not None

    assert not [key for key in module.state_dict() if "class_weights" in key]


def test_weights_round_trip_through_a_checkpoint(tmp_path):
    """A reloaded branch must reproduce the trained one exactly."""
    trained = build_module().eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = trained(images)

    path = tmp_path / "model.ckpt"
    write_checkpoint(trained, path)

    reloaded = load_module(path, module=build_module())
    with torch.no_grad():
        assert torch.allclose(reloaded(images), expected, atol=1e-6)


def test_loading_is_strict_by_default(tmp_path):
    """A silent partial load produces a model that runs and is quietly wrong."""
    path = tmp_path / "model.ckpt"
    write_checkpoint(build_module(), path)

    mismatched = MRIClassificationModule(
        net=MultiscaleClassifier(variant="fixed_3x3", num_classes=4, channels=8),
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        num_classes=4,
    )
    with pytest.raises(RuntimeError):
        load_module(path, module=mismatched)


def test_loading_a_checkpoint_with_a_focal_criterion_round_trips(tmp_path):
    """Both loss families exclude their derived weights, not just cross-entropy."""
    module = MRIClassificationModule(
        net=MultiscaleClassifier(variant="global_gate", num_classes=4, channels=8),
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        criterion=FocalLoss(gamma=2.0, use_class_weights=True),
        num_classes=4,
    )
    module.criterion.set_class_weights(torch.tensor([1.0, 1.0, 2.0, 2.0]))

    path = tmp_path / "focal.ckpt"
    write_checkpoint(module, path)
    load_module(path, module=MRIClassificationModule(
        net=MultiscaleClassifier(variant="global_gate", num_classes=4, channels=8),
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        criterion=FocalLoss(gamma=2.0, use_class_weights=True),
        num_classes=4,
    ))


def test_reloaded_module_is_in_eval_mode(tmp_path):
    """Frozen branches feed cached features; train-mode dropout would randomise them."""
    path = tmp_path / "model.ckpt"
    write_checkpoint(build_module(), path)
    assert not load_module(path, module=build_module()).training


def test_freeze_stops_gradients_and_disables_train_mode():
    """Freezing must also stop BatchNorm's running statistics drifting.

    A branch frozen only by `requires_grad = False` still updates its BN buffers on every
    forward pass, so its "frozen" features would change between epochs and silently
    corrupt the cached features Steps 13-15 are built on.
    """
    module = freeze(build_module().train())

    assert not module.training
    assert all(not p.requires_grad for p in module.parameters())

    bn = module.net.branch.stem.stem[1]
    before = bn.running_mean.clone()
    with torch.no_grad():
        module(torch.randn(4, 3, 64, 64) * 10)
    assert torch.allclose(bn.running_mean, before), "frozen module updated BatchNorm stats"


def test_find_checkpoint_prefers_the_best_epoch(tmp_path):
    """`last.ckpt` is not the selected model; the epoch file is."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for name in ("epoch_003.ckpt", "last.ckpt"):
        torch.save({"state_dict": {}}, checkpoints / name)

    assert find_checkpoint(tmp_path).name == "epoch_003.ckpt"
    assert find_checkpoint(checkpoints).name == "epoch_003.ckpt"
    assert find_checkpoint(tmp_path, prefer="last").name == "last.ckpt"


def test_find_checkpoint_reports_a_missing_run_clearly(tmp_path):
    """Pointing an analysis at the wrong directory is a common mistake."""
    with pytest.raises(FileNotFoundError):
        find_checkpoint(tmp_path)


def test_load_state_dict_rejects_a_non_lightning_file(tmp_path):
    """A raw state dict saved with torch.save is a plausible and confusing mistake."""
    path = tmp_path / "raw.pt"
    torch.save({"weight": torch.zeros(2)}, path)

    with pytest.raises(KeyError, match="state_dict"):
        load_state_dict(path)
