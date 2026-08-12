"""Tests for the Step 9 baselines and the Step 15 fixed protocol."""

import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.models.components.backbones import SimpleCNN
from src.models.components.quantum import (
    CIRCUIT_NAMES,
    DEFAULT_N_QUBITS,
    FixedQCNN,
    QuantumLayer,
    make_circuit,
    scale_to_angles,
)
from src.models.components.transfer import (
    ARCHITECTURES,
    FixedMultiscaleCNN,
    TransferBackbone,
    _last_linear_in_features,
)

# Weights are not downloaded in tests: the study's baselines need ~350MB of ImageNet
# checkpoints, and the wiring under test is independent of the values in them.
NO_WEIGHTS = {"weights": None}


# ------------------------------------------------------------- transfer backbones


@pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0", "vit_b_16", "swin_t"])
def test_transfer_backbones_follow_the_extract_contract(arch):
    """All four architectures must expose the same interface despite differing internals."""
    net = TransferBackbone(arch=arch, num_classes=4, **NO_WEIGHTS)
    images = torch.randn(2, 3, 224, 224)

    net.eval()
    with torch.no_grad():
        outputs = net.extract(images)

    assert set(outputs) >= {"logits", "features"}
    assert outputs["logits"].shape == (2, 4)
    assert outputs["features"].shape == (2, net.feature_dim)
    assert torch.isfinite(outputs["logits"]).all()

    with torch.no_grad():
        assert torch.allclose(net(images), net.extract(images)["logits"])


@pytest.mark.parametrize(
    "arch,expected_dim",
    [("resnet50", 2048), ("efficientnet_b0", 1280), ("vit_b_16", 768), ("swin_t", 768)],
)
def test_feature_dimensions_are_read_from_the_model(arch, expected_dim):
    """Dimensions are discovered, not hard-coded, so a torchvision change cannot go unnoticed."""
    net = TransferBackbone(arch=arch, num_classes=4, **NO_WEIGHTS)
    assert net.feature_dim == expected_dim


@pytest.mark.parametrize(
    "arch,max_trainable_fraction",
    [
        ("resnet50", 0.70),
        # EfficientNet-B0 is the outlier: its last three feature blocks hold most of the
        # network's parameters, so the reference's "fine-tune the last few blocks" in
        # fact leaves roughly 79% trainable. Recorded here rather than smoothed over,
        # because this backbone is also the Step 10 classical branch.
        ("efficientnet_b0", 0.85),
        ("vit_b_16", 0.30),
        ("swin_t", 0.70),
    ],
)
def test_partial_fine_tuning_leaves_part_of_the_backbone_frozen(arch, max_trainable_fraction):
    """Full fine-tuning on a few thousand images would overfit; only late blocks train."""
    net = TransferBackbone(arch=arch, num_classes=4, freeze_backbone=True, **NO_WEIGHTS)

    trainable = sum(p.numel() for p in net.backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.backbone.parameters())

    assert 0 < trainable < total, "expected a partially frozen backbone"
    assert trainable / total <= max_trainable_fraction

    # The fresh head must always train, otherwise nothing maps features to our classes.
    assert all(p.requires_grad for p in net.classifier.parameters())


def test_freezing_can_be_disabled():
    """Full fine-tuning stays available for anyone with the data to justify it."""
    net = TransferBackbone(arch="resnet50", num_classes=4, freeze_backbone=False, **NO_WEIGHTS)
    assert all(p.requires_grad for p in net.backbone.parameters())


def test_only_unfrozen_backbone_blocks_receive_gradients():
    """Verifies the freeze actually holds under a real backward pass."""
    net = TransferBackbone(arch="resnet50", num_classes=4, **NO_WEIGHTS)
    net(torch.randn(2, 3, 224, 224)).sum().backward()

    assert net.backbone.layer1[0].conv1.weight.grad is None, "frozen block received a gradient"
    assert net.backbone.layer4[0].conv1.weight.grad is not None, "unfrozen block got no gradient"


def test_unknown_architecture_is_rejected():
    """A typo in a model config must fail at construction."""
    with pytest.raises(ValueError, match="Unknown architecture"):
        TransferBackbone(arch="resnet51")


def test_architecture_registry_covers_the_specified_options():
    """Step 9 and Step 10 name specific backbone families; both must be available."""
    assert {"resnet50", "densenet121"} <= set(ARCHITECTURES)
    assert {"efficientnet_b0", "efficientnet_v2_s", "convnext_tiny"} <= set(ARCHITECTURES)
    assert {"vit_b_16", "swin_t"} <= set(ARCHITECTURES)

    families = {spec["family"] for spec in ARCHITECTURES.values()}
    assert families == {"cnn", "transformer"}


def test_head_width_discovery_handles_both_head_shapes():
    """Heads are Linear on some models and Sequential on others."""
    assert _last_linear_in_features(torch.nn.Linear(1280, 1000)) == 1280
    assert (
        _last_linear_in_features(
            torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(768, 1000))
        )
        == 768
    )
    with pytest.raises(TypeError):
        _last_linear_in_features(torch.nn.Identity())


# ------------------------------------------------------------- multiscale baseline


def test_fixed_multiscale_concatenates_every_kernel_path():
    """Baseline 6's feature width must reflect all parallel paths, unweighted."""
    net = FixedMultiscaleCNN(num_classes=4, channels=32, kernel_sizes=(3, 5, 7))
    assert net.feature_dim == 32 * 3

    outputs = net.extract(torch.randn(2, 3, 64, 64))
    assert outputs["logits"].shape == (2, 4)
    assert outputs["features"].shape == (2, 96)


def test_fixed_multiscale_has_no_gating_parameters():
    """It is the *ungated* control; a gate here would invalidate the Step 11 comparison."""
    net = FixedMultiscaleCNN(num_classes=4)
    names = [name for name, _ in net.named_parameters()]
    assert not any("gate" in name for name in names)


# --------------------------------------------------------------------- quantum


@pytest.mark.parametrize("name", CIRCUIT_NAMES)
def test_every_circuit_builds_and_returns_one_expectation_per_qubit(name):
    """Step 12: expectation values are the quantum features, one per wire."""
    circuit, weight_shapes = make_circuit(name, DEFAULT_N_QUBITS)
    weights = torch.randn(*weight_shapes["weights"]) * 0.1

    values = circuit(torch.randn(DEFAULT_N_QUBITS), weights)
    assert len(values) == DEFAULT_N_QUBITS


def test_circuits_cover_two_depths_and_two_entanglement_patterns():
    """Step 12: "Test at least two circuit depths and at least two entanglement patterns"."""
    shapes = {name: make_circuit(name)[1]["weights"] for name in CIRCUIT_NAMES}

    # Two depths within the basic-entangler family.
    assert shapes["fixed"][0] == 2
    assert shapes["deep"][0] == 4

    # Strongly-entangling layers carry a third rotation axis, so the pattern differs.
    assert len(shapes["fixed"]) == 2
    assert len(shapes["strong"]) == 3
    assert shapes["strong"][0] == 2
    assert shapes["combined"][0] == 4


def test_unknown_circuit_is_rejected():
    """Circuit names come from config; a typo must not silently pick a default."""
    with pytest.raises(ValueError, match="Unknown circuit"):
        make_circuit("bell")


def test_angle_scaling_stays_within_one_rotation_period():
    """Angle encoding is only stable inside [-pi, pi]."""
    angles = scale_to_angles(torch.tensor([-1e4, -1.0, 0.0, 1.0, 1e4]))
    assert torch.all(angles.abs() <= torch.pi + 1e-5)
    assert angles[2].item() == pytest.approx(0.0)


def test_quantum_layer_is_differentiable():
    """A quantum layer that blocks gradients would train nothing upstream of it."""
    layer = QuantumLayer(circuit_name="fixed", n_qubits=DEFAULT_N_QUBITS)
    angles = torch.randn(3, DEFAULT_N_QUBITS, requires_grad=True)

    output = layer(angles)
    assert output.shape == (3, DEFAULT_N_QUBITS)

    output.sum().backward()
    assert angles.grad is not None
    assert torch.isfinite(angles.grad).all()
    assert any(p.grad is not None for p in layer.parameters())


def test_quantum_layer_returns_output_on_the_input_device():
    """The CPU simulator forces a round trip; the result must come back where it started."""
    layer = QuantumLayer()
    angles = torch.randn(2, DEFAULT_N_QUBITS)
    assert layer(angles).device == angles.device


def test_fixed_qcnn_baseline_end_to_end():
    """Baseline 5 must train like any other model in the study."""
    net = FixedQCNN(num_classes=4, n_qubits=DEFAULT_N_QUBITS)
    images = torch.randn(2, 3, 64, 64)

    outputs = net.extract(images)
    assert outputs["logits"].shape == (2, 4)
    assert outputs["features"].shape == (2, DEFAULT_N_QUBITS)
    assert outputs["angles"].shape == (2, DEFAULT_N_QUBITS)

    outputs["logits"].sum().backward()
    assert net.reduce.weight.grad is not None
    assert torch.isfinite(net.reduce.weight.grad).all()


def test_quantum_layer_adds_few_parameters():
    """Step 20 asks about parameter efficiency; the circuit's own count is tiny."""
    layer = QuantumLayer(circuit_name="fixed", n_qubits=4)
    assert sum(p.numel() for p in layer.parameters()) == 2 * 4


# ------------------------------------------------------------- protocol integrity


def compose_train(*overrides: str):
    """Compose the training config in isolation from other tests.

    Hydra's `GlobalHydra` is process-global, so leaving an `initialize()` context open
    across a module collides with any other test that composes its own config.

    :param overrides: Hydra override strings.
    :return: The composed DictConfig.
    """
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=list(overrides))
    GlobalHydra.instance().clear()
    return cfg


@pytest.fixture(scope="module")
def protocol_cfg():
    """Compose a training config with the fixed protocol applied.

    :return: The composed DictConfig.
    """
    return compose_train("experiment=step09_baselines", "data.data_dir=/tmp/x")


def test_protocol_matches_the_specification(protocol_cfg):
    """Step 15 constrains optimiser, batch size, patience, scheduler and selection metric."""
    cfg = protocol_cfg

    assert cfg.model.optimizer._target_.endswith("AdamW")
    assert cfg.model.optimizer.weight_decay == 1e-4
    assert cfg.model.optimizer.lr in (1e-4, 3e-4), "Step 15 starts from 1e-4 or 3e-4"

    assert cfg.data.batch_size in (16, 32), "Step 15: commonly 16 or 32"

    assert 10 <= cfg.callbacks.early_stopping.patience <= 15, "Step 15: patience of 10 to 15"
    assert cfg.trainer.max_epochs >= cfg.callbacks.early_stopping.patience

    assert "CosineAnnealing" in cfg.model.scheduler._target_


def test_model_selection_uses_macro_f1_not_accuracy(protocol_cfg):
    """Step 15: "not only validation accuracy"."""
    cfg = protocol_cfg

    assert cfg.callbacks.model_checkpoint.monitor == "val/f1_macro"
    assert cfg.callbacks.model_checkpoint.mode == "max"
    assert cfg.callbacks.early_stopping.monitor == "val/f1_macro"
    assert cfg.callbacks.early_stopping.mode == "max"
    assert cfg.optimized_metric == "val/f1_macro"


def test_scheduler_cycle_tracks_the_epoch_ceiling(protocol_cfg):
    """A cosine cycle shorter or longer than the run would decay at the wrong rate."""
    assert protocol_cfg.model.scheduler.T_max == protocol_cfg.trainer.max_epochs


def test_resource_monitor_is_attached(protocol_cfg):
    """Step 20 needs training time and memory captured during training, not after."""
    assert "resource_monitor" in protocol_cfg.callbacks
    assert protocol_cfg.callbacks.resource_monitor._target_.endswith("ResourceMonitor")


@pytest.mark.parametrize(
    "model_name",
    [
        "baseline_simple_cnn",
        "baseline_resnet50",
        "baseline_efficientnet_b0",
        "baseline_vit",
        "baseline_swin",
        "baseline_fixed_qcnn",
        "baseline_fixed_multiscale",
    ],
)
def test_every_baseline_config_composes_under_the_protocol(model_name):
    """All seven Step 9 baselines must be sweepable as one multirun."""
    cfg = compose_train(
        "experiment=step09_baselines", f"model={model_name}", "data.data_dir=/tmp/x"
    )

    assert cfg.model._target_.endswith("MRIClassificationModule")
    assert cfg.model.num_classes == 4
    # The protocol must win over each model config's own defaults.
    assert cfg.callbacks.early_stopping.patience == 12
    assert cfg.trainer.max_epochs == 30
    assert OmegaConf.select(cfg, "model.net._target_") is not None
