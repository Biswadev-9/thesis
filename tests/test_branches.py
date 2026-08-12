"""Tests for the Step 10-12 branches: multiscale gating, arms, and adaptive quantum."""

import pytest
import torch

from src.models.components.multiscale import (
    ARMS,
    PATH_LABELS,
    VARIANTS,
    ConvStem,
    GlobalMultiScaleGate,
    MultiscaleBranch,
    MultiscaleClassifier,
    SpatialMultiScaleGate,
)
from src.models.components.quantum import (
    CIRCUIT_NAMES,
    AdaptiveQuantumBranch,
    AdaptiveQuantumClassifier,
    AdaptiveQuantumSelector,
)

CHANNELS = 8  # narrow for speed; the properties under test are width-independent


# --------------------------------------------------------------------- stem


def test_stem_downsamples_by_four():
    """The gate operates at input/4, which the morphology analysis upsamples from."""
    stem = ConvStem(channels=CHANNELS)
    assert stem(torch.randn(2, 3, 224, 224)).shape == (2, CHANNELS, 56, 56)


# --------------------------------------------------------------- spatial gate


def test_spatial_gate_weights_sum_to_one_at_every_pixel():
    """A softmax over paths per pixel is what makes the weights interpretable.

    If they did not sum to 1, the "which receptive field does the model trust here"
    reading - and Step 11's whole morphology analysis - would be meaningless.
    """
    gate = SpatialMultiScaleGate(CHANNELS)
    fused, weights = gate(torch.randn(2, CHANNELS, 16, 16))

    assert fused.shape == (2, CHANNELS, 16, 16)
    assert weights.shape == (2, 3, 16, 16)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 16, 16), atol=1e-5)
    assert (weights >= 0).all()


def test_spatial_gate_weights_actually_vary_across_space():
    """A per-pixel gate that emits a constant map is a global gate with extra cost."""
    torch.manual_seed(0)
    gate = SpatialMultiScaleGate(CHANNELS)

    # Left half and right half given clearly different content.
    x = torch.randn(1, CHANNELS, 16, 16)
    x[:, :, :, :8] *= 5.0

    _, weights = gate(x)
    assert weights.std(dim=(2, 3)).max() > 1e-4, "gate produced a spatially constant map"


def test_gradients_reach_all_three_paths():
    """A path that never receives gradient is dead weight and would silently skew the arm."""
    gate = SpatialMultiScaleGate(CHANNELS)
    fused, _ = gate(torch.randn(2, CHANNELS, 16, 16))
    fused.sum().backward()

    for name in ("path_fine", "path_medium", "path_broad"):
        conv = getattr(gate, name)[0]
        assert conv.weight.grad is not None, f"{name} received no gradient"
        assert conv.weight.grad.abs().sum() > 0, f"{name} gradient is identically zero"


def test_broad_path_has_a_wider_receptive_field_than_the_fine_path():
    """The three paths must genuinely differ in reach, or the ablation tests nothing."""
    gate = SpatialMultiScaleGate(CHANNELS)
    fine = gate.path_fine[0]
    broad = gate.path_broad[0]

    fine_reach = fine.dilation[0] * (fine.kernel_size[0] - 1) + 1
    broad_reach = broad.dilation[0] * (broad.kernel_size[0] - 1) + 1
    medium_reach = gate.path_medium[0].kernel_size[0]

    assert fine_reach == 3
    assert medium_reach == 5
    assert broad_reach == 7, "dilated path should reach 7x7 at 3x3 parameter cost"


# ---------------------------------------------------------------- global gate


def test_global_gate_emits_one_weight_per_path_per_image():
    """Arm 5's control: adaptive, but with no spatial dimension at all."""
    gate = GlobalMultiScaleGate(CHANNELS)
    fused, weights = gate(torch.randn(4, CHANNELS, 16, 16))

    assert fused.shape == (4, CHANNELS, 16, 16)
    assert weights.shape == (4, 3), "global gate must not carry spatial dimensions"
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


def test_global_gate_has_fewer_parameters_than_the_spatial_gate():
    """The spatial gate's edge must not come from simply being a bigger module.

    It is bigger here, so any gain it shows should be reported alongside this fact rather
    than attributed purely to spatial adaptivity.
    """
    spatial = sum(p.numel() for p in SpatialMultiScaleGate(32).gate_head.parameters())
    glob = sum(p.numel() for p in GlobalMultiScaleGate(32).gate_fc.parameters())
    assert glob < spatial


# --------------------------------------------------------------------- arms


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_produces_the_same_feature_width(variant):
    """All arms must feed an identically shaped head, or they are not comparable."""
    branch = MultiscaleBranch(variant=variant, channels=CHANNELS)
    features, _ = branch(torch.randn(2, 3, 64, 64))

    assert features.shape == (2, CHANNELS)
    assert branch.feature_dim == CHANNELS


def test_concat_arm_projects_back_instead_of_tripling_its_features():
    """Arm 4 uses all three paths; without the 1x1 projection it would get 3x the width."""
    branch = MultiscaleBranch(variant="concat_nogate", channels=CHANNELS)
    features, weights = branch(torch.randn(2, 3, 64, 64))

    assert features.shape == (2, CHANNELS)
    assert weights is None, "the no-gate arm must not report gate weights"


@pytest.mark.parametrize("variant", ["fixed_3x3", "fixed_5x5", "fixed_dilated", "concat_nogate"])
def test_ungated_arms_expose_no_gate_weights(variant):
    """Reporting weights for an ungated arm would be fabricating an explanation."""
    net = MultiscaleClassifier(variant=variant, num_classes=4, channels=CHANNELS)
    assert "gate_maps" not in net.extract(torch.randn(2, 3, 64, 64))


def test_all_eight_arms_are_addressable_and_distinct():
    """The ablation is swept by name, so the registry must be complete and unambiguous."""
    assert len(ARMS) == 8
    assert len({(v, q) for v, q in ARMS.values()}) == 8

    quantum_arms = [name for name, (_, q) in ARMS.items() if q]
    assert len(quantum_arms) == 2, "arms 7 and 8 are the quantum pair"


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_each_arm_builds_and_classifies(arm):
    """Every arm must satisfy the shared extract contract."""
    net = MultiscaleClassifier.from_arm(arm, num_classes=4, channels=CHANNELS)
    outputs = net.extract(torch.randn(2, 3, 64, 64))

    assert outputs["logits"].shape == (2, 4)
    assert outputs["features"].shape == (2, CHANNELS)
    assert torch.isfinite(outputs["logits"]).all()


def test_unknown_arm_and_variant_are_rejected():
    """Arm names come from a sweep; a typo must fail rather than silently pick a default."""
    with pytest.raises(ValueError, match="Unknown arm"):
        MultiscaleClassifier.from_arm("arm9_something")
    with pytest.raises(ValueError, match="Unknown variant"):
        MultiscaleBranch(variant="fixed_7x7")


def test_spatial_and_global_arms_differ_only_in_their_gate():
    """Arm 5 vs arm 6 must isolate spatial adaptivity, not confound it with other changes."""
    spatial = MultiscaleBranch("spatial_gate", channels=CHANNELS)
    glob = MultiscaleBranch("global_gate", channels=CHANNELS)

    def stem_shape(branch):
        return [tuple(p.shape) for p in branch.stem.parameters()]

    assert stem_shape(spatial) == stem_shape(glob)
    for name in ("path_fine", "path_medium", "path_broad"):
        s = [tuple(p.shape) for p in getattr(spatial.gate, name).parameters()]
        g = [tuple(p.shape) for p in getattr(glob.gate, name).parameters()]
        assert s == g


def test_path_labels_match_the_three_paths():
    """The morphology figures index these labels positionally."""
    assert len(PATH_LABELS) == 3


# ---------------------------------------------------------- adaptive quantum


def test_selector_emits_a_distribution_over_experts():
    """Step 12's mixture weights must be a proper distribution to be interpretable."""
    selector = AdaptiveQuantumSelector(feature_dim=CHANNELS, num_experts=5)
    weights = selector(torch.randn(4, CHANNELS))

    assert weights.shape == (4, 5)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (weights >= 0).all()


def test_selector_output_depends_on_the_input():
    """A selector that ignores its input is a fixed circuit choice with extra parameters."""
    torch.manual_seed(0)
    selector = AdaptiveQuantumSelector(feature_dim=CHANNELS, num_experts=5)
    first = selector(torch.randn(1, CHANNELS) * 10)
    second = selector(torch.randn(1, CHANNELS) * 10)
    assert not torch.allclose(first, second, atol=1e-4)


def test_adaptive_branch_mixes_all_five_circuits():
    """Step 12 requires at least two depths and two entanglement patterns to be tested."""
    branch = AdaptiveQuantumBranch(channels=CHANNELS)
    assert len(branch.experts) == len(CIRCUIT_NAMES) == 5
    assert branch.circuit_names == CIRCUIT_NAMES


def test_adaptive_branch_returns_everything_fusion_needs():
    """Step 13 consumes the classical and quantum vectors; Step 12's analysis the weights."""
    branch = AdaptiveQuantumBranch(channels=CHANNELS, n_qubits=4)
    outputs = branch(torch.randn(2, 3, 64, 64))

    assert outputs["classical_features"].shape == (2, CHANNELS)
    assert outputs["quantum_features"].shape == (2, 4)
    assert outputs["quantum_weights"].shape == (2, 5)
    assert outputs["gate_maps"].shape == (2, 3, 16, 16)
    assert torch.isfinite(outputs["quantum_features"]).all()


def test_quantum_features_are_a_weighted_mixture_of_the_experts():
    """Pins the mixture arithmetic: the output must be the selector-weighted expert sum."""
    torch.manual_seed(0)
    branch = AdaptiveQuantumBranch(channels=CHANNELS, n_qubits=4).eval()
    images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        outputs = branch(images)
        classical, gate_maps = branch.spatial_branch(images)

        from src.models.components.quantum import scale_to_angles

        angles = scale_to_angles(branch.reduce(classical))
        stacked = torch.stack([expert(angles) for expert in branch.experts], dim=1)
        expected = (stacked * outputs["quantum_weights"].unsqueeze(-1)).sum(dim=1)

    assert torch.allclose(outputs["quantum_features"], expected, atol=1e-5)


def test_mixture_is_bounded_by_the_experts_it_mixes():
    """A convex combination cannot exceed the range of its inputs; guards the weighting."""
    branch = AdaptiveQuantumBranch(channels=CHANNELS, n_qubits=4).eval()
    with torch.no_grad():
        outputs = branch(torch.randn(3, 3, 64, 64))

    # PauliZ expectation values live in [-1, 1], so any convex mixture must too.
    assert outputs["quantum_features"].abs().max() <= 1.0 + 1e-5


def test_adaptive_classifier_concatenates_classical_and_quantum_features():
    """Step 12: "use expectation values as quantum features and concatenate them with
    classical features"."""
    net = AdaptiveQuantumClassifier(channels=CHANNELS, num_classes=4, n_qubits=4)
    assert net.feature_dim == CHANNELS + 4

    outputs = net.extract(torch.randn(2, 3, 64, 64))
    assert outputs["logits"].shape == (2, 4)
    assert outputs["features"].shape == (2, CHANNELS + 4)


def test_gradients_flow_through_the_quantum_mixture():
    """Both the selector and the upstream spatial branch must learn."""
    net = AdaptiveQuantumClassifier(channels=CHANNELS, num_classes=4, n_qubits=4)
    net.extract(torch.randn(2, 3, 64, 64))["logits"].sum().backward()

    assert net.branch.reduce.weight.grad is not None
    assert net.branch.selector.selector[0].weight.grad is not None
    assert net.branch.spatial_branch.stem.stem[0].weight.grad is not None
    assert all(
        torch.isfinite(p.grad).all()
        for p in net.parameters()
        if p.grad is not None
    )
