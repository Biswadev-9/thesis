"""Tests for the Swin-aware Grad-CAM target-layer selection, and the Swin arm's configs.

``GradCAM`` weights channels by their mean gradient - ``gradients.mean(dim=(2, 3))`` - and
sums over ``dim=1``. That arithmetic is only correct for channels-first activations.

EfficientNet's ``features[-1]`` is channels-first. A torchvision Swin Transformer's is not:
its stages emit ``(B, H, W, C)`` and the model permutes afterwards. Hooking ``features[-1]``
on a Swin would therefore average over width and treat height as the channel axis - and
would do so *silently*, producing a plausible-looking but meaningless heatmap rather than
an error. These tests pin the layout contract so that failure mode cannot return.
"""

import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from torch import nn

from src.analysis.explainability import grad_cam_target_layer
from src.models.components.explain import GradCAM
from src.models.components.transfer import TransferBackbone

# Weights are not downloaded in tests; the wiring under test does not depend on them.
NO_WEIGHTS = {"weights": None}


# ------------------------------------------------------- layer selection, by architecture


def test_swin_target_layer_is_the_permute_not_the_last_stage():
    """Swin's stages are NHWC; its ``permute`` is the first channels-first point."""
    backbone = TransferBackbone(arch="swin_t", num_classes=4, **NO_WEIGHTS).backbone

    layer = grad_cam_target_layer(backbone)

    assert layer is backbone.permute
    assert layer is not backbone.features[-1]


def test_efficientnet_target_layer_is_unchanged():
    """The CNN path must behave exactly as it did before Swin support was added."""
    backbone = TransferBackbone(arch="efficientnet_b0", num_classes=4, **NO_WEIGHTS).backbone

    assert grad_cam_target_layer(backbone) is backbone.features[-1]


@pytest.mark.parametrize("arch", ["swin_t", "efficientnet_b0"])
def test_selected_layer_emits_channels_first_activations(arch):
    """The contract that matters: whatever is hooked must produce ``(B, C, H, W)``.

    This is the assertion that would have caught the Swin bug. It reads the activation the
    hook actually captures and checks the channel axis against the backbone's own feature
    width, rather than trusting the layer's name.
    """
    net = TransferBackbone(arch=arch, num_classes=4, **NO_WEIGHTS).eval()
    layer = grad_cam_target_layer(net.backbone)

    captured = {}
    handle = layer.register_forward_hook(lambda _m, _i, output: captured.update(out=output))
    try:
        net(torch.randn(2, 3, 224, 224))
    finally:
        handle.remove()

    activation = captured["out"]
    assert activation.ndim == 4, "Grad-CAM needs a 4-D feature map"
    assert activation.shape[0] == 2, "batch must come first"
    assert activation.shape[1] == net.feature_dim, (
        f"{arch}: expected the channel axis at dim=1 with width {net.feature_dim}, got "
        f"{tuple(activation.shape)} - GradCAM would reduce over the wrong axes"
    )


@pytest.mark.parametrize("arch", ["swin_t", "efficientnet_b0"])
def test_grad_cam_runs_end_to_end_on_the_selected_layer(arch):
    """A map must actually be produced: hooks fire, gradients arrive, output normalises."""
    net = TransferBackbone(arch=arch, num_classes=4, **NO_WEIGHTS).eval()

    with GradCAM(net, grad_cam_target_layer(net.backbone)) as cam:
        heatmap = cam(torch.randn(2, 3, 224, 224), target_class=1)

    assert heatmap.shape == (2, 224, 224)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


# ------------------------------------------------------------------- unsupported backbones


def test_an_unrecognised_backbone_raises_rather_than_guessing():
    """Hooking a layer of unverified layout is the failure this function exists to stop."""

    class Opaque(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.Sequential(nn.Conv2d(3, 8, 3))

    with pytest.raises(AttributeError, match="grad_cam_target_layer"):
        grad_cam_target_layer(Opaque())


# ------------------------------------------- Swin arm: Step 20 protocol must match Step 15


def _compose(config_name: str, *overrides: str):
    """:param config_name: Root config.

    :param overrides: Hydra overrides.
    :return: The composed DictConfig.
    """
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        return compose(config_name=config_name, overrides=list(overrides))


@pytest.fixture
def step15_swin():
    """:return: The composed Swin Step 15 training config."""
    return _compose("train.yaml", "experiment=step15_final_protocol_swin")


@pytest.fixture
def step20_swin():
    """:return: The composed Swin Step 20 analysis config."""
    return _compose("analyze.yaml", "analysis=step20_quantum_advantage_swin")


def test_swin_step20_protocol_matches_swin_step15(step15_swin, step20_swin):
    """Step 20 retrains a no-quantum control; a drifted protocol invalidates the verdict."""
    protocol = step20_swin.analysis.protocol

    assert protocol.max_epochs == step15_swin.trainer.max_epochs
    assert protocol.min_epochs == step15_swin.trainer.min_epochs
    assert protocol.patience == step15_swin.callbacks.early_stopping.patience
    assert protocol.monitor == step15_swin.callbacks.early_stopping.monitor
    assert protocol.mode == step15_swin.callbacks.early_stopping.mode
    assert protocol.batch_size == step15_swin.data.batch_size
    assert protocol.use_weighted_sampler == step15_swin.data.use_weighted_sampler


def test_swin_step20_control_architecture_matches_swin_step15(step15_swin, step20_swin):
    """The control must differ from the proposed model ONLY by the quantum branch."""
    control = step20_swin.analysis.fusion_model.net
    proposed = step15_swin.model.net

    assert control._target_ == proposed._target_
    for field in ("classical_dim", "spatial_dim", "quantum_dim", "proj_dim", "dropout"):
        assert control[field] == proposed[field], f"{field} differs between control and model"


# --------------------------------------------------- Swin arm: cache and backbone wiring


def test_swin_step15_reads_the_swin_feature_cache(step15_swin):
    """Training the Swin head on the baseline cache would silently reproduce the baseline."""
    assert step15_swin.data.tag == "swin"
    assert step15_swin.model.net.classical_dim == 768


def test_swin_step10_selects_the_swin_backbone():
    """The whole point of the arm."""
    cfg = _compose("train.yaml", "experiment=step10_classical_swin")

    assert cfg.model.net.arch == "swin_t"


def test_swin_extraction_config_pairs_swin_arch_with_the_swin_tag():
    """A swin_t backbone writing into the `default` cache would overwrite the baseline."""
    cfg = _compose("extract_features_swin.yaml")

    assert cfg.tag == "swin"
    assert cfg.classical_model.net.arch == "swin_t"


@pytest.mark.parametrize(
    "analysis_name",
    ["step16_internal_swin", "step17_external_swin", "step19_explainability_swin"],
)
def test_swin_analysis_configs_rebuild_a_swin_pipeline(analysis_name):
    """Rebuilding an EfficientNet shell around Swin weights fails at load_state_dict."""
    cfg = _compose("analyze.yaml", f"analysis={analysis_name}").analysis

    assert cfg.classical_model.net.arch == "swin_t"
    assert cfg.fusion_model.net.classical_dim == 768
    # The spatial/quantum branch is reused unchanged, so its config must NOT drift.
    assert cfg.quantum_model.net.channels == 32
    assert cfg.quantum_model.net.n_qubits == 4


def test_swin_robustness_changes_only_the_proposed_entry():
    """models.efficientnet_b0 and models.vit are Step 18's comparison baselines."""
    models = _compose("analyze.yaml", "analysis=step18_robustness_swin").analysis.models

    assert models.proposed.classical_model.net.arch == "swin_t"
    assert models.proposed.fusion_model.net.classical_dim == 768
    assert models.efficientnet_b0.model_cfg.net.arch == "efficientnet_b0"
    assert models.vit.model_cfg.net.arch == "vit_b_16"


@pytest.mark.parametrize(
    "analysis_name",
    ["step13_fusion_swin", "step14_loss_selection_swin", "step20_quantum_advantage_swin"],
)
def test_swin_cache_consuming_analyses_read_the_swin_tag(analysis_name):
    """These read the cache by `analysis.tag`, not `data.tag` - an easy override to get wrong."""
    assert _compose("analyze.yaml", f"analysis={analysis_name}").analysis.tag == "swin"


# ------------------------------------------------- the baseline arm must remain untouched


def test_baseline_configs_are_unchanged_by_the_swin_arm():
    """The EfficientNet study must stay runnable and reportable exactly as before."""
    step15 = _compose("train.yaml", "experiment=step15_final_protocol")
    step16 = _compose("analyze.yaml", "analysis=step16_internal").analysis

    assert step15.data.tag == "default"
    assert step15.model.net.classical_dim == 1280
    assert step16.classical_model.net.arch == "efficientnet_b0"
    assert step16.fusion_model.net.classical_dim == 1280
