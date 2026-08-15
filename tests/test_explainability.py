"""Tests for Step 19's saliency methods and Step 20's statistical machinery."""

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score
from torch import nn
from torchvision import models

from src.models.components.explain import (
    AttentionCapture,
    GradCAM,
    attention_rollout,
    rollout_to_map,
)
from src.models.components.transfer import TransferBackbone
from src.utils.statistics import (
    MIN_WILCOXON_PAIRS,
    bootstrap_ci,
    mcnemar_test,
    paired_bootstrap,
    summarise_across_seeds,
    wilcoxon_paired,
)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """:param y_true: True labels.
    :param y_pred: Predictions.
    :return: Macro-averaged F1.
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# -------------------------------------------------------------------- Grad-CAM


@pytest.fixture
def cnn():
    """:return: A small CNN with an identifiable target layer."""
    return TransferBackbone(arch="efficientnet_b0", num_classes=4, weights=None).eval()


def test_grad_cam_produces_a_normalised_map(cnn):
    """Maps must be comparable across images, so each is normalised to [0, 1]."""
    images = torch.randn(2, 3, 224, 224)

    with GradCAM(cnn, cnn.backbone.features[-1]) as cam:
        heatmap = cam(images, target_class=1)

    assert heatmap.shape == (2, 224, 224)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-5


def test_grad_cam_removes_its_hooks_on_exit(cnn):
    """The notebook left hooks attached forever, silently corrupting later passes."""
    layer = cnn.backbone.features[-1]
    before = len(layer._forward_hooks)

    with GradCAM(cnn, layer) as cam:
        cam(torch.randn(1, 3, 224, 224), target_class=0)
        assert len(layer._forward_hooks) > before

    assert len(layer._forward_hooks) == before


def test_grad_cam_removes_its_hooks_even_after_an_exception(cnn):
    """Teardown must survive a failure, or one bad image poisons the whole analysis."""
    layer = cnn.backbone.features[-1]
    before = len(layer._forward_hooks)

    with pytest.raises(RuntimeError):
        with GradCAM(cnn, layer):
            raise RuntimeError("boom")

    assert len(layer._forward_hooks) == before


def test_grad_cam_responds_to_the_requested_class(cnn):
    """Explaining different classes must give different maps, or it explains nothing."""
    images = torch.randn(1, 3, 224, 224)

    with GradCAM(cnn, cnn.backbone.features[-1]) as cam:
        first = cam(images, target_class=0)
        second = cam(images, target_class=3)

    assert not np.allclose(first, second)


def test_grad_cam_reports_an_unreached_target_layer(cnn):
    """A layer from another model is a plausible mistake and must not fail silently."""
    orphan = nn.Conv2d(3, 8, 3)

    with pytest.raises(RuntimeError, match="captured no activations"):
        with GradCAM(cnn, orphan) as cam:
            cam(torch.randn(1, 3, 224, 224), target_class=0)


# ------------------------------------------------------------ attention rollout


def test_attention_capture_collects_weights_and_restores_the_model():
    """torchvision discards attention weights; capture must not leave the model patched."""
    vit = models.vit_b_16(weights=None).eval()
    original = [m.forward for m in vit.modules() if isinstance(m, nn.MultiheadAttention)]

    with AttentionCapture(vit) as capture:
        with torch.no_grad():
            vit(torch.randn(1, 3, 224, 224))
        assert len(capture.attentions) == 12, "expected one attention map per encoder layer"
        assert capture.attentions[0].shape[-1] == capture.attentions[0].shape[-2]

    restored = [m.forward for m in vit.modules() if isinstance(m, nn.MultiheadAttention)]
    assert restored == original, "attention modules were left patched"


def test_rollout_composes_layers_into_a_row_stochastic_matrix():
    """Each row must remain a distribution over tokens after composition."""
    tokens = 5
    attentions = [torch.softmax(torch.randn(2, tokens, tokens), dim=-1) for _ in range(3)]

    rollout = attention_rollout(attentions)

    assert rollout.shape == (2, tokens, tokens)
    assert torch.allclose(rollout.sum(dim=-1), torch.ones(2, tokens), atol=1e-4)


def test_rollout_accounts_for_the_residual_path():
    """Rollout adds an identity term; without it a near-zero attention map erases signal."""
    tokens = 4
    # Attention that sends everything to token 0, ignoring the residual path.
    attention = torch.zeros(1, tokens, tokens)
    attention[:, :, 0] = 1.0

    rollout = attention_rollout([attention])
    # With the residual identity, a token must retain some of its own information.
    assert rollout[0, 2, 2] > 0, "residual path was not accounted for"


def test_rollout_map_is_square_and_normalised():
    """The map is plotted over the image, so it must reshape to the patch grid."""
    tokens = 1 + 16  # class token plus a 4x4 patch grid
    rollout = torch.softmax(torch.randn(2, tokens, tokens), dim=-1)

    maps = rollout_to_map(rollout)

    assert maps.shape == (2, 4, 4)
    assert maps.min() >= 0.0 and maps.max() <= 1.0 + 1e-6


def test_rollout_rejects_empty_input():
    """Silence here would produce an all-zero saliency map that looks like a result."""
    with pytest.raises(ValueError, match="No attention weights captured"):
        attention_rollout([])


# ------------------------------------------------------------------- statistics


def test_mcnemar_uses_only_discordant_pairs():
    """Agreements carry no information about which model is better."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # all correct
    b = np.array([0, 0, 1, 1, 1, 1, 0, 0])  # four wrong

    result = mcnemar_test(y_true, a, b)

    assert result["only_a_correct"] == 4
    assert result["only_b_correct"] == 0
    assert result["n_discordant"] == 4
    assert result["favours"] == "a"


def test_mcnemar_handles_identical_models():
    """Two identical models have nothing to test; the result must say so."""
    y_true = np.array([0, 1, 2, 3])
    result = mcnemar_test(y_true, y_true.copy(), y_true.copy())

    assert result["n_discordant"] == 0
    assert result["p_value"] is None
    assert "nothing to compare" in result["note"]


def test_mcnemar_switches_to_the_exact_test_for_small_counts():
    """The chi-square approximation is unreliable when discordant pairs are few."""
    y_true = np.zeros(50, dtype=int)
    a = y_true.copy()
    b = y_true.copy()
    b[:3] = 1  # only three discordant pairs

    assert mcnemar_test(y_true, a, b)["test"] == "exact binomial"


def test_paired_bootstrap_detects_a_real_difference():
    """A consistently better model should produce an interval excluding zero."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=400)

    good = y_true.copy()
    good[:20] = (good[:20] + 1) % 4  # 5% error
    bad = y_true.copy()
    bad[:160] = (bad[:160] + 1) % 4  # 40% error

    result = paired_bootstrap(y_true, good, bad, macro_f1, n_resamples=300, seed=0)

    assert result["observed_delta"] > 0
    assert result["significant"] is True
    assert result["fraction_favouring_a"] > 0.95


def test_paired_bootstrap_reports_no_difference_when_there_is_none():
    """The interval must span zero for equivalent models - the honest negative result."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 4, size=300)
    predictions = y_true.copy()
    predictions[:30] = (predictions[:30] + 1) % 4

    result = paired_bootstrap(y_true, predictions, predictions.copy(), macro_f1, n_resamples=200)

    assert result["observed_delta"] == pytest.approx(0.0)
    assert result["significant"] is False


def test_bootstrap_ci_brackets_the_point_estimate():
    """Step 23 requires 95% intervals on the headline metrics."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 4, size=200)
    predictions = y_true.copy()
    predictions[:20] = (predictions[:20] + 1) % 4

    result = bootstrap_ci(y_true, predictions, macro_f1, n_resamples=300)

    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
    assert result["confidence"] == 95.0


def test_wilcoxon_refuses_a_sample_too_small_to_be_significant():
    """The notebook ran this over four per-class values, where p < 0.05 is unreachable.

    The two-sided floor at n=4 is 0.125, so any p-value reported would be uninformative.
    """
    result = wilcoxon_paired([0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2], label="classes")

    assert result["p_value"] is None
    assert result["n_pairs"] == 4
    assert "at least" in result["note"]
    assert "paired bootstrap" in result["note"]


def test_wilcoxon_runs_when_the_sample_supports_it():
    """With enough pairs the test is meaningful and is performed."""
    first = [0.90, 0.88, 0.91, 0.87, 0.89, 0.92, 0.90]
    second = [0.80, 0.79, 0.82, 0.78, 0.81, 0.83, 0.80]

    result = wilcoxon_paired(first, second)

    assert result["n_pairs"] >= MIN_WILCOXON_PAIRS
    assert result["p_value"] is not None
    assert result["median_difference"] > 0


def test_wilcoxon_rejects_mismatched_inputs():
    """Misaligned pairs would silently compare the wrong runs."""
    with pytest.raises(ValueError, match="match in length"):
        wilcoxon_paired([1.0, 2.0], [1.0])


def test_seed_summary_uses_the_sample_standard_deviation():
    """Seeds are a sample, so ddof=1 - the population form would understate spread."""
    summary = summarise_across_seeds([0.90, 0.92, 0.94])

    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(0.92)
    assert summary["std"] == pytest.approx(np.std([0.90, 0.92, 0.94], ddof=1))
    assert summary["min"] == 0.90 and summary["max"] == 0.94


def test_seed_summary_handles_a_single_seed():
    """A single run has no spread; it must not raise."""
    assert summarise_across_seeds([0.9])["std"] == 0.0


def test_grad_cam_works_on_a_fully_frozen_pipeline():
    """The real usage: every branch frozen, plain (non-grad) input tensors.

    With requires_grad=False everywhere and an input that is not a grad-requiring leaf,
    autograd prunes the subgraph and the backward hook never fires - so Grad-CAM would
    silently have no gradients to weight by. GradCAM re-enables grad on the input itself
    rather than relying on every caller to remember.
    """
    from src.models.components.fusion import FusedFeatureClassifier
    from src.models.components.quantum import AdaptiveQuantumClassifier
    from src.models.full_pipeline import FullPipeline

    classical = TransferBackbone(arch="efficientnet_b0", num_classes=4, weights=None)
    quantum = AdaptiveQuantumClassifier(channels=8, num_classes=4, n_qubits=4)
    fusion = FusedFeatureClassifier(
        classical_dim=classical.feature_dim, spatial_dim=8, quantum_dim=4, num_classes=4
    )
    pipeline = FullPipeline(classical, quantum, fusion).eval()

    assert all(not p.requires_grad for p in pipeline.classical_net.parameters())

    images = torch.randn(1, 3, 64, 64)  # deliberately not requires_grad
    with GradCAM(pipeline, pipeline.classical_net.backbone.features[-1]) as cam:
        heatmap = cam(images, target_class=0)

    assert heatmap.shape == (1, 64, 64)
    assert np.isfinite(heatmap).all()
    assert heatmap.max() > 0, "an all-zero map means no gradient reached the layer"


def test_grad_cam_distinguishes_missing_activations_from_missing_gradients(cnn):
    """The two failures have different causes and different fixes."""
    with pytest.raises(RuntimeError, match="no activations"):
        with GradCAM(cnn, nn.Conv2d(3, 8, 3)) as cam:
            cam(torch.randn(1, 3, 224, 224), target_class=0)
