"""Saliency methods for Step 19.

Two techniques, because the specification names both:

    "Use Grad-CAM or Score-CAM for CNN-based feature maps. Use attention rollout or
     attention maps for Transformer-based components."

:class:`GradCAM` covers the convolutional path - the classical EfficientNet-B0 branch that
Step 13's ablation shows carries most of the signal. :func:`attention_rollout` covers the
ViT baseline, which the reference notebook never explained at all.

**Hooks are always removed.** Both classes are context managers, and both remove their
hooks in ``__exit__`` even if the body raised. The reference notebook registered hooks at
module scope and never detached them, so every subsequent forward pass kept writing into
stale buffers - which produces wrong saliency maps rather than an error.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Gradient-weighted class activation mapping over a convolutional layer.

    Weights each feature-map channel by the mean gradient of the target class score with
    respect to that channel, sums them, and keeps the positive part - the regions whose
    presence pushes the score up.

    Used as a context manager so the forward and backward hooks are guaranteed to be
    removed::

        with GradCAM(model, target_layer) as cam:
            heatmap = cam(images, target_class=2)

    :param model: Model to explain. Must not be wrapped in ``torch.no_grad``.
    :param target_layer: Convolutional layer to read activations and gradients from.
        The last convolutional block gives the best trade-off between semantic content
        and spatial resolution.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "GradCAM":
        """:return: This instance, with hooks registered."""
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        return self

    def __exit__(self, *exc_info) -> None:
        """Remove every hook, including after an exception.

        :param exc_info: Exception details, unused.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.activations = None
        self.gradients = None

    def _save_activation(self, module: nn.Module, inputs, output) -> None:
        """:param module: The hooked layer.
        :param inputs: Layer inputs, unused.
        :param output: Layer output, retained for the CAM.
        """
        self.activations = output

    def _save_gradient(self, module: nn.Module, grad_input, grad_output) -> None:
        """:param module: The hooked layer.
        :param grad_input: Input gradients, unused.
        :param grad_output: Output gradients, retained for the CAM.
        """
        self.gradients = grad_output[0]

    def __call__(
        self, images: torch.Tensor, target_class: Optional[int] = None, output_size: Optional[int] = None
    ) -> np.ndarray:
        """Compute a class activation map.

        :param images: Input batch, ``(B, 3, H, W)``.
        :param target_class: Class to explain. ``None`` explains each sample's own
            prediction. Explaining the **true** class of a misclassified image is often
            more informative - it shows whether the right evidence was present but
            outvoted.
        :param output_size: Edge length to upsample the map to; defaults to the input's.
        :return: Maps in ``[0, 1]``, shape ``(B, size, size)``.
        :raises RuntimeError: If the hooks captured nothing, which means the target layer
            was not reached by the forward pass.
        """
        self.model.zero_grad(set_to_none=True)

        # The branches this explains are frozen (requires_grad=False on every parameter).
        # If the input also does not require grad, autograd prunes the entire subgraph and
        # the backward hook never fires - so there would be no gradients to weight the
        # activations by. Making the input a leaf that requires grad keeps the graph alive.
        if not images.requires_grad:
            images = images.clone().detach().requires_grad_(True)

        logits = self.model(images)
        if isinstance(logits, dict):
            logits = logits["logits"]

        if target_class is None:
            scores = logits.gather(1, logits.argmax(dim=1, keepdim=True)).sum()
        else:
            scores = logits[:, target_class].sum()
        scores.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM captured no activations: the target layer was not reached by the "
                "forward pass. Check that the layer belongs to the model being called."
            )
        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM captured activations but no gradients: nothing backpropagated "
                "through the target layer. This happens when the whole subgraph is frozen "
                "and the input does not require grad, or when the forward path is wrapped "
                "in torch.no_grad()."
            )

        # One weight per channel: how strongly that channel's presence raises the score.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        size = output_size or images.shape[-1]
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        # Normalise per image so maps are comparable across samples.
        flat = cam.flatten(1)
        minimum = flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
        maximum = flat.max(dim=1, keepdim=True).values.unsqueeze(-1)
        cam = (cam - minimum) / (maximum - minimum + 1e-8)

        return cam.detach().cpu().numpy()


class AttentionCapture:
    """Capture per-layer attention weights from a torchvision ViT.

    ``torchvision``'s encoder blocks call ``nn.MultiheadAttention`` with
    ``need_weights=False``, so the weights are computed and discarded. This context manager
    temporarily wraps each attention module's ``forward`` to request and record them, then
    restores the originals - so the model is left exactly as found.

    :param model: A torchvision ViT, or a module containing one.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.attentions: List[torch.Tensor] = []
        self._originals: Dict[nn.Module, object] = {}

    def _attention_modules(self) -> List[nn.Module]:
        """:return: Every ``MultiheadAttention`` inside the model, in definition order."""
        return [m for m in self.model.modules() if isinstance(m, nn.MultiheadAttention)]

    def __enter__(self) -> "AttentionCapture":
        """:return: This instance, with attention modules patched."""
        self.attentions = []

        for module in self._attention_modules():
            original = module.forward
            self._originals[module] = original

            def patched(*args, _original=original, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                output, weights = _original(*args, **kwargs)
                if weights is not None:
                    self.attentions.append(weights.detach())
                return output, weights

            module.forward = patched

        return self

    def __exit__(self, *exc_info) -> None:
        """Restore the original forwards, including after an exception.

        :param exc_info: Exception details, unused.
        """
        for module, original in self._originals.items():
            module.forward = original
        self._originals.clear()


def attention_rollout(
    attentions: List[torch.Tensor], discard_ratio: float = 0.0, head_fusion: str = "mean"
) -> torch.Tensor:
    """Abnar & Zuidema attention rollout across transformer layers.

    Raw attention from a single layer is a poor explanation: information mixes across
    layers, and residual connections carry signal that attention alone does not show.
    Rollout composes the layers, adding an identity term for the residual path and
    renormalising, so the result describes how much each input patch contributes to the
    class token at the output.

    :param attentions: Per-layer attention, each ``(B, tokens, tokens)``, in forward order.
    :param discard_ratio: Fraction of the weakest connections to zero at each layer, which
        sharpens the map. 0 keeps everything.
    :param head_fusion: Reserved; weights are expected already averaged over heads.
    :return: Rollout matrix, ``(B, tokens, tokens)``.
    :raises ValueError: If no attention was captured.
    """
    if not attentions:
        raise ValueError(
            "No attention weights captured. Wrap the forward pass in AttentionCapture, "
            "and check the model actually contains nn.MultiheadAttention modules."
        )

    result = torch.eye(attentions[0].size(-1), device=attentions[0].device)
    result = result.unsqueeze(0).expand(attentions[0].size(0), -1, -1).clone()

    for attention in attentions:
        weights = attention.clone()

        if discard_ratio > 0:
            flat = weights.flatten(1)
            count = int(flat.size(1) * discard_ratio)
            if count > 0:
                _, indices = flat.topk(count, dim=1, largest=False)
                flat.scatter_(1, indices, 0.0)
                weights = flat.view_as(attention)

        # Residual connections mean a token always retains its own information; the
        # identity term is what accounts for that path.
        identity = torch.eye(weights.size(-1), device=weights.device).unsqueeze(0)
        weights = weights + identity
        weights = weights / weights.sum(dim=-1, keepdim=True)

        result = torch.bmm(weights, result)

    return result


def rollout_to_map(rollout: torch.Tensor, grid_size: Optional[int] = None) -> np.ndarray:
    """Extract the class token's attention over patches as a square map.

    :param rollout: Output of :func:`attention_rollout`, ``(B, tokens, tokens)``.
    :param grid_size: Patch grid edge. Inferred from the token count when omitted.
    :return: Normalised maps, ``(B, grid, grid)``.
    """
    # Row 0 is the class token; column 0 is its attention to itself, so it is dropped.
    class_attention = rollout[:, 0, 1:]

    if grid_size is None:
        grid_size = int(round(class_attention.size(-1) ** 0.5))

    maps = class_attention[:, : grid_size * grid_size].reshape(-1, grid_size, grid_size)

    flat = maps.flatten(1)
    minimum = flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
    maximum = flat.max(dim=1, keepdim=True).values.unsqueeze(-1)

    return ((maps - minimum) / (maximum - minimum + 1e-8)).detach().cpu().numpy()
