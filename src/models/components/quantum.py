"""Quantum circuits and the fixed-QCNN baseline (Steps 9 and 12).

Step 12 sets out the design constraints this module implements:

- reduce feature maps to a compact vector before quantum encoding;
- use **angle encoding** for the first implementation, "because it is simple and stable";
- start with **4 qubits** for comparison with prior QCNN work;
- use a parameterised circuit with rotation and entangling gates;
- test at least two circuit depths and at least two entanglement patterns;
- use **expectation values** as the quantum features.

The circuit factories below cover all five variants the study needs. Step 9's Baseline 5
uses only the fixed one; Phase 4 adds the adaptive mixture over all five.

**Simulator device placement.** PennyLane's ``default.qubit`` runs on CPU, so a quantum
layer inside a GPU model must move its input across and its output back. That shuttle
happens once per forward pass in :class:`QuantumLayer`, and it is the dominant cost of
every quantum model here - not parameter count, which is tiny. It also means quantum
models cannot use mixed precision and are impractical under DDP.
"""

from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
import torch
from torch import nn

from src.models.components.base import FeatureNet

#: Step 12: "Start with 4 qubits for comparison with prior QCNN work."
DEFAULT_N_QUBITS = 4

#: The five circuit variants Step 12 asks to be compared. Two depths (2 vs 4 layers) and
#: two entanglement patterns (basic ring vs strongly-entangling), plus data re-uploading.
CIRCUIT_NAMES = ("fixed", "deep", "strong", "combined", "reupload")


@lru_cache(maxsize=None)
def _device(n_qubits: int) -> Any:
    """Return a cached simulator device.

    Devices are cached because constructing one per model instance would multiply
    simulator setup cost across a sweep.

    :param n_qubits: Number of wires.
    :return: A PennyLane device.
    """
    return qml.device("default.qubit", wires=n_qubits)


@lru_cache(maxsize=None)
def make_circuit(name: str, n_qubits: int = DEFAULT_N_QUBITS) -> Tuple[Callable, Dict[str, tuple]]:
    """Build one of the five quantum expert circuits.

    Every circuit angle-encodes the input, applies a parameterised entangling ansatz, and
    measures ``PauliZ`` expectation values - one per wire - which become the quantum
    feature vector.

    :param name: One of :data:`CIRCUIT_NAMES`.

        ``fixed``
            2 basic entangling layers. The Step 9 baseline and the fixed-circuit control.
        ``deep``
            4 basic entangling layers - same pattern, greater depth.
        ``strong``
            2 strongly-entangling layers - richer pattern, same depth as ``fixed``.
        ``combined``
            4 strongly-entangling layers - greater depth and richer pattern.
        ``reupload``
            Data re-uploading: the input is re-encoded before each layer, which raises
            the expressible function class without adding qubits.

    :param n_qubits: Number of wires.
    :return: ``(qnode, weight_shapes)`` ready for ``qml.qnn.TorchLayer``.
    :raises ValueError: If ``name`` is not a known circuit.
    """
    if name not in CIRCUIT_NAMES:
        raise ValueError(f"Unknown circuit {name!r}. Available: {CIRCUIT_NAMES}")

    device = _device(n_qubits)
    wires = range(n_qubits)

    if name in ("fixed", "deep"):
        layers = 2 if name == "fixed" else 4

        @qml.qnode(device, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=wires)
            qml.BasicEntanglerLayers(weights, wires=wires)
            return [qml.expval(qml.PauliZ(i)) for i in wires]

        return circuit, {"weights": (layers, n_qubits)}

    if name in ("strong", "combined"):
        layers = 2 if name == "strong" else 4

        @qml.qnode(device, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=wires)
            qml.StronglyEntanglingLayers(weights, wires=wires)
            return [qml.expval(qml.PauliZ(i)) for i in wires]

        return circuit, {"weights": (layers, n_qubits, 3)}

    @qml.qnode(device, interface="torch")
    def circuit(inputs, weights):
        # Re-uploading: re-encode the data between layers so the circuit is not limited
        # to a single-shot embedding of the input.
        for layer in range(weights.shape[0]):
            qml.AngleEmbedding(inputs, wires=wires)
            qml.BasicEntanglerLayers(weights[layer : layer + 1], wires=wires)
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    return circuit, {"weights": (2, n_qubits)}


class QuantumLayer(nn.Module):
    """A parameterised circuit wrapped for use inside a torch model.

    Handles the two details every quantum model here needs: scaling the incoming features
    into a valid angle range, and moving tensors to CPU for the simulator and back.

    :param circuit_name: One of :data:`CIRCUIT_NAMES`.
    :param n_qubits: Number of wires; also the output width.
    """

    def __init__(self, circuit_name: str = "fixed", n_qubits: int = DEFAULT_N_QUBITS) -> None:
        super().__init__()
        circuit, weight_shapes = make_circuit(circuit_name, n_qubits)
        self.circuit_name = circuit_name
        self.n_qubits = n_qubits
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """:param angles: Pre-scaled rotation angles, ``(B, n_qubits)``.
        :return: PauliZ expectation values, ``(B, n_qubits)``, on the input's device.
        """
        device = angles.device
        # default.qubit is a CPU simulator; a GPU tensor must cross over and come back.
        expectations = self.qlayer(angles.cpu())
        return expectations.to(device)

    def extra_repr(self) -> str:
        return f"circuit={self.circuit_name}, n_qubits={self.n_qubits}"


def scale_to_angles(features: torch.Tensor) -> torch.Tensor:
    """Squash arbitrary features into the ``[-pi, pi]`` rotation range.

    Angle encoding is only stable when inputs stay inside one rotation period; ``tanh``
    bounds them smoothly without clipping gradients to zero the way a hard clamp would.

    :param features: Reduced feature vector.
    :return: Angles in ``[-pi, pi]``.
    """
    return torch.tanh(features) * np.pi


class FixedQCNN(FeatureNet):
    """Baseline 5: a small CNN feeding a fixed quantum circuit.

    The specification lists a fixed QCNN as a required baseline, and it is the control the
    Step 12 adaptive quantum branch is measured against. Its convolutional stem is
    deliberately small - the point is to test the quantum layer, not the CNN in front of
    it.

    :param num_classes: Number of output classes.
    :param n_qubits: Circuit width; also the reduced feature width.
    :param channels: Channel widths of the two convolutional blocks.
    :param circuit_name: Which circuit to use; ``fixed`` is the baseline.
    """

    def __init__(
        self,
        num_classes: int = 4,
        n_qubits: int = DEFAULT_N_QUBITS,
        channels: tuple = (16, 32),
        circuit_name: str = "fixed",
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
            nn.AdaptiveAvgPool2d(1),
        )
        # Step 12: "Use dimensionality reduction before quantum encoding ... or a
        # learnable projection layer."
        self.reduce = nn.Linear(channels[1], n_qubits)
        self.quantum = QuantumLayer(circuit_name=circuit_name, n_qubits=n_qubits)
        self.classifier = nn.Linear(n_qubits, num_classes)
        self.feature_dim = n_qubits

    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """:param x: Image batch, ``(B, 3, H, W)``.
        :return: ``{"logits", "features", "angles"}`` where features are the circuit's
            expectation values.
        """
        pooled = torch.flatten(self.cnn(x), 1)
        angles = scale_to_angles(self.reduce(pooled))
        quantum_features = self.quantum(angles)
        return {
            "logits": self.classifier(quantum_features),
            "features": quantum_features,
            "angles": angles,
        }


class AdaptiveQuantumSelector(nn.Module):
    """Learns how much to trust each quantum expert, per image (Step 12).

    Step 12 asks for a set of circuit variants - fixed, adaptive depth, adaptive
    entanglement, re-uploading - with "the final model using only the configuration
    selected by validation performance". The reference notebook goes further: rather than
    picking one circuit offline, it learns a softmax mixture over all five, conditioned on
    the input's classical features, so the choice can differ per image.

    This module follows the notebook by instruction.

    :param feature_dim: Width of the conditioning classical features.
    :param num_experts: Number of circuits to mix over.
    :param hidden_dim: Width of the selector's hidden layer.
    :param dropout: Dropout inside the selector.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        num_experts: int = 5,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.selector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """:param features: Classical features, ``(B, feature_dim)``.
        :return: Mixture weights, ``(B, num_experts)``, summing to 1 per image.
        """
        return torch.softmax(self.selector(features), dim=1)


class AdaptiveQuantumBranch(nn.Module):
    """Spatial-gate features feeding a learned mixture of five quantum circuits (Step 12).

    The pipeline is: the Step 11 spatial-gate branch produces a compact classical feature
    vector, a learnable projection reduces it to the qubit count, ``tanh`` scales it into a
    valid angle range, all five circuits evaluate those angles, and the selector's weights
    combine their expectation values into one quantum feature vector.

    **Cost warning.** Every forward pass evaluates *five* circuits on a CPU simulator, so
    this branch is roughly five times slower than the single-circuit baseline, which was
    already the slowest model in the study. Extract its features once and cache them
    rather than recomputing them per epoch.

    :param channels: Channel width of the spatial-gate branch.
    :param n_qubits: Circuit width; also the quantum feature width.
    :param circuit_names: Circuits to mix over; defaults to all five.
    :param selector_hidden: Selector hidden width.
    :param selector_dropout: Selector dropout.
    """

    def __init__(
        self,
        channels: int = 32,
        n_qubits: int = DEFAULT_N_QUBITS,
        circuit_names: Optional[Sequence[str]] = None,
        selector_hidden: int = 64,
        selector_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Imported here rather than at module scope: multiscale.py imports this module for
        # its quantum arms, so a top-level import would be circular.
        from src.models.components.multiscale import MultiscaleBranch

        self.circuit_names = tuple(circuit_names or CIRCUIT_NAMES)
        self.n_qubits = n_qubits
        self.channels = channels

        self.spatial_branch = MultiscaleBranch(variant="spatial_gate", channels=channels)
        self.reduce = nn.Linear(channels, n_qubits)
        self.experts = nn.ModuleList(
            [QuantumLayer(circuit_name=name, n_qubits=n_qubits) for name in self.circuit_names]
        )
        self.selector = AdaptiveQuantumSelector(
            feature_dim=channels,
            num_experts=len(self.circuit_names),
            hidden_dim=selector_hidden,
            dropout=selector_dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """:param x: Image batch, ``(B, 3, H, W)``.
        :return: ``{"quantum_features", "classical_features", "quantum_weights",
            "gate_maps"}``. ``classical_features`` are the spatial-gate branch's output,
            which Step 13 fuses alongside the quantum vector.
        """
        classical_features, gate_maps = self.spatial_branch(x)

        angles = scale_to_angles(self.reduce(classical_features))
        expert_outputs = torch.stack([expert(angles) for expert in self.experts], dim=1)

        weights = self.selector(classical_features)
        quantum_features = (expert_outputs * weights.unsqueeze(-1)).sum(dim=1)

        return {
            "quantum_features": quantum_features,
            "classical_features": classical_features,
            "quantum_weights": weights,
            "gate_maps": gate_maps,
        }


class AdaptiveQuantumClassifier(FeatureNet):
    """The Step 12 branch with a classification head, for standalone validation.

    The head is discarded in Step 13; only the branch's features carry forward into the
    fusion module. It exists so Step 12 can be trained and evaluated on its own, and so
    the learned expert weights can be inspected against class labels.

    :param channels: Channel width of the spatial-gate branch.
    :param num_classes: Number of output classes.
    :param n_qubits: Circuit width.
    :param circuit_names: Circuits to mix over.
    :param hidden_dims: Widths of the classifier hidden layers.
    :param dropout: Dropout in the first hidden layer; the second uses two thirds of it.
    """

    def __init__(
        self,
        channels: int = 32,
        num_classes: int = 4,
        n_qubits: int = DEFAULT_N_QUBITS,
        circuit_names: Optional[Sequence[str]] = None,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.branch = AdaptiveQuantumBranch(
            channels=channels, n_qubits=n_qubits, circuit_names=circuit_names
        )
        # Step 12: "concatenate [expectation values] with classical features".
        self.feature_dim = channels + n_qubits

        first, second = hidden_dims[0], hidden_dims[1]
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, first),
            nn.BatchNorm1d(first),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.GELU(),
            nn.Dropout(dropout * 2 / 3),
            nn.Linear(second, num_classes),
        )

    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """:param x: Image batch, ``(B, 3, H, W)``.
        :return: ``{"logits", "features", "quantum_features", "classical_features",
            "quantum_weights", "gate_maps"}``.
        """
        outputs = self.branch(x)
        fused = torch.cat([outputs["classical_features"], outputs["quantum_features"]], dim=1)
        return {"logits": self.classifier(fused), "features": fused, **outputs}
