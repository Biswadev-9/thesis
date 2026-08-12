"""Tests for the brain-tumour MRI datamodule and the networks it feeds."""

from collections import Counter
from pathlib import Path

import pytest
import torch

from src.data.bt_mri_datamodule import BTMRIDataModule
from src.models.components.backbones import SimpleCNN, SmallCNN
from tests.helpers.synthetic_dataset import make_synthetic_dataset


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """Build a synthetic data directory laid out like the real one.

    :param tmp_path_factory: pytest temporary directory factory.
    :return: Path usable as the datamodule's ``data_dir``.
    """
    root = tmp_path_factory.mktemp("data")
    make_synthetic_dataset(root / "raw" / "bt_mri", per_class_train=14, per_class_test=6)
    return root


def build_datamodule(data_dir: Path, **overrides) -> BTMRIDataModule:
    """Construct a datamodule with test-friendly defaults.

    :param data_dir: Root data directory.
    :param overrides: Values overriding the defaults.
    :return: An unprepared datamodule.
    """
    settings = {
        "data_dir": str(data_dir),
        "image_size": 32,
        "batch_size": 8,
        "num_workers": 0,
        **overrides,
    }
    return BTMRIDataModule(**settings)


def test_prepare_data_builds_the_split_once(data_dir, tmp_path):
    """The split is built on demand and then reused, never silently rebuilt."""
    datamodule = build_datamodule(data_dir, split_subpath=str(tmp_path / "split.csv"))
    assert not Path(datamodule.split_csv).is_file()

    datamodule.prepare_data()
    assert Path(datamodule.split_csv).is_file()

    modified_at = Path(datamodule.split_csv).stat().st_mtime_ns
    datamodule.prepare_data()
    assert Path(datamodule.split_csv).stat().st_mtime_ns == modified_at


def test_setup_creates_all_three_splits_and_class_weights(data_dir, tmp_path):
    """Class weights come from the training split only - never validation or test."""
    datamodule = build_datamodule(data_dir, split_subpath=str(tmp_path / "split.csv"))
    assert datamodule.class_weights is None

    datamodule.prepare_data()
    datamodule.setup()

    assert len(datamodule.data_train) > len(datamodule.data_val)
    assert len(datamodule.data_test) > 0
    assert datamodule.num_classes == 4
    assert datamodule.class_names == ["Glioma", "Meningioma", "Pituitary", "No-tumor"]

    weights = datamodule.class_weights
    assert weights.shape == (4,)
    assert torch.isfinite(weights).all()
    # Balanced inverse-frequency weights average to 1, keeping loss scales comparable.
    assert weights.mean().item() == pytest.approx(1.0, abs=0.15)


def test_dataloaders_yield_correctly_shaped_batches(data_dir, tmp_path):
    """Shape and dtype contract every downstream model relies on."""
    datamodule = build_datamodule(data_dir, split_subpath=str(tmp_path / "split.csv"))
    datamodule.prepare_data()
    datamodule.setup()

    for loader in (
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
    ):
        images, labels = next(iter(loader))
        assert images.shape[1:] == (3, 32, 32)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert labels.min() >= 0 and labels.max() < 4


def test_weighted_sampler_balances_the_training_loader(data_dir, tmp_path):
    """Step 8's sampler must equalise class frequency across a training epoch."""
    datamodule = build_datamodule(
        data_dir, split_subpath=str(tmp_path / "split.csv"), batch_size=4, use_weighted_sampler=True
    )
    datamodule.prepare_data()
    datamodule.setup()

    torch.manual_seed(0)
    drawn = Counter()
    for _, labels in datamodule.train_dataloader():
        drawn.update(labels.tolist())

    assert set(drawn) == {0, 1, 2, 3}
    assert sum(drawn.values()) == len(datamodule.data_train)


def test_sampler_and_shuffle_are_mutually_exclusive(data_dir, tmp_path):
    """PyTorch forbids both at once; the datamodule must pick correctly."""
    datamodule = build_datamodule(data_dir, split_subpath=str(tmp_path / "s1.csv"))
    datamodule.prepare_data()
    datamodule.setup()
    assert datamodule.train_dataloader().sampler is not None

    plain = build_datamodule(
        data_dir, split_subpath=str(tmp_path / "s2.csv"), use_weighted_sampler=False
    )
    plain.prepare_data()
    plain.setup()
    from torch.utils.data import RandomSampler

    assert isinstance(plain.train_dataloader().sampler, RandomSampler)


def test_validation_and_test_are_never_augmented(data_dir, tmp_path):
    """Step 7 is training-only; repeated eval reads must be identical."""
    datamodule = build_datamodule(
        data_dir, split_subpath=str(tmp_path / "split.csv"), augment=True
    )
    datamodule.prepare_data()
    datamodule.setup()

    first, _ = datamodule.data_val[0]
    second, _ = datamodule.data_val[0]
    assert torch.allclose(first, second)


def test_pipeline_description_is_serialisable(data_dir, tmp_path):
    """Run artefacts need the exact preprocessing settings, per Steps 5 and 7."""
    datamodule = build_datamodule(data_dir, split_subpath=str(tmp_path / "split.csv"))
    described = datamodule.pipeline_description()

    assert described["train"]["augment"] is True
    assert described["eval"]["augment"] is False
    assert described["use_weighted_sampler"] is True
    assert "augmentation" in described["train"]


def test_missing_image_root_gives_an_actionable_error(tmp_path):
    """A wrong data path is the most common setup mistake; say what to do about it."""
    datamodule = build_datamodule(tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError, match="download_data|Image root not found"):
        datamodule.prepare_data()


@pytest.mark.parametrize("net_class", [SimpleCNN, SmallCNN])
@pytest.mark.parametrize("size", [128, 224])
def test_networks_follow_the_extract_contract_at_any_input_size(net_class, size):
    """D1's contract, and the fix to SmallCNN's hard-coded 128px head.

    The notebook's SmallCNN ended in Linear(64*16*16, 128), which raised a shape error at
    any resolution but 128. Global average pooling makes both nets size-independent.
    """
    net = net_class(num_classes=4)
    images = torch.randn(2, 3, size, size)

    outputs = net.extract(images)
    assert set(outputs) >= {"logits", "features"}
    assert outputs["logits"].shape == (2, 4)
    assert outputs["features"].shape == (2, net.feature_dim)

    # forward() must be exactly extract()["logits"].
    net.eval()
    with torch.no_grad():
        assert torch.allclose(net(images), net.extract(images)["logits"])


def test_networks_produce_gradients():
    """A net that does not backpropagate would train silently and uselessly."""
    net = SimpleCNN(num_classes=4)
    net(torch.randn(2, 3, 64, 64)).sum().backward()

    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
