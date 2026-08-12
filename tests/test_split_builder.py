"""Tests for Step 3 (leak-free splitting) and Step 4 (data audit)."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.components.split_builder import (
    CLASS_MAP,
    audit_images,
    build_split,
    class_distribution,
    deduplicate,
    imbalance_ratio,
    load_split,
    normalize_class_folder,
    pool_image_records,
    verify_no_leakage,
)
from tests.helpers.synthetic_dataset import make_synthetic_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> dict:
    """Build the synthetic dataset once for the whole module.

    :param tmp_path_factory: pytest temporary directory factory.
    :return: Dict with ``root`` and the generator's report.
    """
    root = tmp_path_factory.mktemp("bt_mri_raw")
    report = make_synthetic_dataset(root, per_class_train=14, per_class_test=6, duplicates_per_class=2)
    return {"root": Path(root), "report": report}


def test_pool_collects_every_image_from_both_vendor_folders(dataset):
    """Pooling must merge Training and Testing, since we build our own split."""
    pooled = pool_image_records(dataset["root"])

    assert len(pooled) == dataset["report"]["files_written"]
    assert set(pooled["class_name"]) == set(CLASS_MAP)
    assert set(pooled["source_split"]) == {"Training", "Testing"}
    # Paths are stored relative so the split table survives being moved between machines.
    assert not any(Path(p).is_absolute() for p in pooled["rel_path"])


@pytest.mark.parametrize(
    "folder,expected",
    [
        ("Glioma", "Glioma"),
        ("glioma_tumor", "Glioma"),
        ("notumor", "No-tumor"),
        ("No-tumor", "No-tumor"),
        ("pituitary_tumor", "Pituitary"),
        ("not_a_class", None),
    ],
)
def test_class_folder_aliases_resolve(folder, expected):
    """Published mirrors disagree on folder naming; all variants must map to one label."""
    assert normalize_class_folder(folder) == expected


def test_deduplicate_removes_exactly_the_planted_duplicates(dataset):
    """The synthetic tree plants a known number of byte-identical cross-split copies."""
    from src.data.components.split_builder import add_content_hashes

    hashed = add_content_hashes(pool_image_records(dataset["root"]), dataset["root"])
    deduped, report = deduplicate(hashed)

    assert report["rows_removed"] == dataset["report"]["duplicate_files"]
    assert len(deduped) == dataset["report"]["unique_images"]
    assert deduped["md5"].is_unique


def test_build_split_is_leak_free_and_correctly_proportioned(dataset, tmp_path):
    """Dedup-before-split is what makes the split leak-free; verify both properties."""
    out_csv = tmp_path / "splits" / "dataset_split.csv"
    split_df, report = build_split(dataset["root"], out_csv, seed=42, val_frac=0.15, test_frac=0.15)

    assert out_csv.is_file()
    verify_no_leakage(split_df)

    total = len(split_df)
    assert total == dataset["report"]["unique_images"]
    assert report["rows_removed"] == dataset["report"]["duplicate_files"]

    sizes = report["split_sizes"]
    assert sizes["train"] + sizes["val"] + sizes["test"] == total
    # Two-stage splitting rounds, so allow a couple of images of slack.
    assert sizes["train"] == pytest.approx(0.70 * total, abs=3)
    assert sizes["val"] == pytest.approx(0.15 * total, abs=3)
    assert sizes["test"] == pytest.approx(0.15 * total, abs=3)


def test_split_is_stratified_across_all_splits(dataset, tmp_path):
    """Every class must appear in every split, or metrics become undefined."""
    split_df, _ = build_split(dataset["root"], tmp_path / "split.csv", seed=42)

    for split in ("train", "val", "test"):
        present = set(split_df[split_df["split"] == split]["label"])
        assert present == set(CLASS_MAP.values()), f"{split} is missing classes {present}"


def test_split_is_deterministic_for_a_fixed_seed(dataset, tmp_path):
    """Step 15 requires split files to be reproducible and logged."""
    first, _ = build_split(dataset["root"], tmp_path / "a.csv", seed=42)
    second, _ = build_split(dataset["root"], tmp_path / "b.csv", seed=42)
    third, _ = build_split(dataset["root"], tmp_path / "c.csv", seed=7)

    key = lambda df: dict(zip(df["md5"], df["split"]))  # noqa: E731
    assert key(first) == key(second)
    assert key(first) != key(third)


def test_audit_records_bit_depth_and_intensity(dataset, tmp_path):
    """Step 4 asks for bit depth explicitly; the reference notebook omitted it."""
    split_df, _ = build_split(dataset["root"], tmp_path / "split.csv")
    per_image, corrupted = audit_images(split_df, dataset["root"])

    assert not corrupted
    assert len(per_image) == len(split_df)
    for column in ("width", "height", "mode", "bits_per_channel", "channels", "min_value", "max_value"):
        assert column in per_image.columns
    assert set(per_image["bits_per_channel"]) == {8}


def test_audit_reports_corrupted_files(dataset, tmp_path):
    """An unreadable file must be reported, not silently crash the audit."""
    split_df, _ = build_split(dataset["root"], tmp_path / "split.csv")

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "not_an_image.png").write_bytes(b"definitely not a PNG")
    rows = pd.DataFrame([{"rel_path": "not_an_image.png", "class_name": "Glioma"}])

    per_image, corrupted = audit_images(rows, broken)
    assert len(per_image) == 0
    assert len(corrupted) == 1


def test_class_distribution_and_imbalance_ratio(dataset, tmp_path):
    """Step 4 requires the class distribution and the imbalance ratio to be reported."""
    split_df, report = build_split(dataset["root"], tmp_path / "split.csv")
    distribution = class_distribution(split_df)

    assert set(distribution["split"]) == {"train", "val", "test"}
    assert len(distribution) == 3 * len(CLASS_MAP)

    for split in ("train", "val", "test"):
        fractions = distribution[distribution["split"] == split]["fraction"].sum()
        assert fractions == pytest.approx(1.0, abs=1e-9)

    # The synthetic set is balanced by construction, so the ratio must be near 1.
    assert imbalance_ratio(split_df, "train") == pytest.approx(
        report["train_imbalance_ratio"], abs=1e-9
    )
    assert 1.0 <= imbalance_ratio(split_df, "train") < 1.6


def test_load_split_filters_and_raises_clearly(dataset, tmp_path):
    """A missing split file should say how to build it, not raise a bare IOError."""
    out_csv = tmp_path / "split.csv"
    build_split(dataset["root"], out_csv)

    assert set(load_split(out_csv, split="train")["split"]) == {"train"}
    assert len(load_split(out_csv)) > len(load_split(out_csv, split="val"))

    with pytest.raises(FileNotFoundError, match="prepare_data"):
        load_split(tmp_path / "missing.csv")


def test_pool_rejects_a_directory_with_no_images(tmp_path):
    """A wrong data path is a common setup error; fail loudly and early."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        pool_image_records(tmp_path / "empty")


# ------------------------------------------------------- dataset root resolution


def _make_class_folders(parent: Path) -> None:
    """Create the four class folders, each holding one image.

    :param parent: Directory to populate.
    """
    from PIL import Image

    for class_name in CLASS_MAP:
        folder = parent / class_name
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("L", (8, 8)).save(folder / "img.png")


def test_resolver_returns_the_root_when_split_folders_are_already_there(tmp_path):
    """A flat layout needs no descent."""
    from src.data.components.split_builder import locate_dataset_root

    for split in ("Training", "Testing"):
        _make_class_folders(tmp_path / split)

    assert locate_dataset_root(tmp_path) == tmp_path


def test_resolver_returns_the_root_when_class_folders_are_directly_there(tmp_path):
    """Some mirrors ship class folders with no Training/Testing division."""
    from src.data.components.split_builder import locate_dataset_root

    _make_class_folders(tmp_path)
    assert locate_dataset_root(tmp_path) == tmp_path


def test_resolver_descends_into_the_doubly_nested_archive(tmp_path):
    """The published archive unpacks to <name>/<name>/Training, two levels down."""
    from src.data.components.split_builder import locate_dataset_root

    nested = tmp_path / "BT-MRI Dataset" / "BT-MRI Dataset"
    for split in ("Training", "Testing"):
        _make_class_folders(nested / split)

    assert locate_dataset_root(tmp_path) == nested


def test_resolver_never_selects_a_degraded_copy_of_the_dataset(tmp_path):
    """The real archive ships a 'Challenging Datasets' tree with the SAME class folders.

    Selecting it would train the study on deliberately blurred and noise-corrupted
    images while every log still said 'Glioma, Meningioma, Pituitary, No-tumor'. Only a
    directory containing the Training/Testing split folders may be chosen automatically.
    """
    from src.data.components.split_builder import locate_dataset_root

    primary = tmp_path / "BT-MRI Dataset" / "BT-MRI Dataset"
    for split in ("Training", "Testing"):
        _make_class_folders(primary / split)

    # Sorts before "BT-MRI..." so a naive search would reach it first.
    for degraded in ("Blurred Dataset", "Noisy Dataset"):
        _make_class_folders(tmp_path / "AAA Challenging" / "AAA Challenging" / degraded)

    assert locate_dataset_root(tmp_path) == primary


def test_pooling_resolves_the_nested_root_automatically(tmp_path):
    """The documented quickstart must work without quoting a nested path."""
    nested = tmp_path / "BT-MRI Dataset" / "BT-MRI Dataset"
    make_synthetic_dataset(nested, per_class_train=3, per_class_test=1, duplicates_per_class=0)

    pooled = pool_image_records(tmp_path)
    assert len(pooled) == 4 * 4
    # Paths are relative to the resolved root, so a recipe mirror lines up with them.
    assert all(p.startswith(("Training/", "Testing/")) for p in pooled["rel_path"])
