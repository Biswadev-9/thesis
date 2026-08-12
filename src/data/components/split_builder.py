"""Dataset pooling, deduplication, splitting and auditing.

Implements Steps 3 and 4 of the thesis specification:

- Step 3: the dataset is split **before** any augmentation or synthetic generation,
  70% train / 15% validation / 15% internal test, stratified by class.
- Step 4: an initial data audit records dimensions, colour mode, bit depth, intensity
  range, class distribution, imbalance ratio, corrupted files and exact duplicates.

Two details matter and are easy to get wrong:

1. **Deduplicate before splitting, not after.** The raw dataset ships the same image
   under both ``Training/`` and ``Testing/``. Splitting first and deduplicating second
   leaves near-copies straddling the train/test boundary, which silently inflates test
   scores. Hashing first and splitting the unique set is what makes the split leak-free.

2. **Store paths relative to the image root.** The reference notebook wrote absolute
   Colab paths into its split CSV, so the file stopped resolving the moment the runtime
   was recycled. Paths here are stored relative and resolved at load time.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Step 1: the four-class task. Label indices are fixed by the specification and must
# stay identical across training, validation, internal testing and external validation.
CLASS_MAP: Dict[str, int] = {
    "Glioma": 0,
    "Meningioma": 1,
    "Pituitary": 2,
    "No-tumor": 3,
}

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Published mirrors of this dataset disagree on folder naming. Matching is done on a
# normalised key so that "no_tumor", "notumor" and "No-tumor" all resolve to class 3.
_FOLDER_ALIASES: Dict[str, str] = {
    "glioma": "Glioma",
    "gliomatumor": "Glioma",
    "meningioma": "Meningioma",
    "meningiomatumor": "Meningioma",
    "pituitary": "Pituitary",
    "pituitarytumor": "Pituitary",
    "notumor": "No-tumor",
    "no": "No-tumor",
    "normal": "No-tumor",
    "healthy": "No-tumor",
}

# PIL mode -> (bits per channel, channel count). Used for the bit-depth column the
# specification asks for in Step 4; the notebook's audit omitted it.
_MODE_DEPTH: Dict[str, Tuple[int, int]] = {
    "1": (1, 1),
    "L": (8, 1),
    "P": (8, 1),
    "LA": (8, 2),
    "RGB": (8, 3),
    "RGBA": (8, 4),
    "CMYK": (8, 4),
    "I;16": (16, 1),
    "I": (32, 1),
    "F": (32, 1),
}


def normalize_class_folder(name: str) -> Optional[str]:
    """Map a raw class-folder name onto a canonical class name.

    :param name: Folder name as it appears on disk.
    :return: The canonical class name, or ``None`` if the folder is not a class folder.
    """
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    key = key.removesuffix("tumour").removesuffix("tumor") or key
    for candidate in (name, _FOLDER_ALIASES.get(key), _FOLDER_ALIASES.get(key + "tumor")):
        if candidate in CLASS_MAP:
            return candidate
    return None


def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Content hash of a file, read in chunks so large images do not spike memory.

    :param path: File to hash.
    :param chunk_size: Bytes per read.
    :return: Hex-encoded MD5 digest.
    """
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool_image_records(
    raw_dir: Path,
    source_dirs: Sequence[str] = ("Training", "Testing"),
) -> pd.DataFrame:
    """Pool every image from the raw dataset tree into one table.

    The dataset's own ``Training``/``Testing`` division is deliberately discarded: the
    specification requires our own stratified 70/15/15 split over the whole pool, and
    the vendor division is not leak-free.

    :param raw_dir: Root containing the ``source_dirs`` folders.
    :param source_dirs: Vendor split folders to pool. If none exist, ``raw_dir`` is
        treated as holding class folders directly.
    :return: Table with ``rel_path``, ``class_name``, ``label`` and ``source_split``.
    :raises FileNotFoundError: If ``raw_dir`` does not exist or contains no images.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    search_roots: List[Tuple[str, Path]] = [
        (name, raw_dir / name) for name in source_dirs if (raw_dir / name).is_dir()
    ]
    if not search_roots:
        search_roots = [("all", raw_dir)]

    records: List[Dict[str, object]] = []
    for source_split, root in search_roots:
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            class_name = normalize_class_folder(class_dir.name)
            if class_name is None:
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                records.append(
                    {
                        "rel_path": image_path.relative_to(raw_dir).as_posix(),
                        "class_name": class_name,
                        "label": CLASS_MAP[class_name],
                        "source_split": source_split,
                    }
                )

    if not records:
        raise FileNotFoundError(
            f"No images found under {raw_dir}. Expected class folders such as "
            f"{sorted(CLASS_MAP)} inside {list(source_dirs)}."
        )
    return pd.DataFrame.from_records(records)


def add_content_hashes(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Attach an MD5 content hash to every row.

    :param df: Table carrying a ``rel_path`` column.
    :param raw_dir: Root the relative paths resolve against.
    :return: Copy of ``df`` with an ``md5`` column.
    """
    raw_dir = Path(raw_dir)
    out = df.copy()
    out["md5"] = [md5_file(raw_dir / rel) for rel in out["rel_path"]]
    return out


def deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop exact duplicate images, keeping the first occurrence of each hash.

    :param df: Table carrying an ``md5`` column.
    :return: The deduplicated table and a report of what was removed.
    """
    group_sizes = df.groupby("md5").size()
    duplicate_groups = group_sizes[group_sizes > 1]

    deduped = df.drop_duplicates(subset="md5", keep="first").reset_index(drop=True)
    report = {
        "images_before_dedup": int(len(df)),
        "images_after_dedup": int(len(deduped)),
        "rows_removed": int(len(df) - len(deduped)),
        "duplicate_groups": int(len(duplicate_groups)),
        "rows_in_duplicate_groups": int(duplicate_groups.sum()),
    }
    return deduped, report


def stratified_split(
    df: pd.DataFrame,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """Assign a leak-free, class-stratified ``split`` column.

    :param df: Deduplicated table carrying a ``label`` column.
    :param seed: Random state, recorded in the audit for reproducibility.
    :param val_frac: Validation fraction of the whole pool.
    :param test_frac: Internal-test fraction of the whole pool.
    :return: Copy of ``df`` with a ``split`` column of ``train``/``val``/``test``.
    :raises ValueError: If the requested fractions do not leave a training set.
    """
    holdout_frac = val_frac + test_frac
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"val_frac + test_frac must lie in (0, 1), got {holdout_frac}")

    train_df, holdout_df = train_test_split(
        df, test_size=holdout_frac, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_frac / holdout_frac,
        stratify=holdout_df["label"],
        random_state=seed,
    )

    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def verify_no_leakage(df: pd.DataFrame) -> None:
    """Assert that no image content appears in more than one split.

    Deduplicating before splitting makes this true by construction, so a failure here
    means the split was built by some other path and cannot be trusted.

    :param df: Split table carrying ``md5`` and ``split`` columns.
    :raises AssertionError: If any hash spans multiple splits.
    """
    spans = df.groupby("md5")["split"].nunique()
    leaking = spans[spans > 1]
    assert leaking.empty, (
        f"{len(leaking)} image hashes appear in more than one split - the split is "
        "not leak-free. Deduplicate before splitting."
    )


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split, per-class counts plus each split's imbalance ratio.

    :param df: Split table carrying ``split``, ``class_name`` and ``label`` columns.
    :return: Long-form table with ``split``, ``class_name``, ``count``, ``fraction``.
    """
    rows: List[Dict[str, object]] = []
    for split in ("train", "val", "test"):
        subset = df[df["split"] == split]
        if subset.empty:
            continue
        counts = subset["class_name"].value_counts()
        total = int(counts.sum())
        for class_name in sorted(CLASS_MAP, key=CLASS_MAP.get):
            count = int(counts.get(class_name, 0))
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "label": CLASS_MAP[class_name],
                    "count": count,
                    "fraction": count / total if total else 0.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def imbalance_ratio(df: pd.DataFrame, split: str = "train") -> float:
    """Majority-to-minority class-count ratio for one split.

    :param df: Split table.
    :param split: Split to measure.
    :return: ``max(count) / min(count)``, or ``inf`` if a class is absent.
    """
    counts = df[df["split"] == split]["label"].value_counts()
    counts = counts.reindex(range(len(CLASS_MAP)), fill_value=0)
    minimum = int(counts.min())
    return float("inf") if minimum == 0 else float(counts.max() / minimum)


def audit_images(df: pd.DataFrame, raw_dir: Path) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """Inspect every image for the Step 4 quality-control table.

    Records width, height, colour mode, bit depth, channel count and intensity range,
    and collects any file that cannot be decoded.

    :param df: Table carrying ``rel_path`` and ``class_name`` columns.
    :param raw_dir: Root the relative paths resolve against.
    :return: The per-image audit table and a list of unreadable files.
    """
    import numpy as np

    raw_dir = Path(raw_dir)
    rows: List[Dict[str, object]] = []
    corrupted: List[Dict[str, str]] = []

    for rel_path, class_name in zip(df["rel_path"], df["class_name"]):
        path = raw_dir / rel_path
        try:
            with Image.open(path) as probe:
                probe.verify()  # cheap structural check; consumes the handle
            with Image.open(path) as image:
                mode, width, height = image.mode, image.width, image.height
                array = np.asarray(image)
        except Exception as error:  # noqa: BLE001 - the point is to record any failure
            corrupted.append({"rel_path": rel_path, "error": f"{type(error).__name__}: {error}"})
            continue

        bits, channels = _MODE_DEPTH.get(mode, (8, array.ndim))
        rows.append(
            {
                "rel_path": rel_path,
                "class_name": class_name,
                "width": width,
                "height": height,
                "mode": mode,
                "bits_per_channel": bits,
                "channels": channels,
                "min_value": float(array.min()),
                "max_value": float(array.max()),
            }
        )

    return pd.DataFrame.from_records(rows), corrupted


def build_split(
    raw_dir: Path,
    out_csv: Path,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    source_dirs: Sequence[str] = ("Training", "Testing"),
    deduplicate_first: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build the leak-free split table and write it to disk.

    This is the single source of truth for which image belongs to which split. Every
    datamodule reads it; nothing else assigns splits.

    :param raw_dir: Root of the raw dataset tree.
    :param out_csv: Destination CSV path; parent directories are created.
    :param seed: Random state for the stratified split.
    :param val_frac: Validation fraction.
    :param test_frac: Internal-test fraction.
    :param source_dirs: Vendor split folders to pool.
    :param deduplicate_first: Drop exact duplicates before splitting. Leave enabled;
        the flag exists so the leakage effect can be demonstrated, not disabled.
    :return: The split table and a provenance report.
    """
    raw_dir, out_csv = Path(raw_dir), Path(out_csv)

    pooled = pool_image_records(raw_dir, source_dirs=source_dirs)
    hashed = add_content_hashes(pooled, raw_dir)

    if deduplicate_first:
        unique, dedup_report = deduplicate(hashed)
    else:
        unique, dedup_report = hashed, {"images_before_dedup": len(hashed), "rows_removed": 0}

    split_df = stratified_split(unique, seed=seed, val_frac=val_frac, test_frac=test_frac)
    verify_no_leakage(split_df)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(out_csv, index=False)

    report: Dict[str, object] = {
        "raw_dir": str(raw_dir),
        "split_csv": str(out_csv),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "deduplicate_first": deduplicate_first,
        **dedup_report,
        "split_sizes": {
            split: int((split_df["split"] == split).sum()) for split in ("train", "val", "test")
        },
        "train_imbalance_ratio": imbalance_ratio(split_df, "train"),
    }
    return split_df, report


def load_split(split_csv: Path, split: Optional[str] = None) -> pd.DataFrame:
    """Read the split table, optionally filtered to one split.

    :param split_csv: Path written by :func:`build_split`.
    :param split: ``train``, ``val``, ``test``, or ``None`` for all rows.
    :return: The requested rows with a reset index.
    :raises FileNotFoundError: If the split CSV does not exist.
    """
    split_csv = Path(split_csv)
    if not split_csv.is_file():
        raise FileNotFoundError(
            f"Split file not found: {split_csv}. Run the datamodule's prepare_data(), "
            "or `python src/analyze.py analysis=step04_audit`, to build it."
        )
    df = pd.read_csv(split_csv)
    if split is not None:
        df = df[df["split"] == split]
    return df.reset_index(drop=True)
