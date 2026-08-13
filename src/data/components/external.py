"""The Figshare external validation dataset (Step 17).

Step 17 requires evaluation on data the model has never seen, from a different source:

    "Cross-dataset validation is necessary because benchmark accuracy alone does not prove
     clinical robustness."

The Figshare brain-tumour dataset ships as MATLAB v7.3 files, which are HDF5 underneath, so
``scipy.io.loadmat`` cannot read them and ``h5py`` is used instead. Each file holds a
``cjdata`` group with ``label``, ``image``, ``PID`` (patient identifier) and
``tumorMask``.

Two mapping details decide whether the evaluation is valid at all:

**Label indices differ between the datasets.** Figshare encodes 1=meningioma, 2=glioma,
3=pituitary; this project's Step 1 mapping is glioma=0, meningioma=1, pituitary=2. Getting
this backwards would silently produce a plausible-looking but meaningless confusion matrix,
so the mapping goes through the canonical class *names* rather than raw indices.

**Figshare has no non-tumour class.** Step 17 anticipates this: "If the external dataset
has only glioma, meningioma, and pituitary classes, evaluate a three-class external task
using the same trained feature extractor." How that restriction is applied is the analysis's
decision, not the dataset's - this loader simply reports the true labels.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from src.data.components.split_builder import CLASS_MAP
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

#: Figshare's own label encoding, mapped to this project's canonical class names.
FIGSHARE_LABELS: Dict[int, str] = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}

#: Classes present in the external set. No-tumor is absent by construction.
EXTERNAL_CLASSES: Tuple[str, ...] = ("Glioma", "Meningioma", "Pituitary")


def scan_figshare(root: Path) -> List[Dict[str, Any]]:
    """Index every readable ``.mat`` scan under a directory tree.

    :param root: Directory containing the Figshare release, at any nesting depth.
    :return: One record per scan, with ``filepath``, ``class_name``, ``label`` and ``pid``.
    :raises FileNotFoundError: If the directory or any scans are missing.
    """
    import h5py

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Figshare dataset not found at {root}. Download it with "
            "`scripts/download_data.ps1 -IncludeExternal` (or `--external` on Linux)."
        )

    files = sorted(root.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found under {root}")

    records: List[Dict[str, Any]] = []
    unreadable = 0

    for path in files:
        try:
            with h5py.File(path, "r") as handle:
                data = handle["cjdata"]
                raw_label = int(np.array(data["label"]).ravel()[0])
                # PID is stored as an array of character codes.
                pid = "".join(chr(int(code)) for code in np.array(data["PID"]).ravel())
        except Exception:  # noqa: BLE001 - a corrupt scan must not abort the scan pass
            unreadable += 1
            continue

        class_name = FIGSHARE_LABELS.get(raw_label)
        if class_name is None:
            unreadable += 1
            continue

        records.append(
            {
                "filepath": str(path),
                "class_name": class_name,
                "label": CLASS_MAP[class_name],
                "pid": pid,
            }
        )

    if unreadable:
        log.warning(f"Skipped {unreadable} unreadable or unmapped Figshare scans")
    log.info(f"Indexed {len(records)} Figshare scans from {root}")

    return records


class FigshareDataset(Dataset):
    """Figshare brain-tumour scans, presented like the internal dataset.

    Scans are 16-bit and vary in intensity range, so each is min-max normalised to 0-255
    before the shared Step 5 pipeline runs. Applying the internal dataset's fixed
    normalisation to raw 16-bit values would make the domain shift look far worse than it
    is, for a reason that has nothing to do with the model.

    :param records: Output of :func:`scan_figshare`.
    :param transform: Evaluation transform, the same one used for the internal test split.
    """

    def __init__(self, records: List[Dict[str, Any]], transform: Optional[Callable] = None) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        """:return: Number of scans."""
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """:param index: Scan index.
        :return: ``(image, label)`` with the label in this project's class indexing.
        """
        import h5py

        record = self.records[index]
        with h5py.File(record["filepath"], "r") as handle:
            array = np.array(handle["cjdata"]["image"]).astype(np.float32)

        # Per-image min-max to 8-bit: these scans are 16-bit with varying ranges.
        array -= array.min()
        maximum = array.max()
        if maximum > 0:
            array /= maximum
        image = Image.fromarray((array * 255).astype(np.uint8)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(record["label"])

    @property
    def labels(self) -> List[int]:
        """:return: Labels in dataset order."""
        return [record["label"] for record in self.records]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self)})"
