import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import rootutils
from omegaconf import DictConfig
from PIL import Image

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# See src/train.py for what setup_root does.
# ------------------------------------------------------------------------------------ #

from src.data.components.preprocessing import build_recipe, is_identity_recipe
from src.data.components.split_builder import load_split
from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def materialise_recipe(
    recipe: str,
    raw_dir: Path,
    output_dir: Path,
    split_csv: Path,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply a preprocessing recipe to every image and cache the result to disk.

    This is what makes Step 6's chosen preprocessing usable in real training. Anisotropic
    diffusion at 10 iterations costs roughly a tenth of a second per image; applied
    on the fly it would dominate every epoch of every run, which is why the reference
    notebook selected diffusion in Step 6 and then never applied it in Steps 7 onward.
    Filtering once into a cached mirror removes that cost from the training loop.

    The mirror reproduces the raw tree's relative layout exactly, so the single split
    table addresses raw images and every recipe alike.

    :param recipe: Recipe name, e.g. ``diffusion_i10_k15`` or ``clahe``.
    :param raw_dir: Root of the raw dataset tree.
    :param output_dir: Destination directory for this recipe's mirror.
    :param split_csv: Split table listing the images to process.
    :param overwrite: Re-filter images whose output already exists.
    :param limit: Process at most this many images; for smoke tests.
    :return: A manifest describing what was written.
    """
    raw_dir, output_dir, split_csv = Path(raw_dir), Path(output_dir), Path(split_csv)
    df = load_split(split_csv)
    if limit is not None:
        df = df.head(limit)

    filter_fn = build_recipe(recipe)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = skipped = failed = 0
    started = time.time()

    for position, rel_path in enumerate(df["rel_path"], start=1):
        destination = output_dir / rel_path
        if destination.exists() and not overwrite:
            skipped += 1
            continue

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(raw_dir / rel_path) as image:
                filter_fn(image.convert("RGB")).save(destination)
            processed += 1
        except Exception as error:  # noqa: BLE001 - one bad file must not lose the batch
            log.warning(f"Failed on {rel_path}: {type(error).__name__}: {error}")
            failed += 1

        if position % 500 == 0:
            log.info(f"{position}/{len(df)} images ({processed} written, {skipped} skipped)")

    elapsed = time.time() - started
    manifest: Dict[str, Any] = {
        "recipe": recipe,
        "filter": repr(filter_fn),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "split_csv": str(split_csv),
        "images_total": int(len(df)),
        "images_written": processed,
        "images_skipped": skipped,
        "images_failed": failed,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_image": round(elapsed / processed, 4) if processed else None,
    }

    with open(output_dir / "recipe_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


@task_wrapper
def prepare(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Materialises one preprocessing recipe into a cached image mirror.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with the manifest and a dict with all instantiated objects.
    """
    recipe = cfg.recipe

    data_dir = Path(cfg.paths.data_dir)
    raw_dir = data_dir / cfg.raw_subdir
    split_csv = data_dir / cfg.split_subpath

    if is_identity_recipe(recipe):
        # 'raw' and 'conventional' apply no image-space filter; they differ only in the
        # Step 5 intensity treatment, which lives in the transform pipeline. Writing an
        # identical copy of the dataset would waste disk and invite drift.
        log.info(
            f"Recipe {recipe!r} applies no image-space filtering, so no mirror is needed. "
            f"Use `data.recipe=null data.normalize="
            f"{'none' if recipe == 'raw' else 'imagenet'}` instead."
        )
        return {"recipe": recipe, "skipped": True}, {"cfg": cfg}

    if not split_csv.is_file():
        raise FileNotFoundError(
            f"Split table not found at {split_csv}. Build it first with "
            "`python src/analyze.py analysis=step04_audit`."
        )

    output_dir = data_dir / "processed" / recipe
    log.info(f"Materialising recipe {recipe!r} -> {output_dir}")

    manifest = materialise_recipe(
        recipe=recipe,
        raw_dir=raw_dir,
        output_dir=output_dir,
        split_csv=split_csv,
        overwrite=cfg.get("overwrite", False),
        limit=cfg.get("limit"),
    )

    log.info(
        f"Recipe {recipe!r} ready: {manifest['images_written']} written, "
        f"{manifest['images_skipped']} already present, {manifest['images_failed']} failed "
        f"({manifest['elapsed_seconds']}s)"
    )
    log.info(f"Train on it with: python src/train.py data.recipe={recipe}")

    return manifest, {"cfg": cfg}


@hydra.main(version_base="1.3", config_path="../configs", config_name="prepare_dataset.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for dataset preparation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    prepare(cfg)


if __name__ == "__main__":
    main()
