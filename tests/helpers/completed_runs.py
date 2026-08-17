"""Fabricate training runs the pipeline will accept as finished.

A ``checkpoints/`` directory is what an *interrupted* run leaves behind: a run killed at
epoch two has one, holding an ``epoch_002.ckpt`` that is indistinguishable from a
converged run's best epoch. Completion is recorded separately, by the driver's
``.pipeline_done.json`` marker, and the stages that consume trained branches check for it -
so a test that wants "this branch was trained" has to say so, not merely leave a directory.

Used by the orchestration tests, which wire stages together without training anything.
"""

import json
from pathlib import Path
from typing import Iterable, Union


def mark_run_complete(run_dir: Union[str, Path], checkpoint: str = "epoch_003.ckpt") -> Path:
    """Make one run directory look like a training run that finished.

    :param run_dir: The run directory.
    :param checkpoint: Name of the checkpoint file to leave in ``checkpoints/``.
    :return: The run directory.
    """
    run_dir = Path(run_dir)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / checkpoint).write_bytes(b"")
    (run_dir / ".pipeline_done.json").write_text(
        json.dumps({"stage": run_dir.name, "status": "done", "returncode": 0}),
        encoding="utf-8",
    )
    return run_dir


def complete_train_runs(pipe, stage_ids: Iterable[str]) -> None:
    """Mark several of a pipeline's training stages as finished.

    :param pipe: The pipeline, for its log root.
    :param stage_ids: Training stage ids, as they appear under ``logs/train/runs/``.
    """
    for stage_id in stage_ids:
        mark_run_complete(pipe.log_root / "train" / "runs" / stage_id)
