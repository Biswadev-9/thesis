#!/usr/bin/env python
"""Run the whole study end to end with one command, on a machine that will be killed.

Kaggle stops a session after 12 hours whether or not the work is finished, so a
"run all" driver for this project has to be more than a list of commands. Three
things make it work:

**Pinned output directories.** Hydra normally writes each run to a timestamped
directory, which means the only way to give Step 16 the Step 15 checkpoint is to
read the timestamp off the console and paste it back. Every stage here is given
an explicit ``hydra.run.dir``, so ``logs/train/runs/step15_final/seed_42`` is
known before the run starts and downstream stages can just refer to it.

**Per-run resumability.** Each stage writes ``.pipeline_done.json`` when it
finishes. A second invocation skips completed stages, so restarting after the
session dies costs nothing. Training runs additionally resume mid-run from
``last.ckpt``, and are given a ``trainer.max_time`` matching the remaining time
budget so they stop cleanly and save rather than being killed.

**Selection propagation.** Steps 6, 8, 13 and 14 are studies that *choose*
something. Their choices are read back out of the summary JSON they write and
applied to the stages downstream, which is what the specification intends and
what a human otherwise does by hand.

Usage::

    python scripts/kaggle_pipeline.py --profile smoke     # ~15 min, proves the wiring
    python scripts/kaggle_pipeline.py --profile fast      # 1 seed, short schedules
    python scripts/kaggle_pipeline.py --profile full      # the study, 3 seeds

    python scripts/kaggle_pipeline.py --list              # show the stage graph
    python scripts/kaggle_pipeline.py --only step16_internal
    python scripts/kaggle_pipeline.py --from step13_fusion

Only ``--profile full`` produces reportable results. The other two shorten the
fixed training protocol, which ``configs/protocol/fixed.yaml`` says is a protocol
amendment; runs made under them are stamped ``protocol_intact: false`` in the
manifest and must not be quoted in the thesis.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]

#: Entry points, keyed by the Hydra ``task_name`` they run under - which is also the
#: directory they write into, so the two must not drift apart.
ENTRYPOINTS: Dict[str, str] = {
    "analyze": "src/analyze.py",
    "train": "src/train.py",
    "extract_features": "src/extract_features.py",
    "prepare_dataset": "src/prepare_dataset.py",
}

#: Recipes that need no materialised mirror. Mirrors ``IDENTITY_RECIPES`` in
#: ``src/data/components/preprocessing.py``.
IDENTITY_RECIPES = ("raw", "conventional")

#: Step 9's seven baselines, in specification order.
BASELINES = [
    "baseline_simple_cnn",
    "baseline_resnet50",
    "baseline_efficientnet_b0",
    "baseline_vit",
    "baseline_swin",
    "baseline_fixed_qcnn",
    "baseline_fixed_multiscale",
]

#: Step 11's eight ablation arms.
ARMS = [
    "arm1_fixed_3x3",
    "arm2_fixed_5x5",
    "arm3_fixed_dilated",
    "arm4_concat_nogate",
    "arm5_global_gate",
    "arm6_spatial_gate",
    "arm7_spatial_gate_quantum",
    "arm8_global_gate_quantum",
]

#: Models whose forward pass runs a circuit on the CPU simulator. Sending these to the
#: GPU moves tensors off the accelerator and back on every batch, so they stay on CPU.
QUANTUM_MODELS = {"baseline_fixed_qcnn"}
QUANTUM_ARMS = {"arm7_spatial_gate_quantum", "arm8_global_gate_quantum"}

#: Step 8 strategy -> (loss config, weighted sampler, augmentation). Mirrors
#: ``STRATEGIES`` in ``src/analysis/imbalance_study.py``; the study names a strategy,
#: this is how that name becomes training overrides.
IMBALANCE_OVERRIDES: Dict[str, Any] = {
    "baseline": ("plain_ce", False, False),
    "class_weighting": ("weighted_ce", False, False),
    "focal_loss": ("focal", False, False),
    "focal_loss_legacy": ("focal_legacy", False, False),
    "weighted_sampler": ("plain_ce", True, False),
    "augmentation": ("plain_ce", False, True),
    "combined_sampler_weighting": ("weighted_ce", True, False),
}

#: Seconds held back from the time budget so the manifest and bundle still get written.
RESERVE_SECONDS = 480

#: A stage shorter than this is not worth starting near the end of a session.
MIN_STAGE_SECONDS = 240


class BudgetExhausted(RuntimeError):
    """Raised when the remaining wall-clock budget cannot fit another stage."""


class StageFailed(RuntimeError):
    """Raised when a required stage exits non-zero."""


# ------------------------------------------------------------------------------------ #
# Stage definition
# ------------------------------------------------------------------------------------ #


@dataclass
class Stage:
    """One invocation of one entry point.

    :param id: Unique name; also the run directory under ``logs/<entry>/runs/``.
    :param entry: Key into :data:`ENTRYPOINTS`.
    :param build: Returns the Hydra overrides, or ``None`` to skip the stage.
    :param group: Coarse label used by ``--only`` / ``--from`` / ``--skip``.
    :param is_train: Training stages get a time cap and mid-run resume.
    :param optional: A failure is recorded and the pipeline continues.
    :param note: One line shown in ``--list``.
    """

    id: str
    entry: str
    build: Callable[["Pipeline"], Optional[List[str]]]
    group: str
    is_train: bool = False
    optional: bool = False
    note: str = ""


# ------------------------------------------------------------------------------------ #
# The pipeline
# ------------------------------------------------------------------------------------ #


class Pipeline:
    """Executes the stage graph, carrying selections forward between stages."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = ROOT
        self.started = time.time()
        self.deadline = self.started + args.budget_hours * 3600.0
        self.records: List[Dict[str, Any]] = []
        self.selections: Dict[str, Any] = {}
        self._summary_cache: Dict[Path, Dict[str, Any]] = {}

        # Only `full` writes to the canonical logs/ tree USAGE.md documents. A shortened
        # profile gets its own root, because completion markers would otherwise make a
        # later `full` run skip every stage and quietly hand back a study built from
        # one-epoch models.
        logs = self.root / "logs"
        self.log_root = logs if self.protocol_intact else logs / f"_{args.profile}"

        # Same reasoning for the feature cache: Steps 13-15 would train over whatever
        # tensors happened to be in data/features/default.
        self.tag = args.tag if (self.protocol_intact or args.tag != "default") else args.profile

        # Created on first write, not here: constructing a Pipeline to inspect the graph
        # (--list, the tests) should not leave directories behind.
        self.pipeline_dir = self.log_root / "pipeline"
        self.console_log = self.pipeline_dir / "pipeline.log"

        self.seeds: List[int] = (
            [int(s) for s in args.seeds.split(",")] if args.seeds else self._profile_seeds()
        )
        self.gpu = self._detect_gpu() if args.accelerator == "auto" else args.accelerator == "gpu"

    # -- profile ---------------------------------------------------------------------

    def _profile_seeds(self) -> List[int]:
        return {"smoke": [42], "fast": [42], "full": [42, 123, 7]}[self.args.profile]

    @property
    def protocol_intact(self) -> bool:
        """:return: Whether the fixed protocol was left untouched (``full`` only)."""
        return self.args.profile == "full"

    def train_shape(self) -> List[str]:
        """:return: Profile overrides that shorten training. Empty under ``full``."""
        if self.args.profile == "smoke":
            return [
                "++trainer.max_epochs=1",
                "++trainer.limit_train_batches=3",
                "++trainer.limit_val_batches=3",
                "++trainer.limit_test_batches=3",
                "++callbacks.early_stopping.patience=1",
            ]
        if self.args.profile == "fast":
            return ["++trainer.max_epochs=8", "++callbacks.early_stopping.patience=4"]
        return []

    def analysis_shape(self, stage_id: str) -> List[str]:
        """:param stage_id: Stage being built.

        :return: Profile overrides that shrink an analysis stage.
        """
        smoke = self.args.profile == "smoke"
        fast = self.args.profile == "fast"
        if not (smoke or fast):
            return []

        table: Dict[str, Dict[str, List[str]]] = {
            "step06_preprocessing": {
                "smoke": [
                    "analysis.epochs=1",
                    "analysis.per_class_train=24",
                    "analysis.per_class_val=12",
                    "analysis.diffusion_iterations=[10]",
                    "analysis.diffusion_kappas=[15.0]",
                    "analysis.comparators=[conventional,clahe]",
                    "analysis.edge_sample_size=4",
                ],
                "fast": [
                    "analysis.epochs=3",
                    "analysis.diffusion_iterations=[10,15]",
                    "analysis.diffusion_kappas=[15.0]",
                ],
            },
            "step08_imbalance": {
                "smoke": [
                    "analysis.epochs=1",
                    "analysis.per_class_train=24",
                    "analysis.per_class_val=12",
                ],
                "fast": ["analysis.epochs=3"],
            },
            # max_points only thins the t-SNE; the forward pass over every split is what
            # costs, so smoke drops to one split.
            "step10_embeddings": {
                "smoke": ["analysis.max_points=200", "analysis.splits=[val]"],
                "fast": [],
            },
            "step11_gate_morphology": {"smoke": ["analysis.samples_per_class=4"], "fast": []},
            "step13_fusion": {"smoke": ["analysis.epochs=2"], "fast": ["analysis.epochs=10"]},
            "step14_loss_selection": {
                "smoke": ["analysis.epochs=2"],
                "fast": ["analysis.epochs=12"],
            },
            "step18_robustness": {"smoke": ["analysis.limit_batches=2"], "fast": []},
        }
        return table.get(stage_id, {}).get("smoke" if smoke else "fast", [])

    # -- environment -----------------------------------------------------------------

    @staticmethod
    def _detect_gpu() -> bool:
        try:
            import torch  # imported lazily: --list must work without a torch install

            return bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover - environment probe
            return False

    def trainer_override(self, quantum: bool) -> List[str]:
        """:param quantum: Whether the model runs circuits on the CPU simulator.

        :return: The trainer group override for this model.
        """
        if quantum and self.args.quantum_accelerator == "cpu":
            return ["trainer=default"]
        return ["trainer=gpu"] if self.gpu else ["trainer=default"]

    def loader_overrides(self) -> List[str]:
        """:return: Dataloader overrides for stages that read images from disk."""
        out = [f"data.num_workers={self.args.num_workers}"]
        if self.gpu:
            out.append("data.pin_memory=true")
        return out

    # -- selections ------------------------------------------------------------------

    def out_dir(self, stage: Stage) -> Path:
        """:param stage: Stage to locate.

        :return: The pinned Hydra run directory for the stage.
        """
        return self.log_root / stage.entry / "runs" / stage.id

    def read_summary(self, stage_id: str, entry: str, filename: str) -> Optional[Dict[str, Any]]:
        """Read an analysis summary written by a previous stage.

        :param stage_id: Stage that produced it.
        :param entry: Entry point the stage ran under.
        :param filename: Summary file name inside the run directory.
        :return: The parsed summary, or ``None`` if it is not there.
        """
        path = self.log_root / entry / "runs" / stage_id / filename
        if path in self._summary_cache:
            return self._summary_cache[path]
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        self._summary_cache[path] = payload
        return payload

    def selected_recipe(self) -> Optional[str]:
        """:return: The preprocessing recipe to train on, or ``None`` for raw images."""
        if "recipe" in self.selections:
            return self.selections["recipe"]

        recipe: Optional[str] = None
        if self.args.recipe is not None:
            recipe = None if self.args.recipe in ("null", "none", "") else self.args.recipe
        elif self.args.apply_selections:
            summary = self.read_summary(
                "step06_preprocessing", "analyze", "step06_preprocessing_summary.json"
            )
            if summary:
                candidate = str(summary.get("selected_recipe", ""))
                recipe = None if candidate in IDENTITY_RECIPES or not candidate else candidate

        self.selections["recipe"] = recipe
        return recipe

    def recipe_override(self) -> List[str]:
        """:return: ``data.recipe=...`` for image stages, honouring the Step 6 choice."""
        recipe = self.selected_recipe()
        return [f"data.recipe={recipe}" if recipe else "data.recipe=null"]

    def imbalance_overrides(self, with_data: bool = True) -> List[str]:
        """Translate Step 8's chosen strategy into training overrides.

        :param with_data: Include the datamodule flags. Off for the feature datamodule,
            which has no ``augment`` key.
        :return: Hydra overrides, empty when nothing was selected.
        """
        strategy: Optional[str] = self.selections.get("imbalance")
        if strategy is None:
            if self.args.imbalance is not None:
                strategy = self.args.imbalance
            elif self.args.apply_selections:
                summary = self.read_summary(
                    "step08_imbalance", "analyze", "step08_imbalance_summary.json"
                )
                strategy = str(summary["selected_strategy"]) if summary else ""
            else:
                strategy = ""
            self.selections["imbalance"] = strategy

        if not strategy or strategy not in IMBALANCE_OVERRIDES:
            return []

        loss, sampler, augment = IMBALANCE_OVERRIDES[strategy]
        out = [f"loss@model.criterion={loss}"]
        if with_data:
            out += [
                f"data.use_weighted_sampler={str(sampler).lower()}",
                f"data.augment={str(augment).lower()}",
            ]
        else:
            out += [f"data.use_weighted_sampler={str(sampler).lower()}"]
        return out

    def selected_loss(self) -> Optional[str]:
        """:return: The loss Step 14 selected, or the ``--loss`` override."""
        if self.args.loss:
            return self.args.loss
        summary = self.read_summary(
            "step14_loss_selection", "analyze", "step14_loss_selection_summary.json"
        )
        return str(summary["selected_loss"]) if summary else None

    def branch_ckpt(self, stage_id: str) -> Optional[str]:
        """:param stage_id: A training stage id.

        :return: Its run directory as a POSIX path, or ``None`` if it never ran.
        """
        path = self.log_root / "train" / "runs" / stage_id
        return path.as_posix() if (path / "checkpoints").is_dir() else None

    # -- execution -------------------------------------------------------------------

    def remaining(self) -> float:
        """:return: Seconds left in the budget, less the archiving reserve."""
        return self.deadline - time.time() - RESERVE_SECONDS

    def echo(self, message: str) -> None:
        """:param message: Line to print and append to the pipeline log."""
        print(message, flush=True)
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        with self.console_log.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def run_stage(self, stage: Stage) -> str:
        """Execute one stage, honouring markers, budget and resume.

        :param stage: The stage to run.
        :return: ``done``, ``skipped``, ``cached`` or ``failed``.
        :raises BudgetExhausted: If there is no time left to start it.
        :raises StageFailed: If a required stage exits non-zero.
        """
        out = self.out_dir(stage)
        marker = out / ".pipeline_done.json"

        if marker.exists() and not self.args.force:
            self.echo(f"  [cached]  {stage.id}  ->  {out.relative_to(self.root).as_posix()}")
            self.records.append({"stage": stage.id, "status": "cached", "out_dir": str(out)})
            return "cached"

        try:
            overrides = stage.build(self)
        except StageFailed as error:
            # A stage whose inputs were never produced - worth recording as such rather
            # than losing it, since --keep-going will carry on past it.
            self.records.append({"stage": stage.id, "status": "blocked", "reason": str(error)})
            raise

        if overrides is None:
            self.echo(f"  [skip]    {stage.id}  (not applicable)")
            self.records.append({"stage": stage.id, "status": "skipped"})
            return "skipped"

        budget = self.remaining()
        if budget < MIN_STAGE_SECONDS:
            raise BudgetExhausted(f"{budget / 60:.1f} min left; not starting {stage.id}")

        out.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, ENTRYPOINTS[stage.entry], *overrides]

        cap: Optional[float] = None
        if stage.is_train:
            last = out / "checkpoints" / "last.ckpt"
            if last.is_file() and not self.args.force:
                argv.append(f"ckpt_path={last.as_posix()}")
                self.echo(f"  [resume]  {stage.id} from last.ckpt")
            if not self.args.no_time_cap:
                cap = budget
                argv.append(f"++trainer.max_time={_hms(cap)}")
        if stage.is_train and not self.args.progress:
            # Sixty-odd runs of Rich redrawing a progress bar overflows a notebook's
            # output buffer. Lightning refuses `enable_progress_bar=false` while a
            # progress-bar callback is configured, so the callback goes too. The
            # depth -1 model summary is the other offender: ViT alone prints several
            # hundred lines of it, and the parameter counts land in resource_usage.json
            # regardless.
            argv += [
                "~callbacks.rich_progress_bar",
                "++trainer.enable_progress_bar=false",
                "++callbacks.model_summary.max_depth=1",
            ]

        argv.append(f"hydra.run.dir={out.as_posix()}")

        self.echo("")
        self.echo(f"  [run]     {stage.id}")
        self.echo(f"            {' '.join(argv[1:])}")

        began = time.time()
        code = self._spawn(argv, out / "stage.log")
        elapsed = time.time() - began

        truncated = bool(cap and elapsed >= 0.95 * cap)
        record: Dict[str, Any] = {
            "stage": stage.id,
            "status": "done" if code == 0 else "failed",
            "returncode": code,
            "seconds": round(elapsed, 1),
            "out_dir": str(out),
            "overrides": overrides,
        }

        if code != 0:
            self.echo(f"  [FAIL]    {stage.id}  exit {code}  after {elapsed / 60:.1f} min")
            self.records.append(record)
            if stage.optional:
                return "failed"
            raise StageFailed(stage.id)

        if truncated:
            record["status"] = "partial"
            record["truncated_by_budget"] = True
            self.records.append(record)
            (out / ".pipeline_partial.json").write_text(json.dumps(record, indent=2))
            self.echo(f"  [partial] {stage.id} hit the time cap; it will resume next session")
            raise BudgetExhausted(f"{stage.id} was cut short by the time budget")

        record["profile"] = self.args.profile
        record["protocol_intact"] = self.protocol_intact
        record["finished"] = _now()
        marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
        (out / ".pipeline_partial.json").unlink(missing_ok=True)
        self.records.append(record)
        self.echo(f"  [ok]      {stage.id}  in {elapsed / 60:.1f} min")
        return "done"

    def _spawn(self, argv: List[str], log_path: Path) -> int:
        """Run a subprocess, echoing its output and teeing it to a file.

        :param argv: Command to run.
        :param log_path: File the output is copied into.
        :return: The process exit code.
        """
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HYDRA_FULL_ERROR", "1")

        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"\n=== {_now()} :: {' '.join(argv)}\n")
            handle.flush()
            process = subprocess.Popen(
                argv,
                cwd=str(self.root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                # A notebook cell's stdout is a pipe, so Python block-buffers it and a
                # long stage looks hung for minutes at a time. Flush per line.
                sys.stdout.flush()
                handle.write(line)
            process.stdout.close()
            return process.wait()


def _hms(seconds: float) -> str:
    """:param seconds: A duration.

    :return: Lightning's ``DD:HH:MM:SS`` ``max_time`` format.
    """
    seconds = max(int(seconds), 60)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{secs:02d}"


def _now() -> str:
    """:return: An ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------ #
# The stage graph
# ------------------------------------------------------------------------------------ #


def build_stages(pipe: Pipeline) -> List[Stage]:
    """Assemble the ordered stage graph for the configured seeds and profile.

    :param pipe: The pipeline, consulted for seeds and profile.
    :return: Stages in execution order.
    """
    stages: List[Stage] = []
    add = stages.append
    seeds = pipe.seeds

    # -- Step 4: audit, and the split every later stage depends on ---------------------
    add(
        Stage(
            "step04_audit",
            "analyze",
            lambda p: ["analysis=step04_audit", *p.loader_overrides()],
            group="step04",
            note="data audit; builds data/splits/dataset_split.csv",
        )
    )

    # -- Step 6: rank preprocessing, then materialise the winner -----------------------
    add(
        Stage(
            "step06_preprocessing",
            "analyze",
            lambda p: [
                "analysis=step06_preprocessing",
                *p.loader_overrides(),
                *p.analysis_shape("step06_preprocessing"),
            ],
            group="step06",
            note="proxy sweep over preprocessing recipes",
        )
    )

    def _prepare(p: Pipeline) -> Optional[List[str]]:
        recipe = p.selected_recipe()
        if not recipe:
            return None
        return [f"recipe={recipe}"]

    add(
        Stage(
            "step06_materialise",
            "prepare_dataset",
            _prepare,
            group="step06",
            note="write the chosen recipe to data/processed/ (skipped if identity)",
        )
    )

    # -- Step 8: imbalance strategy ----------------------------------------------------
    add(
        Stage(
            "step08_imbalance",
            "analyze",
            lambda p: [
                "analysis=step08_imbalance",
                *p.loader_overrides(),
                *p.analysis_shape("step08_imbalance"),
            ],
            group="step08",
            note="compare imbalance strategies on macro-F1 and worst-class recall",
        )
    )

    # -- Step 9 + 15: the seven baselines ----------------------------------------------
    for model in BASELINES:
        for seed in seeds:
            add(
                Stage(
                    f"step09_baselines/{model}/seed_{seed}",
                    "train",
                    _train_builder(
                        [
                            "experiment=step09_baselines",
                            f"model={model}",
                            f"seed={seed}",
                        ],
                        quantum=model in QUANTUM_MODELS,
                    ),
                    group="step09",
                    is_train=True,
                    note=f"baseline {model}, seed {seed}",
                )
            )

    # -- Step 10: classical branch, then its embeddings --------------------------------
    for seed in seeds:
        add(
            Stage(
                f"step10_classical/seed_{seed}",
                "train",
                _train_builder(["experiment=step10_classical", f"seed={seed}"]),
                group="step10",
                is_train=True,
                note=f"classical feature branch, seed {seed}",
            )
        )

    add(
        Stage(
            "step10_embeddings",
            "analyze",
            _ckpt_analysis(
                "analysis=step10_embeddings",
                "model=branch_classical",
                ckpt_key="analysis.ckpt_path",
                ckpt_stage=lambda p: f"step10_classical/seed_{p.seeds[0]}",
                extra=lambda p: p.analysis_shape("step10_embeddings"),
            ),
            group="step10",
            note="t-SNE projection and silhouette score",
        )
    )

    # -- Step 11: the eight-arm ablation, then gate morphology -------------------------
    for arm in ARMS:
        for seed in seeds:
            add(
                Stage(
                    f"step11_arm_ablation/{arm}/seed_{seed}",
                    "train",
                    _train_builder(
                        [
                            "experiment=step11_arm_ablation",
                            f"model.net.arm={arm}",
                            f"seed={seed}",
                        ],
                        quantum=arm in QUANTUM_ARMS,
                    ),
                    group="step11",
                    is_train=True,
                    note=f"arm {arm}, seed {seed}",
                )
            )

    add(
        Stage(
            "step11_gate_morphology",
            "analyze",
            _ckpt_analysis(
                "analysis=step11_gate_morphology",
                "model=branch_multiscale",
                "model.net.arm=arm6_spatial_gate",
                ckpt_key="analysis.ckpt_path",
                ckpt_stage=lambda p: f"step11_arm_ablation/arm6_spatial_gate/seed_{p.seeds[0]}",
                extra=lambda p: p.analysis_shape("step11_gate_morphology"),
            ),
            group="step11",
            note="learned scale weights vs proxy tumour extent",
        )
    )

    # -- Step 12: adaptive quantum branch ----------------------------------------------
    for seed in seeds:
        add(
            Stage(
                f"step12_adaptive_quantum/seed_{seed}",
                "train",
                _train_builder(
                    ["experiment=step12_adaptive_quantum", f"seed={seed}"], quantum=True
                ),
                group="step12",
                is_train=True,
                note=f"adaptive quantum branch, seed {seed} (slowest stage in the study)",
            )
        )

    # -- The feature cache Steps 13-15 train over --------------------------------------
    def _features(p: Pipeline) -> Optional[List[str]]:
        classical = p.branch_ckpt(f"step10_classical/seed_{p.seeds[0]}")
        quantum = p.branch_ckpt(f"step12_adaptive_quantum/seed_{p.seeds[0]}")
        if not (classical and quantum):
            raise StageFailed(
                "extract_features needs the Step 10 and Step 12 checkpoints; "
                "run those stages first"
            )
        return [
            f"classical_ckpt={classical}",
            f"quantum_ckpt={quantum}",
            f"tag={p.tag}",
            *p.recipe_override(),
            *p.loader_overrides(),
        ]

    add(
        Stage(
            "features",
            "extract_features",
            _features,
            group="features",
            note="cache the three frozen branches to data/features/<tag>/",
        )
    )

    # -- Step 13: fusion strategy ------------------------------------------------------
    add(
        Stage(
            "step13_fusion",
            "analyze",
            lambda p: [
                "analysis=step13_fusion",
                f"analysis.tag={p.tag}",
                *p.analysis_shape("step13_fusion"),
            ],
            group="step13",
            note="concat vs SE vs gated, then branch-contribution ablation",
        )
    )

    # -- Step 14: loss ------------------------------------------------------------------
    add(
        Stage(
            "step14_loss_selection",
            "analyze",
            lambda p: [
                "analysis=step14_loss_selection",
                f"analysis.tag={p.tag}",
                *p.analysis_shape("step14_loss_selection"),
            ],
            group="step14",
            note="validation-only choice between weighted CE and focal",
        )
    )

    # -- Step 15: the final model -------------------------------------------------------
    def _final(seed: int) -> Callable[[Pipeline], Optional[List[str]]]:
        def build(p: Pipeline) -> Optional[List[str]]:
            loss = p.selected_loss()
            if loss is None:
                raise StageFailed(
                    "Step 15 needs the loss Step 14 selected; run step14_loss_selection "
                    "first or pass --loss"
                )
            return [
                "experiment=step15_final_protocol",
                f"seed={seed}",
                f"loss@model.criterion={loss}",
                f"data.tag={p.tag}",
                *p.trainer_override(quantum=False),
                *p.train_shape(),
                "logger=csv",
            ]

        return build

    for seed in seeds:
        add(
            Stage(
                f"step15_final/seed_{seed}",
                "train",
                _final(seed),
                group="step15",
                is_train=True,
                note=f"final fused classifier, seed {seed}",
            )
        )

    # -- Step 16: the one internal test -------------------------------------------------
    def _internal(p: Pipeline) -> Optional[List[str]]:
        return [
            "analysis=step16_internal",
            *_pipeline_ckpts(p, prefix="analysis."),
            *p.recipe_override(),
            *p.loader_overrides(),
            f"analysis.force={str(p.args.force_test).lower()}",
        ]

    add(
        Stage(
            "step16_internal",
            "analyze",
            _internal,
            group="step16",
            note="the full metric battery; runs once and locks the checkpoint",
        )
    )

    # -- Step 17: external validation ----------------------------------------------------
    def _external(p: Pipeline) -> Optional[List[str]]:
        figshare = p.root / "data" / "raw" / "figshare"
        if not (figshare.is_dir() and any(figshare.rglob("*.mat"))):
            return None
        summary = p.log_root / "analyze" / "runs" / "step16_internal"
        summary = summary / "step16_internal_summary.json"
        overrides = [
            "analysis=step17_external",
            "data=figshare",
            *_pipeline_ckpts(p, prefix="analysis."),
            f"data.num_workers={p.args.num_workers}",
        ]
        if summary.is_file():
            overrides.append(f"analysis.internal_summary={summary.as_posix()}")
        return overrides

    add(
        Stage(
            "step17_external",
            "analyze",
            _external,
            group="step17",
            optional=True,
            note="Figshare cross-dataset validation (skipped if the dataset is absent)",
        )
    )

    # -- Step 18: robustness ---------------------------------------------------------------
    def _robustness(p: Pipeline) -> Optional[List[str]]:
        effnet = p.branch_ckpt(f"step09_baselines/baseline_efficientnet_b0/seed_{p.seeds[0]}")
        vit = p.branch_ckpt(f"step09_baselines/baseline_vit/seed_{p.seeds[0]}")
        if not (effnet and vit):
            raise StageFailed(
                "Step 18 compares against the EfficientNet-B0 and ViT baselines; "
                "run the Step 9 stages first"
            )
        return [
            "analysis=step18_robustness",
            *_pipeline_ckpts(p, prefix="analysis.models.proposed."),
            f"analysis.models.efficientnet_b0.ckpt={effnet}",
            f"analysis.models.vit.ckpt={vit}",
            *p.analysis_shape("step18_robustness"),
            f"data.num_workers={p.args.num_workers}",
        ]

    add(
        Stage(
            "step18_robustness",
            "analyze",
            _robustness,
            group="step18",
            note="five degradation families against the CNN and Transformer baselines",
        )
    )

    return stages


def _train_builder(
    base: Sequence[str], quantum: bool = False
) -> Callable[[Pipeline], Optional[List[str]]]:
    """Wrap a training stage's fixed overrides with the pipeline's contextual ones.

    :param base: Overrides specific to this stage.
    :param quantum: Whether the model runs circuits on the CPU simulator.
    :return: A build callable.
    """

    def build(p: Pipeline) -> Optional[List[str]]:
        return [
            *base,
            *p.trainer_override(quantum=quantum),
            *p.recipe_override(),
            *p.imbalance_overrides(with_data=True),
            *p.loader_overrides(),
            *p.train_shape(),
            "logger=csv",
        ]

    return build


def _ckpt_analysis(
    *base: str,
    ckpt_key: str,
    ckpt_stage: Callable[[Pipeline], str],
    extra: Callable[[Pipeline], List[str]],
) -> Callable[[Pipeline], Optional[List[str]]]:
    """Build an analysis stage that reloads one trained checkpoint.

    :param base: Fixed overrides.
    :param ckpt_key: Config key the checkpoint path goes into.
    :param ckpt_stage: Returns the training stage id supplying the checkpoint.
    :param extra: Returns profile-dependent overrides.
    :return: A build callable that skips the stage when the checkpoint is missing.
    """

    def build(p: Pipeline) -> Optional[List[str]]:
        ckpt = p.branch_ckpt(ckpt_stage(p))
        if ckpt is None:
            return None
        return [*base, f"{ckpt_key}={ckpt}", *extra(p), *p.loader_overrides()]

    return build


def _pipeline_ckpts(pipe: Pipeline, prefix: str) -> List[str]:
    """The three checkpoints that define the proposed model.

    :param pipe: The running pipeline.
    :param prefix: Config prefix the three keys hang off.
    :return: Overrides naming the classical, quantum and fusion run directories.
    :raises StageFailed: If any of the three has not been trained.
    """
    seed = pipe.seeds[0]
    parts = {
        "classical_ckpt": pipe.branch_ckpt(f"step10_classical/seed_{seed}"),
        "quantum_ckpt": pipe.branch_ckpt(f"step12_adaptive_quantum/seed_{seed}"),
        "fusion_ckpt": pipe.branch_ckpt(f"step15_final/seed_{seed}"),
    }
    missing = [key for key, value in parts.items() if value is None]
    if missing:
        raise StageFailed(f"missing trained checkpoints for: {', '.join(missing)}")
    return [f"{prefix}{key}={value}" for key, value in parts.items()]


# ------------------------------------------------------------------------------------ #
# Results
# ------------------------------------------------------------------------------------ #

#: Headline fields lifted out of each summary into the run report.
HEADLINES: Dict[str, Sequence[str]] = {
    "step04_audit_summary.json": (
        "n_images",
        "n_corrupted",
        "split_sizes",
        "train_imbalance_ratio",
        "leak_free",
    ),
    "step06_preprocessing_summary.json": ("selected_recipe", "selected_macro_f1"),
    "step08_imbalance_summary.json": (
        "selected_strategy",
        "selected_macro_f1",
        "selected_min_class_recall",
    ),
    "step13_fusion_summary.json": ("selected_strategy",),
    "step14_loss_selection_summary.json": ("selected_loss", "rationale"),
    "step16_internal_summary.json": (
        "overall.accuracy",
        "overall.balanced_accuracy",
        "overall.macro_f1",
        "overall.mcc",
        "calibration.expected_calibration_error",
    ),
    "step17_external_summary.json": (
        "restricted.macro_f1",
        "unrestricted.macro_f1",
        "drop",
        "predicted_absent_class_count",
    ),
}


def _dig(payload: Dict[str, Any], dotted: str) -> Any:
    """:param payload: A summary.

    :param dotted: A key path such as ``overall.macro_f1``.
    :return: The value, or ``None`` if any part of the path is missing.
    """
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def write_report(pipe: Pipeline) -> Path:
    """Write the manifest and a readable summary of everything that ran.

    :param pipe: The finished (or interrupted) pipeline.
    :return: Path to the Markdown report.
    """
    manifest = {
        "generated": _now(),
        "profile": pipe.args.profile,
        "protocol_intact": pipe.protocol_intact,
        "seeds": pipe.seeds,
        "gpu": pipe.gpu,
        "selections": pipe.selections,
        "elapsed_hours": round((time.time() - pipe.started) / 3600, 2),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "stages": pipe.records,
    }
    pipe.pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipe.pipeline_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    lines = [
        "# Pipeline report",
        "",
        f"Generated {manifest['generated']} · profile **{pipe.args.profile}** · "
        f"seeds {pipe.seeds} · {'GPU' if pipe.gpu else 'CPU'}",
        "",
    ]
    if not pipe.protocol_intact:
        lines += [
            "> **Not reportable.** This profile shortens the fixed training protocol, so "
            "these numbers check the wiring, not the science. Re-run with "
            "`--profile full` for results.",
            "",
        ]

    counts: Dict[str, int] = {}
    for record in pipe.records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    lines += ["## Stages", "", " · ".join(f"{v} {k}" for k, v in sorted(counts.items())), ""]
    lines += ["| Stage | Status | Minutes |", "|---|---|---|"]
    for record in pipe.records:
        seconds = record.get("seconds")
        minutes = f"{seconds / 60:.1f}" if seconds else "-"
        lines.append(f"| `{record['stage']}` | {record['status']} | {minutes} |")

    lines += ["", "## Headline results", ""]
    found = False
    for name, keys in HEADLINES.items():
        # Read the pinned stage directory only. Globbing logs/ would also pick up
        # summaries from earlier timestamped runs and report several contradictory
        # answers to the same question.
        stage_id = name.removesuffix("_summary.json")
        path = pipe.log_root / "analyze" / "runs" / stage_id / name
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            picked = {k: _dig(payload, k) for k in keys}
            picked = {k: v for k, v in picked.items() if v is not None}
            if not picked:
                continue
            found = True
            lines.append(
                f"**{name.replace('_summary.json', '')}** — "
                + ", ".join(f"{k.split('.')[-1]}: `{v}`" for k, v in picked.items())
            )
            lines.append("")
    if not found:
        lines.append("_No summaries written yet._")

    report = pipe.pipeline_dir / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


#: Extensions carried into the results bundle. Checkpoints and feature caches are far
#: too large to ship alongside tables and figures.
BUNDLE_SUFFIXES = {".json", ".csv", ".png", ".md", ".yaml", ".txt", ".log", ".svg", ".pdf"}


def bundle_results(pipe: Pipeline, destination: Path) -> Path:
    """Zip every lightweight artefact into one file that outlives the session.

    :param pipe: The finished pipeline.
    :param destination: Directory the archive is written to.
    :return: Path to the archive.
    """
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = destination / f"thesis_results_{stamp}.zip"

    sources: List[Path] = [pipe.root / "logs"]
    splits = pipe.root / "data" / "splits"
    if splits.is_dir():
        sources.append(splits)

    written = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for source in sources:
            for path in source.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in BUNDLE_SUFFIXES:
                    continue
                if path.stat().st_size > 64 * 1024 * 1024:
                    continue
                zf.write(path, path.relative_to(pipe.root).as_posix())
                written += 1

    print(f"\nBundled {written} files -> {archive}  ({archive.stat().st_size / 1e6:.1f} MB)")
    return archive


def restore_state(source: Path, root: Path) -> None:
    """Copy a previous session's outputs back in, so the pipeline resumes rather than repeats.

    Only state is restored - logs, splits, materialised recipes and feature caches. Code
    comes from the repository, so a restored session still picks up any commits since.

    :param source: A previous session's project directory.
    :param root: This session's project directory.
    """
    for relative in ("logs", "data/splits", "data/processed", "data/features"):
        src = source / relative
        if not src.is_dir():
            continue
        dst = root / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copytree(src, dst)
        print(f"Restored {relative} from {src}")


# ------------------------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------------------------ #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """:param argv: Command line, or ``None`` for ``sys.argv``.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "fast", "full"),
        default="fast",
        help="smoke: minutes, proves the wiring. fast: one seed, short schedules. "
        "full: the study as specified - the only reportable profile.",
    )
    parser.add_argument("--seeds", default=None, help="Comma-separated, overrides the profile.")
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=11.0,
        help="Wall-clock budget. Kaggle kills a session at 12h; the default leaves room "
        "to write the bundle.",
    )
    parser.add_argument("--tag", default="default", help="Feature cache name.")
    parser.add_argument(
        "--accelerator", choices=("auto", "gpu", "cpu"), default="auto", help="Trainer device."
    )
    parser.add_argument(
        "--quantum-accelerator",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Quantum models run their circuits on a CPU simulator; keeping the whole "
        "model on CPU avoids moving tensors off the accelerator every batch.",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")

    parser.add_argument("--only", default=None, help="Comma-separated stage ids or groups.")
    parser.add_argument("--from", dest="from_stage", default=None, help="Start at this stage.")
    parser.add_argument("--until", default=None, help="Stop after this stage.")
    parser.add_argument("--skip", default=None, help="Comma-separated stage ids or groups.")
    parser.add_argument("--list", action="store_true", help="Print the stage graph and exit.")

    parser.add_argument("--force", action="store_true", help="Re-run stages already marked done.")
    parser.add_argument(
        "--no-time-cap", action="store_true", help="Do not cap training runs to the budget."
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="Continue past a failing stage."
    )
    parser.add_argument("--progress", action="store_true", help="Show Lightning progress bars.")

    parser.add_argument(
        "--no-apply-selections",
        dest="apply_selections",
        action="store_false",
        help="Do not feed the Step 6 and Step 8 choices into later stages.",
    )
    parser.add_argument("--recipe", default=None, help="Force a preprocessing recipe.")
    parser.add_argument("--imbalance", default=None, help="Force a Step 8 strategy.")
    parser.add_argument("--loss", default=None, help="Force the Step 15 loss.")
    parser.add_argument(
        "--force-test",
        action="store_true",
        help="Override the once-only internal test lock. Recorded in the summary.",
    )

    parser.add_argument(
        "--restore-from",
        default=None,
        help="A previous session's project directory; its logs and caches are copied in "
        "first so the run resumes.",
    )
    parser.add_argument(
        "--bundle-to",
        default=None,
        help="Directory for the results archive. Defaults to /kaggle/working when present.",
    )
    parser.add_argument("--no-bundle", action="store_true", help="Skip the results archive.")
    return parser.parse_args(argv)


def select_stages(stages: List[Stage], args: argparse.Namespace) -> List[Stage]:
    """Apply ``--only`` / ``--from`` / ``--until`` / ``--skip``.

    :param stages: The full graph.
    :param args: Parsed arguments.
    :return: The stages to run, in order.
    """

    def matches(stage: Stage, token: str) -> bool:
        return stage.id == token or stage.group == token or stage.id.startswith(token)

    chosen = list(stages)
    if args.only:
        tokens = [t.strip() for t in args.only.split(",") if t.strip()]
        chosen = [s for s in chosen if any(matches(s, t) for t in tokens)]
    if args.from_stage:
        index = next((i for i, s in enumerate(chosen) if matches(s, args.from_stage)), None)
        if index is None:
            raise SystemExit(f"--from: no stage matches {args.from_stage!r}")
        chosen = chosen[index:]
    if args.until:
        hits = [i for i, s in enumerate(chosen) if matches(s, args.until)]
        if not hits:
            raise SystemExit(f"--until: no stage matches {args.until!r}")
        chosen = chosen[: hits[-1] + 1]
    if args.skip:
        tokens = [t.strip() for t in args.skip.split(",") if t.strip()]
        chosen = [s for s in chosen if not any(matches(s, t) for t in tokens)]
    return chosen


def main(argv: Optional[Sequence[str]] = None) -> int:
    """:param argv: Command line, or ``None`` for ``sys.argv``.

    :return: Process exit code.
    """
    args = parse_args(argv)

    if args.restore_from:
        restore_state(Path(args.restore_from).resolve(), ROOT)

    pipe = Pipeline(args)
    stages = select_stages(build_stages(pipe), args)

    if args.list:
        width = max(len(s.id) for s in stages)
        for stage in stages:
            done = (pipe.out_dir(stage) / ".pipeline_done.json").exists()
            print(f"{'x' if done else ' '} {stage.id:<{width}}  {stage.note}")
        print(f"\n{len(stages)} stages · seeds {pipe.seeds} · profile {args.profile}")
        return 0

    pipe.echo("=" * 88)
    pipe.echo(f"Thesis pipeline · profile {args.profile} · seeds {pipe.seeds} · "
              f"{'GPU' if pipe.gpu else 'CPU'} · budget {args.budget_hours}h")
    pipe.echo(f"{len(stages)} stages · started {_now()}")
    if not pipe.protocol_intact:
        pipe.echo("WARNING: this profile shortens the fixed protocol. Results are not reportable.")
    pipe.echo("=" * 88)

    status = 0
    try:
        for stage in stages:
            try:
                pipe.run_stage(stage)
            except StageFailed as error:
                if not args.keep_going and not stage.optional:
                    pipe.echo(f"\nStopping: {error}")
                    status = 1
                    break
                pipe.echo(f"  [continue] after failure in {stage.id}: {error}")
    except BudgetExhausted as error:
        pipe.echo(f"\nOut of time: {error}")
        pipe.echo("Re-run the same command in a new session; finished stages are skipped.")
        status = 2
    except KeyboardInterrupt:
        pipe.echo("\nInterrupted.")
        status = 130

    report = write_report(pipe)
    pipe.echo(f"\nReport: {report}")

    if not args.no_bundle:
        default = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else ROOT / "logs"
        bundle_results(pipe, Path(args.bundle_to) if args.bundle_to else default)

    settled = {r["stage"] for r in pipe.records if r["status"] in ("done", "cached", "skipped")}
    pipe.echo(f"{len(settled)}/{len(stages)} stages settled.")
    if len(settled) < len(stages):
        pipe.echo("Re-run the same command to continue; finished stages are skipped.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
