"""Step 21: evaluate the A0-A8 + P ablation matrix.

    "The ablation study should report the same metrics for every configuration. Use
     macro-F1 and class-wise recall as primary selection metrics because the task is
     multiclass and may be imbalanced."

*Same metrics* is taken literally: every row goes through
:mod:`src.analysis.metric_battery`, the identical code Step 16 uses for the headline
result, so an ablation row and the headline are directly comparable rather than merely
similarly named.

Three constraints shape this analysis.

**The test set informs nothing.** Checkpoints are chosen by the training run's own
``val/f1_macro`` callback, and this analysis only ever *reads* the resulting file. It
never inspects test metrics to pick a checkpoint, an epoch or a row, and it never reads
the ``test/*`` columns that ``trainer.test()`` writes into every run's ``metrics.csv`` -
those exist, and harvesting them would make the provenance of an ablation number
unauditable. Predictions here are computed fresh from the validation-selected checkpoint.

**Row P is not recomputed.** P is the shipped model, already evaluated once by Step 16.
Reporting it means reading that summary, not re-running it: a second evaluation would
either duplicate the headline or quietly contradict it, and would spend the once-only
test budget the lock file exists to protect.

**A8 has no delta.** Explanations change no weights, so its classification metrics are
identical to A7's by construction. It is reported as such rather than given a
manufactured difference.

The output is a deterministic, machine-readable matrix: per-seed metrics for every row,
mean/std/95% interval across seeds, and a flat table keyed on ``(row_id, seed)``.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.analysis import metric_battery as battery
from src.analysis.ablation_rows import (
    PROTOCOL_SEEDS,
    ROWS,
    AblationContext,
    AblationRow,
    get_row,
    resolve_loss,
    resolve_recipe,
)
from src.analysis.base import Analysis
from src.utils import RankedLogger
from src.utils.checkpoints import find_checkpoint, load_module

log = RankedLogger(__name__, rank_zero_only=True)

#: Metrics lifted into the flat comparison table. Macro-F1 and class-wise recall are the
#: specification's primary pair; the rest are the Step 16 battery, reported for every row
#: so the table is comparable end to end.
TABLE_METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "macro_precision",
    "macro_recall_sensitivity",
    "weighted_f1",
    "macro_specificity",
    "mcc",
    "auc_ovr_macro",
)

#: The two metrics Step 21 names as primary.
PRIMARY_METRICS = ("macro_f1", "per_class_recall")

#: Columns of the flat per-seed table, in order. Pinned so downstream steps can rely on
#: the schema rather than on whatever the dict happened to iterate as.
TABLE_COLUMNS = (
    "row_id",
    "label",
    "seed",
    "recipe",
    "loss",
    "model",
    "feature_tag",
    "checkpoint",
    "provenance",
    "n_samples",
    *TABLE_METRICS,
    "expected_calibration_error",
    "brier_score",
    "min_class_recall",
    "worst_class",
)


class AblationStudy(Analysis):
    """Evaluate every ablation row on the internal test set.

    :param name: Analysis identifier.
    :param step06_summary: Path to ``step06_preprocessing_summary.json``, which decides
        the diffusion recipe rows A2-A6 use and the recipe row P describes.
    :param step14_summary: Path to ``step14_loss_selection_summary.json``, which decides
        row A7's loss.
    :param step16_summary: Path to ``step16_internal_summary.json``. Row P is read from
        here rather than re-evaluated.
    :param run_root: Directory holding the Step 21 training runs, laid out as
        ``<run_root>/<row_id>/seed_<n>``.
    :param seeds: Protocol seeds to look for. Defaults to Step 15's three.
    :param row_models: Mapping of row id to the Hydra model config used to rebuild that
        row's module from its checkpoint.
    :param n_calibration_bins: Bins for the ECE estimate.
    :param accelerator: ``auto``, ``cpu`` or ``gpu``.
    :param strict: Fail if a row's checkpoints are missing, rather than recording the row
        as absent and continuing.
    """

    def __init__(
        self,
        name: str = "step21_ablation",
        step06_summary: Optional[str] = None,
        step14_summary: Optional[str] = None,
        step16_summary: Optional[str] = None,
        run_root: Optional[str] = None,
        seeds: Sequence[int] = PROTOCOL_SEEDS,
        row_models: Optional[Dict[str, Any]] = None,
        n_calibration_bins: int = 10,
        accelerator: str = "auto",
        strict: bool = False,
    ) -> None:
        super().__init__(name=name)
        self.step06_summary = step06_summary
        self.step14_summary = step14_summary
        self.step16_summary = step16_summary
        self.run_root = run_root
        self.seeds = tuple(int(s) for s in seeds)
        self.row_models = dict(row_models or {})
        self.n_calibration_bins = n_calibration_bins
        self.accelerator = accelerator
        self.strict = strict

    # ------------------------------------------------------------------- inputs

    def context(self) -> AblationContext:
        """Resolve the Step 6 and Step 14 answers the matrix depends on.

        :return: The resolved context.
        :raises FileNotFoundError: If either summary is missing.
        """
        step06 = _read_json(self.step06_summary, "Step 6 preprocessing")
        step14 = _read_json(self.step14_summary, "Step 14 loss selection")
        return AblationContext.from_summaries(step06, step14)

    # -------------------------------------------------------------------- compute

    def compute(self, datamodule: Any = None) -> Dict[str, Any]:
        """Evaluate every row and assemble the matrix.

        :param datamodule: Unused; each row supplies its own data through
            :meth:`build_datamodule`, because rows differ in preprocessing.
        :return: The ablation matrix.
        """
        context = self.context()
        log.info(
            f"Ablation context: diffusion={context.diffusion_recipe} "
            f"selected={context.selected_recipe} step14_loss={context.step14_loss}"
        )

        rows: Dict[str, Any] = {}
        for row in ROWS:
            rows[row.row_id] = self._evaluate_row(row, context, rows)

        table = self._flat_table(rows, context)
        self.save_table(table, "step21_ablation_matrix.csv")

        summary = {
            "context": {
                "diffusion_recipe": context.diffusion_recipe,
                "selected_recipe": context.selected_recipe,
                "step14_loss": context.step14_loss,
            },
            "seeds": list(self.seeds),
            "primary_metrics": list(PRIMARY_METRICS),
            "rows": rows,
            "coverage": self._coverage(rows),
            "notes": self._notes(context),
        }
        self._log(summary)
        return summary

    # ------------------------------------------------------------------ per row

    def _evaluate_row(
        self, row: AblationRow, context: AblationContext, done: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate one row across its seeds, or read it from where it already lives.

        :param row: The row.
        :param context: Resolved Step 6 / Step 14 selections.
        :param done: Rows already evaluated this run, for A8 to mirror.
        :return: The row's record.
        """
        record: Dict[str, Any] = {
            "row_id": row.row_id,
            "label": row.label,
            "recipe": resolve_recipe(row, context),
            "loss": resolve_loss(row, context),
            "model": row.model,
            "feature_tag": row.feature_tag,
            "trains": row.trains,
            "rqs": list(row.rqs),
            "note": row.note,
        }

        if row.row_id == "P":
            record.update(self._read_shipped_row())
            return record

        if row.row_id == "A8":
            record.update(self._mirror_row("A7", done))
            return record

        record.update(self._evaluate_seeds(row, context))
        return record

    def _evaluate_seeds(self, row: AblationRow, context: AblationContext) -> Dict[str, Any]:
        """Evaluate a trained row at each protocol seed.

        :param row: The row.
        :param context: Resolved Step 6 / Step 14 selections.
        :return: Per-seed results plus the across-seed summary.
        :raises FileNotFoundError: In strict mode, if a seed's checkpoint is absent.
        """
        per_seed: Dict[str, Any] = {}
        missing: List[int] = []

        for seed in self.seeds:
            run_dir = self.run_dir(row, seed)
            try:
                checkpoint = find_checkpoint(run_dir, prefer="best")
            except FileNotFoundError as error:
                if self.strict:
                    raise
                log.warning(f"{row.row_id} seed {seed}: {error}")
                missing.append(seed)
                continue

            outputs = self.predict_row(row, seed, checkpoint, context)
            metrics = battery.full_battery(
                outputs["y_true"],
                outputs["y_pred"],
                outputs["y_prob"],
                outputs["class_names"],
                self.n_calibration_bins,
            )
            metrics["checkpoint"] = str(checkpoint)
            # The selection rule, recorded per row so the table carries its own proof that
            # no test metric chose this checkpoint.
            metrics["selection"] = "val/f1_macro (training-time ModelCheckpoint)"
            metrics["provenance"] = "computed by step21 from the validation-selected checkpoint"
            per_seed[str(seed)] = metrics

            np.savez_compressed(
                self.output_dir / f"step21_predictions_{row.row_id}_seed{seed}.npz",
                y_true=outputs["y_true"],
                y_pred=outputs["y_pred"],
                y_prob=outputs["y_prob"],
            )

        return {
            "per_seed": per_seed,
            "across_seeds": self._across_seeds(per_seed),
            "missing_seeds": missing,
            "evaluated": bool(per_seed),
        }

    def _read_shipped_row(self) -> Dict[str, Any]:
        """Read row P from Step 16 rather than re-evaluating the shipped model.

        Re-running it would spend the once-only test budget a second time and could report
        a number that disagrees with the study's own headline.

        :return: P's record, in the same shape as an evaluated row.
        """
        try:
            step16 = _read_json(self.step16_summary, "Step 16 internal test")
        except FileNotFoundError as error:
            if self.strict:
                raise
            log.warning(f"Row P: {error}")
            return {"per_seed": {}, "across_seeds": {}, "missing_seeds": list(self.seeds),
                    "evaluated": False}

        seed = _seed_from_path(step16.get("checkpoint"))
        metrics = {
            "n_samples": step16.get("n_test_samples"),
            "overall": step16.get("overall", {}),
            "per_class": step16.get("per_class", []),
            "confusion": step16.get("confusion", {}),
            "calibration": step16.get("calibration", {}),
            "checkpoint": step16.get("checkpoint"),
            "selection": "val/f1_macro (training-time ModelCheckpoint)",
            "provenance": f"reused from Step 16 ({self.step16_summary})",
        }
        per_seed = {str(seed if seed is not None else self.seeds[0]): metrics}

        return {
            "per_seed": per_seed,
            "across_seeds": self._across_seeds(per_seed),
            # Step 16 evaluated one checkpoint, by design. Recorded rather than hidden:
            # P's spread is not comparable to a three-seed A-row's.
            "missing_seeds": [s for s in self.seeds if str(s) not in per_seed],
            "evaluated": True,
            "reused_from": "step16_internal",
        }

    def _mirror_row(self, source_id: str, done: Dict[str, Any]) -> Dict[str, Any]:
        """A8's metrics are A7's, because explanations change no weights.

        Copied from the already-computed record rather than recomputed. Re-evaluating
        would run the whole test split through A7's checkpoint a second time - the same
        arithmetic, at twice the cost, and a second read of the test set for a row that
        cannot differ from the first.

        :param source_id: Row whose metrics are mirrored.
        :param done: Rows already evaluated this run.
        :return: The mirrored record.
        :raises KeyError: If the source row has not been evaluated yet.
        """
        import copy

        if source_id not in done:
            raise KeyError(
                f"A8 mirrors {source_id}, which must be evaluated first; check ROWS order."
            )

        source = copy.deepcopy(
            {key: done[source_id][key] for key in ("per_seed", "across_seeds", "missing_seeds", "evaluated")}
        )
        for metrics in source["per_seed"].values():
            metrics["provenance"] = (
                f"identical to {source_id} by construction; explanations change no weights"
            )
        source["mirrors"] = source_id
        return source

    # -------------------------------------------------------------- overridable

    def run_dir(self, row: AblationRow, seed: int) -> Path:
        """:param row: The row.

        :param seed: Protocol seed.
        :return: Where that row's training run for that seed lives.
        """
        return Path(self.run_root or "logs/train/runs/step21_ablation") / row.row_id / f"seed_{seed}"

    def build_datamodule(self, row: AblationRow, context: AblationContext) -> Any:
        """Instantiate the datamodule a row is evaluated on.

        Each row carries its own preprocessing, so each gets its own datamodule rather
        than sharing one - which is precisely what makes A0, A1 and A2 different rows.

        :param row: The row.
        :param context: Resolved Step 6 / Step 14 selections.
        :return: An un-setup datamodule.
        """
        if row.feature_tag is not None:
            from src.data.bt_mri_feature_datamodule import BTMRIFeatureDataModule

            return BTMRIFeatureDataModule(tag=row.feature_tag)

        from src.data.bt_mri_datamodule import BTMRIDataModule

        recipe = resolve_recipe(row, context)
        return BTMRIDataModule(
            recipe=None if recipe in ("raw", "conventional") else recipe,
            normalize=row.normalize,
            augment=False,  # never on an evaluation split
            use_weighted_sampler=False,
        )

    def predict_row(
        self, row: AblationRow, seed: int, checkpoint: Path, context: AblationContext
    ) -> Dict[str, Any]:
        """Run one row's checkpoint over the internal test split.

        :param row: The row.
        :param seed: Protocol seed.
        :param checkpoint: The validation-selected checkpoint.
        :param context: Resolved Step 6 / Step 14 selections.
        :return: ``{"y_true", "y_pred", "y_prob", "class_names"}``.
        """
        device = torch.device(
            "cuda" if (self.accelerator != "cpu" and torch.cuda.is_available()) else "cpu"
        )
        datamodule = self.build_datamodule(row, context)
        datamodule.prepare_data()
        datamodule.setup()

        module = load_module(checkpoint, model_cfg=self.row_models.get(row.row_id)).to(device)

        if row.feature_tag is not None:
            from src.analysis.sweep_utils import collect_feature_predictions

            outputs = collect_feature_predictions(module, datamodule.test_dataloader(), device)
        else:
            from src.models.full_pipeline import predict

            outputs = predict(module, datamodule.test_dataloader(), device)

        outputs["class_names"] = list(datamodule.class_names)
        return outputs

    # ------------------------------------------------------------- aggregation

    def _across_seeds(self, per_seed: Dict[str, Any]) -> Dict[str, Any]:
        """Mean, std and 95% interval for each metric across the seeds present.

        :param per_seed: Per-seed metric records.
        :return: Across-seed summary, empty when nothing was evaluated.
        """
        if not per_seed:
            return {}

        seeds = sorted(per_seed, key=int)
        summary: Dict[str, Any] = {
            metric: battery.summarise_seeds(
                [per_seed[s].get("overall", {}).get(metric) for s in seeds]
            )
            for metric in TABLE_METRICS
        }

        for metric in ("expected_calibration_error", "brier_score"):
            summary[metric] = battery.summarise_seeds(
                [per_seed[s].get("calibration", {}).get(metric) for s in seeds]
            )

        # Class-wise recall is a primary metric, so it gets the same treatment per class
        # rather than being collapsed into the macro average.
        classes = [entry["class_name"] for entry in per_seed[seeds[0]].get("per_class", [])]
        summary["per_class_recall"] = {
            name: battery.summarise_seeds(
                [_class_recall(per_seed[s], name) for s in seeds]
            )
            for name in classes
        }

        return summary

    def _flat_table(self, rows: Dict[str, Any], context: AblationContext) -> pd.DataFrame:
        """Assemble the deterministic per-seed table.

        :param rows: Evaluated row records.
        :param context: Resolved Step 6 / Step 14 selections.
        :return: One row per ``(row_id, seed)``, columns in :data:`TABLE_COLUMNS` order.
        """
        records: List[Dict[str, Any]] = []

        for row in ROWS:
            record = rows[row.row_id]
            for seed in sorted(record.get("per_seed", {}), key=int):
                metrics = record["per_seed"][seed]
                overall = metrics.get("overall", {})
                worst = _worst_class(metrics)
                records.append(
                    {
                        "row_id": row.row_id,
                        "label": row.label,
                        "seed": int(seed),
                        "recipe": record["recipe"],
                        "loss": record["loss"],
                        "model": row.model,
                        "feature_tag": row.feature_tag,
                        "checkpoint": metrics.get("checkpoint"),
                        "provenance": metrics.get("provenance"),
                        "n_samples": metrics.get("n_samples"),
                        **{name: overall.get(name) for name in TABLE_METRICS},
                        "expected_calibration_error": metrics.get("calibration", {}).get(
                            "expected_calibration_error"
                        ),
                        "brier_score": metrics.get("calibration", {}).get("brier_score"),
                        "min_class_recall": worst[1],
                        "worst_class": worst[0],
                    }
                )

        frame = pd.DataFrame(records, columns=list(TABLE_COLUMNS))
        return frame

    def _coverage(self, rows: Dict[str, Any]) -> Dict[str, Any]:
        """What was actually evaluated, so a partial matrix cannot be mistaken for a full one.

        :param rows: Evaluated row records.
        :return: Coverage record.
        """
        return {
            "expected_rows": [row.row_id for row in ROWS],
            "evaluated_rows": [rid for rid, rec in rows.items() if rec.get("evaluated")],
            "incomplete_rows": {
                rid: rec.get("missing_seeds", [])
                for rid, rec in rows.items()
                if rec.get("missing_seeds")
            },
            "complete": all(rec.get("evaluated") for rec in rows.values())
            and not any(rec.get("missing_seeds") for rec in rows.values()),
        }

    @staticmethod
    def _log(summary: Dict[str, Any]) -> None:
        """Print the matrix in the shape the write-up quotes it.

        :param summary: Computed summary.
        """
        log.info("=== Step 21: ablation matrix ===")
        log.info(f"  {'row':<4} {'macro-F1 (mean+-std)':<24} {'worst-class recall':<20} note")

        for row_id, record in summary["rows"].items():
            across = record.get("across_seeds", {})
            macro = across.get("macro_f1") or {}
            if macro.get("mean") is None:
                log.info(f"  {row_id:<4} {'not evaluated':<24}")
                continue

            recalls = across.get("per_class_recall", {})
            worst = min(
                ((name, stats.get("mean")) for name, stats in recalls.items()
                 if stats.get("mean") is not None),
                key=lambda pair: pair[1],
                default=(None, None),
            )
            worst_text = f"{worst[1]:.4f} ({worst[0]})" if worst[0] else "-"
            log.info(
                f"  {row_id:<4} {macro['mean']:.4f} +- {macro['std'] or 0.0:.4f} (n={macro['n']})"
                f"   {worst_text:<20} {record['recipe']}/{record['loss']}"
            )

        coverage = summary["coverage"]
        if not coverage["complete"]:
            log.warning(
                f"Matrix is INCOMPLETE: {coverage['incomplete_rows']}. "
                "Do not quote it as a full ablation."
            )

    def _notes(self, context: AblationContext) -> Dict[str, str]:
        """Standing caveats a reader needs before quoting any number here.

        :param context: Resolved Step 6 / Step 14 selections.
        :return: Note mapping.
        """
        return {
            "test_set": (
                "Every metric here was computed by this analysis from a checkpoint the "
                "training run selected on val/f1_macro. No test metric influenced any "
                "checkpoint, epoch or row choice, and the test/* columns written into "
                "each training run's metrics.csv were not read."
            ),
            "a6_vs_p": (
                f"A6 follows the specification and uses {context.diffusion_recipe}; row P "
                f"is the shipped model and uses {context.selected_recipe}, which is what "
                "Step 6 selected. They are different configurations and A6 is not the "
                "proposed model."
            ),
            "a8": (
                "A8's classification metrics are A7's by construction. Its contribution is "
                "qualitative - deletion/insertion and MC-dropout uncertainty from Step 19."
            ),
            "row_p_seeds": (
                "Row P is reported from Step 16, which evaluated one checkpoint by design. "
                "Its across-seed spread is therefore not comparable with a three-seed "
                "A-row's; treat the P comparison as single-seed."
            ),
            "imbalance_settings": (
                "Every A-row pins augment=false and use_weighted_sampler=false, matching "
                "the observed Step 8 'baseline' selection. If a full-profile Step 8 selects "
                "otherwise, the rows do NOT follow it - the conflict is reported rather "
                "than silently applied."
            ),
        }


# ------------------------------------------------------------------------ helpers


def _read_json(path: Optional[str], role: str) -> Dict[str, Any]:
    """:param path: File to read.

    :param role: What it is, for the error message.
    :return: Parsed JSON.
    :raises FileNotFoundError: If the path is unset or absent.
    """
    if not path:
        raise FileNotFoundError(f"Step 21 needs the {role} summary; none was configured.")
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Step 21 needs the {role} summary; {resolved} does not exist.")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _seed_from_path(path: Optional[str]) -> Optional[int]:
    """:param path: A checkpoint path or run directory.

    :return: The seed encoded as ``seed_<n>``, or ``None``.
    """
    if not path:
        return None
    for part in reversed(Path(path).parts):
        if part.startswith("seed_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def _class_recall(metrics: Dict[str, Any], class_name: str) -> Optional[float]:
    """:param metrics: One seed's metric record.

    :param class_name: Class to look up.
    :return: That class's recall, or ``None`` if absent.
    """
    for entry in metrics.get("per_class", []):
        if entry.get("class_name") == class_name:
            return entry.get("recall_sensitivity")
    return None


def _worst_class(metrics: Dict[str, Any]) -> tuple:
    """The lowest class-wise recall, which Step 21 names a primary selection metric.

    :param metrics: One seed's metric record.
    :return: ``(class_name, recall)``, or ``(None, None)``.
    """
    entries = [e for e in metrics.get("per_class", []) if e.get("recall_sensitivity") is not None]
    if not entries:
        return (None, None)
    worst = min(entries, key=lambda e: e["recall_sensitivity"])
    return (worst["class_name"], float(worst["recall_sensitivity"]))
