"""Step 6 confirmation: turn the real-backbone sweep into an authoritative decision.

`analysis=step06_preprocessing` ranks preprocessing candidates with a `SmallCNN` proxy at
128px on a class-balanced subset. That is affordable and it is *not* the model the study
reports, which is why its own summary ends with:

    "Ranking comes from a reduced-scale proxy ... Confirm the top candidates with the real
     backbone on the full validation split before committing."

`configs/experiment/step06_confirm.yaml` is the confirmation, and it is a **training
experiment**: a Hydra multirun over `data.recipe=`. It produces one run directory per
candidate and nothing else - no summary, no winner, no `selected_recipe`. Whoever ran it
was expected to read the numbers and remember the answer.

That gap is what this module closes. It reads the completed confirmation runs, compares
them on validation macro-F1, and writes `step06_confirm_summary.json` carrying an explicit
`selected_recipe` and the provenance behind it. Downstream stages consume that artefact
rather than a person's recollection.

Three constraints shape it.

**No winner exists before the runs do.** This module invents nothing. Point it at an empty
directory and it fails; point it at one recipe and it refuses to call that a comparison.
The `selected_recipe` field exists only once real runs support it.

**Validation only.** The confirmation decides preprocessing, so it may read `val/*` and
nothing else. Every run's `metrics.csv` also carries `test/*` columns - `test: True` in
`configs/train.yaml` is never overridden - and reading those would let the internal test
set influence a design decision that precedes Step 16. The metric key is validated to start
with `val/`, and a test enforces it.

**Only genuine confirmation runs count.** Each run's `.hydra/overrides.yaml` must name
`experiment=step06_confirm`. Pointing this at the Step 9 baseline sweep, which also varies
by recipe, would otherwise produce a confident answer to a different question.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.analysis.base import Analysis
from src.utils import RankedLogger
from src.utils.statistics import summarise_across_seeds

log = RankedLogger(__name__, rank_zero_only=True)

#: The experiment a run must declare to count as confirmation evidence.
CONFIRMATION_EXPERIMENT = "step06_confirm"

#: Where the authoritative decision is written, and what downstream stages look for.
SUMMARY_FILENAME = "step06_confirm_summary.json"

#: Selection metric. `val/f1_macro_best` is the running best the ModelCheckpoint selected
#: on, so it is the score belonging to the checkpoint the run would actually ship.
DEFAULT_METRIC = "val/f1_macro_best"

#: Fallback when a run predates the `_best` column.
FALLBACK_METRIC = "val/f1_macro"

#: Status written into the summary. Only `confirmed` licenses downstream use.
STATUS_CONFIRMED = "confirmed"

#: `data.recipe=null` is the conventional reference - the raw tree plus the Step 5
#: intensity treatment - and is recorded under this name rather than as a null.
CONVENTIONAL = "conventional"


class ConfirmationIncomplete(RuntimeError):
    """Raised when no authoritative Step 6 decision can be produced or read.

    Deliberately distinct from ``FileNotFoundError``: a caller that catches this knows the
    confirmation has not happened, rather than that some path was mistyped.
    """


class PreprocessingConfirmation(Analysis):
    """Read the Step 6 confirmation sweep and write the authoritative decision.

    :param name: Analysis identifier; also the summary's filename stem.
    :param run_root: Directory holding the confirmation runs. A Hydra multirun root
        (``logs/train/multiruns/<timestamp>/``) or any directory containing run
        directories, searched one level deep and then recursively.
    :param run_dirs: Explicit run directories, used instead of searching ``run_root``.
    :param metric: Validation metric deciding the winner. Must start with ``val/``.
    :param min_candidates: Fewest distinct recipes that count as a comparison.
    :param require_experiment: Only accept runs whose overrides name this experiment.
        Set to ``None`` to accept any run, which should be a deliberate choice.
    """

    def __init__(
        self,
        name: str = "step06_confirm",
        run_root: Optional[str] = None,
        run_dirs: Optional[Sequence[str]] = None,
        metric: str = DEFAULT_METRIC,
        min_candidates: int = 2,
        require_experiment: Optional[str] = CONFIRMATION_EXPERIMENT,
    ) -> None:
        super().__init__(name=name)
        self.run_root = run_root
        self.run_dirs = list(run_dirs) if run_dirs else None
        self.metric = metric
        self.min_candidates = min_candidates
        self.require_experiment = require_experiment

    # -------------------------------------------------------------------- compute

    def compute(self, datamodule: Any = None) -> Dict[str, Any]:
        """Assemble the confirmation decision.

        :param datamodule: Unused; this stage reads artefacts, not data.
        :return: The confirmation summary, carrying ``selected_recipe``.
        :raises ValueError: If the metric is not a validation metric.
        :raises ConfirmationIncomplete: If the runs cannot support a decision.
        """
        if not self.metric.startswith("val/"):
            raise ValueError(
                f"Step 6 confirmation selects on validation only; {self.metric!r} is not a "
                "val/ metric. The internal test set must not influence a decision that "
                "precedes Step 16."
            )

        runs = self._collect()
        recipes = self._aggregate(runs)
        winner = max(recipes.values(), key=lambda r: r["mean"])

        summary: Dict[str, Any] = {
            "selected_recipe": winner["recipe"],
            "selected_metric_value": winner["mean"],
            "confirmation_status": STATUS_CONFIRMED,
            "selection_metric": self.metric,
            "selection_scope": "validation split only; the test set was not read",
            "model": self._single_value(runs, "model") or "see per-run provenance",
            "n_candidates": len(recipes),
            "n_runs": len(runs),
            "seeds": sorted({r["seed"] for r in runs if r["seed"] is not None}),
            "candidates": [
                recipes[name] for name in sorted(recipes, key=lambda k: -recipes[k]["mean"])
            ],
            "runs": [
                {
                    "run_dir": r["run_dir"],
                    "recipe": r["recipe"],
                    "seed": r["seed"],
                    "metric_value": r["value"],
                    "metric": self.metric if r["metric_used"] is None else r["metric_used"],
                }
                for r in runs
            ],
            "provenance": {
                "source_stage": CONFIRMATION_EXPERIMENT,
                "run_root": str(self.run_root) if self.run_root else None,
                "note": (
                    "Read from the completed confirmation runs' validation metrics. No "
                    "test metric was read, no model was retrained, and no winner was "
                    "assumed before these runs existed."
                ),
            },
            "supersedes": (
                "step06_preprocessing_summary.json, whose ranking comes from a "
                "reduced-scale proxy. This confirmation is the authoritative decision."
            ),
        }

        self.save_table(pd.DataFrame(summary["candidates"]), "step06_confirm_candidates.csv")
        self._log(summary)
        return summary

    # -------------------------------------------------------------------- inputs

    def _candidate_dirs(self) -> List[Path]:
        """Locate the run directories to read.

        :return: Directories that look like Hydra runs.
        :raises ConfirmationIncomplete: If none can be found.
        """
        if self.run_dirs:
            return [Path(d) for d in self.run_dirs]

        if not self.run_root:
            raise ConfirmationIncomplete(
                "Step 6 confirmation needs the runs it should read: set "
                "analysis.run_root to the sweep directory (for example "
                "logs/train/multiruns/<timestamp>) or analysis.run_dirs explicitly."
            )

        root = Path(self.run_root)
        if not root.is_dir():
            raise ConfirmationIncomplete(
                f"Step 6 confirmation run root {root} does not exist. Run the sweep first:\n"
                "  python src/train.py -m experiment=step06_confirm "
                "data.recipe=<candidates> trainer=gpu"
            )

        found = sorted({p.parent.parent for p in root.glob("*/.hydra/overrides.yaml")})
        if not found:
            found = sorted({p.parent.parent for p in root.rglob("*/.hydra/overrides.yaml")})
        if not found:
            raise ConfirmationIncomplete(
                f"No Hydra run directories under {root}. Expected one subdirectory per "
                "candidate, each with .hydra/overrides.yaml and csv/version_0/metrics.csv."
            )
        return found

    def _collect(self) -> List[Dict[str, Any]]:
        """Read every usable confirmation run.

        :return: One record per run.
        :raises ConfirmationIncomplete: If too few distinct recipes are usable.
        """
        records: List[Dict[str, Any]] = []
        skipped: List[str] = []

        for directory in self._candidate_dirs():
            overrides = _read_overrides(directory)
            if overrides is None:
                skipped.append(f"{directory}: no .hydra/overrides.yaml")
                continue

            experiment = _override_value(overrides, "experiment")
            if self.require_experiment and experiment != self.require_experiment:
                # A Step 9 sweep also varies by recipe; reading it would answer a
                # different question with the same confidence.
                skipped.append(f"{directory}: experiment={experiment!r}, not confirmation")
                continue

            value, metric_used = _read_metric(directory, self.metric)
            if value is None:
                skipped.append(f"{directory}: no {self.metric} in metrics.csv")
                continue

            records.append(
                {
                    "run_dir": str(directory),
                    "recipe": _recipe_of(overrides),
                    "seed": _seed_of(overrides),
                    "model": _override_value(overrides, "model"),
                    "value": value,
                    "metric_used": metric_used if metric_used != self.metric else None,
                }
            )

        for message in skipped:
            log.warning(f"Step 6 confirmation skipped {message}")

        distinct = {r["recipe"] for r in records}
        if len(distinct) < self.min_candidates:
            raise ConfirmationIncomplete(
                f"Step 6 confirmation needs at least {self.min_candidates} distinct recipes "
                f"to compare; found {len(distinct)} ({sorted(distinct)}) across "
                f"{len(records)} usable run(s)."
                + (f" Skipped: {'; '.join(skipped)}" if skipped else "")
                + "\nRun the sweep before confirming:\n"
                "  python src/train.py -m experiment=step06_confirm "
                "data.recipe=<candidates> trainer=gpu"
            )
        return records

    def _aggregate(self, runs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Group runs by recipe and summarise across seeds.

        :param runs: Per-run records.
        :return: One record per recipe.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in runs:
            grouped.setdefault(record["recipe"], []).append(record)

        aggregated: Dict[str, Dict[str, Any]] = {}
        for recipe, entries in grouped.items():
            values = [e["value"] for e in entries]
            spread = summarise_across_seeds(values)
            aggregated[recipe] = {
                "recipe": recipe,
                "mean": spread["mean"],
                "std": spread["std"],
                "n_runs": spread["n"],
                "min": spread["min"],
                "max": spread["max"],
                "seeds": sorted({e["seed"] for e in entries if e["seed"] is not None}),
            }
        return aggregated

    @staticmethod
    def _single_value(runs: Sequence[Dict[str, Any]], key: str) -> Optional[str]:
        """:param runs: Per-run records.

        :param key: Field to read.
        :return: The shared value, or ``None`` when the runs disagree.
        """
        values = {r.get(key) for r in runs if r.get(key)}
        return values.pop() if len(values) == 1 else None

    @staticmethod
    def _log(summary: Dict[str, Any]) -> None:
        """:param summary: The computed summary."""
        log.info("=== Step 6 confirmation (real backbone, validation only) ===")
        log.info(f"  metric: {summary['selection_metric']}  model: {summary['model']}")
        for candidate in summary["candidates"]:
            std = candidate["std"] or 0.0
            log.info(
                f"  {candidate['recipe']:<24} {candidate['mean']:.4f} +- {std:.4f} "
                f"(n={candidate['n_runs']})"
            )
        log.info(f"  CONFIRMED: {summary['selected_recipe']}")
        log.info(
            "  This supersedes the proxy ranking. Downstream stages should now consume "
            f"{SUMMARY_FILENAME}."
        )


# ------------------------------------------------------------------- the reader


def load_confirmed_recipe(summary_path: Optional[str]) -> str:
    """Read the authoritative Step 6 decision, or refuse.

    The single place downstream stages learn which preprocessing was confirmed. It never
    falls back to the proxy ranking: an unconfirmed recipe silently becoming the basis of a
    fifteen-run experiment is exactly the failure this artefact exists to prevent.

    :param summary_path: Path to ``step06_confirm_summary.json``.
    :return: The confirmed recipe name.
    :raises ConfirmationIncomplete: If the summary is unset, missing, malformed, not
        marked confirmed, or carries no recipe.
    """
    if not summary_path:
        raise ConfirmationIncomplete(
            "Step 6 confirmation has not produced an authoritative selected_recipe; "
            "no confirmation summary was configured. Run the confirmation sweep and then "
            f"analysis=step06_confirm to produce {SUMMARY_FILENAME}."
        )

    path = Path(summary_path)
    if not path.is_file():
        raise ConfirmationIncomplete(
            "Step 6 confirmation has not produced an authoritative selected_recipe; "
            f"{path} does not exist. Run the confirmation sweep on the real backbone, then "
            "analysis=step06_confirm. The proxy ranking is NOT a substitute."
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfirmationIncomplete(
            f"Step 6 confirmation summary at {path} is not valid JSON: {error}"
        ) from error

    status = payload.get("confirmation_status")
    if status != STATUS_CONFIRMED:
        raise ConfirmationIncomplete(
            f"Step 6 confirmation at {path} reports status {status!r}, not "
            f"{STATUS_CONFIRMED!r}; it cannot be used as the authoritative decision."
        )

    recipe = payload.get("selected_recipe")
    if not recipe:
        raise ConfirmationIncomplete(
            f"Step 6 confirmation at {path} carries no selected_recipe."
        )
    return str(recipe)


# ------------------------------------------------------------------- internals


def _read_overrides(run_dir: Path) -> Optional[List[str]]:
    """:param run_dir: A Hydra run directory.

    :return: Its override list, or ``None`` if absent or unreadable.
    """
    path = run_dir / ".hydra" / "overrides.yaml"
    if not path.is_file():
        return None
    try:
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - unreadable override file
        return None
    return [str(entry) for entry in loaded] if isinstance(loaded, list) else None


def _override_value(overrides: Sequence[str], key: str) -> Optional[str]:
    """:param overrides: A run's Hydra overrides.

    :param key: Override key.
    :return: Its value, or ``None``.
    """
    prefix = f"{key}="
    for entry in overrides:
        if entry.startswith(prefix):
            return entry[len(prefix) :]
    return None


def _recipe_of(overrides: Sequence[str]) -> str:
    """The recipe a run trained on.

    ``data.recipe=null`` is the conventional reference - the raw tree with the Step 5
    intensity treatment - and is named rather than left null so it appears in the table
    like every other candidate.

    :param overrides: A run's Hydra overrides.
    :return: The recipe name.
    """
    value = _override_value(overrides, "data.recipe")
    if value is None or value in ("null", "none", "None", ""):
        return CONVENTIONAL
    return value


def _seed_of(overrides: Sequence[str]) -> Optional[int]:
    """:param overrides: A run's Hydra overrides.

    :return: The seed, or ``None`` if not overridden.
    """
    value = _override_value(overrides, "seed")
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _read_metric(run_dir: Path, metric: str) -> Tuple[Optional[float], Optional[str]]:
    """Read one run's best validation score.

    Only the requested ``val/`` column and its fallback are touched. The same file carries
    ``test/*`` columns, and this function must never return one.

    :param run_dir: A Hydra run directory.
    :param metric: Validation metric to read.
    :return: ``(value, metric_actually_used)``, or ``(None, None)``.
    """
    path = run_dir / "csv" / "version_0" / "metrics.csv"
    if not path.is_file():
        return (None, None)

    try:
        frame = pd.read_csv(path)
    except Exception:  # pragma: no cover - unreadable metrics file
        return (None, None)

    for column in (metric, FALLBACK_METRIC):
        if not column.startswith("val/"):
            continue
        if column in frame.columns:
            series = frame[column].dropna()
            if not series.empty:
                return (float(series.max()), column)
    return (None, None)
