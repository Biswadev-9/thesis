"""Shared base for the study's analysis stages.

Steps 4, 6, 8 and 16 through 23 are analyses rather than training runs: they read
artefacts produced upstream, compute tables and figures, and write them into the Hydra
run directory. Giving them one interface lets a single entry point (``src/analyze.py``)
drive all of them through ``configs/analysis/``.

Every analysis writes into ``${paths.output_dir}``. The reference notebook wrote to
hard-coded Google Drive paths, so its outputs were not tied to the run that produced
them and could be silently overwritten by a later re-run.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils import RankedLogger
from src.utils.atomic import atomic_write, atomic_write_json

log = RankedLogger(__name__, rank_zero_only=True)


class Analysis:
    """Base class for an analysis stage.

    :param name: Short identifier used for output filenames and log lines.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._output_dir: Optional[Path] = None

    @property
    def output_dir(self) -> Path:
        """:return: Directory this analysis writes into.
        :raises RuntimeError: If accessed before :meth:`run` has bound it.
        """
        if self._output_dir is None:
            raise RuntimeError(f"{self.name}: output_dir is only available inside run()")
        return self._output_dir

    def run(self, datamodule: Any = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Execute the analysis.

        :param datamodule: Instantiated datamodule, when the analysis needs data.
        :param output_dir: Directory for artefacts; defaults to the working directory.
        :return: Summary metrics, also written as ``<name>_summary.json``.
        """
        self._output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        self._output_dir.mkdir(parents=True, exist_ok=True)

        summary = self.compute(datamodule)

        self.save_json(summary, f"{self.name}_summary.json")
        return summary

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Do the analysis work.

        :param datamodule: Instantiated datamodule, or ``None``.
        :return: Summary metrics.
        :raises NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute()")

    # ------------------------------------------------------------- artefacts

    def save_table(
        self, df: pd.DataFrame, filename: str, index: bool = False, quiet: bool = False
    ) -> Path:
        """Write a table into the run directory.

        :param df: Table to write.
        :param filename: File name, including the ``.csv`` suffix.
        :param index: Whether to write the index column.
        :param quiet: Suppress the log line. Used when a long sweep rewrites the same
            table after every candidate so an interrupted run keeps its completed work -
            logging each rewrite would bury the actual progress messages.
        :return: The path written.
        """
        path = self.output_dir / filename
        # Atomic: a sweep rewrites this file after every candidate, so a kill during a
        # rewrite would otherwise destroy the completed rows it exists to preserve.
        atomic_write(path, lambda temporary: df.to_csv(temporary, index=index))
        if not quiet:
            log.info(f"[{self.name}] wrote {path.name} ({len(df)} rows)")
        return path

    def save_json(self, payload: Dict[str, Any], filename: str) -> Path:
        """Write a JSON artefact into the run directory.

        :param payload: JSON-serialisable mapping.
        :param filename: File name, including the ``.json`` suffix.
        :return: The path written.
        """
        path = self.output_dir / filename
        # Atomic: downstream stages decide what to train from these summaries, and a
        # truncated one is worse than an absent one - `read_summary` would raise on the
        # absent file and silently misread nothing at all.
        atomic_write_json(path, payload, default=str)
        log.info(f"[{self.name}] wrote {path.name}")
        return path

    def figure_path(self, filename: str) -> Path:
        """:param filename: File name, including the image suffix.
        :return: Path inside the run directory for a figure.
        """
        return self.output_dir / filename
