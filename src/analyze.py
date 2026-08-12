from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# See src/train.py for what setup_root does. Same rationale here: analyses import from
# `src`, so the project root must be on the path before those imports run.
# ------------------------------------------------------------------------------------ #

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def analyze(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs one analysis stage.

    Analyses are the study's non-training steps - the Step 4 audit, the Step 6 and 8
    selection studies, and the Steps 16-23 evaluation and reporting suite. They read
    artefacts produced upstream and write tables and figures into the Hydra run
    directory, so each set of outputs stays tied to the config that produced it.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with the analysis summary and a dict with all instantiated objects.
    """
    datamodule: Optional[Any] = None
    if cfg.get("data"):
        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating analysis <{cfg.analysis._target_}>")
    analysis = hydra.utils.instantiate(cfg.analysis)

    output_dir = Path(cfg.paths.output_dir)
    log.info(f"Running analysis; artefacts will be written to {output_dir}")
    summary = analysis.run(datamodule=datamodule, output_dir=output_dir)

    object_dict = {"cfg": cfg, "datamodule": datamodule, "analysis": analysis}
    return summary, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="analyze.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for analysis.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    analyze(cfg)


if __name__ == "__main__":
    main()
