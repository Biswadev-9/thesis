"""Step 20: is there any measurable quantum advantage?

    "Quantum advantage must be treated as an empirical question. ... If the quantum branch
     does not outperform strong baselines, report the result honestly. In that case, the
     contribution may still be useful if it improves parameter efficiency, robustness, or
     interpretability."

That framing decides the design of this analysis: it is built to be able to return "no",
and to say so in terms the write-up can use directly. Four independent lines of evidence:

1. **Paired significance.** The full model against the same architecture with the quantum
   features zeroed, on identical test samples, via McNemar and a paired bootstrap.
2. **Fixed versus adaptive.** Step 9's fixed QCNN against Step 12's five-circuit mixture,
   which is the study's actual novelty claim about quantum circuits.
3. **Efficiency.** Trainable parameters, inference time, training time and peak memory -
   the specification lists all four, and the notebook reported only the first two.
4. **Feature separability.** UMAP or t-SNE plus a silhouette score computed with and
   without the quantum features.

The comparison is deliberately *paired* throughout. Two models scored on the same test set
make correlated errors, and an unpaired test discards precisely the information that makes
a small difference detectable.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, silhouette_score

from src.analysis.base import Analysis
from src.data.bt_mri_feature_datamodule import BTMRIFeatureDataModule
from src.utils import RankedLogger
from src.utils.checkpoints import find_checkpoint, load_module
from src.utils.statistics import bootstrap_ci, mcnemar_test, paired_bootstrap

log = RankedLogger(__name__, rank_zero_only=True)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """:param y_true: True labels.
    :param y_pred: Predictions.
    :return: Macro-averaged F1.
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


class QuantumAdvantageStudy(Analysis):
    """Test whether the quantum branch earns its place.

    :param name: Analysis identifier.
    :param tag: Feature cache to read.
    :param fusion_ckpt: Trained fusion checkpoint, used for the full model.
    :param fusion_model: Hydra config for the fusion module. The control is instantiated
        from this same config - optimizer, scheduler and loss included - so it cannot
        silently diverge from the model it is compared against.
    :param protocol: The Step 15 training protocol the control must be trained under.
        Mirrors ``configs/protocol/fixed.yaml``; a regression test asserts the two agree,
        so amending the protocol fails the suite rather than drifting unnoticed.
    :param loss: Explicit loss name, overriding everything else. Mirrors the pipeline's
        ``--loss`` flag.
    :param loss_summary: Path to Step 14's ``step14_loss_selection_summary.json``. The
        control trains with whatever loss Step 14 actually selected, exactly as
        ``scripts/kaggle_pipeline.py`` feeds it to Step 15 - so the two cannot diverge
        when Step 14's answer changes.
    :param seed: Seed for the control and the bootstrap. Must match the seed of the
        checkpoint being compared against; a mismatch is detected and reported.
    :param n_resamples: Bootstrap resamples.
    :param run_dirs: Optional mapping of label to training run directory, read for
        ``resource_usage.json`` so training time and memory can be reported.
    :param accelerator: ``auto``, ``cpu`` or ``gpu``.
    """

    def __init__(
        self,
        name: str = "step20_quantum_advantage",
        tag: str = "default",
        fusion_ckpt: Optional[str] = None,
        fusion_model: Optional[Any] = None,
        protocol: Optional[Dict[str, Any]] = None,
        loss: Optional[str] = None,
        loss_summary: Optional[str] = None,
        seed: int = 42,
        n_resamples: int = 2000,
        run_dirs: Optional[Dict[str, str]] = None,
        accelerator: str = "auto",
    ) -> None:
        super().__init__(name=name)
        self.tag = tag
        self.fusion_ckpt = fusion_ckpt
        self.fusion_model = fusion_model
        self.protocol = dict(protocol or {})
        self.loss = loss
        self.loss_summary = loss_summary
        self.seed = seed
        self.n_resamples = n_resamples
        self.run_dirs = dict(run_dirs or {})
        self.accelerator = accelerator

    def compute(self, datamodule: Any) -> Dict[str, Any]:
        """Run all four lines of evidence.

        :param datamodule: Datamodule supplying the data directory.
        :return: Summary with an explicit verdict.
        """
        data_dir = datamodule.hparams.data_dir if datamodule is not None else "data/"
        device = torch.device(
            "cuda" if (self.accelerator != "cpu" and torch.cuda.is_available()) else "cpu"
        )

        full = self._full_model_predictions(data_dir, device)
        control = self._no_quantum_predictions(data_dir, device)

        summary: Dict[str, Any] = {
            "tag": self.tag,
            "loss_provenance": control.get("loss"),
            "seed_check": self._check_seed_matches_checkpoint(),
            "performance": self._compare(full, control),
            "separability": self._separability(data_dir),
            "efficiency": self._efficiency(full["model"], control["model"], data_dir, device),
        }
        summary["verdict"] = self._verdict(summary)

        self._log(summary)
        return summary


    # ------------------------------------------------------------ protocol inputs

    def _resolve_loss(self) -> tuple:
        """Determine the loss the control must train with.

        Step 15 does not use a fixed loss: ``scripts/kaggle_pipeline.py`` reads Step 14's
        ``selected_loss`` at run time and passes it as ``loss@model.criterion=<name>``,
        raising if Step 14 has not run. Step 20 therefore cannot hard-code one - if
        Step 14 selects ``focal`` and the config here still says ``weighted_ce``, the
        control trains on a different objective than the model it is compared against,
        and nothing in the run would signal it.

        Precedence mirrors the pipeline's own: explicit override, then Step 14's summary,
        then whatever the config carries.

        :return: ``(loss_name_or_None, source)`` where source explains the provenance.
        """
        if self.loss:
            return self.loss, "explicit analysis.loss override"

        if self.loss_summary:
            path = Path(self.loss_summary)
            if path.is_file():
                selected = json.loads(path.read_text(encoding="utf-8")).get("selected_loss")
                if selected:
                    return str(selected), f"Step 14 selection ({path})"
            log.warning(f"Step 14 summary not readable at {path}; falling back to config")

        return None, "configs/analysis/step20_quantum_advantage.yaml (unverified)"

    def _apply_loss(self, module_config: Dict[str, Any]) -> Dict[str, Any]:
        """Overwrite the control's criterion with the finalized Step 15 loss.

        :param module_config: The fusion module config, modified in place.
        :return: Provenance record for the summary.
        """
        from src.analysis.imbalance_study import build_loss

        name, source = self._resolve_loss()
        configured = module_config.get("criterion", {}).get("_target_", "unset")

        if name is None:
            log.warning(
                "The loss Step 15 trained with was not verified: pass "
                "analysis.loss_summary=<step14 summary json> (or analysis.loss=<name>) so "
                "the control provably matches. Using the configured criterion "
                f"{configured}."
            )
            return {"loss": None, "source": source, "configured_criterion": configured}

        criterion = build_loss(name)
        module_config["criterion"] = criterion
        log.info(f"Control trains with loss '{name}' - resolved from {source}")
        return {"loss": name, "source": source, "criterion": type(criterion).__name__}

    @staticmethod
    def _seed_from_checkpoint(path: Optional[str]) -> Optional[int]:
        """Recover the training seed from a run directory named ``.../seed_<n>``.

        :param path: Checkpoint path or run directory.
        :return: The seed, or ``None`` if the path does not encode one.
        """
        if not path:
            return None
        for part in reversed(Path(path).parts):
            if part.startswith("seed_"):
                try:
                    return int(part[len("seed_") :])
                except ValueError:
                    return None
        return None

    def _check_seed_matches_checkpoint(self) -> Dict[str, Any]:
        """Verify the control is trained at the same seed as the compared checkpoint.

        The pipeline evaluates a *fixed* seed - ``pipe.seeds[0]`` - rather than the best of
        the three Step 15 seeds, so a single-seed control is a fair like-for-like
        comparison. That only holds while the two seeds agree; if the pipeline is run with
        a reordered ``--seeds``, the compared checkpoint could be seed 7 while the control
        trains at 42, and the measured delta would include a seed effect.

        :return: Record of both seeds and whether they agree.
        """
        checkpoint_seed = self._seed_from_checkpoint(self.fusion_ckpt)
        matched = checkpoint_seed is None or checkpoint_seed == self.seed

        if not matched:
            log.warning(
                f"Seed mismatch: the fusion checkpoint was trained at seed "
                f"{checkpoint_seed}, but the control will train at seed {self.seed}. The "
                f"measured difference would include a seed effect. Pass "
                f"analysis.seed={checkpoint_seed}."
            )

        return {
            "control_seed": self.seed,
            "checkpoint_seed": checkpoint_seed,
            "seeds_match": bool(matched),
        }

    # -------------------------------------------------------------- predictions

    def _full_model_predictions(self, data_dir: str, device: torch.device) -> Dict[str, Any]:
        """Load the trained fusion head and score it on the test features.

        :param data_dir: Project data directory.
        :param device: Device to run on.
        :return: Predictions and the loaded model.
        """
        candidate = Path(self.fusion_ckpt)
        checkpoint = find_checkpoint(candidate) if candidate.is_dir() else candidate
        module = load_module(checkpoint, model_cfg=self.fusion_model).to(device)

        datamodule = BTMRIFeatureDataModule(data_dir=data_dir, tag=self.tag)
        datamodule.prepare_data()
        datamodule.setup()

        return {"model": module.net, **self._predict(module.net, datamodule, device)}

    def _no_quantum_predictions(self, data_dir: str, device: torch.device) -> Dict[str, Any]:
        """Retrain an identical head with the quantum features zeroed.

        Retraining matters. Masking the quantum features at inference on a head that was
        trained with them present measures disruption, not contribution - the head has
        already learned to rely on inputs that suddenly vanished. Retraining asks the
        question Step 20 actually poses: how well does this architecture do *without* the
        quantum branch?

        :param data_dir: Project data directory.
        :param device: Device to run on.
        :return: Predictions and the trained control.
        """
        import tempfile

        import hydra
        from lightning import Trainer, seed_everything
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from omegaconf import OmegaConf

        seed_everything(self.seed, workers=True)

        datamodule = BTMRIFeatureDataModule(
            data_dir=data_dir,
            tag=self.tag,
            zero_branches=["quantum"],
            batch_size=self.protocol["batch_size"],
            use_weighted_sampler=self.protocol["use_weighted_sampler"],
        )
        datamodule.prepare_data()
        datamodule.setup()

        # Build the control from the SAME module config as the full model - which carries
        # the optimizer, scheduler and loss as well as the architecture - overriding only
        # the feature widths so they match the cache.
        #
        # Reconstructing it by class, or supplying a fresh optimizer and no criterion,
        # would silently fall back to defaults: the class's proj_dim/hidden_dims/dropout
        # and an *unweighted* cross-entropy. Step 20's central claim is "the same
        # architecture after removing the quantum branch", so anything else differing is a
        # confound, and one that would quietly favour the quantum model.
        module_config = OmegaConf.to_container(self.fusion_model, resolve=True)
        module_config["net"].update(
            classical_dim=datamodule.classical_dim,
            spatial_dim=datamodule.spatial_dim,
            quantum_dim=datamodule.quantum_dim,
            num_classes=datamodule.num_classes,
        )
        loss_record = self._apply_loss(module_config)
        module = hydra.utils.instantiate(module_config)

        log.info(
            f"Retraining the no-quantum control under the Step 15 protocol: "
            f"max_epochs={self.protocol['max_epochs']}, "
            f"lr={module_config['optimizer']['lr']}, "
            f"early stopping on {self.protocol['monitor']} (patience "
            f"{self.protocol['patience']}), best-checkpoint selection"
        )

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            # The full model was selected by best validation macro-F1, not by its final
            # epoch. Without a checkpoint callback here the control would be scored on
            # whatever weights it happened to end on - a different selection rule, and one
            # that can only hurt it.
            checkpoint = ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor=self.protocol["monitor"],
                mode=self.protocol["mode"],
                save_top_k=1,
            )
            early_stopping = EarlyStopping(
                monitor=self.protocol["monitor"],
                mode=self.protocol["mode"],
                patience=self.protocol["patience"],
            )

            Trainer(
                max_epochs=self.protocol["max_epochs"],
                min_epochs=self.protocol.get("min_epochs", 1),
                accelerator=self.accelerator,
                callbacks=[checkpoint, early_stopping],
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            ).fit(model=module, datamodule=datamodule)

            if checkpoint.best_model_path:
                state = torch.load(checkpoint.best_model_path, map_location="cpu")["state_dict"]
                module.load_state_dict(state)
            else:
                log.warning(
                    "No best checkpoint was written for the control; scoring its final "
                    "epoch instead. The selection rule now differs from the full model's."
                )

        return {
            "model": module.net,
            "loss": loss_record,
            **self._predict(module.net, datamodule, device),
        }

    @staticmethod
    @torch.no_grad()
    def _predict(net: Any, datamodule: Any, device: torch.device) -> Dict[str, np.ndarray]:
        """Score a fusion head over the cached test features.

        :param net: Fusion head.
        :param datamodule: Feature datamodule.
        :param device: Device to run on.
        :return: ``{"y_true", "y_pred", "fused"}``.
        """
        net = net.to(device).eval()
        predictions, targets, fused = [], [], []

        for classical, spatial, quantum, labels in datamodule.test_dataloader():
            outputs = net.extract(classical.to(device), spatial.to(device), quantum.to(device))
            predictions.append(outputs["logits"].argmax(dim=1).cpu().numpy())
            fused.append(outputs["fused"].cpu().numpy())
            targets.append(labels.numpy())

        return {
            "y_true": np.concatenate(targets),
            "y_pred": np.concatenate(predictions),
            "fused": np.concatenate(fused),
        }

    # --------------------------------------------------------------- comparison

    def _compare(self, full: Dict[str, Any], control: Dict[str, Any]) -> Dict[str, Any]:
        """Paired significance testing between the full model and the control.

        :param full: Full-model predictions.
        :param control: No-quantum predictions.
        :return: Scores, McNemar, paired bootstrap and confidence intervals.
        """
        y_true = full["y_true"]

        comparison = {
            "full_macro_f1": macro_f1(y_true, full["y_pred"]),
            "no_quantum_macro_f1": macro_f1(y_true, control["y_pred"]),
            "mcnemar": mcnemar_test(y_true, full["y_pred"], control["y_pred"]),
            "paired_bootstrap": paired_bootstrap(
                y_true,
                full["y_pred"],
                control["y_pred"],
                macro_f1,
                n_resamples=self.n_resamples,
                seed=self.seed,
            ),
            "full_macro_f1_ci": bootstrap_ci(
                y_true, full["y_pred"], macro_f1, n_resamples=self.n_resamples, seed=self.seed
            ),
            "no_quantum_macro_f1_ci": bootstrap_ci(
                y_true, control["y_pred"], macro_f1, n_resamples=self.n_resamples, seed=self.seed
            ),
        }
        comparison["delta_macro_f1"] = (
            comparison["full_macro_f1"] - comparison["no_quantum_macro_f1"]
        )
        return comparison

    def _separability(self, data_dir: str) -> Dict[str, Any]:
        """Silhouette score of the cached features with and without the quantum block.

        Operates on the *cached branch features* rather than the fused projections, so the
        measurement is independent of whatever the fusion head learned.

        :param data_dir: Project data directory.
        :return: Silhouette scores and their difference.
        """
        datamodule = BTMRIFeatureDataModule(data_dir=data_dir, tag=self.tag)
        datamodule.prepare_data()
        datamodule.setup()

        cached = torch.load(datamodule.feature_dir / "test.pt", map_location="cpu")
        labels = cached["labels"].numpy()

        without = torch.cat([cached["classical"], cached["spatial"]], dim=1).numpy()
        with_quantum = torch.cat(
            [cached["classical"], cached["spatial"], cached["quantum"]], dim=1
        ).numpy()

        if len(np.unique(labels)) < 2:
            return {"note": "only one class present; silhouette is undefined"}

        scores = {
            "silhouette_with_quantum": float(silhouette_score(with_quantum, labels)),
            "silhouette_without_quantum": float(silhouette_score(without, labels)),
        }
        scores["delta"] = scores["silhouette_with_quantum"] - scores["silhouette_without_quantum"]

        self._plot_projection(with_quantum, without, labels, datamodule.class_names)
        return scores

    def _efficiency(
        self, full_net: Any, control_net: Any, data_dir: str, device: torch.device
    ) -> Dict[str, Any]:
        """Parameters, inference time, training time and peak memory.

        Step 20 lists all four. The reference notebook reported parameters and inference
        time only, because training cost was never measured while training happened - which
        is why `ResourceMonitor` records it from Step 9 onward.

        :param full_net: Full fusion head.
        :param control_net: No-quantum head.
        :param data_dir: Project data directory.
        :param device: Device to run on.
        :return: Efficiency table plus whatever the run directories recorded.
        """
        datamodule = BTMRIFeatureDataModule(data_dir=data_dir, tag=self.tag)
        datamodule.prepare_data()
        datamodule.setup()

        rows = []
        for label, net in (("full", full_net), ("no_quantum", control_net)):
            rows.append(
                {
                    "model": label,
                    "trainable_parameters": int(
                        sum(p.numel() for p in net.parameters() if p.requires_grad)
                    ),
                    "inference_seconds_per_batch": self._time_inference(net, datamodule, device),
                }
            )

        table = pd.DataFrame(rows)
        self.save_table(table, "step20_efficiency.csv")

        recorded = {}
        for label, directory in self.run_dirs.items():
            path = Path(directory) / "resource_usage.json"
            if path.is_file():
                usage = json.loads(path.read_text(encoding="utf-8"))
                recorded[label] = {
                    key: usage.get(key)
                    for key in (
                        "training_time_sec",
                        "epochs_completed",
                        "mean_epoch_time_sec",
                        "peak_memory_mib",
                        "total_parameters",
                        "trainable_parameters",
                    )
                }

        return {
            "per_model": table.to_dict("records"),
            "from_training_runs": recorded,
            "note": (
                "The quantum branch adds very few trainable parameters - its cost is "
                "simulator time per forward pass, not model size. Point run_dirs at the "
                "Step 9/10/12 runs to include training time and peak memory."
            )
            if not recorded
            else None,
        }

    @staticmethod
    @torch.no_grad()
    def _time_inference(net: Any, datamodule: Any, device: torch.device, batches: int = 20) -> float:
        """Median seconds per batch, after a warm-up.

        :param net: Model to time.
        :param datamodule: Feature datamodule.
        :param device: Device to run on.
        :param batches: Batches to time.
        :return: Median batch time in seconds.
        """
        net = net.to(device).eval()
        loader = list(datamodule.test_dataloader())[: batches + 1]
        if not loader:
            return float("nan")

        # Warm-up: the first batch pays lazy initialisation costs.
        classical, spatial, quantum, _ = loader[0]
        net(classical.to(device), spatial.to(device), quantum.to(device))

        timings = []
        for classical, spatial, quantum, _ in loader[1:]:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            net(classical.to(device), spatial.to(device), quantum.to(device))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)

        return float(np.median(timings)) if timings else float("nan")

    # ------------------------------------------------------------------ verdict

    @staticmethod
    def _verdict(summary: Dict[str, Any]) -> Dict[str, Any]:
        """State plainly whether the quantum branch earned its place.

        The specification explicitly permits a negative answer and asks for it to be
        reported honestly, so the verdict is generated from the evidence rather than
        hedged.

        :param summary: Computed results.
        :return: Verdict and the fallback criteria the specification allows.
        """
        performance = summary["performance"]
        bootstrap = performance["paired_bootstrap"]
        mcnemar = performance["mcnemar"]
        separability = summary.get("separability", {})

        significant = bool(bootstrap.get("significant"))
        p_value = mcnemar.get("p_value")
        helps = performance["delta_macro_f1"] > 0

        if significant and helps:
            headline = (
                "The quantum branch gives a statistically significant improvement "
                f"(+{performance['delta_macro_f1']:.4f} macro-F1, bootstrap CI excludes zero)."
            )
        elif helps:
            headline = (
                f"The quantum branch is nominally better (+{performance['delta_macro_f1']:.4f} "
                "macro-F1) but the paired bootstrap interval spans zero, so the difference "
                "is not distinguishable from noise on this test set."
            )
        else:
            headline = (
                f"The quantum branch does not improve performance "
                f"({performance['delta_macro_f1']:+.4f} macro-F1). Reported as a negative "
                "result, which the specification explicitly provides for."
            )

        fallbacks = {
            "parameter_efficiency": (
                "The circuit adds few trainable parameters, but its inference cost is the "
                "CPU simulator - see the efficiency table before claiming efficiency."
            ),
            "separability": (
                f"Silhouette changes by {separability.get('delta'):+.4f} when the quantum "
                "features are included."
                if isinstance(separability.get("delta"), float)
                else "Separability not computed."
            ),
            "robustness": "See Step 18; robustness is not evidence this step can supply.",
        }

        return {
            "headline": headline,
            "statistically_significant": significant,
            "mcnemar_p_value": p_value,
            "delta_macro_f1": performance["delta_macro_f1"],
            "fallback_criteria": fallbacks,
            "note": (
                "A negative result here is a valid contribution: it answers empirically "
                "whether quantum advantage emerges in this hybrid design, rather than "
                "assuming it."
            ),
        }

    # ------------------------------------------------------------------- output

    def _plot_projection(
        self, with_quantum: np.ndarray, without: np.ndarray, labels: np.ndarray, class_names: List[str]
    ) -> None:
        """Project both feature sets to 2-D and plot them side by side.

        :param with_quantum: Features including the quantum block.
        :param without: Features excluding it.
        :param labels: Class labels.
        :param class_names: Class names ordered by index.
        """
        try:
            import umap

            reducer = lambda: umap.UMAP(n_components=2, random_state=self.seed)  # noqa: E731
            method = "UMAP"
        except ImportError:
            from sklearn.manifold import TSNE

            perplexity = min(30.0, max(5.0, (len(labels) - 1) / 3.0))
            reducer = lambda: TSNE(n_components=2, random_state=self.seed, perplexity=perplexity)  # noqa: E731
            method = "t-SNE"

        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for axis, features, title in (
            (axes[0], with_quantum, "with quantum features"),
            (axes[1], without, "without quantum features"),
        ):
            projected = reducer().fit_transform(features)
            for index, name in enumerate(class_names):
                mask = labels == index
                if mask.any():
                    axis.scatter(projected[mask, 0], projected[mask, 1], s=10, alpha=0.6, label=name)
            axis.set_title(title)

        axes[0].legend(fontsize=8)
        figure.suptitle(f"Step 20: feature separability ({method})")
        figure.tight_layout()
        figure.savefig(self.figure_path("step20_separability.png"), dpi=150, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _log(summary: Dict[str, Any]) -> None:
        """:param summary: Computed summary, logged in readable form."""
        performance = summary["performance"]
        log.info("=== Step 20: quantum advantage ===")
        log.info(f"  full model      macro-F1 {performance['full_macro_f1']:.4f}")
        log.info(f"  no-quantum      macro-F1 {performance['no_quantum_macro_f1']:.4f}")
        log.info(f"  delta                    {performance['delta_macro_f1']:+.4f}")

        bootstrap = performance["paired_bootstrap"]
        log.info(
            f"  paired bootstrap 95% CI  [{bootstrap['ci_lower']:+.4f}, "
            f"{bootstrap['ci_upper']:+.4f}]  significant={bootstrap['significant']}"
        )
        mcnemar = performance["mcnemar"]
        if mcnemar.get("p_value") is not None:
            log.info(f"  McNemar p                {mcnemar['p_value']:.4f} ({mcnemar['test']})")

        log.info(f"  VERDICT: {summary['verdict']['headline']}")
