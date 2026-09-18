# Swin-T backbone experiment — execution guide

Replaces the Step 10 classical branch (EfficientNet-B0, 1280-d) with **Swin-T (768-d)**.
The spatial branch, the quantum branch, the fusion strategy and the classifier head are
**unchanged**, so this arm isolates the effect of the backbone.

Swin-T was chosen on the Step 9 baselines, on validation only: **99.07 ± 0.10** macro-F1
against EfficientNet-B0's 98.71 ± 0.22.

> **Status:** the repository work is complete. Nothing in this arm has been executed or
> runtime-verified. Run §3 before §4 and do not report results until §3 passes.

---

## 1. What changes, and what does not

```
MRI 224x224x3
├── Swin-T, frozen except features[-2:]              -> 768   <- the only change
└── AdaptiveQuantumBranch   (Step 12 checkpoint REUSED verbatim)
      ├── MultiscaleBranch("spatial_gate", ch=32)    -> 32    <- unchanged
      └── Linear(32->4) -> tanh -> 5 circuits -> mix -> 4     <- unchanged
                 |
BranchProjections:  Linear(768->64) | Linear(32->64) | Linear(4->64)
                 -> concat -> 192                             <- width unchanged
                 |
FinalClassifier: [->128, BN, GELU, Drop0.4] [->64, BN, GELU, Drop0.4] -> 4  <- unchanged
```

Held identical to the baseline so the comparison is controlled: dataset,
`data/splits/dataset_split.csv`, `recipe: null`, `image_size: 224`,
`normalize: imagenet`, `DEFAULT_AUGMENTATION`, `configs/protocol/fixed.yaml`
(AdamW, lr 1e-4, wd 1e-4, bs 32, <=30 epochs, cosine, patience 12, select on
`val/f1_macro`), seeds 42/123/7.

**Input size is deliberately not changed.** `configs/data/bt_mri.yaml` documents that 224
was chosen over the specification's 256 precisely because `swin_t` and `vit_b_16` carry
positional embeddings baked for 224.

---

## 2. Isolation rules

| Resource | Baseline | Swin arm |
|---|---|---|
| Feature cache | `data/features/default/` | `data/features/swin/` |
| Step 10 runs | `logs/train/runs/step10_classical/seed_*` | `logs/train/runs/step10_classical_swin/seed_*` |
| Step 15 runs | `logs/train/runs/step15_final/seed_*` | `logs/train/runs/step15_final_swin/seed_*` |
| Analysis runs | `logs/analyze/runs/<step>` | `logs/analyze/runs/<step>_swin[/seed_*]` |

Every command below pins `hydra.run.dir`. **Do not drop it** — Hydra's default is a
timestamped directory, which breaks resumption and makes provenance unreadable.

Nothing here writes into a baseline path. The EfficientNet configs, caches, checkpoints
and results are untouched and remain runnable.

---

## 3. Runtime checks — run these first

```bash
cd /kaggle/working/thesis

# 3.1 Swin feature dim, layout and config wiring (fast, no training)
python -m pytest tests/test_grad_cam_target_layer.py -v
python -m pytest tests/test_baselines.py -k swin -v

# 3.2 Nothing regressed
python -m pytest tests/ -q

# 3.3 End-to-end shape check
python - <<'PY'
import hydra, torch
from hydra import compose, initialize
with initialize(version_base="1.3", config_path="configs"):
    cfg = compose("train.yaml", overrides=["experiment=step10_classical_swin"])
net = hydra.utils.instantiate(cfg.model.net)
assert net.feature_dim == 768, net.feature_dim
print("backbone:", net.extract(torch.randn(2, 3, 224, 224))["features"].shape)  # (2, 768)

from src.models.components.quantum import AdaptiveQuantumBranch
o = AdaptiveQuantumBranch(channels=32, n_qubits=4)(torch.randn(2, 3, 224, 224))
print("spatial:", o["classical_features"].shape, "quantum:", o["quantum_features"].shape)

from src.models.components.fusion import FusedFeatureClassifier
f = FusedFeatureClassifier(classical_dim=768, spatial_dim=32, quantum_dim=4)
out = f.extract(torch.randn(2, 768), torch.randn(2, 32), torch.randn(2, 4))
print("fused:", out["fused"].shape, "logits:", out["logits"].shape)  # (2,192) (2,4)

from src.analysis.explainability import grad_cam_target_layer
print("grad-cam layer:", type(grad_cam_target_layer(net.backbone)).__name__)  # Permute
PY
```

Expected: `768`, `(2, 768)`, `(2, 32)`, `(2, 4)`, `(2, 192)`, `(2, 4)`, `Permute`.

**If any check fails, stop.** Do not start training.

---

## 4. Execution order

Restore the previous session first (see §7), then:

```bash
cd /kaggle/working/thesis
QCKPT=logs/train/runs/step12_adaptive_quantum/seed_42     # REUSED, not retrained
```

### Step 10 — Swin classical branch, 3 seeds ▶️

```bash
for SEED in 42 123 7; do
  python src/train.py \
    experiment=step10_classical_swin seed=$SEED \
    trainer=gpu logger=csv test=false data.num_workers=0 \
    hydra.run.dir=logs/train/runs/step10_classical_swin/seed_$SEED
done
```

Output: `logs/train/runs/step10_classical_swin/seed_*/checkpoints/*.ckpt`

### Feature cache — tag `swin` ▶️

```bash
python src/extract_features.py --config-name extract_features_swin \
  classical_ckpt=logs/train/runs/step10_classical_swin/seed_42 \
  quantum_ckpt=$QCKPT \
  data.num_workers=0 \
  hydra.run.dir=logs/extract_features/runs/features_swin
```

**Gate — verify before continuing:**

```bash
python -c "import json;m=json.load(open('data/features/swin/manifest.json'));print(m['splits']['train']);assert m['splits']['train']['classical_dim']==768"
```

`classical_dim` must be **768**. If it says 1280 the wrong config was used — stop.

### Step 13 — fusion strategy ▶️

```bash
python src/analyze.py analysis=step13_fusion_swin \
  hydra.run.dir=logs/analyze/runs/step13_fusion_swin
```

> Reads the cache via `analysis.tag`, **not** `data.tag`. Already set to `swin` in the config.

### Step 14 — loss selection ▶️

```bash
python src/analyze.py analysis=step14_loss_selection_swin \
  hydra.run.dir=logs/analyze/runs/step14_loss_selection_swin

LOSS=$(python -c "import json;print(json.load(open('logs/analyze/runs/step14_loss_selection_swin/step14_loss_selection_summary.json'))['selected_loss'])")
echo "Step 14 selected: $LOSS"
```

### Step 15 — final model, 3 seeds ▶️

```bash
for SEED in 42 123 7; do
  python src/train.py \
    experiment=step15_final_protocol_swin seed=$SEED \
    loss@model.criterion=$LOSS \
    trainer=gpu logger=csv test=false \
    hydra.run.dir=logs/train/runs/step15_final_swin/seed_$SEED
done
```

`test=false` matters: it stops Lightning auto-evaluating the test set and spending the
once-only budget before Step 16.

### Step 16 — internal test, all three seeds ▶️

```bash
for SEED in 42 123 7; do
  python src/analyze.py analysis=step16_internal_swin \
    analysis.classical_ckpt=logs/train/runs/step10_classical_swin/seed_42 \
    analysis.quantum_ckpt=$QCKPT \
    analysis.fusion_ckpt=logs/train/runs/step15_final_swin/seed_$SEED \
    data.recipe=null data.num_workers=0 \
    hydra.run.dir=logs/analyze/runs/step16_internal_swin/seed_$SEED
done
```

**Never pass `analysis.force=true`.** The lock is written beside the *fusion* checkpoint,
so each `step15_final_swin/seed_*` has its own once-only budget. Needing `force` means a
path is wrong.

Report the mean ± sd over the three seeds, not seed 42 alone.

### Steps 17–20 ▶️

```bash
C=logs/train/runs/step10_classical_swin/seed_42
F=logs/train/runs/step15_final_swin/seed_42

python src/analyze.py analysis=step17_external_swin data=figshare \
  analysis.classical_ckpt=$C analysis.quantum_ckpt=$QCKPT analysis.fusion_ckpt=$F \
  analysis.internal_summary=logs/analyze/runs/step16_internal_swin/seed_42/step16_internal_summary.json \
  hydra.run.dir=logs/analyze/runs/step17_external_swin

python src/analyze.py analysis=step18_robustness_swin \
  analysis.models.proposed.classical_ckpt=$C \
  analysis.models.proposed.quantum_ckpt=$QCKPT \
  analysis.models.proposed.fusion_ckpt=$F \
  analysis.models.efficientnet_b0.ckpt=logs/train/runs/step09_baselines/baseline_efficientnet_b0/seed_42 \
  analysis.models.vit.ckpt=logs/train/runs/step09_baselines/baseline_vit/seed_42 \
  hydra.run.dir=logs/analyze/runs/step18_robustness_swin

python src/analyze.py analysis=step19_explainability_swin \
  analysis.classical_ckpt=$C analysis.quantum_ckpt=$QCKPT analysis.fusion_ckpt=$F \
  hydra.run.dir=logs/analyze/runs/step19_explainability_swin

python src/analyze.py analysis=step20_quantum_advantage_swin \
  analysis.fusion_ckpt=$F \
  analysis.loss_summary=logs/analyze/runs/step14_loss_selection_swin/step14_loss_selection_summary.json \
  "analysis.run_dirs={classical: $C, quantum: $QCKPT}" \
  hydra.run.dir=logs/analyze/runs/step20_quantum_advantage_swin
```

Step 18 keeps `models.efficientnet_b0` and `models.vit` as EfficientNet and ViT on
purpose — they are the CNN/Transformer comparison baselines Step 18 exists to make.

---

## 5. Steps 21–23 are NOT wired for this arm

This is deliberate. `ROWS` in `src/analysis/ablation_rows.py` is a frozen tuple, and
`tests/test_ablation_matrix.py` asserts it is exactly
`["A0","A1","A2","A3","A4","A5","A6","A7","A8","P"]`. `AblationContext` carries only
`diffusion_recipe`, `selected_recipe` and `step14_loss` — no backbone or cache-tag field.
`statistical_report.py` hard-codes H1–H4 with a Holm-Bonferroni `family_size` of 4.

Adding a Swin row would modify shared code, break that test, and change the baseline's
adjusted p-values — violating "do not change existing EfficientNet results".

**What is therefore missing from the final report if these stay unchanged:**

- No Swin row in `step21_ablation_matrix.csv`; no A0–A8 ladder for Swin, so **RQ10**
  (ablation) is answered for the EfficientNet arm only.
- No Swin entry in the Step 22 RQ mapping.
- No Holm-corrected significance test of Swin against the baseline in Step 23.

**The head-to-head comparison is still available without touching shared code.** Step 16
writes `test_predictions.npz` (`y_true`, `y_pred`, `y_prob`) at
`src/analysis/internal_test.py:125`, for both arms, over the same 990-sample test split.
`src/utils/statistics.py` already provides `mcnemar_test`, `paired_bootstrap` and
`bootstrap_ci`:

```bash
python - <<'PY'
import numpy as np
from src.utils.statistics import mcnemar_test, paired_bootstrap

base = np.load("logs/analyze/runs/step16_internal/test_predictions.npz")
swin = np.load("logs/analyze/runs/step16_internal_swin/seed_42/test_predictions.npz")
assert np.array_equal(base["y_true"], swin["y_true"]), "different test sets - not pairable"

print(mcnemar_test(base["y_true"], base["y_pred"], swin["y_pred"]))
print(paired_bootstrap(base["y_true"], base["y_pred"], swin["y_pred"]))
PY
```

That is a genuine paired test on identical samples. Report it as a **single planned
comparison**, not as part of the Step 23 corrected family.

Extending Steps 21–23 properly is a separate piece of work: it needs a backbone field on
`AblationRow`, a parallel matrix keyed by arm, and an updated `test_ablation_matrix.py`.

---

## 6. Reuse vs regenerate

| Reused unchanged | Must be regenerated for Swin |
|---|---|
| `data/splits/dataset_split.csv` | Step 10 (x3 seeds) |
| `data/processed/*` recipe mirrors | `data/features/swin/` |
| Step 4 audit | Step 13, Step 14 |
| Step 6 preprocessing + confirmation | Step 15 (x3 seeds) |
| Step 8 imbalance | Steps 16 (x3), 17, 18, 19, 20 |
| Step 9 baselines (all 7) | |
| Step 11 arm ablation + gate morphology | |
| **Step 12 quantum branch (x3 seeds)** | |
| Steps 21–25 (baseline arm) | |

Reusing the Step 12 checkpoint is the single biggest saving: it is the slowest stage in
the study (five quantum circuits per forward pass on a CPU simulator) and the backbone
swap does not touch it.

---

## 7. Kaggle session setup

Attach as inputs:

1. **Notebook output, the version whose `thesis/logs/train/runs/step12_adaptive_quantum/**/checkpoints/` contains `.ckpt` files.**
   The `thesis_results_*.zip` bundle is **not** sufficient — `BUNDLE_SUFFIXES` in
   `scripts/kaggle_pipeline.py` excludes `.ckpt` and `.pt`, so the zip carries no weights
   and no feature cache.
2. `mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images`
3. `ashkhagan/figshare-brain-tumor-dataset` (Step 17)

In the notebook, set `BRANCH = "swin-backbone"` and `EXTRA_ARGS = ["--list"]` so cell 18
performs the restore and exits without running the baseline pipeline. Then verify:

```bash
find /kaggle/working/thesis/logs/train/runs/step12_adaptive_quantum -name "*.ckpt"
md5sum /kaggle/working/thesis/data/splits/dataset_split.csv
```

If the first returns nothing, attach a different version — the Swin arm depends on it.

`scripts/kaggle_pipeline.py` has **no Swin stages**. Driving this arm through the runner
would collide with the baseline's completion markers, so run the commands above directly.

---

## 8. Expected outcome

Swin beat EfficientNet-B0 by 0.36 validation macro-F1 in Step 9. Because the Step 13
branch ablation shows the classical branch carries essentially all of the fused model's
signal (zeroing spatial or quantum leaves validation macro-F1 at 98.43, unchanged), expect
roughly that gain to carry: somewhere near **98.2–98.7** test accuracy, against the current
Swin-less proposed model's 97.81 ± 0.35 and the A0 CNN baseline's 98.32 ± 0.31.

On 990 test samples a 0.5-point difference is five images, which is within seed noise.
Treat any such gap as inconclusive unless the paired test in §5 supports it.
