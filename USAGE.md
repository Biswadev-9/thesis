# Usage

How to run the thesis study from this repository.

**Status: Phases 1–4 implemented — specification Steps 1, 3–12 and 15.**
Steps 13–14 and 16–23 are not built yet; their sections below say so explicitly.

> This file is updated at the end of every phase. If a command appears here, it has been
> run and works. If a step is not listed, it is not implemented yet.

Related documents:

- `docs/Instruction BY asif vai.md` — the specification. It is the authority.
- `docs/IMPLEMENTATION_PLAN.md` — how the notebook maps onto this repo, and the 14 fixes.
- `docs/DEVIATIONS.md` — every deliberate departure from the notebook or the spec, with reasons.

---

## 1. Setup

### Environment

```bash
# The project venv already exists at ./env
./env/Scripts/python.exe -m pip install -r requirements.txt     # Windows
# source env/bin/activate && pip install -r requirements.txt    # Linux/macOS
```

Verify:

```bash
./env/Scripts/python.exe -c "import torch, lightning, pennylane; print(torch.__version__, torch.cuda.is_available())"
```

> **This machine reports `2.13.0+cpu` — no CUDA.** Everything below runs, but full
> training needs a GPU box. Install a CUDA build of torch there before starting §3.

### Data

```powershell
.\scripts\download_data.ps1                  # primary dataset
.\scripts\download_data.ps1 -IncludeExternal # + Figshare, needed for Step 17
```

```bash
./scripts/download_data.sh --external        # Linux/macOS
```

Needs Kaggle credentials — either `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` /
`KAGGLE_KEY` in `.env` (copy `.env.example`).

The archive unpacks into a doubly-nested folder and **also ships a
`Challenging Datasets/` tree containing the same four class names** but deliberately
blurred, noisy and motion-corrupted images. The loader finds the real training data
automatically by requiring `Training/` and `Testing/` folders, and will never select a
degraded copy. Point `data.raw_subdir` somewhere explicit if your layout differs.

Expected after download: **7,023** images in the primary tree.

---

## 2. How the repository is driven

Four entry points, all Hydra-configured. Every run writes into its own timestamped
directory under `logs/`, so results are always tied to the config that produced them.

| Entry point | Purpose | Config |
|---|---|---|
| `src/analyze.py` | Non-training stages: audits, selection studies, evaluation | `configs/analyze.yaml` |
| `src/prepare_dataset.py` | Materialise a preprocessing recipe to disk | `configs/prepare_dataset.yaml` |
| `src/train.py` | Train one model | `configs/train.yaml` |
| `src/eval.py` | Evaluate a checkpoint | `configs/eval.yaml` |

Anything can be overridden from the command line:

```bash
python src/train.py trainer=gpu data.batch_size=16 model.optimizer.lr=3e-4
```

`-m` turns any override into a sweep:

```bash
python src/train.py -m model=baseline_resnet50,baseline_vit seed=42,123,7
```

### The one rule

`configs/protocol/fixed.yaml` holds Step 15's fixed training protocol — optimiser,
batch size, epochs, patience, scheduler, selection metric, seeds. **Do not override
those values per model.** Every comparison in the study depends on all models sharing
them. Changing that file is a protocol amendment: re-run everything downstream.

---

## 3. Running the study

Run in order. Each stage consumes what the previous produced.

### Step 4 — Data audit *(also builds the split)*

```bash
python src/analyze.py analysis=step04_audit
```

Builds `data/splits/dataset_split.csv` on first run — pooling both vendor folders,
hashing, **deduplicating, then** splitting 70/15/15 stratified. That order is what makes
the split leak-free, and it is asserted, not assumed.

Produces: `dataset_audit_table.csv`, `image_audit.csv`, `class_distribution.{csv,png}`,
`sample_images.png`, `step04_audit_summary.json`.

Verified on the real dataset:

| Check | Result |
|---|---|
| Pooled → unique | 7,023 → **6,597** (426 exact duplicates removed) |
| Split sizes | train **4617** / val **990** / test **990** |
| Corrupted files | 0 |
| Dimensions | all **224×224** |
| Colour / depth | all RGB, 8-bit, range 0–255 |
| Duplicates remaining / leakage | 0 / 0 |
| Train imbalance ratio | **1.14** (mild) |

> The notebook's audit prose claims 726 duplicates. That is wrong and contradicts its own
> split sizes — use the generated table. See `docs/DEVIATIONS.md` (F1).

**Background cropping fails its validation on this dataset** — up to 5.1 % of bright
tissue would be lost on the worst image, against a 0.1 % tolerance. Step 5 permits
cropping "only if it does not remove tumor regions", so leave `data.crop_background=false`.
The check exists precisely to make that call on evidence.

### Step 6 — Choose preprocessing

Rank the candidates (a fast proxy sweep — eleven-plus methods, small model, 128px):

```bash
python src/analyze.py analysis=step06_preprocessing
```

Ranks on validation macro-F1 and reports a Sobel edge-preservation score alongside, so a
candidate that "wins" by blurring detail away is visible. The summary names the winner and
prints the exact command to materialise it.

> **This takes roughly 10–15 minutes on CPU** — 13 candidates at about a minute each. The
> log prints `[n/13] <recipe>  (~Xm remaining)` as it goes, and the results table is
> rewritten after every candidate, so interrupting it keeps whatever finished. Shorten it
> with `analysis.epochs=3` or a smaller grid:
>
> ```bash
> python src/analyze.py analysis=step06_preprocessing \
>     'analysis.diffusion_iterations=[10,15]' 'analysis.diffusion_kappas=[15.0]'
> ```

Then **confirm the winner with the real backbone** — the proxy ranks, it does not decide:

```bash
python src/prepare_dataset.py -m recipe=diffusion_i10_k15,diffusion_i15_k15
python src/train.py -m experiment=step06_confirm \
    data.recipe=null,diffusion_i10_k15,diffusion_i15_k15 trainer=gpu
```

Compare on **validation** macro-F1 only. The test set stays untouched until Step 16.

> Materialising is what makes diffusion usable at all: applied on the fly it would
> dominate every epoch, which is why the notebook selected it and then never used it.
> `raw` and `conventional` need no mirror — they apply no filter and differ only in
> `data.normalize`.

### Step 8 — Choose the imbalance strategy

```bash
python src/analyze.py analysis=step08_imbalance
```

Compares class weighting, focal loss, balanced sampler, augmentation, and a combined arm,
on macro-F1, balanced accuracy **and class-wise recall**. The combined arm is only selected
if it beats every component on both macro-F1 and worst-class recall — a strategy that lifts
the average while collapsing one class is rejected.

Also runs the notebook's original focal formulation as `focal_loss_legacy` and reports the
difference, quantifying the F6 correction.

### Step 9 + 15 — Train the baselines

```bash
python src/train.py -m experiment=step09_baselines \
    model='glob(baseline_*)' seed=42,123,7 trainer=gpu
```

21 runs: seven baselines × three seeds, all under the fixed protocol.

| Config | Baseline |
|---|---|
| `baseline_simple_cnn` | 1 — CNN from scratch |
| `baseline_resnet50` | 2 — ResNet50 transfer |
| `baseline_efficientnet_b0` | 3 — EfficientNet-B0 transfer |
| `baseline_vit` | 4 — ViT-B/16 transfer |
| `baseline_swin` | 4b — Swin-T transfer |
| `baseline_fixed_qcnn` | 5 — fixed QCNN |
| `baseline_fixed_multiscale` | 6 — multiscale CNN, no gating |

> **Run Baseline 5 separately.** The quantum simulator is CPU-only, so every forward pass
> leaves the GPU and comes back — one batch took 13 s against under 2 s for the CNNs. It
> will dominate the sweep's wall-clock time, and it supports neither mixed precision nor DDP.

```bash
python src/train.py -m experiment=step09_baselines model=baseline_fixed_qcnn seed=42,123,7
```

Each run writes checkpoints plus `resource_usage.json` (training time, per-epoch times,
parameter counts, peak memory) — collected later by Step 20.

### Step 10 — Classical feature branch

```bash
python src/train.py -m experiment=step10_classical seed=42,123,7 trainer=gpu
```

EfficientNet-B0, features taken from the global average pooling layer (1280-d). Its
classification head is temporary — Step 13 discards it and keeps the features.

Then export the embeddings, which Step 10 requires and Step 20 reuses:

```bash
python src/analyze.py analysis=step10_embeddings model=branch_classical \
    analysis.ckpt_path=logs/train/runs/<timestamp>
```

Writes `embeddings.npz`, a t-SNE projection, and a silhouette score per split. The
silhouette is computed in the **original feature space** — t-SNE does not preserve global
distances, so scoring its output would measure the projection, not the features.

### Step 11 — Adaptive multiscale branch + 8-arm ablation

```bash
python src/train.py -m experiment=step11_arm_ablation seed=42,123,7 trainer=gpu \
    model.net.arm=arm1_fixed_3x3,arm2_fixed_5x5,arm3_fixed_dilated,arm4_concat_nogate,arm5_global_gate,arm6_spatial_gate
```

24 runs across three seeds. The arms:

| Arm | What it isolates |
|---|---|
| `arm1_fixed_3x3` / `arm2_fixed_5x5` / `arm3_fixed_dilated` | a single fixed receptive field |
| `arm4_concat_nogate` | multiscale with **no** gating |
| `arm5_global_gate` | one weight per path **per image** (SKNet-style) |
| `arm6_spatial_gate` | one weight per path **per pixel** — proposed |
| `arm7_spatial_gate_quantum` / `arm8_global_gate_quantum` | the same two, feeding a fixed circuit |

**The comparison that carries the claim is arm6 vs arm5.** Both are adaptive, so their
difference isolates *spatial* adaptivity rather than adaptivity in general. Arms 1–3 say
what one fixed kernel achieves; arm4 says what multiscale achieves ungated.

Run the quantum arms separately — they are far slower:

```bash
python src/train.py -m experiment=step11_arm_ablation seed=42,123,7 \
    model.net.arm=arm7_spatial_gate_quantum,arm8_global_gate_quantum
```

Then report the learned scale weights, which Step 11 asks for explicitly:

```bash
python src/analyze.py analysis=step11_gate_morphology model=branch_multiscale \
    model.net.arm=arm6_spatial_gate analysis.ckpt_path=logs/train/runs/<arm6 timestamp>
```

Produces the weight-map figure plus Spearman correlations between proxy tumour extent and
each path's weight, pooled **and per class**. Read the `limitations` field in the summary
before quoting any of it: the tumour mask is an Otsu proxy, the three weights are a
softmax and therefore not independent, and pooled correlations can be manufactured by
between-class differences.

Requires a spatial-gate arm — the global-gate arms emit one weight per image, with no
spatial structure to correlate against.

### Step 12 — Adaptive quantum branch

```bash
python src/train.py -m experiment=step12_adaptive_quantum seed=42,123,7
```

The Step 11 spatial-gate branch feeding a learned softmax mixture over five circuits
(fixed, deeper, strongly-entangling, combined, re-uploading), conditioned per image.

> **Budget this.** Five circuits per forward pass on a CPU simulator — roughly five times
> the fixed-QCNN baseline, which was already an order of magnitude slower than the CNNs.
> Consider reducing `trainer.max_epochs` for the seed sweep. No mixed precision, no DDP.

### Steps 13–14, 16–23 — not implemented yet

| Steps | What | Phase |
|---|---|---|
| 13, 14 | Feature fusion; final classifier | 5 |
| 16, 17, 18 | Internal test; external Figshare; robustness | 6 |
| 19, 20 | Explainability; quantum advantage and efficiency | 7 |
| 21, 22, 23 | Ablation table A0–A8; RQ mapping; statistics | 8 |

---

## 4. Where results go

```
logs/
├── analyze/runs/<timestamp>/      tables, figures, <name>_summary.json
├── train/runs/<timestamp>/        checkpoints/, resource_usage.json, csv/, .hydra/
├── train/multiruns/<timestamp>/   one numbered directory per sweep job
└── prepare_dataset/runs/<ts>/

data/
├── raw/bt_mri/                    downloaded archive (untouched)
├── splits/dataset_split.csv       the single source of split membership
└── processed/<recipe>/            cached preprocessing mirrors
```

Every run directory contains `.hydra/config.yaml` — the fully resolved config, which is
what makes a result reproducible.

---

## 5. Common tasks

**Train one model quickly to check wiring**

```bash
python src/train.py experiment=step09_baselines model=baseline_simple_cnn \
    trainer=cpu +trainer.fast_dev_run=true
```

**Use a preprocessing recipe**

```bash
python src/prepare_dataset.py recipe=clahe
python src/train.py data.recipe=clahe
```

**Change the intensity treatment** — `data.normalize=imagenet|zscore|minmax|none`.
`none` with `augment=false` is ablation row A0's raw-image condition.

**Run the tests**

```bash
./env/Scripts/python.exe -m pytest tests/ -q
```

218 tests. They cover leakage, the corrected focal loss, protocol conformance to Step 15,
dataset-root disambiguation, the gate's per-pixel softmax, all eight ablation arms, the
quantum mixture arithmetic, and checkpoint round-tripping. Run them after changing
anything in `src/`.

---

## 6. Gotchas

**Image size is 224, not the spec's suggested 256.** ViT and Swin carry positional
embeddings baked for 224. Configurable via `data.image_size`, but changing it breaks those
two baselines. See `docs/DEVIATIONS.md` (F2).

**`num_workers` defaults to 0** because Windows spawns rather than forks workers. On Linux,
set `data.num_workers=8` for a large speedup.

**The test set is off limits until Step 16.** Selection studies read validation only, and
the proxy datamodule deliberately returns the validation loader from `test_dataloader()`.

**Nothing is selected until you run it.** The Step 6 and Step 8 studies have only been
smoke-tested on synthetic data, so no preprocessing recipe or imbalance strategy is
actually chosen yet. Run them on the real data before Phase 4 results are meaningful.

**Freezing a branch means `requires_grad = False` *and* `.eval()`.** Use
`src.utils.checkpoints.freeze`. Without eval mode, BatchNorm keeps updating its running
statistics on every forward pass, so a "frozen" branch's features drift between epochs
and quietly corrupt the cached features Steps 13–15 are built on.

**Torch here is CPU-only.** Move to a CUDA machine before §3.

---

## 7. Changelog

| Phase | Steps | Added |
|---|---|---|
| 1 | 1, 3, 4, 5, 7 | Split builder, transforms, datamodule, audit, `analyze.py` |
| 2 | 6, 8 | Preprocessing recipes, `prepare_dataset.py`, proxy datamodule, both studies |
| 3 | 9, 15 | Seven baselines, fixed protocol, resource monitor, quantum circuits |
| 4 | 10, 11, 12 | Classical branch, 8-arm multiscale ablation, adaptive quantum branch, embedding + gate-morphology analyses, checkpoint loader |
