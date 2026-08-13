# Usage

How to run the thesis study from this repository.

**Status: Phases 1–6 implemented — specification Steps 1, 3–18.**
Steps 19–23 are not built yet; their section below says so explicitly.

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
| `src/extract_features.py` | Cache the frozen tri-branch features | `configs/extract_features.yaml` |
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

> **Quantum models need a large batch size.** The simulator has a fixed per-call cost that
> amortises across the batch: five circuits serve 2 images or 32 images at nearly the same
> price. At the protocol's `batch_size=32` the quantum models are *not* the bottleneck —
> measured on real data, the adaptive quantum branch costs 0.11 s/image against 0.16 s/image
> for EfficientNet-B0. At `batch_size=2` it looks ten times worse. Never shrink the batch
> for a quantum model to save memory; it is the one change that makes them slow.
>
> They still support neither mixed precision nor DDP, because the simulator runs on CPU
> and each forward pass moves tensors off the accelerator and back.

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

> **Keep `batch_size` at 32 or higher.** Five circuits run per forward pass, but their cost
> is dominated by per-call overhead rather than per-image work, so it amortises across the
> batch. Measured on real data at bs=32 this branch costs *less* per image than
> EfficientNet-B0. No mixed precision, no DDP.

### Feature cache — build this before Steps 13–15

```bash
python src/extract_features.py \
    classical_ckpt=logs/train/runs/<step10 run> \
    quantum_ckpt=logs/train/runs/<step12 run> \
    tag=default
```

Runs the three frozen branches once and caches their outputs to
`data/features/<tag>/{train,val,test}.pt`. Steps 13, 14, 15 and 20 all train many small
heads over these same tensors — recomputing them per epoch would mean re-running
EfficientNet-B0 and five quantum circuits every time. Caching turns those stages from
hours into seconds and takes the CPU simulator out of the training loop entirely.

Two things worth knowing:

- **Augmentation is off during extraction.** Cached features must be deterministic, or
  every head would train on a different random view of the same image.
- **The spatial features come from inside the Step 12 model**, which carries its own
  spatial-gate branch trained jointly with the quantum mixture. The separately trained
  Step 11 checkpoint is used only for the arm ablation and gate-morphology analysis. This
  is inherited from the notebook and preserved deliberately.

Re-extract whenever either branch is retrained — a stale cache trains the fusion head on
the wrong features, silently.

### Step 13 — Choose the fusion strategy

```bash
python src/analyze.py analysis=step13_fusion analysis.tag=default
```

Trains concatenation, SE attention and gated fusion on identical features, then re-trains
the winner with each branch zeroed at input. Zeroing keeps the architecture, parameter
count and protocol identical, so the measured drop reflects that branch's *information*
rather than a smaller model.

Concatenation is the baseline and is only displaced if something beats it by more than
`analysis.improvement_threshold` — the specification adds attention or gating "only if it
improves validation performance". Raise the threshold to avoid adopting complexity on noise.

Also reports the mean learned per-branch weights from gated fusion, which is what Step 13's
"report the learned fusion weights" asks for. SE attention's weights are per *channel* and
cannot answer "which branch".

For full protocol runs across seeds instead:

```bash
python src/train.py -m experiment=step13_fusion \
    model=fusion_concat,fusion_se,fusion_gated seed=42,123,7
```

### Step 14 — Choose the loss

```bash
python src/analyze.py analysis=step14_loss_selection analysis.tag=default
```

> **Only weighted CE and focal loss are selectable.** Plain CE is trained and reported as
> a reference row but cannot win — the specification restricts the final choice to those
> two, and the reference notebook selected plain CE *by comparing test scores*, then
> reported that same test set as its result.

The tie-break is fixed in advance and never reads test data:
validation macro-F1 → validation balanced accuracy → **lower validation ECE**. The third
matters: the notebook hit a genuine three-way tie, and calibration is a principled way to
break it — its own spot-check found a confidently wrong prediction at probability 1.000.

### Step 15 — Train the final model

```bash
python src/train.py -m experiment=step15_final_protocol seed=42,123,7 \
    loss@model.criterion=<whatever Step 14 selected>
```

Nothing about the protocol may change after this point. Step 16 evaluates the resulting
checkpoint on the internal test set exactly once.

### Step 16 — Internal test ⚠️ once only

> **Do not run this until Steps 6, 8, 13 and 14 are settled and the final model is
> trained.** This is the one evaluation the study cannot repeat. Everything before it is
> decided on validation.

```bash
python src/analyze.py analysis=step16_internal \
    analysis.classical_ckpt=logs/train/runs/<step10> \
    analysis.quantum_ckpt=logs/train/runs/<step12> \
    analysis.fusion_ckpt=logs/train/runs/<step15>
```

Reports the full Step 16 battery: accuracy, balanced accuracy, macro precision/recall/F1,
weighted F1, sensitivity, specificity, MCC, one-vs-rest AUC, the class-wise table with
support, the confusion matrix with its worst pairs named, and calibration (ECE + Brier),
including a count of predictions that were **confidently wrong** above 0.99.

A `test_evaluated.lock` is written beside the fusion checkpoint. A second run against the
same checkpoint is **refused**:

```
RuntimeError: This checkpoint has already been evaluated on the internal test set.
Step 16 requires the test set to be used once.
```

`analysis.force=true` overrides it, and the lock records that the evaluation was forced —
so a result that is no longer a single-use estimate is visible as such.

### Step 17 — External validation

```bash
python src/analyze.py analysis=step17_external data=figshare \
    analysis.classical_ckpt=<step10> analysis.quantum_ckpt=<step12> \
    analysis.fusion_ckpt=<step15> \
    analysis.internal_summary=logs/analyze/runs/<step16>/step16_internal_summary.json
```

Needs the Figshare download (`-IncludeExternal`). No retraining — the model is used exactly
as Step 15 left it.

**Two scores are reported.** Restricting the argmax to the three present classes is standard
for a missing class and is what the notebook did, but it is also *favourable*: it forgives
every case the model would have answered "No-tumor". The unrestricted score counts those as
errors, and `predicted_absent_class_count` says how often it happened. Quote both.

The internal-to-external drop is computed on the **same three classes**, so it measures
domain shift rather than the missing class.

### Step 18 — Robustness

```bash
python src/analyze.py analysis=step18_robustness \
    analysis.models.proposed.classical_ckpt=<step10> \
    analysis.models.proposed.quantum_ckpt=<step12> \
    analysis.models.proposed.fusion_ckpt=<step15> \
    analysis.models.efficientnet_b0.ckpt=<step9 effnet run> \
    analysis.models.vit.ckpt=<step9 vit run>
```

Five degradation families at three severities each, against the CNN and Transformer
baselines Step 18 requires. Results are reported as absolute macro-F1 **and** as a drop from
each model's own clean score — the drop is the robustness measure, since a weaker model that
degrades gently is more robust and absolute scores hide that.

To answer *"does diffusion preprocessing improve robustness under noisy inputs"*, add a
second entry differing **only** in `preprocess` (see the commented `proposed_diffusion`
block in the config). The filter runs *after* the degradation, so it actually sees the
noise. Without that pair the analysis reports `answered: false` rather than implying a
verdict.

> **The archive's pre-degraded "Challenging Datasets" are deliberately unused.** Their
> filenames (`bilateral_glioma (1).jpg`) do not correspond to the primary set's
> (`BT-MRI GL Train (1).jpg`), and there are 3,354 against 7,023 originals — so it is
> impossible to tell which are degraded copies of *training* images. Evaluating on them
> could silently score the model on its own training data.

### Steps 19–23 — not implemented yet

| Steps | What | Phase |
|---|---|---|
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

322 tests. They cover leakage, the corrected focal loss, protocol conformance to Step 15,
dataset-root disambiguation, the gate's per-pixel softmax, all eight ablation arms, the
quantum mixture arithmetic, checkpoint round-tripping, and the Step 14 selection rule -
including that plain CE cannot be selected and that ties break on calibration rather
than on test data, the once-only test lock, degradation determinism, and that gradients
reach the input pixels through the full pipeline. `test_kaggle_pipeline.py` additionally
holds the §6 driver's stage graph to the study: its baseline, arm and Step 8 strategy
tables are checked against the definitions they restate, so adding a ninth arm fails a
test rather than silently never training it. It also covers dataset discovery against a
synthetic replica of the deeper Kaggle mount, including the degraded-copy decoy and the
loader's descent limit. Run them after changing anything in `src/`.

> **`test_train_resume` is flaky.** It comes from the project template and asserts that
> epoch 1 beats epoch 0. That held on MNIST; on 1 % of a brain-MRI split with a
> from-scratch CNN and `save_top_k=1`, epoch 1 often does not improve, so no
> `epoch_001.ckpt` is written. Nothing in the study depends on it.

---

## 6. Running the whole study with one command

§3 is the manual path: run each stage, read the timestamp off the console, paste it into
the next command. `scripts/kaggle_pipeline.py` does that for you.

```bash
python scripts/kaggle_pipeline.py --list                 # the stage graph, ticked where done
python scripts/kaggle_pipeline.py --profile smoke        # ~20 min, proves the wiring
python scripts/kaggle_pipeline.py --profile full         # the study, three seeds
```

It is written for Kaggle — see `notebooks/kaggle_run.ipynb`, which is a Run-All notebook
around it — but nothing in it is Kaggle-specific. It works on any GPU box.

Three things it does that a shell script cannot:

**Pins every output directory.** Each stage runs with an explicit `hydra.run.dir`, so
Step 15's checkpoint is at `logs/train/runs/step15_final/seed_42` before the run starts
and Step 16 can be handed it without a human reading a timestamp.

**Survives being killed.** Every finished stage writes `.pipeline_done.json` and is
skipped on the next invocation; unfinished training runs resume from `last.ckpt`. Each
training run is also given a `trainer.max_time` matching the time left in
`--budget-hours`, so it stops and saves rather than being cut off mid-epoch. Kaggle ends
a session at 12 hours and the full study needs three or four of them, so this is the
difference between the study finishing and never finishing.

**Carries the selections forward.** Steps 6, 8 and 14 *choose* something. The pipeline
reads `selected_recipe`, `selected_strategy` and `selected_loss` back out of the summary
JSON each study writes and applies them to every stage downstream — materialising the
recipe first if it needs a mirror. `--no-apply-selections` turns that off;
`--recipe/--imbalance/--loss` force a value by hand.

**Finds the dataset wherever the host mounted it.** `--setup-data` searches `/kaggle/input`
for the directory holding **both** `Training/` and `Testing/` with all four class folders
inside each, and links that directory — not the mount point — to `data/raw/bt_mri`:

```bash
python scripts/kaggle_pipeline.py --setup-data          # --input-root to search elsewhere
```

Matching on structure rather than on a path is what makes it portable: Kaggle mounts at
`/kaggle/input/<slug>/` for some accounts and `/kaggle/input/datasets/<owner>/<slug>/` for
others, and the archive then nests `BT-MRI Dataset/BT-MRI Dataset/` inside that. Requiring
*both* split folders is also what keeps the archive's `Challenging Datasets/` tree out — it
carries the same four class names but deliberately degraded images.

Linking the split folders' parent is not cosmetic. `locate_dataset_root` descends
`LOADER_MAX_DEPTH = 3` levels; on the deeper mount the split folders sit at level four, so
linking the mount point produces `No images found` from a perfectly good dataset. The
preflight now checks reachability at the loader's own limit, not just that files exist, and
says which of the two problems it found.

Useful flags:

| Flag | Effect |
|---|---|
| `--setup-data` | find and link the attached datasets, then exit |
| `--no-preflight` | skip the dataset check before the first stage |
| `--only step16_internal` | one stage, or one group (`--only step09`) |
| `--from step13_fusion` | start partway down the graph |
| `--skip step11,step17` | leave stages out |
| `--force` | re-run stages already marked done |
| `--progress` | Lightning progress bars, off by default |
| `--seeds 42` | override the profile's seed list |
| `--force-test` | override Step 16's once-only lock (recorded in the summary) |

Afterwards, `logs/pipeline/REPORT.md` lists every stage with its duration and lifts the
headline numbers out of the summaries; `logs/pipeline/manifest.json` is the machine-readable
version. A `thesis_results_*.zip` of every table, figure and resolved config — no
checkpoints — is written to `/kaggle/working` when it exists, otherwise `logs/`.

> **Only `--profile full` is reportable.** `smoke` and `fast` shorten `max_epochs` and
> early-stopping patience, which `configs/protocol/fixed.yaml` says is a protocol
> amendment, not a tweak. Runs made under them are stamped `protocol_intact: false` in
> the manifest and the report says so at the top. Use them to prove the wiring, never to
> produce a number for the thesis.
>
> They are also kept physically apart, because the completion markers that make the
> pipeline resumable would otherwise let a one-epoch run satisfy a `full` one — every
> stage would report itself already done. `full` writes to `logs/` and
> `data/features/default`; `smoke` and `fast` write to `logs/_smoke/`, `logs/_fast/` and
> `data/features/<profile>/`. Run smoke first and then full, in the same checkout,
> without either touching the other.

---

## 7. Gotchas

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

**Measured cost, real data, CPU, batch 32.** EfficientNet-B0 ~0.16 s/image, adaptive
quantum branch ~0.11 s/image. With 4,617 training images that is roughly 12 and 8 minutes
per epoch respectively, so a single 20-epoch run is 3–4 hours and the 21-run baseline
sweep is 3–4 days. **The full study is not viable on CPU** — it is an overnight job on a
mid-range GPU. The proxy studies (Steps 6 and 8) are CPU-feasible at 10–25 minutes each.

**Torch here is CPU-only.** Move to a CUDA machine before the training steps.

---

## 8. Changelog

| Phase | Steps | Added |
|---|---|---|
| 1 | 1, 3, 4, 5, 7 | Split builder, transforms, datamodule, audit, `analyze.py` |
| 2 | 6, 8 | Preprocessing recipes, `prepare_dataset.py`, proxy datamodule, both studies |
| 3 | 9, 15 | Seven baselines, fixed protocol, resource monitor, quantum circuits |
| 4 | 10, 11, 12 | Classical branch, 8-arm multiscale ablation, adaptive quantum branch, embedding + gate-morphology analyses, checkpoint loader |
| 5 | 13, 14, 15 | Feature cache, three fusion strategies, final classifier, branch-contribution ablation, validation-only loss selection, calibration metrics |
| 6 | 16, 17, 18 | Full metric battery with once-only test lock, Figshare external validation, degradation sweep, full image-to-logits pipeline |
| — | 4–18 | `scripts/kaggle_pipeline.py` and `notebooks/kaggle_run.ipynb`: the whole study as one resumable command, plus the Kaggle Run-All notebook around it (§6). Fixed `train.yaml`/`eval.yaml` still defaulting to the deleted `data/mnist` config. |
