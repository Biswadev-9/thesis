# Implementation Plan — MRI Thesis Notebook → Lightning-Hydra-Template

**Sources**
- `docs/Instruction BY asif vai.md` — **authoritative specification** (Steps 1–23)
- `notebooks/mri_thesis_notebook.ipynb` — reference implementation (231 cells)
- Template: `ashleve/lightning-hydra-template`, already in this repo

**Agreed parameters**
- **Scope**: full port, Steps 1–23, phased.
- **Fidelity**: the instruction wins. Methodological gaps in the notebook get **fixed**.
  Exception: **Steps 11 and 12 follow the notebook implementation.**
- **Execution**: implement + smoke-test only (`fast_dev_run`, 1 % subsets, CPU). No full
  training runs here.

> **Consequence you need to accept up front**: fixing the methodology changes results.
> Every number currently in your thesis that touches diffusion preprocessing, focal
> loss, the loss-function choice, or the baseline comparison will move. §7 lists
> exactly which. A `docs/DEVIATIONS.md` register will be maintained so each change is
> traceable to an instruction clause.

---

## 1. Instruction vs. notebook — the gap analysis

This drives everything. ✅ = notebook already complies. ⚠️ = must be fixed. 🔒 = follow
notebook per your override.

| Step | Instruction requires | Notebook state | Action |
|---|---|---|---|
| 1 | 4 classes, consistent labels | Glioma 0 / Meningioma 1 / Pituitary 2 / No-tumor 3 | ✅ keep |
| 2 | Primary + external A (Figshare); external B optional | Both present; no BraTS | ✅ keep, B stays out of scope |
| 3 | Split **before** augmentation/synthesis, 70/15/15 | Yes, plus MD5 dedup before split | ✅ keep (dedup is a strengthening) |
| 4 | Dims, grayscale/RGB, **bit depth**, intensity ranges, class-distribution **plot**, imbalance ratio, corrupted/duplicate/mislabel checks, audit table | Bit depth and distribution plot missing; audit-table counts internally inconsistent | ⚠️ **F1** |
| 5 | Fixed size ("such as 256×256"), 3-ch when backbone needs it, min-max **or** z-score, optional validated background crop, no aggressive skull-stripping | 224 px, ImageNet z-score, **no background crop** | ⚠️ **F2**, **F3** |
| 6 | Anisotropic diffusion, tune iters {5,10,15,20} + kappa on **validation**, λ ≤ 0.25, compare vs none/Wiener/gamma/CLAHE/log, select on validation performance **and** boundary preservation | Study done, diffusion selected (iter=10, κ=15) — **but never applied to real training**; selection made with a 128 px proxy CNN only | ⚠️ **F4** (the big one), **F5** |
| 7 | Conservative train-only augmentation, exact ranges reported | Rotation ±10°, translate 5 %, scale 0.95–1.05, hflip 0.5, jitter 0.1 | ✅ keep |
| 8 | Stratified split + class-weighted CE; evaluate focal loss `FL = −α_t(1−p_t)^γ log(p_t)`; sampler on train loader only; ablate all four; judge by macro-F1 / balanced acc / class recall | Ablation done, sampler selected. **Focal loss formula is wrong** — uses `p_t = exp(−CE_weighted)` | ⚠️ **F6** |
| 9 | 6 baselines | 7 built (Swin added) | ✅ keep all 7 |
| 10 | Strong pretrained backbone, features from final conv / GAP, dropout + weight decay, **save embeddings for t-SNE/UMAP** | EfficientNet-B0, 1280-d GAP, embeddings saved, t-SNE done | ✅ keep |
| 11 | Parallel multiscale paths + gating; ablate fixed 3×3 / 5×5 / dilated vs adaptive; report learned scale weights | 3×3 / 5×5 / dilated-3, per-pixel softmax gate, 8-arm × 3-seed ablation, Phase-4 morphology correlation | 🔒 **follow notebook** |
| 12 | Compact vectors → dim-reduction → angle encoding → PQC; ≥2 depths, ≥2 entanglement patterns; expectation values as features; fixed / adaptive-depth / adaptive-entanglement / re-uploading experiments; select by validation | 5 experts (Q0 fixed, Q1 depth-4, Q2 strong-entangle, Q3 combined, Q4 re-uploading) + learned softmax mixture, 4 qubits, 3 seeds | 🔒 **follow notebook** (it already satisfies the clause) |
| 13 | Concat baseline first, then SE, then gated; report branch contribution + learned fusion weights | All three built, branch ablation + gated weights reported | ✅ keep |
| 14 | FC + norm + dropout + softmax; **loss = class-weighted CE or focal, chosen on validation** | Plain CE selected — **and tie-broken on test-set performance** | ⚠️ **F7** |
| 15 | Fixed protocol before testing; AdamW; lr ∈ {1e-4, 3e-4}; bs 16/32; patience 10–15; cosine; select on val macro-F1; **≥3 seeds for final model *and major baselines***; log hyperparams/seeds/splits/versions | Protocol locked for the final model only. **Baselines: 1 seed, patience 5**, pre-protocol | ⚠️ **F8** |
| 16 | Full metric battery incl. sensitivity + specificity, class-wise table, confusion matrix, ECE/Brier; **evaluate once** | All metrics present, but the test set was touched repeatedly during selection | ⚠️ **F9** |
| 17 | External eval, matched labels, report drop, discuss domain shift | Figshare 3-class restricted-argmax, drop reported | ✅ keep |
| 18 | Noise / contrast / blur / resolution / intensity shift; compare vs CNN **and** Transformer baselines; **report whether diffusion improves robustness under noise** | Sweep done vs EffNet-B0 + ViT. Diffusion only testable as a *test-time* filter | ⚠️ resolved by **F4** |
| 19 | Grad-CAM/Score-CAM; **attention rollout for Transformer components**; SHAP/LIME; MC-dropout; deletion/insertion; correct + incorrect per class | All present **except attention rollout** | ⚠️ **F10** |
| 20 | Remove-quantum comparison; fixed vs adaptive QCNN; report params, inference time, **training time, memory**; UMAP/t-SNE; paired bootstrap / Wilcoxon / McNemar / CIs | Params + inference time only. Wilcoxon on n=4 classes had no power. `fixed_vs_adaptive` referenced but never built (`NameError`) | ⚠️ **F11**, **F12** |
| 21 | A0–A8, all rows with the same metrics | A0≡A1 collapsed; A2 is proxy-CNN only; A3–A6 have **no diffusion** | ⚠️ resolved by **F4**, **F13** |
| 22 | RQ→experiment→evidence mapping | Hand-written prose with hard-coded numbers | ⚠️ **F14** — regenerate from result files |
| 23 | mean±std, 95 % CI, McNemar, Wilcoxon/paired bootstrap, honest reporting | Bootstrap CI + McNemar ✓; Wilcoxon underpowered | ⚠️ **F12** |

---

## 2. The fixes, specified

### F1 — Complete the data audit *(Step 4)*
Add bit-depth and per-channel intensity statistics; emit a class-distribution bar plot
and an explicit imbalance ratio; recompute duplicate counts from scratch. The notebook's
audit prose (726 duplicates / 363 groups) is arithmetically inconsistent with the split
sizes it produced (train 4617 / val 990 / test 990 implies N≈6597, i.e. ~426 removed).
**The new audit table is generated, never transcribed.**

### F2 — Input size and normalization *(Step 5)*
Instruction says "such as 256 × 256". I will keep **224 × 224** and document it as a
justified deviation: torchvision's `vit_b_16` and `swin_t` carry fixed positional
embeddings for 224 and cannot ingest 256 without interpolation, which would confound
Baselines 4/4b. `data.image_size` stays a config knob so 256 is one override away.
Normalization: ImageNet mean/std is z-score with fixed statistics and is *required* by
the pretrained weights — this satisfies the clause. `data.normalize` exposes
`imagenet | zscore | minmax` for the ablation.

### F3 — Validated background cropping *(Step 5)*
Implement optional brain-bounding-box cropping (threshold → largest connected component
→ bbox + margin). Gated by a validation check that asserts the crop never removes
foreground above the Otsu threshold. **Default off**; enabling it is a config flag with
its own ablation row.

### F4 — Bake diffusion preprocessing into real training *(Steps 6, 18, 21)* ⭐
This is the largest single change and it unblocks three downstream steps.

- Preprocessing becomes a **cached, first-class data stage**: `src/prepare_dataset.py`
  materialises `data/processed/<recipe>/…` mirroring the raw tree, where `<recipe>` ∈
  `{raw, conventional, diffusion_i{n}_k{κ}, wiener, clahe, gamma, log}`.
  Diffusion at 10 iterations over ~6 600 images is far too slow to run per-epoch, so
  caching is not an optimisation — it is what makes the fix feasible at all.
- Diffusion runs on the **single grayscale channel**, then replicates to 3, instead of
  the notebook's three independent per-RGB-channel passes over already-replicated
  grayscale (wasteful and inconsistent).
- The duplicate `anisotropic_diffusion` definitions (cells 29 vs 31, opposite `np.roll`
  signs for N/S) collapse into one implementation matching the instruction's
  divergence formulation, with both `c(s)` options and `λ ≤ 0.25` enforced.
- **Every model in ablation rows A2–A6 is trained on diffusion-preprocessed data.**
  Step 18 then becomes a real comparison — diffusion-trained vs conventional-trained
  under degradation — instead of a test-time-filter proxy.

### F5 — Confirm the Step 6 selection with the real backbone *(Step 6)*
Keep the cheap proxy sweep (SmallCNN, 128 px, stratified subset) for the full 11-method
grid, then **confirm the top-2 candidates plus `none` and `conventional` with the real
EfficientNet-B0 on the full validation set.** Selection metric = validation macro-F1,
reported alongside the Sobel edge-preservation score, per "validation performance **and**
boundary/texture preservation checks". Also fix `SmallCNN`'s hard-coded
`Linear(64·16·16, …)` (valid only at 128 px) with `AdaptiveAvgPool2d`.

### F6 — Correct the focal loss *(Step 8)*
Implement the instruction's formula literally:
`p_t = softmax(logits)[target]`, `FL = −α_t (1−p_t)^γ log(p_t)`, with `α` applied as an
explicit per-class factor. The notebook's `p_t = exp(−CE_weighted)` is only equal to the
true `p_t` when all class weights are 1, so its focal term was systematically distorted.
The old behaviour remains available as `loss=focal_legacy` purely for reproducing the
notebook's Step 8/14 tables.

### F7 — Loss selection on validation, from the permitted set *(Step 14)*
Instruction: *"Loss: class-weighted cross-entropy or focal loss based on validation
results."* Candidates are therefore **weighted CE** and **focal**; plain CE is reported
as a reference row but is not selectable. Tie-break order is fixed in advance:
validation macro-F1 → validation balanced accuracy → lower validation ECE. **Test
metrics are never consulted** — the notebook broke this at cell 158.

### F8 — One fixed protocol for everything, ≥3 seeds for major baselines *(Step 15)*
`configs/protocol/fixed.yaml` is the single source: AdamW, wd 1e-4, batch 32, cosine
annealing, max 30 epochs, early-stopping patience 12, selection on `val/f1_macro`,
seeds `[42, 123, 7]`. LR is chosen per stage from `{1e-4, 3e-4}` on validation only.
**All 7 baselines are re-run under this protocol at 3 seeds** (notebook: 1 seed,
patience 5). Every run logs the environment (torch/pennylane/sklearn versions, GPU) and
the split-file MD5.

### F9 — Test-set discipline *(Step 16)*
Test evaluation moves out of training entirely into `src/analyze.py analysis=step16_internal`,
which writes a `test_evaluated.lock` marker into the run directory and refuses a second
evaluation of the same checkpoint without `--force`. Mechanically enforces "evaluate the
final model once".

### F10 — Attention rollout *(Step 19)*
Add Abnar & Zuidema attention rollout for the ViT baseline and attention-map extraction
for Swin, satisfying *"Use attention rollout or attention maps for Transformer-based
components."* The proposed model has no transformer branch, so this attaches to the
transformer baselines — which is where the clause can be honoured.

### F11 — Full efficiency accounting *(Step 20)*
Log **training wall-clock time** and **peak memory** (`torch.cuda.max_memory_allocated`,
`tracemalloc` on CPU) via a Lightning callback, alongside the parameter counts and
inference timings the notebook already had. Build the fixed-vs-adaptive quantum
comparison table properly — cell 207 references `fixed_vs_adaptive`, which is never
defined and would raise `NameError`.

### F12 — Paired bootstrap *(Steps 20, 23)*
The notebook's Wilcoxon test paired only 4 per-class F1 values and had no power. Add a
**paired bootstrap over test predictions** (resample sample indices, recompute Δmacro-F1,
report the CI and the fraction favouring the full model) as the primary paired test
alongside McNemar. Keep Wilcoxon across the 3 seeds where the design supports it.

### F13 — Genuine A0 vs A1 *(Step 21)*
With cached recipes these become distinct real runs:
`A0` = resize + `ToTensor` only (no intensity normalization, no augmentation) +
EfficientNet-B0; `A1` = full Step-5 conventional pipeline + EfficientNet-B0;
`A2` = diffusion recipe + EfficientNet-B0 (**real backbone**, not the 128 px proxy).
A3–A6 all run on the diffusion recipe, matching the instruction's table verbatim.

### F14 — Generated RQ mapping *(Step 22)*
`analysis/step22_rq_mapping.py` reads the result CSV/JSON artefacts and emits the
RQ→experiment→evidence table with live numbers. No hand-typed metrics.

---

## 3. Architecture decisions

### D1 — One `net` contract
Notebook modules have five different forward signatures. Every image-space net will
implement:

```python
def forward(self, x) -> Tensor              # logits
def extract(self, x) -> Dict[str, Tensor]   # {"logits", "features", ...aux}
```

Aux keys: `features`, `gate_maps` (Step 11 Phase 4, Step 19), `quantum_weights`
(Step 12), `branch_weights` (Step 13). `forward` returns `extract(x)["logits"]`.
One `training_step` then serves every model in the repo.

### D2 — Two LightningModules
- **`MRIClassificationModule`** — batch `(image, label)`. Covers all 7 baselines, the
  8 Step-11 arms, the Step-10 classical branch, the Step-12 quantum branch, and the
  Step-20 fixed-quantum control. All differences live in `net` → pure config swaps.
- **`FeatureFusionModule`** — batch `(c, s, q, label)`. Covers Step 13's three fusion
  strategies, Step 14's loss ablation, Step 15's protocol runs, and zero-branch ablations.

Both use torchmetrics: `MulticlassF1Score(average="macro")` as selection metric, plus
accuracy, `MulticlassRecall(average="macro")` for balanced accuracy, precision, AUROC,
MCC. Loss is injected via `configs/loss/*` so Step 8 and Step 14 ablations are multiruns.

### D3 — Feature extraction is cached
`src/extract_features.py` loads the three frozen branch checkpoints and writes
`data/features/<tag>/{train,val,test}.pt` (`c`, `s`, `q`, `quantum_weights`, `labels`).
Steps 13/14/15/20 train many heads over the *same* frozen features; re-extracting each
time (as the notebook does) would dominate runtime, and it keeps the CPU-bound PennyLane
pass out of the training loop. `zero_branches: [quantum]` on the datamodule reproduces
Step 13 Block 9 and Step 20 Block 2 with no extra code.

> **Preserved subtlety**: `TriBranchFeatures` takes the 32-d spatial features from
> *inside* the Step-12 adaptive-quantum model, **not** from the separately saved Step-11
> checkpoint (which is used only for the arm ablation and gate-map analysis). Changing
> this changes results, so it stays — and gets a docstring.

### D4 — Three entry points
`src/train.py` (existing) · `src/extract_features.py` · `src/analyze.py`
(`configs/analyze.yaml` with `defaults: - analysis: <name>`, each analysis object
exposing `.run(cfg) -> dict`). Plus `src/prepare_dataset.py` for the recipe cache.
All artefacts land in `${paths.output_dir}`, replacing hard-coded Google Drive paths.

### D5 — Paths, secrets, Windows
Kaggle creds via `.env` (template auto-loads it). `num_workers` defaults to **0**
(Windows spawn); all preprocessing callables are module-level or `functools.partial`,
never lambdas, so `num_workers > 0` works on Linux. `pandas.groupby(...).apply(lambda …
.sample())` is replaced with `groupby(...).sample(n=…)`.

---

## 4. Target layout

```
configs/
├── data/       bt_mri · bt_mri_proxy · bt_mri_features · bt_mri_degraded · figshare
├── model/      baseline_{simple_cnn,resnet50,efficientnet_b0,vit,swin,fixed_qcnn,fixed_multiscale}
│               classical_branch · multiscale_arm · adaptive_quantum
│               fusion_{concat,se,gated} · final_classifier
├── loss/       plain_ce · weighted_ce · focal · focal_legacy
├── protocol/   fixed.yaml                       ← F8, the single source of truth
├── preprocess/ raw · conventional · diffusion · wiener · clahe · gamma · log
├── experiment/ step06_* · step08_* · step09_baselines · step10_* · step11_arm_ablation
│               step12_* · step13_fusion · step14_loss_ablation · step15_final_protocol
│               step21_a0 … step21_a8
├── analysis/   step04_audit · step06_selection · step08_imbalance · step16_internal
│               step17_external · step18_robustness · step19_explainability
│               step20_quantum · step21_ablation_table · step22_rq_mapping · step23_statistics
├── prepare_dataset.yaml · extract_features.yaml · analyze.yaml
src/
├── data/components/  split_builder · datasets · transforms · preprocessing
│                     degradations · cropping · sampling
├── data/             bt_mri_datamodule · bt_mri_proxy_datamodule
│                     bt_mri_feature_datamodule · bt_mri_degraded_datamodule
│                     figshare_datamodule
├── models/components/ backbones · conv_stem · multiscale · branches · quantum
│                      fusion · losses · attention_rollout
├── models/           mri_classification_module · feature_fusion_module
├── analysis/         one module per study (Steps 4, 6, 8, 16–23)
├── utils/            metrics (specificity, ECE, Brier, bootstrap, McNemar)
│                     checkpoints · resource_monitor (F11)
└── prepare_dataset.py · extract_features.py · analyze.py
scripts/  download_data.{ps1,sh} · run_pipeline.{ps1,sh}
docs/     IMPLEMENTATION_PLAN.md · DEVIATIONS.md · Instruction BY asif vai.md
```

The MNIST example stays until the port is green, then is removed in Phase 8.

---

## 5. Phases

Each phase ends in something runnable and smoke-tested.

| # | Phase | Instruction steps | Exit criterion |
|---|---|---|---|
| **0** | Environment & data | — | Deps installed, **PennyLane verified on Py 3.13 / torch 2.13**, both Kaggle datasets downloaded |
| **1** | Data foundation + audit | 1–5, 7 | `train.py data=bt_mri model=baseline_simple_cnn trainer=cpu +trainer.fast_dev_run=true` passes; audit table + distribution plot generated; split counts reproducible |
| **2** | Preprocessing recipes & studies | 6, 8 | All 7 recipes cached; proxy sweep + real-backbone confirmation (F5) run at 1 % data; corrected focal loss unit-tested (F6) |
| **3** | Baselines under fixed protocol | 9, 15 | `protocol/fixed.yaml` locked; 7 baselines × 3 seeds wired as one multirun; smoke-tested |
| **4** | Proposed branches | 10, 11 🔒, 12 🔒 | Step-11 unit test ported (shapes, gate softmax sums to 1, gradient flow through all 3 paths); 8-arm × 3-seed ablation as multirun; quantum branch forward-tested |
| **5** | Fusion & final model | 13, 14 | Feature cache built; 3 fusion strategies + loss selection on validation only (F7) |
| **6** | Internal + external + robustness | 16, 17, 18 | Full metric battery; once-only test guard (F9); degradation sweep comparing diffusion- vs conventional-trained models |
| **7** | Explainability & quantum advantage | 19, 20 | Grad-CAM + **attention rollout** (F10) + SHAP + MC-dropout + deletion/insertion; params/time/memory table (F11); paired bootstrap (F12) |
| **8** | Ablation, RQ mapping, statistics, cleanup | 21, 22, 23 | A0–A8 all genuine rows (F13); generated RQ table (F14); CIs + McNemar + bootstrap; MNIST removed, README rewritten, tests adapted |

---

## 6. Risks

1. **PennyLane on Python 3.13 / torch 2.13** — highest risk, gates Phase 4. Verified
   first thing in Phase 0. Fallbacks: a second pinned venv, or `pennylane-lightning`.
2. **Compute budget grew.** Fixing the methodology multiplies runs: 7 baselines × 3
   seeds, plus A0/A1/A2 real-backbone runs, plus the diffusion-trained proposed path.
   Mitigated by the recipe cache and the feature cache, but this needs a GPU box.
3. **Diffusion cache size** — a full preprocessed mirror per recipe. ~6 600 images ×
   7 recipes; PNG at 224², modest, but the build is CPU-hours.
4. **Numbers will not reproduce.** Different torch/CUDA, plus the deliberate fixes.
   Old notebook numbers are never back-filled into new outputs.
5. **`data/` is empty.** Both Kaggle pulls
   (`mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images`,
   `ashkhagan/figshare-brain-tumor-dataset`) are prerequisites for Phase 1 validation.

---

## 7. What changes in your thesis

Chapters resting on these will need re-running once you have GPU time:

- **Step 6 / RQ2, RQ3** — diffusion is now actually in the pipeline. The "never
  integrated" limitation disappears; the claim becomes testable rather than caveated.
- **Step 8 / RQ6** — focal-loss numbers change (F6). The strategy ranking may reorder.
- **Step 9 / RQ1** — baselines re-run at 3 seeds under the fixed protocol, so they gain
  ±std and become genuinely comparable to the proposed model (F8).
- **Step 14** — the selected loss will likely no longer be plain CE (F7 restricts the
  candidate set and forbids the test-set tie-break).
- **Step 18 / RQ2** — diffusion robustness becomes a trained-in comparison, not a
  test-time filter.
- **Step 20 / RQ8** — gains training-time and memory columns, plus a properly powered
  paired bootstrap. The honest "no measurable quantum advantage" conclusion may well
  survive; it will simply be better evidenced.
- **Step 21 / RQ10** — A0–A8 become nine distinct configurations instead of the
  collapsed/proxy rows.

Steps 11 and 12 results are **unaffected** — they follow the notebook by your
instruction.
