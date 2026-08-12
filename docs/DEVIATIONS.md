# Deviation register

Every place the implementation differs from either the specification
(`docs/Instruction BY asif vai.md`) or the reference notebook
(`notebooks/mri_thesis_notebook.ipynb`), with the reason.

Two kinds of entry:

- **Fix (F*)** — the notebook did not meet the specification; the specification wins.
  Listed in `docs/IMPLEMENTATION_PLAN.md` §2.
- **Deviation (D*)** — the implementation departs from the specification's literal
  wording for a stated technical reason.

Status values: `done`, `partial`, `planned`.

---

## Phase 1 — data foundation (Steps 1, 3, 4, 5, 7)

### F1 — Data audit completed · `done`

**Specification** Step 4 requires image dimensions, grayscale/RGB status, **bit depth**,
intensity ranges, a class-distribution plot, the imbalance ratio, duplicate and
corruption checks, and a dataset audit table.

**Notebook** Recorded dimensions, mode and intensity range, but never bit depth, and had
no distribution plot. Its audit table was hand-written prose.

**Now** `src/analysis/data_audit.py` computes every row at run time and writes
`dataset_audit_table.csv`, `image_audit.csv`, `class_distribution.{csv,png}` and
`sample_images.png`. Bit depth and channel count come from the PIL mode via
`_MODE_DEPTH` in `split_builder.py`.

**Consequence** The notebook's audit narrative — 726 duplicate images across 363 groups —
does not reconcile with the split sizes it produced. Train 4617 / val 990 / test 990
implies ~6597 unique images, i.e. roughly **426** rows removed from 7023, not 726. The
regenerated table will state the true figure. **Do not carry the old sentence into the
thesis.**

### F2 — Input size held at 224 · `done`

**Specification** Step 5: "Resize all images to a fixed input size, such as 256 × 256."

**Implementation** 224 × 224, configurable via `data.image_size`.

**Reason** `torchvision.models.vit_b_16` and `swin_t` carry positional embeddings baked
for 224 and cannot accept 256 without interpolating them. Interpolated embeddings would
change Baselines 4 and 4b relative to their published ImageNet behaviour and confound
the comparison against the CNN baselines that Step 9 exists to make. The specification
says "such as", so 256 reads as illustrative rather than mandatory.

**Also** ImageNet mean/std normalisation is retained as the default. It *is* z-score
normalisation with fixed statistics, which Step 5 permits, and the pretrained backbones
require it. `data.normalize` additionally offers `zscore` (per-image), `minmax`
(per-image) and `none`.

### F3 — Validated background cropping added · `done`

**Specification** Step 5: "Apply background cropping only if it does not remove tumor
regions."

**Notebook** No cropping at all.

**Now** `src/data/components/cropping.py` provides `BrainBoundingBoxCrop` plus
`validate_crop_preserves_foreground`, which checks that no pixel above the in-brain Otsu
threshold is lost. **Default off**; the Step 4 audit reports whether enabling it would be
safe, so the decision rests on evidence.

### F13 (partial) — A0 and A1 can now differ · `done` for the mechanism

`data.normalize=none` with `data.augment=false` expresses ablation row A0's raw-image
condition, which the notebook could not represent — it collapsed A0 and A1 into a single
number. The runs themselves land in Phase 8.

### D1 — Split paths stored relative, not absolute · `done`

**Notebook** Wrote absolute Colab paths into `dataset_split.csv`, so the file stopped
resolving whenever the runtime was recycled; the split had to be rebuilt, and at least
once was rebuilt inconsistently (notebook cells 218–220).

**Now** `rel_path` is stored relative to an `image_root` supplied at load time. The same
split table addresses the raw tree and any preprocessing-recipe mirror.

### D2 — Class-folder aliases accepted · `done`

Published mirrors of the dataset name the folders variously `notumor`, `no_tumor`,
`No-tumor`, `glioma_tumor`. `normalize_class_folder` maps all variants onto the four
canonical class names, so label indices stay fixed per Step 1 regardless of the mirror.

### D3 — Vendor Training/Testing division discarded · `done`

Matches the notebook. The dataset ships its own split, but it is not leak-free — the same
image appears in both folders. Step 3 requires our own stratified 70/15/15, so all images
are pooled and re-split after deduplication. `source_split` is retained for provenance.

---

## Phase 1 — model layer (Steps 8, 15)

### F6 — Focal loss corrected · `done`

**Specification** Step 8: `FL = -alpha_t (1 - p_t)^gamma log(p_t)`.

**Notebook** Computed `pt = torch.exp(-ce_loss)` where `ce_loss` was *already*
class-weighted. Weighted cross-entropy for a sample is `-w_t log(p_t)`, so
`exp(-CE) = p_t ** w_t`, not `p_t`. The modulating factor was therefore applied to a
distorted quantity whenever class weighting was active — exactly the configuration Step 8
puts under test. Unweighted, the two forms coincide, which is why the defect stayed
invisible.

**Now** `src.models.components.losses.FocalLoss` reads `p_t` from the softmax directly and
applies `alpha_t` as a separate factor. `LegacyFocalLoss` preserves the notebook's
behaviour, available as `loss=focal_legacy`, solely to regenerate its published tables.

`tests/test_losses.py::test_legacy_focal_modulates_against_p_to_the_weight` pins the
exact nature of the defect, so this entry is verifiable rather than asserted.

**Consequence** Step 8's imbalance ablation and Step 14's loss ablation must be re-run.
The strategy ranking may change.

### F8 (partial) — Selection metric is macro-F1 · `done` for the mechanism

**Specification** Step 15: "Save the best model using validation macro-F1 or balanced
accuracy, not only validation accuracy."

`configs/callbacks/mri.yaml` monitors `val/f1_macro` for both checkpointing and early
stopping. The template's `callbacks/default.yaml` (monitoring `val/acc`) is left in place
only while the MNIST example survives. The full fixed protocol and the ≥3-seed baseline
requirement land in Phase 3.

### D4 — Single `forward`/`extract` contract · `done`

Architecture decision D1 in the plan. The notebook grew five different forward
signatures; every net here returns logits from `forward` and a dict from `extract`. This
is a structural change with no effect on results.

### D5 — `SmallCNN` made resolution-independent · `done`

The notebook's proxy CNN ended in `Linear(64 * 16 * 16, 128)`, hard-coding a 128 px
input; it raised a shape error at any other resolution. Replaced with
`AdaptiveAvgPool2d`, keeping the same block structure and parameter scale.

**Consequence** Step 6's proxy numbers will shift slightly. They were only ever used to
rank preprocessing candidates, and F5 (Phase 2) re-confirms the winner with the real
backbone anyway.

---

## Phase 2 — preprocessing and imbalance (Steps 6, 8)

### F4 — Diffusion preprocessing is now usable in real training · `done` (mechanism)

**Specification** Step 6 selects an edge-preserving preprocessing module, and Steps 18
and 21 depend on it having actually been applied.

**Notebook** Selected diffusion (iterations 10, kappa 15) in Step 6 and then never
applied it. Steps 7 onward used plain resize and normalise, so ablation rows A2–A6, all
of which read "Diffusion + …", contained no diffusion at all.

**Now** Preprocessing is a cached data stage. `src/prepare_dataset.py` materialises a
recipe into `data/processed/<recipe>/`, mirroring the raw tree's relative layout so the
single split table addresses raw images and every recipe alike. `data.recipe=<name>`
then trains on it.

Caching is what makes the fix feasible rather than merely tidy: diffusion at 10
iterations costs order 0.1 s per image, which applied on the fly would dominate every
epoch of every run — the practical reason the notebook's selection was never adopted.

Recipe names encode their parameters (`diffusion_i10_k15`), so two configurations cannot
share a cache directory. `raw` and `conventional` are recognised as identity recipes and
deliberately materialise nothing: they differ only in the Step 5 intensity treatment, so
copying the dataset twice would waste disk and invite drift.

**Still to come**: the A0–A6 runs that consume these mirrors land in Phase 8.

### F4a — Diffusion filters luminance once, not RGB three times · `done`

The notebook replicated grayscale to three channels and then ran diffusion
independently on each identical channel — triple the cost for an identical result.
Filtering happens once on the luminance channel, then replicates.

### F4b — One diffusion implementation, and the duplicate was harmless · `done`

The notebook defined `anisotropic_diffusion` twice (cells 29 and 31) with the north and
south `np.roll` directions swapped, which looked like a genuine inconsistency.

**It was not.** All four directional terms are summed and the coefficient depends only on
`|delta|`, so swapping the two labels yields an identical result. The discrepancy was
cosmetic. `tests/test_preprocessing.py::test_north_south_roll_direction_is_cosmetic`
demonstrates this against the reference formulation, so the claim is verified rather
than asserted. One implementation now replaces both.

**Correction**: `docs/IMPLEMENTATION_PLAN.md` §2 (F4) implies this discrepancy mattered.
It did not — no result was ever affected by it.

**Known minor artefact, retained**: `np.roll` wraps at the image border rather than
replicating the edge. On MRI slices the border is background, so the effect is
negligible; kept for fidelity to the reference and recorded here rather than silently
changed.

### F5 — Step 6 selection is explicitly a proxy, with its limits recorded · `done`

**Specification** Step 6: *"Do not choose preprocessing based only on visual appearance.
Select it using validation performance and boundary/texture preservation checks."*

`src/analysis/preprocessing_study.py` reports both criteria: validation macro-F1 from a
trained proxy model decides the ranking, and a Sobel edge-preservation score is reported
alongside so a candidate that wins by blurring detail away is visible as such.

The proxy protocol — `SmallCNN`, 128 px, 200 images per class — matches the reference so
the sweep stays affordable across eleven-plus candidates. Unlike the reference, the
summary states this limitation explicitly and names the confirmation step: re-run the top
candidates with the real backbone on the full validation split before committing. Those
confirmation runs are ordinary training runs (`data.recipe=<name>`) and are scheduled for
Phase 3.

### F6a — Step 8 quantifies what the focal-loss correction changed · `done`

The Step 8 ablation runs the corrected focal loss and, alongside it, the notebook's
formulation as `focal_loss_legacy`. The summary reports the macro-F1 difference between
them, so the effect of F6 on this dataset is measured rather than assumed.

### D6 — Combining imbalance strategies requires two criteria, not one · `done`

**Specification** Step 8: *"Use more than one strategy only when ablation confirms
benefit."*

A first implementation required a combined arm to beat its components on macro-F1. That
guard was vacuous: an arm that ranks first beats everything on macro-F1 by definition, so
the branch was unreachable.

The rule now requires a combined arm to beat every component on **both** macro-F1 and
worst-class recall. This is what catches the failure the notebook actually hit — its
combined arm looked acceptable in aggregate while collapsing Meningioma recall to 0.373 —
and it is why the specification names class-wise recall as a judging criterion. Both
branches are unit-tested.

### D7 — Preprocessing sweeps filter in memory, not on disk · `done`

The Step 6 sweep applies candidates on the fly to a balanced subset rather than
materialising thirteen full mirrors. A selection sweep touches each candidate once, so
caching them all would cost far more disk and time than it saves. Only the winning recipe
is materialised, by `src/prepare_dataset.py`.

### D8 — Selection studies never touch the internal test split · `done`

`BTMRIProxyDataModule.test_dataloader` returns the validation loader. Step 16 requires
the internal test set to stay unseen until the final model is evaluated once, and a
selection study has no legitimate reason to read it. Unit-tested.

---

## Environment and template repairs

These are not methodological, but they changed files and are recorded for traceability.

### E1 — Checkpoints made loadable under torch ≥ 2.6 · `done`

`torch.load` now defaults to `weights_only=True`. Lightning stores the
`save_hyperparameters` payload inside the checkpoint, and Hydra supplies optimizers and
schedulers as `functools.partial` and configs as `omegaconf` containers — so training
completed, the checkpoint was written, and only the reload for the test pass failed.

Two changes:

1. Modules exclude `net`, `criterion`, `optimizer` and `scheduler` from
   `save_hyperparameters`, keeping them as plain attributes. Checkpoints are always
   reloaded by reconstructing the model from config and loading the state dict, so
   nothing is lost. Applied to `MRIClassificationModule` and to the template's
   `MNISTLitModule`, which had the same latent defect.
2. `src/utils/serialization.py` allow-lists the remaining types our own checkpoints
   legitimately contain (`omegaconf` containers, `typing.Any`, `collections.defaultdict`),
   registered on import of `src.utils`.

### E2 — `pkg_resources` removed from test helpers · `done`

`tests/helpers/{package_available,run_if}.py` imported `pkg_resources`, which setuptools
removed and which is absent on Python 3.12+. On the pristine template the entire test
suite failed to collect. Replaced with `importlib.metadata`.

### E3 — DDP worker processes could not load checkpoints · `done`

Symptom: `tests/test_train.py::test_train_ddp_sim` failed in the spawned child with
`Unsupported global: omegaconf.listconfig.ListConfig`, despite the parent having
registered it.

Cause: `torch.serialization`'s allow-list is per-process. Lightning's DDP launcher spawns
a fresh interpreter that unpickles the model class directly. `MNISTLitModule` imports
nothing from `src.utils`, so the child reached `torch.load` with an empty allow-list.
`MRIClassificationModule` does import `src.utils`, which is why our own models were
unaffected and the defect only surfaced through the template example.

Fix: the registration moved to `src/__init__.py`, so it runs however the package is
entered. This makes DDP viable for the project's own models rather than only fixing the
example. Suite now green.

### E4 — Dependencies added · `done`

`requirements.txt` gained pandas, scikit-learn, scipy, Pillow, numpy, opencv,
scikit-image, SimpleITK, h5py, matplotlib, seaborn and kaggle. PennyLane, shap,
umap-learn and statsmodels are listed but **commented out**, to be enabled by the phase
that needs them — PennyLane only after its Python 3.13 support is verified.

Installed into `env/`: pandas 3.0.5, scikit-learn 1.9.0, scikit-image 0.26.0, Pillow,
matplotlib. Note `torch 2.13.0+cpu` — **this environment has no CUDA**, which is why
execution is limited to smoke tests.
