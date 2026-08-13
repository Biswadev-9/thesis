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

**Reason 1 — the source images are natively 224 × 224.** Confirmed by the Step 4 audit on
the real dataset: all 6,597 images are exactly 224×224, RGB, 8-bit. Resizing to 256 would
*upsample every image in the study*, fabricating detail that was never acquired and paying
about 30 % more compute per forward pass for it. 224 is not a compromise here; it is the
native resolution.

**Reason 2 — the Transformer baselines require it.** `torchvision.models.vit_b_16` and
`swin_t` carry positional embeddings baked for 224 and cannot accept 256 without
interpolating them, which would change Baselines 4 and 4b relative to their published
ImageNet behaviour and confound exactly the comparison Step 9 exists to make.

The specification says "such as", so 256 reads as illustrative rather than mandatory.

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

**Outcome on the real dataset**: the validation **fails** — up to 5.1 % of bright tissue
would be lost on the worst-case training image, against a 0.1 % tolerance. Cropping
therefore stays disabled, and this is now an evidence-backed decision rather than the
notebook's silent omission. Worth stating in the methods section: background cropping was
implemented, tested against the specification's own condition, and rejected on measurement.

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

## Phase 3 — baselines and the fixed protocol (Steps 9, 15)

### F8 — One protocol, applied to everything · `done`

**Specification** Step 15: *"The training protocol must be fixed before final testing"*,
with AdamW or Adam, lr from {1e-4, 3e-4} tuned on validation, batch 16 or 32, early
stopping with patience 10–15, cosine annealing or ReduceLROnPlateau, selection on
validation macro-F1 or balanced accuracy, and *"at least three seeds for the final model
and major baselines"*.

**Notebook** Locked a protocol for the final model only. Its Step 9 baselines had already
been trained with `patience=5` and a **single seed**, before the protocol existed — so the
baselines were not strictly comparable to the model they benchmarked. The notebook flagged
this itself as an optional follow-up and never did it.

**Now** `configs/protocol/fixed.yaml` is the single source. It is a `@package _global_`
config listed after the groups it constrains, so it overrides each model's own defaults,
and before `experiment`, so an experiment can still amend it deliberately. Every Step 9
baseline, every branch and the final classifier compose against it.

`tests/test_baselines.py` asserts the protocol against the specification's clauses
directly — optimiser family, learning rate, batch size, patience range, scheduler,
selection metric — so a later edit that silently violates Step 15 fails the suite rather
than quietly changing results.

Three seeds for all seven baselines is one multirun:

```
python src/train.py -m experiment=step09_baselines model='glob(baseline_*)' seed=42,123,7 trainer=gpu
```

### F11 — Training time and memory are measured during training · `done` (mechanism)

**Specification** Step 20: report *"trainable parameters, inference time, training time,
memory usage, and performance metrics"*.

**Notebook** Reported parameter counts and inference time only. Training cost and memory
were never captured, and by the time Step 20 ran the models had long since trained.

**Now** `src/utils/resource_monitor.py` attaches to every run from Step 9 onward and
writes `resource_usage.json` into the Hydra run directory, beside `checkpoints/`. It
records wall-clock training time, per-epoch times, total and trainable parameter counts,
and CUDA peak memory. Capturing it during training is the only way to have it later
without retraining everything. The Step 20 aggregation lands in Phase 7.

### D9 — Seven baselines retained, not six · `done`

The specification lists six. The notebook built seven, adding Swin-T alongside ViT where
Step 9 offers "ViT **or** Swin Transformer". Both are kept: the extra run is cheap and it
strengthens the Transformer comparison rather than weakening it.

### D10 — One wrapper for four pretrained architectures · `done`

ResNet50, EfficientNet-B0, ViT-B/16 and Swin-T have materially different forward paths —
ViT and Swin pool tokens internally. `TransferBackbone` replaces each model's
classification head with `Identity`, so all four emit a plain feature vector and satisfy
the D1 `extract` contract identically. Feature widths are **read from the head being
replaced** rather than hard-coded, so a torchvision definition change surfaces as a test
failure instead of a silent mismatch.

### Observation — "fine-tune the last few blocks" is not equally careful across backbones

Following the reference's per-architecture unfreezing depth, the fraction of backbone
parameters left trainable varies far more than the phrasing suggests:

| Baseline | Unfrozen | Trainable / total |
|---|---|---|
| ResNet50 | `layer4` | 14,964,736 / 23,508,032 — 64 % |
| EfficientNet-B0 | `features[-3:]` | 3,155,740 / 4,007,548 — **79 %** |
| ViT-B/16 | last 2 encoder layers | 14,175,744 / 85,798,656 — 17 % |
| Swin-T | final stage | 15,366,576 / 27,519,354 — 56 % |

EfficientNet-B0 is the outlier: its last three feature blocks hold most of the network, so
"fine-tune only the last few blocks" leaves nearly four fifths of it trainable. That is
worth stating because EfficientNet-B0 is **also the Step 10 classical branch backbone**,
and Step 10 asks for dropout and weight decay specifically "to reduce overfitting". The
reference behaviour is preserved rather than changed — this is recorded so the overfitting
discussion can be accurate, and `tests/test_baselines.py` pins the figures per
architecture so they cannot drift unnoticed.

### PennyLane gate — cleared

The plan flagged PennyLane on Python 3.13 / torch 2.13 as the project's highest risk,
gating Baseline 5 and all of Phase 4. **Verified working**: PennyLane 0.45.1, with
`qml.qnn.TorchLayer` forward and backward passes and finite gradients reaching the input.
Now an active dependency in `requirements.txt`.

`src/models/components/quantum.py` provides all five Step 12 circuit variants up front —
two depths (2 and 4 layers) and two entanglement patterns (basic ring vs
strongly-entangling), plus data re-uploading — so Step 12's requirement to "test at least
two circuit depths and at least two entanglement patterns" is satisfiable. Step 9 uses the
fixed circuit only; Phase 4 adds the adaptive mixture over all five.

**Cost, measured**: a single `fast_dev_run` batch through the fixed QCNN took 13 s against
under 2 s for the CNN baselines. The simulator runs on CPU, so every forward pass moves
tensors off the accelerator and back. Quantum models therefore cannot use mixed precision,
are impractical under DDP, and will dominate the wall-clock time of any sweep containing
them. Budget for this when scheduling Steps 11–12.

---

## Phase 4 — the three branches (Steps 10, 11, 12)

Steps 11 and 12 **follow the reference notebook rather than the specification**, by
explicit instruction. Entries here record where notebook and specification differ, so the
write-up can be accurate about which was followed.

### Step 10 — classical branch reuses the Step 9 wrapper · `done`

EfficientNet-B0 with features taken from global average pooling (1280-d), exactly as the
notebook. It is configured as the same `TransferBackbone` used for Baseline 3, so the
branch and the baseline are provably the same architecture rather than two similar
definitions that could drift apart.

One deliberate difference: the notebook applied dropout *before* returning the features,
making the extracted embedding stochastic. Here dropout sits between the features and the
head, so `features` are deterministic. At extraction time the branch is frozen and in eval
mode, where dropout is the identity, so the cached features are identical either way — but
the deterministic version cannot be accidentally sampled in train mode.

### Step 11 — paths are 3x3 / 5x5 / dilated, not 3x3 / 5x5 / 7x7 · notebook

The specification suggests "3 x 3, 5 x 5, and 7 x 7, or ... dilated convolutions with
different dilation rates". The notebook mixes the two: two plain kernels plus one dilated
3x3 at dilation 3. That reaches a 7x7 receptive field at 3x3 parameter cost, so the
intended fine/medium/broad span is preserved. Followed as in the notebook; asserted in
`tests/test_branches.py` so the three paths cannot silently collapse to the same reach.

### Step 11 — the ungated arm projects back to the shared width · `done`

Arm 4 concatenates all three paths, which would give it three times the feature width of
every other arm and therefore a larger classifier head. A 1x1 projection returns it to the
shared width, so the ablation compares *gating* rather than head capacity. The notebook
does the same; it is recorded because the alternative is an easy and invisible mistake.

### Step 11 — the spatial gate is larger than its control · recorded

Arm 6's gate head (two 1x1 convolutions) has more parameters than Arm 5's gate
(two linear layers on pooled features). The arms are otherwise identical — same stem, same
three paths, verified parameter-shape-by-shape in the tests. Any advantage Arm 6 shows
should therefore be reported alongside the fact that it is also the slightly larger module.
Both remain far smaller than the classifier they feed.

### Step 12 — a learned mixture, not a selected circuit · notebook

The specification says to test the circuit variants and that "the final model should use
only the configuration selected by validation performance" — i.e. pick one offline. The
notebook instead learns a per-image softmax mixture over all five circuits. Followed as in
the notebook.

Worth stating plainly in the write-up: this is a *different and stronger* claim than the
specification asks for, and it costs five circuit evaluations per forward pass rather than
one. The mixture weights are exposed as `quantum_weights` so the analysis can report which
circuits the model actually relies on, and whether the mixture degenerates onto a single
expert — in which case the specification's simpler design would have sufficed.

### F11a — class weights are no longer written into checkpoints · `done`

Found by reloading a trained branch: `criterion.class_weights` was a persistent buffer, so
it entered the checkpoint, and a freshly constructed module — whose buffer is still `None`
and has no matching key — could not load it under `strict=True`. Every branch reload from
Step 13 onward would have hit this.

The buffer is now non-persistent. It still moves with `.to(device)`, but stays out of the
checkpoint, which is correct: class weights are *derived* from the training split by
`MRIClassificationModule.setup`, not learned. Pinned by
`tests/test_checkpoints.py::test_checkpoints_carry_no_derived_class_weights`.

### D11 — freezing means `requires_grad = False` **and** `.eval()` · `done`

`src.utils.checkpoints.freeze` does both. Disabling gradients alone leaves BatchNorm
updating its running statistics on every forward pass, so a nominally frozen branch would
produce drifting features — silently corrupting the cached features Steps 13-15 are built
on, in a way that no error would report. Pinned by a test that asserts the running mean is
unchanged after a forward pass.

### D12 — Step 11's morphology analysis states its limits in its own output · `done`

The notebook's morphology conclusions rest on three weak points. All three are now emitted
in the analysis summary rather than living only in prose:

1. the tumour region is an **Otsu intensity proxy**, not a segmentation mask — this dataset
   ships none;
2. the three gate weights are a **softmax and sum to 1**, so they are not independent; a
   negative correlation on one path is partly the arithmetic consequence of positive
   correlations on the others, not separate evidence;
3. **pooled correlations can be manufactured** by between-class differences in both lesion
   size and gate behaviour, so correlations are reported per class as well, and only
   directions holding *within* classes should be claimed.

`No-tumor` is excluded from the correlation entirely: there is no lesion whose extent could
correlate with anything.

### D13 — silhouette is computed in feature space, not on the projection · `done`

Step 10's separability analysis scores the raw embeddings. t-SNE and UMAP do not preserve
global distances, so a silhouette taken on their 2-D output measures the projection's
layout rather than the features' separability. The projection is kept for the figure only.

---

## Phase 5 — fusion and the final classifier (Steps 13, 14, 15)

### F4b — the feature cache · `done`

Architecture decision D3 from the plan, now built. `src/extract_features.py` runs the
frozen branches once and writes `data/features/<tag>/{train,val,test}.pt`.

Steps 13, 14, 15 and 20 all train many small heads over the *same* frozen outputs.
Recomputing them per epoch would re-run EfficientNet-B0 and five quantum circuits every
time, and the quantum branch is already the slowest component in the study. Extracting
once removes the CPU simulator from the training loop entirely.

**Augmentation is disabled during extraction.** Cached features must be deterministic;
with augmentation on, every fusion head would train on a different random view of the same
image and the cache would silently mean nothing.

### D14 — the spatial features come from the Step 12 model, not the Step 11 checkpoint · notebook

`TriBranchExtractor` reads the 32-d spatial features from *inside* the Step 12
adaptive-quantum model, which carries its own spatial-gate branch trained jointly with the
quantum mixture. The separately trained Step 11 checkpoint feeds only the arm ablation and
the gate-morphology analysis.

Inherited from the reference notebook and preserved deliberately - substituting the Step 11
weights would change every downstream result. It reads like a bug, so it is documented in
the extractor's own docstring and asserted in the tests.

### D15 — every fusion strategy projects before fusing · `done`

The branches arrive at 1280, 32 and 4 dimensions. Concatenating them raw would make the
classical branch 97 % of the fused vector, and any "contribution" measured afterwards would
largely be measuring width. All three strategies project to a shared 64-d space first, so
the comparison is about information rather than dimensionality. The notebook does the same;
recorded because the alternative is an easy and invisible mistake.

Worth noting when reading Step 13's table: **gated fusion fuses by weighted sum**, so its
classifier sees 64 dimensions where concat and SE see 192. It is therefore the smallest of
the three heads, and its score should be read with that in mind. Parameter counts are
reported per strategy for exactly this reason.

### D16 — concatenation is displaced only on evidence · `done`

Step 13: *"Then add attention-based or gated fusion only if it improves validation
performance."* The selection rule keeps concatenation unless a more complex strategy beats
it by more than `improvement_threshold` on validation macro-F1. Setting the threshold above
0 guards against adopting attention or gating on noise.

### F7 — the loss is selected on validation, from the permitted set · `done`

**Specification** Step 14: *"Loss: class-weighted cross-entropy or focal loss based on
validation results."*

**Notebook** Selected **plain cross-entropy** - which the clause does not permit - and
selected it by comparing **test** macro-F1 after finding a three-way validation tie
(0.9897 for all three candidates). It then reported that same test set as the final
result, which inflates the reported performance and violates Step 16's requirement that
the test set stay unseen.

**Now** `src/analysis/loss_selection.py` enforces both halves:

- Only `weighted_ce` and `focal` are *selectable*. Plain CE is trained and reported as a
  reference row - it is useful to know what the imbalance handling buys - but cannot win.
  A run containing only reference losses raises rather than silently selecting one.
- The tie-break is fixed in advance and never reads test data: validation macro-F1 →
  validation balanced accuracy → **lower validation ECE**.

The calibration tie-break is a deliberate choice rather than a coin toss. When two losses
classify equally well, the better-calibrated one is the more useful model - and the
notebook's own spot-check found a confidently *wrong* prediction (true Glioma, predicted
Meningioma at probability 1.000), which is exactly what poor calibration looks like.

Four tests pin this, including that plain CE loses even when it is best on every metric,
and that a full tie resolves on ECE.

**Consequence**: the selected loss will most likely differ from the notebook's, and
Step 14's table changes.

### D17 — branch contribution is measured by zeroing, not by removal · `done`

The Step 13 ablation replaces a branch's features with zeros rather than deleting the
branch and shrinking the head. Architecture, parameter count and training protocol stay
identical, so the measured drop is attributable to that branch's *information*. Deleting
the branch instead would confound information with capacity. Step 20 reuses the same
mechanism for the no-quantum control.

A near-zero or negative contribution is a finding, not a failure: it means the branch
carries nothing the others do not already supply.

---

## Phase 6 — evaluation (Steps 16, 17, 18)

### F9 — the test set is spent once, and that is enforced · `done`

**Specification** Step 16: *"evaluate the final model **once** on the internal test set.
The test set should remain unseen during training and hyperparameter tuning."*

**Notebook** Read test metrics while choosing the Step 14 loss, then reported that same
test set as the final result.

**Now** `src/analysis/internal_test.py` writes a `test_evaluated.lock` beside the fusion
checkpoint. A second evaluation of the same checkpoint raises, with a message pointing at
the lock and explaining the two legitimate options. `analysis.force=true` overrides it and
records `"forced": true` in the lock's history, so a result that is no longer a single-use
estimate is visible as such rather than indistinguishable from a clean one.

Verified end to end: first run succeeds, second is refused, third with `force` succeeds and
the lock accumulates both entries.

### D18 — Step 17 reports the restricted *and* unrestricted score · `done`

Figshare has no non-tumour class, so Step 17 permits a three-class external task.
Restricting the argmax to the three present classes is standard, and it is what the
notebook did - but it is also a *favourable* choice, because it silently forgives every
case where the model would have answered "No-tumor".

Both are now reported, along with `predicted_absent_class_count`. A large gap between them
means the model frequently reaches for a class that cannot be correct here, which is itself
a finding about transfer. Reporting only the restricted figure would overstate
generalisation.

The internal-to-external drop is computed against the internal per-class F1 **restricted to
the same three classes**, so it measures domain shift rather than the absent class.

### D19 — Figshare labels are mapped through class names, not indices · `done`

Figshare encodes 1=meningioma, 2=glioma, 3=pituitary; this project's Step 1 mapping is
glioma=0, meningioma=1, pituitary=2. Mapping index to index would silently swap glioma and
meningioma and produce a plausible-looking but meaningless confusion matrix. The mapping
goes through the canonical class names and is unit-tested.

Figshare scans are 16-bit with varying intensity ranges, so each is min-max normalised to
8-bit before the shared pipeline. Applying the internal set's fixed normalisation to raw
16-bit values would inflate the apparent domain shift for a reason unrelated to the model.

### D20 — degradation happens before preprocessing · `done`

Step 18 asks whether diffusion preprocessing "improves robustness under noisy inputs". That
only means anything if the filter sees the noise, so `DegradedDataset` runs
**raw → degrade → preprocess → Step 5 transform**. Reading from a pre-materialised recipe
mirror and degrading afterwards would corrupt an already-denoised image and answer the
opposite question. It also matches deployment: a noisy scan arrives, the pipeline denoises
it, the model classifies it.

The notebook could only apply diffusion as a *test-time* filter because it never trained
with it; with F4 in place this is now a genuine trained-in comparison.

### D21 — noise is deterministic per image, shared across models · `done`

Gaussian noise takes a per-image seed derived from the sample index. Every model in the
comparison therefore sees the *same* corrupted pixels, and a model cannot look more robust
by drawing an easier noise sample. Unit-tested in both directions: identical across calls
for a given index, different between indices.

### D22 — robustness is reported as a drop, not only as a score · `done`

Each condition is reported as absolute macro-F1 and as the drop from that model's **own**
clean baseline. The drop is the robustness measure: a weaker model that degrades gently is
more robust than a stronger one that collapses, and absolute scores hide that. Unit-tested
with exactly that scenario.

### D23 — the shipped "Challenging Datasets" are excluded · `done`

The archive ships `Challenging Datasets/{Blurred,Noisy,Patient Motion Artifact}/`, each
containing the same four class folders - superficially an ideal real-world robustness set,
and initially identified as a bonus for Step 18.

**They are not usable.** Their filenames (`bilateral_glioma (1).jpg`) do not correspond to
the primary set's (`BT-MRI GL Train (1).jpg`), and there are 3,354 of them against 7,023
originals, so there is no way to determine which are degraded copies of *training* images.
Evaluating on them could silently score the model on its own training data and inflate the
robustness result.

The synthetic degradations are applied to the held-out test split only, where provenance is
known. If the dataset authors publish a filename correspondence, this decision is worth
revisiting - real acquisition artefacts would be stronger evidence than synthetic ones.

### D24 — no `torch.no_grad` in the full pipeline's forward path · `done`

`FullPipeline` freezes the branches with `requires_grad = False`, which stops their weights
updating, but deliberately does **not** wrap the forward in `no_grad`. Doing both would also
block gradients from reaching the input, breaking Phase 7's Grad-CAM - and the failure would
present as a uniformly blank saliency map rather than as an error. A test asserts gradients
reach the input pixels and are not identically zero.

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
