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

## Phase 7 — explainability and quantum advantage (Steps 19, 20)

### F10 — attention rollout for the Transformer baseline · `done`

**Specification** Step 19: *"Use attention rollout or attention maps for Transformer-based
components."*

**Notebook** Never explained its ViT or Swin baselines at all.

**Now** `AttentionCapture` temporarily wraps each `nn.MultiheadAttention` forward to
request weights `torchvision` otherwise computes and discards, then restores the originals
so the model is left exactly as found. `attention_rollout` composes the layers with the
identity term that accounts for residual connections - without it, a layer with diffuse
attention would appear to erase all signal.

### F11 — the efficiency table is complete · `done`

Step 20 lists trainable parameters, inference time, **training time** and **memory usage**.
Parameters and inference time are measured directly; training time and peak memory are read
from each run's `resource_usage.json`, which `ResourceMonitor` has been writing since
Phase 3. Measuring during training was the only way to have them without retraining.

### F12 — a properly powered paired test · `done`

**Notebook** Ran a Wilcoxon signed-rank test over **four** per-class F1 values. That design
cannot reach significance at any effect size: the two-sided floor for n=4 is 0.125.

**Now** `src/utils/statistics.py` provides a **paired bootstrap** over test predictions -
resampling sample indices and scoring both models on the same draw, which preserves the
correlation between two models that saw identical images. Over ~1000 test samples it
resolves differences of a fraction of a point. McNemar sits alongside it, switching to the
exact binomial test when discordant pairs are few, where the chi-square approximation is
unreliable.

`wilcoxon_paired` **refuses** samples below six pairs and returns an explanation instead of
an uninformative p-value. Pinned by a test using the notebook's exact four-value scenario.

### D25 — the no-quantum control is retrained, not masked · `done`

Step 20 compares the full model against "the same architecture after removing the quantum
branch". Zeroing the quantum features on a head *already trained with them present*
measures disruption - the head has learned to rely on inputs that suddenly vanished - not
contribution. The control is therefore retrained from scratch on zeroed features, which
asks the question the specification poses.

### D26 — Step 20 is built to be able to return "no" · `done`

The specification states plainly that "if the quantum branch does not outperform strong
baselines, report the result honestly", and lists parameter efficiency, robustness and
interpretability as fallback criteria. The verdict is therefore *generated from the
evidence*, distinguishing three cases: significant improvement, nominal improvement whose
bootstrap interval spans zero, and no improvement. Each is phrased for direct use in the
write-up, with the fallback criteria attached.

### D27 — Grad-CAM re-enables gradients on its own input · `done`

Found during the Phase 7 smoke run. Every branch in the pipeline is frozen with
`requires_grad = False`; with an input that is also not a grad-requiring leaf, autograd
prunes the whole subgraph, the backward hook never fires, and Grad-CAM has no gradients to
weight activations by.

`GradCAM` now makes the input a grad-requiring leaf itself rather than relying on callers
to remember. Its two failure modes are also distinguished in the error message - "no
activations" (wrong layer) versus "no gradients" (pruned graph) - because they have
different causes and different fixes. A test exercises the fully frozen pipeline, which is
the real usage.

### D28 — saliency hooks are removed even on failure · `done`

Both `GradCAM` and `AttentionCapture` are context managers that clean up in `__exit__`,
including after an exception. The notebook registered hooks at module scope and never
detached them, so every later forward pass kept writing into stale buffers - which produces
*wrong saliency maps* rather than an error. Two tests cover the normal and the exception
path.

Related: the MC-dropout pass restores every dropout layer to eval mode in a `finally`
block. Leaving them in train mode would silently randomise every analysis that ran
afterwards.

### D29 — Step 20's control is trained under the Step 15 protocol, not a slacker one · `done`

Found by auditing Step 20 against Step 15 rather than against its own config. The control
had been given its own training settings, and every difference ran the same direction: it
trained for fewer epochs, without early stopping, without best-checkpoint selection, at a
different learning rate, without the weighted sampler, and it kept its final weights rather
than its best. A handicapped control does not measure the quantum branch - it manufactures
a positive result. Six mismatches, all corrected in Step 20's favour of Step 15.

Step 15 is the source of truth and was **not** modified. Its real protocol is what
`configs/protocol/fixed.yaml` composes, which is not what the YAML files read individually
say: `configs/model/final_classifier.yaml` declares `lr: 1e-3`, and the protocol overrides
it globally to `1e-4`. The values were established by composing the config, not by reading
it.

Two of the six could not be fixed by copying a number:

- **The loss is not a constant.** Step 15 has no fixed loss - `scripts/kaggle_pipeline.py`
  reads Step 14's `selected_loss` at run time and passes `loss@model.criterion=<name>`, so
  hard-coding today's answer (`weighted_ce`) in Step 20 would diverge silently the day
  Step 14 chose `focal`. Step 20 now resolves the loss the same way, from
  `analysis.loss_summary`, with `analysis.loss` mirroring the pipeline's `--loss` flag. The
  config's criterion remains only as a fallback, and taking it logs a warning and records
  `loss_provenance.source: "…(unverified)"` in the summary, so an unverified comparison is
  never mistaken for a verified one.
- **The seed had to be shown to be fair before it could be matched.** The full profile
  trains three seeds, which raised the question of whether one control was being compared
  against the best of three. It is not: every downstream stage reads `pipe.seeds[0]`, a
  fixed position, so the checkpoint under evaluation comes from one predetermined seed and
  a single seed-matched control is the correct comparison. The control trains at
  `analysis.seed`, and `seed_check` in the summary flags it if a reordered `--seeds` ever
  breaks the match.

The duplication of Step 15's protocol into Step 20's config is unavoidable - Hydra cannot
reach a training experiment's composition from an analysis config - so
`tests/test_protocol_consistency.py` composes the real Step 15 config and asserts every
value matches. Changing the protocol now fails the suite instead of quietly invalidating
Step 20. Verified by perturbation: altering `max_epochs`, `lr`, `use_class_weights`,
`patience`, `dropout` or `T_max`, breaking the Step 14 loss lookup, or switching
`seeds[0]` to `seeds[-1]` each fails one or two tests.

### D31 — Step 14's selection reaches Step 20 through the runner, not through a human · `done`

Steps 19 and 20 were the only implemented steps the pipeline did not run, so Step 20 had to
be launched by hand with `analysis.loss_summary=<step14 summary>` attached. D29 made a
forgotten flag *visible* in the output; it could not make it impossible. Both stages are
now in the graph, after Step 18, and the runner supplies the summary path itself. If
Step 14 has not run and no `--loss` was given, the stage refuses to build with the same
error Step 15 raises - failing loudly beats a defensible-looking quantum-advantage number
whose control trained on an unverified objective.

The *path* is passed, not the resolved name, so Step 14 remains the single place the answer
lives and Step 20's summary records which file it read. `--loss` short-circuits both stages
identically, so they cannot disagree.

Neither stage can influence anything upstream: both only read finalized checkpoints, and a
test asserts they stay after Steps 14-18 in the graph. The shortened profiles thin Step 19's
SHAP and MC-dropout sampling and Step 20's bootstrap, which only widens intervals. Step 20's
control keeps Step 15's full training protocol in **every** profile - a control trained for
fewer epochs would lose for a reason unrelated to the quantum branch, which is a fake
positive even in a run already marked unreportable.

Found while wiring this: Step 20's config carries only `fusion_ckpt` - the branches reach it
through the feature cache - so passing it the classical and quantum checkpoints made Hydra
reject the whole stage. On Kaggle that surfaces hours into a run. A test now composes both
stages from the overrides the pipeline actually emits.

### D30 — Step 18 builds its degraded datamodule from the real dataset paths · `done`

`DegradedTestDataModule` was constructed without `data_dir`, `raw_subdir` or
`split_subpath`, so it fell back to defaults. Whenever the study ran against a dataset at a
non-default location - which is every Kaggle run - robustness would have been measured on
the wrong data or failed outright. The paths are now taken from the source datamodule the
study was given. Two tests record what is passed through and assert it matches.

---

## Phase 8 — ablation, RQ mapping and statistics (Steps 21, 22, 23)

### D32 — A2-A6 keep diffusion, and row P is added for the shipped model · `done`

**Specification** Step 21 writes "Diffusion" into rows A2-A6, and A6 reads as the full
proposed model.

**Reality** Step 6 selected CLAHE on measurement (clahe 0.257, `diffusion_i10_k15` 0.247,
conventional 0.100), so the model the study ships is CLAHE-based. Taken literally the
ablation ladder terminates at a configuration the study does not ship.

**Now** Both are represented. A2-A6 stay on diffusion exactly as written, and a tenth row
**P** carries the shipped model. Substituting the selection into A2-A6 would have deleted
the study's diffusion evidence while leaving the labels claiming it; substituting diffusion
into P would have misreported what was trained.

The gap is useful rather than awkward. Step 6's own summary carries the caveat that its
ranking comes from "a reduced-scale proxy (SmallCNN, 128px, 24 images/class)" and "should
be confirmed with the real backbone on the full validation split". A7 against P is that
confirmation, at full scale. It is reported descriptively, not as a hypothesis test.

Which diffusion configuration is read from Step 6's `ranking`, not hard-coded, so a better
variant cannot be silently ignored.

### D33 — A6 uses plain CE so that A7 measures something · `done`

A7 reads "core model + imbalance-aware loss". The core model already carries the loss
Step 14 selected, so as literally specified A7 and A6 are the same configuration and their
delta is zero by construction - a number that looks like evidence and is not. A6 is
therefore pinned to `plain_ce` and A7 to whatever Step 14 selected, making H3 a real
comparison. This defines the ablation rows; **Step 15 is untouched** and still trains with
Step 14's selection.

### D34 — A8 gets no fabricated performance delta · `done`

Explanations change no weights, so A8's classification metrics are A7's *by construction*.
Step 21 mirrors A7's record rather than recomputing it - re-running would be identical
arithmetic at twice the cost and a second read of the test set - and Step 23 excludes A8
from testing entirely. Its statistical contribution is Step 19's deletion/insertion and
MC-dropout output, which are not classification metrics.

### D35 — Ablation rows pin their own settings instead of inheriting selections · `done`

Every training stage in `scripts/kaggle_pipeline.py` applies `recipe_override()` and
`imbalance_overrides()`, injecting Step 6's and Step 8's *selections*. The real run confirms
it: Step 9's EfficientNet trained with `data.recipe=clahe`, not the `recipe: null` its
experiment config declares. That is right for the main study and wrong for an ablation - a
row would be labelled "diffusion" and trained on whatever won Step 6, and would change
silently when Step 6 was re-run.

Each row therefore pins recipe, normalization, augmentation, sampler and loss explicitly.
`augment=false` and `use_weighted_sampler=false` are pinned literals matching the observed
Step 8 `baseline` selection; if a full-profile Step 8 selects otherwise the rows do **not**
follow it, and the conflict is reported rather than applied.

Consequence worth stating: no existing Step 9-12 checkpoint satisfies any A-row, because
all of them were trained on CLAHE. A0-A7 are new runs.

### D36 — The primary hypothesis family is pre-registered, and small · `done`

**Specification** Step 23 asks for McNemar, paired bootstrap and confidence intervals, and
warns against overstating minor improvements. It does not say which comparisons are
hypotheses.

**Now** Four, each isolating exactly one factor, declared in
`configs/analysis/step23_statistics.yaml` before any of them is computed:

| | comparison | isolates | RQ |
|---|---|---|---|
| H1 | A2 vs A1 | diffusion vs conventional preprocessing | RQ2 |
| H2 | A5 vs A4 | adaptive vs fixed circuits | RQ4 |
| H3 | A7 vs A6 | imbalance-aware loss | RQ6 |
| H4 | A6 vs A3 | quantum + fusion over multiscale alone | RQ8 |

Holm-Bonferroni across the four at alpha=0.05. Testing every row against A6 instead would
be eight hypotheses chosen after seeing the table, and a correction applied to a family
assembled that way controls nothing. `_check_family` refuses a configured family that
differs from the registered one, because the family size scales every adjusted p-value.

Everything else - A1 vs A0, A3 vs A2, A7 vs P, the full ladder, A8 - is descriptive: an
effect size and an interval, no p-value, `significant: null` rather than `false`, because
there is no claim rather than a claim of no effect.

### D37 — Three seeds describe; the bootstrap tests · `done`

Step 23 permits Wilcoxon "for repeated fold/seed comparisons", but a two-sided Wilcoxon over
three pairs has a floor of p=0.25 and cannot reach significance at any effect size - the
same defect F12 fixed for Step 20. Seed spread is reported as mean and standard deviation
and explicitly marked `role: descriptive`; the powered paired test is the bootstrap over
the ~1000 test samples, which is where the resolution actually is. `wilcoxon_paired`
refuses below six pairs and returns the reason instead of a p-value.

Every primary comparison reports **both** its raw and its Holm-adjusted p-value, and the
verdict uses the adjusted one. The p-value is McNemar's, over discordant errors, while the
effect size and interval are macro-F1 - different questions, so `p_value_source` and an
`interpretation` field say so rather than letting the two be conflated.

### D38 — Pairing is verified rather than assumed · `done`

H4 pairs A6, whose predictions come through the feature cache, with A3, whose come through
the image loader. Both use `shuffle=False` and preserve test order today, so the pairing is
valid - but a paired test over misaligned samples returns a confident wrong p-value rather
than an error, which is exactly the failure that survives review. The two label vectors are
compared element-wise before anything is computed, and a mismatch is refused.

### D39 — Row P has no saved predictions, and none are invented · `open`

Step 21 reports P from Step 16's summary rather than re-evaluating it, which is what keeps
the once-only test budget intact - so no prediction file exists for P and there is nothing
to resample. A7-vs-P therefore degrades to a **metrics-only** comparison: a difference of
point estimates, with `ci_low`/`ci_high` null and the reason recorded.

Step 16 did save its own `test_predictions.npz`, so a paired descriptive interval is
available without any new evaluation. That is exposed as `analysis.step16_predictions` and
left **off by default**, because reading it consumes an artefact from outside Step 21 -
a scope decision rather than a technical one. Either way P stays single-seed and the
comparison stays descriptive.

Related: P is single-seed by Step 16's design while the A-rows are three-seed. No variance
is imputed for it - `seed_spread` is `None`, meaning "not estimable", not zero.

### D40 — The Step 16 metric battery is shared, not reimplemented · `done`

Step 21 must "report the same metrics for every configuration", and *same* has to mean the
same code: two implementations that agree today drift the first time one changes a
zero-division policy or an averaging mode, and the table would then compare numbers that
only look alike. `src/analysis/metric_battery.py` holds the arithmetic and both
`InternalTest` and the ablation call it. Step 16's output was captured before the
refactor and re-checked after: byte-identical.

### D41 — RQ5 is metadata-limited, and says so · `done`

**Specification** RQ5: *"Analyze performance by tumor size/appearance if metadata or masks
are available."*

The dataset ships neither. No tumour-size annotation, no appearance label, no segmentation
mask - so the subgroup half of RQ5 is not assessable, and the tempting move is to
substitute a proxy (lesion area from a threshold, say) that nobody asked for and that would
be reported as though it answered the question.

**Now** Step 22 lists the absent evidence as a row with a null value and a stated reason,
and RQ5's status is `partially_supported` with an explicit limitation. Class-wise metrics,
which *are* available, are reported as the half of the question the data can answer.

RQ3 is limited the same way: with no masks, Grad-CAM localization is assessed by
deletion/insertion sanity checks rather than against annotated boundaries.

### D42 — RQ1's baseline coverage is partial, and stated · `open`

RQ1 asks for the proposed model against "all baselines". Only two of the seven Step 9
baselines - EfficientNet-B0 and ViT - appear in an artefact Step 22 is permitted to read,
via Step 18's clean scores. The other five recorded their test metrics only in their
training runs' `metrics.csv`, and `test: True` in `configs/train.yaml` means those columns
exist for every run.

Harvesting them would be one line and would make RQ1 look complete. It is refused: a number
whose provenance is a training log cannot be audited, and mixing harvested values with
Step 21's freshly computed ones would make the table's numbers incomparable. RQ1 therefore
carries the limitation instead. Closing it means either extending the ablation ladder to
the remaining baselines or having Step 16 evaluate them - both cost test-set reads, and
both are scope decisions rather than fixes.

### D43 — Ablation rows do not inherit the study's running selections · `done`

Every training stage in `scripts/kaggle_pipeline.py` applies `recipe_override()` and
`imbalance_overrides()`, which inject Step 6's and Step 8's selections. The real run shows
the mechanism plainly: Step 9's EfficientNet trained with `data.recipe=clahe` although its
experiment config declares `recipe: null`.

That is right for the main study and wrong for an ablation. A row whose label reads
"diffusion" would train on whatever won Step 6, and the label would still read "diffusion"
in the final table. Phase 8's rows therefore bypass `_train_builder` entirely and emit
`row_overrides(...)` from the manifest, pinning recipe, normalization, augmentation,
sampler and loss per row. Twenty-four orchestration tests compose each row under Hydra and
assert the pinned values; perturbing the runner to re-apply either helper fails 17-24 of
them.

Consequence: no existing Step 9-12 checkpoint satisfies any A-row, because all of them were
trained on CLAHE. A0-A7 are new runs - 24 of them, of which A4 and A5 run circuits on the
CPU simulator.

### D44 — Phase 8 writes to its own namespace; the result bundles are immutable · `done`

Phase 8 outputs live under `logs/train/runs/step21_ablation/<row>/seed_<n>`,
`logs/analyze/runs/{step21_ablation,step23_statistics,step22_rq_mapping}` and the
`a6_diffusion` feature cache. The two shipped result bundles - `thesis_results_20260813_090056/`
and `thesis_results_20260814_075721/` - are the study's record of what was actually run, and
are never written to, overwritten or deleted. A test resolves every Phase 8 stage's output
directory and asserts none falls inside them; pointing the namespace at a bundle fails 69
tests.

### D45 — The runner points at the analyses rather than reimplementing them · `done`

Steps 21, 22 and 23 are wired as three stages whose ordering encodes their dependencies:
the eight rows, then evaluation, then statistics, then the RQ map. Each refuses to build
when its inputs are absent - Step 21 lists the missing checkpoints, Step 23 names Step 21,
Step 22 names Step 23 - so a partial run fails at the stage that is missing something rather
than producing a thinner table downstream.

The Holm correction, the hypothesis family and the RQ bindings stay in the analyses. A test
parses the runner's AST and asserts it imports no `scipy`/`sklearn` and calls none of the
statistical primitives: the one thing it imports from `src/` is the row manifest, and only
to emit overrides from it.

---

## Step 24 — receptive-field strategy ablation

### D46 — Why Step 24 exists · `done` (not yet run)

Step 11 built an eight-arm ablation of the multiscale module, and Phase 8 carries the
proposed branch as row A3. Neither answers the question the thesis actually needs answered.

Phase 8's only comparison involving the branch is **H4 (A6 vs A3)**, which changes three
things at once - backbone, quantum branch and fusion head - so nothing in it is
attributable to the receptive-field mechanism. And **every Phase 8 row from A3 upward uses
`spatial_gate`**: the gate is never varied or removed, so no row can measure its
contribution. The evidence for RQ4 therefore rests entirely on Step 11's arm sweep, which
Phase 8 references but does not re-run under Phase 8's controlled settings.

Step 24 asks one question directly: *does spatially adaptive selection among multiple
receptive fields improve classification over conventional fixed-receptive-field convolution
and over ungated multi-scale fusion?*

**No results are claimed. Step 24 has not been trained.**

### D47 — All five conditions reuse existing Step 11 arms · `done`

The ladder needed no new architecture. Every condition is an arm
`src/models/components/multiscale.py` already implements:

| Condition | Arm | Receptive field | Fusion |
|---|---|---|---|
| `FIXED_3X3` | `arm1_fixed_3x3` | single 3x3 | n/a |
| `FIXED_5X5` | `arm2_fixed_5x5` | single 5x5 | n/a |
| `FIXED_DILATED_3X3` | `arm3_fixed_dilated` | 3x3 dilation 3 (7x7 effective) | n/a |
| `MULTISCALE_NO_GATE` | `arm4_concat_nogate` | 3x3 + 5x5 + dilated | concat + learned 1x1 projection |
| `ADAPTIVE_MULTISCALE` | `arm6_spatial_gate` | 3x3 + 5x5 + dilated | per-pixel softmax gate |

Two naming corrections the implementation forces:

- **`FIXED_DILATED_3X3` is not a literal 7x7 convolution.** It is a dilated 3x3 reaching a
  7x7 field at 3x3 parameter cost. Substituting a real 7x7 would change the condition's
  capacity and stop it matching the same path inside the two multi-scale conditions.
- **`MULTISCALE_NO_GATE` is not equal-weight fusion.** It concatenates the three paths and
  learns a 1x1 projection back to the shared width. The mixer is *learned* but
  input-independent once trained. Describing it as equal-weight averaging would misstate
  what the control is. A true equal-weight arm does not exist in the repository.

Terminology throughout: the convolutions are fixed in every condition. What adapts is the
per-pixel weighting over their outputs. "Spatially adaptive multi-scale receptive-field
selection" is accurate; "dynamic kernels" and "attention" are not - there is no query-key
computation anywhere in the module.

### D48 — One formal hypothesis, in its own family · `done`

**H24: `ADAPTIVE_MULTISCALE` vs `MULTISCALE_NO_GATE`.** Both carry the identical three
receptive fields, so the gate is the only difference - which makes it the one causal
comparison in the ladder.

The three fixed-kernel comparisons (`S24a/b/c`) change the receptive field *and* the
parameter budget together, so they cannot isolate gating and are **descriptive**: effect
size and 95% interval, no p-value, `significant: null` rather than `false`.

**This family is separate from Phase 8's H1-H4 and does not touch it.** Appending would
turn four Holm-corrected tests into five and weaken every one; `StatisticalReport`'s
`_check_family` refuses exactly that, and a test asserts Step 24's identifiers do not
intersect Phase 8's. Holm over a family of one is the identity - applied and reported
anyway, with a note, because an adjusted p-value that silently equals the raw one is worse
than one that explains why.

### D49 — The conditions are not parameter-matched, and the output says so · `done`

Counts measured by building the arms (`channels=32`, `num_classes=4`):

| Condition | Parameters | vs adaptive |
|---|---|---|
| `FIXED_3X3` | 19,716 | −36,643 (−65.0%) |
| `FIXED_DILATED_3X3` | 19,716 | −36,643 (−65.0%) |
| `FIXED_5X5` | 36,100 | −20,259 (−35.9%) |
| `ADAPTIVE_MULTISCALE` | 56,359 | — |
| `MULTISCALE_NO_GATE` | **57,828** | **+1,469 (+2.6%)** |

The fixed conditions are smaller because a single path is smaller - which is precisely why
their comparisons are descriptive rather than formal.

**For H24 the relationship runs the other way: the control is the larger model.** The
ungated arm carries 2.6% more parameters than the adaptive one, so a win for adaptivity
cannot be explained by extra capacity. The adaptive gate head itself is 1,635 parameters,
2.9% of the model - most of what separates the adaptive condition from a fixed one is
*having three paths at all*, not gating. Counts are computed at run time, never tabulated
in code, and a test asserts they match the real modules.

### D50 — Nothing is inherited, and the ladder shares one recipe · `done`

`configs/experiment/step11_arm_ablation.yaml` declares `augment: true` and
`use_weighted_sampler: true`, and the pipeline then overrides both with Step 8's selection.
Fine for an arm sweep; fatal for a controlled comparison, because two conditions run under
different selections are not comparable for a reason unrelated to receptive fields.

Step 24's conditions therefore pin recipe, normalization, augmentation, sampler and loss
explicitly (`plain_ce`, `augment=false`, `use_weighted_sampler=false`,
`normalize=imagenet`), bypass `_train_builder` entirely, and share one recipe resolved from
Step 6's ranking - the same diffusion recipe the Phase 8 rows use. That makes
`ADAPTIVE_MULTISCALE` configuration-identical to Phase 8's row A3, so the two experiments
are mutually checkable.

`configs/experiment/step24_receptive_field.yaml` deliberately declares no data defaults at
all, for the same reason.

### D51 — Step 24 is appended, not merged into Phase 8 · `done`

It runs after Step 22 in its own namespace (`step24_receptive_field`), reads none of
Steps 21-23's outputs and is read by none of them. The Phase 8 dependency graph is
unchanged and nothing in Step 24 can retrain it. Fifteen training runs (5 conditions x 3
seeds), all classical - unlike Phase 8, no condition here runs a quantum simulator.

### D52 — Step 24 uses the recipe Step 6 *confirms*, and blocks until it exists · `done`

**Superseded the original D52**, which framed this as "diffusion (the specification) versus
CLAHE (what the study ships)". That framing rested on a false premise: **the study ships
nothing yet.** Auditing the artefacts showed there is no finalized preprocessing selection
at all.

Four Step 6 summaries exist and they disagree - `conventional`, `conventional`, `clahe`,
`clahe` - and *all four are reduced-scale proxy runs*. The one most often quoted came from
the smoke profile, at 0.2570 against 0.2469 on four balanced classes, where chance is 0.25;
both numbers are noise, and that run's own report is stamped "Not reportable". The
strongest run (13 candidates, 5 epochs, 200 images/class) does favour CLAHE at 0.647
against 0.5598 for the best diffusion variant - but it is still a `SmallCNN` at 128px, and
every Step 6 summary ends with the same caveat:

> *"Confirm the top candidates with the real backbone on the full validation split before
> committing."*

**`step06_confirm` has never been run.** No such run directory exists anywhere.

**And it could not have settled anything if it had.** `configs/experiment/step06_confirm.yaml`
is a *training experiment*, not an analysis: a Hydra multirun that produces one run
directory per candidate and no summary, no winner, no `selected_recipe`. It is not wired
into the pipeline, and grepping `src/`, `configs/` and `tests/` finds no consumer - its only
mention anywhere is its own docstring. Meanwhile `Pipeline.selected_recipe()` reads only the
*proxy* summary, so even after a confirmation ran, every downstream stage would still have
resolved the proxy's answer. The confirmation was advisory, and the study had no artefact
recording what it advised.

**Now** `src/analysis/preprocessing_confirmation.py` reads the completed confirmation runs,
compares them on validation macro-F1 and writes `step06_confirm_summary.json` with an
explicit `selected_recipe` and its provenance. Step 24 consumes that artefact and **nothing
else**:

- no fallback to the proxy ranking - the resolver cannot even reach it, asserted by a test
  that walks the AST of both Step 24 modules;
- with no confirmation and no explicit override, Step 24 **refuses to build**, so fifteen
  training runs cannot start on an unconfirmed preprocessing;
- `--recipe` still wins, because an explicit operator decision is not an implicit fallback;
- all five conditions receive one recipe from one context object - there is no per-condition
  recipe field that could diverge.

No winner is assumed anywhere. The analysis invents nothing: an empty sweep fails, a
single-recipe sweep is refused as not being a comparison, and runs from another experiment
are ignored (the Step 9 sweep also varies by recipe and would otherwise answer a different
question with equal confidence). Tests parametrise over several unrelated recipes precisely
so no test encodes a preprocessing decision, and the winner reverses when the scores do.

Selection is validation-only. Every run's `metrics.csv` also carries `test/*` columns -
`test: True` in `configs/train.yaml` is never overridden - and reading those would let the
internal test set decide a question that precedes Step 16. The metric is validated to start
with `val/` and anything else is rejected.

**Phase 8 is untouched.** Its A2-A6 rows still resolve the diffusion recipe from Step 6's
ranking, per the specification's Step 21 table. Whether they should also follow the
confirmation is a separate decision, still open.

Related, and inherited from Step 11: **no test proved the gate is input-dependent across
images** before Step 24. Step 11's suite checks the weight map varies across space *within*
one input; nothing checked that two different images produce different maps, which is the
property the word "adaptive" claims. Step 24 adds that test.

---

## Step 25 — quantum circuit adaptivity ablation

### D53 — Why Step 25 exists · `done` (not yet run)

Step 12 runs **five** quantum circuits on every image and combines their outputs with a
learned per-image softmax. Nothing in the study tested whether that beats committing to one
circuit.

Phase 8's H2 (A5 vs A4) looks like that test and is not: A4 is `baseline_fixed_qcnn`, a
2-conv CNN stem with 5,248 parameters, against A5's 72,408-parameter Step 12 branch. They
differ in backbone, feature dimension (4 vs 36), fusion and classifier. H2 measures
*architectures*, not circuit adaptivity.

Step 25 asks directly, with the confound removed.

**No results are claimed. Step 25 has not been trained.**

### D54 — All four conditions are the existing Step 12 class · `done`

No new architecture. `AdaptiveQuantumClassifier` already accepts `circuit_names`, and
setting it to a single circuit makes the mixture **mathematically inert** - a softmax over
one element is identically 1.0. The same class therefore expresses both the treatment and
its controls, and they cannot drift apart.

| Condition | `circuit_names` | Circuit | q-params | Total |
|---|---|---|---|---|
| `FIXED_BASIC` | `['fixed']` | 2× BasicEntangler | 8 | 72,052 |
| `FIXED_DEEP` | `['deep']` | 4× BasicEntangler | 16 | 72,060 |
| `FIXED_STRONG` | `['strong']` | 2× StronglyEntangling | 24 | 72,068 |
| `ADAPTIVE_QUANTUM` | `null` | all five, softmax-mixed | 104 | 72,408 |

Spatial branch (56,227), projection (132), fusion (concat→36) and classifier (13,508) are
**identical across all four**, verified by measurement rather than assertion.

### D55 — Terminology: a soft mixture, not dynamic selection · `done`

All five circuits execute on every forward pass. Nothing is skipped, no circuit is chosen
at run time, and there is no conditional execution - a test hooks every expert's `forward`
and asserts all five are called. The correct description is an **adaptive soft mixture of
quantum circuits**. "Dynamic circuit selection" and "conditional quantum execution" would
both misstate the mechanism *and* its cost, which is five simulator evaluations per image
against one.

### D56 — What the experiment does and does not license · `done`

The Step 11 spatial gate is hard-coded inside `AdaptiveQuantumBranch` and is **77.65% of
the parameters**; the quantum experts are **0.14%**. The gate is identical across all four
conditions, so the comparison is valid - but the claim it supports is about
*circuit-mixture adaptivity only*.

It must not be reported as showing the model as a whole is more powerful because of
adaptivity. Given the parameter share, a null result is unsurprising and remains a valid,
reportable outcome.

### D57 — Three primary comparisons, Holm-corrected · `done`

**H25a/b/c: `ADAPTIVE_QUANTUM` against each fixed circuit.** One family of three, so unlike
Step 24's family of one the correction genuinely bites: three tests at alpha=0.05 give
roughly a one-in-seven chance of a spurious positive uncorrected.

The three fixed-versus-fixed comparisons are **descriptive**. Whether depth or entanglement
alone explains a difference is a different question, and promoting them would double the
family from three to six. Separate from Phase 8's H1-H4 and Step 24's H24; a test asserts
the identifier sets do not intersect.

### D58 — Validation only; the test set stays sealed · `done`

Step 25 selects among architectures, like Steps 6, 8, 13 and 14, so it is decided on the
**validation** split and the internal test set remains Step 16's to spend once. `split` is
asserted to be `val` and anything else is refused. Every training run's `metrics.csv` also
carries `test/*`; none is read.

### D59 — Capacity is not matched, and the gap favours the treatment · `done`

The adaptive condition is the larger model by at most 356 parameters (0.49%), because it
carries five circuits' weights. Small, but in the treatment's favour - the opposite of
Step 24, where the control is larger. Counts are measured by building the models, and a
positive H25 result must be read with the gap in view.

### D60 — Adaptivity is analysed, not assumed · `done`

The claimed contribution is the per-image weighting, so the mixture weights are summarised
directly: per-circuit mean/std/min/max, normalised entropy, the dominant circuit, and
whether the weights vary between images at all.

Two failure modes matter and neither shows up in accuracy: a selector **collapsed** onto one
circuit is a fixed circuit with extra parameters, and one that stayed **uniform** is an
unweighted average. Both are detected, and tests drive each case explicitly.

Weights that vary are **not** evidence the mixture helps - only the paired comparison
answers that, and the output says so in the note a reader will quote.

### D61 — Step 25 follows Step 6's confirmed preprocessing · `done`

Step 25 does not select a recipe of its own; it uses the same authoritative confirmation
Step 24 does, with no fallback to the proxy ranking, and refuses to run without one. The
explicit `analysis.recipe` / `--recipe` development override is preserved and recorded in
the output as the recipe's source, so a development run can never be mistaken for the final
study. All four conditions receive the same recipe, structurally: one context object
supplies it and no condition carries a recipe field.

### D62 — Step 6 confirmation is wired into the graph, and gates Steps 24 and 25 · `done`

Before this, `step06_confirm` existed as an experiment config and an analysis, and was in
**no** pipeline stage. Run All would have gone proxy -> materialise -> ... -> Step 24 and
stopped there with `StageFailed`. Safe, but not a dependency - the graph did not know one
existed.

The chain is now explicit:

```
step06_preprocessing  (proxy ranking)
        |
step06_confirm_materialise/<candidate>       one per non-identity candidate
        |
step06_confirm/<candidate>/seed_<n>          real backbone, fixed protocol
        |
step06_confirm  ->  step06_confirm_summary.json     AUTHORITATIVE
        |
        +---------------> step24_receptive_field/*      (5 conditions x 3 seeds)
        +---------------> step25_quantum_circuit_ablation/*  (4 conditions x 3 seeds)
```

Enforced two ways, not one. **Ordering**: the confirmation stages sit in `step06`, long
before either consumer, and its summary stage follows its own training runs. **Refusal**:
every Step 24 and Step 25 *training* stage resolves the recipe through
`confirmed_recipe_context()` and raises `StageFailed` if the artefact is absent - so Run All
stops at the first consumer rather than proceeding on the proxy's ranking. Tests assert
both, and moving the confirmation stage to the end of the graph fails the suite.

**Resumable, not repeated.** The confirmation stages carry `.pipeline_done.json` like every
other stage, so a completed confirmation is skipped on the next invocation. No new caching
mechanism was introduced.

**The candidate set is data-driven and overridable.** `confirm_candidates()` takes the top
`--confirm-top-k` (default 3) of the proxy's own ranking and always appends the conventional
reference - a confirmation that cannot say "no preprocessing was as good" has confirmed
nothing. `--confirm-recipes` states the set explicitly instead.

**`selected_recipe` is NOT YET ESTABLISHED.** No confirmation has been run. No module,
config or test in the Step 24 / Step 25 / confirmation chain names a preprocessing recipe;
an AST test enforces it, and the propagation tests are parametrised across several
unrelated recipes precisely so none of them encodes a guess.

### D63 — Open: the confirmation's candidate set and seed count · `open`

Two parameters of the confirmation are scientific choices with defaults rather than
decisions:

- **How many candidates** (`--confirm-top-k`, default 3, plus the conventional reference).
  Three real-backbone runs is affordable; more is better evidence.
- **How many seeds** (`--confirm-seeds`). **Settled: the full protocol set.** A single run
  ranks the candidates with no error bar, and this repository's own proxy runs produced
  three different winners across configurations - so a small gap must be distinguishable
  from a seed effect. The default follows `--seeds`, so a shortened profile confirms at
  that profile's seeds.

The candidate count remains a default rather than a signed-off decision. Both are exposed
on the command line and recorded in the summary's provenance.


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

### E5 — `test_train_resume` updated after the MNIST example was removed · `done`

`configs/data/mnist.yaml` was deleted and `configs/train.yaml`'s defaults became
`data: bt_mri, model: baseline_simple_cnn`. The template's `tests/test_train.py` therefore
began training *this project's* pipeline, and `test_train_resume` started failing: it
asserted `epoch_001.ckpt` existed after resuming, which under `save_top_k=1` requires
validation accuracy to improve. With `limit_train_batches=0.01` a second epoch sees roughly
one batch of 4,617 MRI images, so whether it improves is chance.

The assertion held for MNIST and stopped holding for this data. The test now asserts what
it is actually for - that training resumes and advances past the first run's epoch - by
comparing the epoch recorded in `last.ckpt`. Confirmed by running the same test against the
Phase 2 commit, where it passes, so the change is attributable to the config swap rather
than to a pipeline regression.

### E4 — Dependencies added · `done`

`requirements.txt` gained pandas, scikit-learn, scipy, Pillow, numpy, opencv,
scikit-image, SimpleITK, h5py, matplotlib, seaborn and kaggle. PennyLane, shap,
umap-learn and statsmodels are listed but **commented out**, to be enabled by the phase
that needs them — PennyLane only after its Python 3.13 support is verified.

Installed into `env/`: pandas 3.0.5, scikit-learn 1.9.0, scikit-image 0.26.0, Pillow,
matplotlib. Note `torch 2.13.0+cpu` — **this environment has no CUDA**, which is why
execution is limited to smoke tests.
