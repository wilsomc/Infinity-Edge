# Infinity Edge

Monocular depth estimation: predict a depth map from one RGB image using a
pretrained ResNet-34 encoder and a U-Net-style decoder with skip connections.
Baseline training freezes the encoder; refinement can unfreeze its last block.
The decoder learns from paired RGB/depth targets using
masked Smooth L1 loss, optionally blended with a depth-gradient loss.
MiDaS Small can be evaluated with the relative-depth comparison below.

## Setup

Run commands from the repository root. Python 3.10+ is recommended.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first pretrained training run downloads ImageNet ResNet-34 weights.
If Python has no default certificate bundle (as can happen with macOS Python.org
installs), the training CLI uses the installed `certifi` CA bundle for HTTPS.
Certificate verification stays enabled; explicit SSL certificate settings are preserved.
The CLI selects CUDA, then Apple MPS, then CPU; override with `--device cpu`.
Data and generated results are ignored by Git.

## Data and depth encoding

The local dataset layout is:

```text
data/
  rgb/train/*.jpg
  rgb/test/*.jpg
  depth_bfx/train/*.png
  depth_bfx/test/*.png
```

`data/train/{rgb,depth_bfx}` and `data/test/{rgb,depth_bfx}` are also supported.
RGB and depth must share a filename stem and original spatial dimensions.
Windows `:Zone.Identifier` files are ignored. Missing pairs cause an error.

The default `--depth-encoding sunrgbd` decodes original uint16 SUN RGB-D depth
using `((raw >> 3) | (raw << 13)) & 65535`, then divides by 1000 for meters.
This follows the [SUN RGB-D toolbox read3dPoints.m](https://github.com/charlesq34/frustum-pointnets/blob/master/sunrgbd/sunrgbd_data/matlab/SUNRGBDtoolbox/readData/read3dPoints.m).
The local sampled PNGs appear consistent with this encoding; archive provenance
has not been independently verified. Check the inspection output and scene
distances before committing to a full experiment. For already-decoded exports,
choose `--depth-encoding millimeters` or `meters` explicitly; do not decode twice.

The configured valid target interval defaults to **[0.001, 8] meters**.
Non-finite, zero, and out-of-range targets are excluded, not trained as zero.
Depth is resized with nearest-neighbor interpolation; RGB uses ImageNet
normalization. Both are resized to 320×384, which can distort aspect ratios.
Evaluation is at this resized resolution, with no crop or ground-truth alignment.

## 1. Inspect data and save splits

```sh
python -m script.CNN inspect
```

Writes `results/baseline/data_inspection.json` with sample depth ranges and valid
fractions, and `results/splits.json` with paired paths. The default validation
split is a seeded 10% of the supplied training images (seed 174); supplied test
images are preserved. Subsequent runs reuse and verify the manifest.

This is an **image-level split**: scene/sequence metadata is unavailable, so
scene independence and absence of near-duplicate leakage are unverified.
The supplied 8,001/2,334 split is not claimed to be the official benchmark split.
Use scene metadata before making benchmark/generalization claims. Keep the
manifest fixed during tuning. A new dataset or different split needs a new
`--manifest` path. Saved manifests contain absolute local paths.

## 2. Train and validate

```sh
python -m script.CNN train --epochs 1 --output results/baseline
```

One epoch is a pipeline check, not evidence of convergence. Increase the epoch
budget only after checking targets, loss, and predictions. Training evaluates
validation after each epoch and selects `best.pt` by **raw-output validation
Smooth L1**, computed over valid pixels. `latest.pt` stores weights and optimizer state after each completed epoch.
`recovery.pt` is saved every 100 batches (configurable with `--checkpoint-every`)
and before validation. Checkpoint writes replace the previous file atomically.
Existing checkpoints are protected against accidental replacement.

Training and evaluation fully decode all pairs in their selected splits before
starting model computation, writing `integrity.json`. This catches unreadable
images, incompatible depth encoding, and mismatched RGB/depth dimensions early.
`inspect` checks all three splits. Errors name the pair and both file paths.
Source files are never silently skipped or modified.

If a pair is damaged, restore it from the original archive, or explicitly exclude
it in a **new** manifest, retaining all other existing seeded split assignments:

```sh
python -m script.CNN inspect --manifest results/clean_splits.json --exclude-ids 01279
python -m script.CNN train --epochs 1 --manifest results/clean_splits.json
```

Exclusions are recorded in the manifest with their original split and paths.
Report these exclusions with evaluation results, especially for test data.
The local scan found one unreadable depth file, `data/depth_bfx/train/01279.png`;
the prepared `results/clean_splits.json` excludes only that training pair.
This leaves 7,200 training, 800 validation, and 2,334 test pairs.

To continue after an interruption:

```sh
python -m script.CNN train --epochs 1 --output results/baseline \
  --manifest results/baseline/splits.json --resume results/baseline/recovery.pt
```

Resume restores weights, optimizer, batch position, training accumulators and
CPU random state, using the saved configuration and deterministic epoch order.
It skips batches already included in the checkpoint. Up to 99 batches can be
lost between periodic saves with the default interval. CPU resume equivalence
is tested; bit-identical accelerator behavior across devices is not guaranteed.
`--epochs` is the total target epoch count. To extend a completed run to five
epochs, use `--epochs 5 --resume results/baseline/latest.pt` with the same output
and manifest. Old runs that never saved a checkpoint cannot be recovered.

Outputs include configuration, a copy of the split manifest, loss/metric history,
`best.pt`, `latest.pt`, `validation_metrics.json`, and four fixed validation panels.
The test split is not evaluated during training. By default, training uses Smooth L1 only.

For a quick offline pipeline check using very little data:

```sh
python -m script.CNN train --limit 2 --epochs 1 --size 64 96 --batch-size 1 \
  --no-pretrained --device cpu --output results/smoke
```

`--limit` restricts every processed split. `--no-pretrained` uses a frozen random
encoder. Either flag marks checkpoint/metric outputs as `smoke_test: true`;
these outputs are not model-quality results.

## 3. Evaluate a saved checkpoint

```sh
python -m script.CNN evaluate --checkpoint results/baseline/best.pt \
  --manifest results/baseline/splits.json --split validation --output results/baseline
```

After choosing the final model with validation, run held-out test evaluation:

```sh
python -m script.CNN evaluate --checkpoint results/baseline/best.pt \
  --manifest results/baseline/splits.json --split test --output results/baseline
```

Evaluation reloads preprocessing/depth settings from the checkpoint, verifies
the manifest, uses eval mode without gradients, and saves `<split>_metrics.json`
and RGB/target/prediction/absolute-error panels. Numeric metrics use dataset-wide
valid-pixel aggregation, not averages of batch scores:

| Metric | Definition | Better |
| --- | --- | --- |
| MAE (m) | mean absolute depth error | Lower |
| RMSE (m) | square root of mean squared depth error | Lower |
| AbsRel | mean absolute error divided by target depth | Lower |
| δ1 / δ2 / δ3 | fraction with max(pred/target, target/pred) < 1.25^k | Higher |

Predictions are clamped to the configured valid range for these metrics and
panels. The clipped-pixel fraction is reported so clipping cannot silently hide
out-of-range outputs. Non-finite predictions on valid targets raise an error.
Smooth L1 is computed on raw predictions. Empty valid-pixel evaluations fail.
Panels share a fixed meter scale. MiDaS relative inverse depths must not be
passed directly into these metric-depth metrics without a documented protocol.

## Controlled gradient-loss experiment

Train a fresh model with the same pretrained encoder, decoder initialization seed,
data manifest, resolution, optimizer, learning rate, batch size, and five-epoch
budget as the baseline. Change only the training objective:

```sh
.venv/bin/python -m script.CNN train --epochs 5 --gradient-weight 0.2 \
  --manifest results/baseline/splits.json --output results/gradient_loss
```

This uses `0.8 * masked Smooth L1 + 0.2 * masked gradient error`. Gradient error
compares horizontal and vertical finite differences in predicted and target
depth, counting an adjacent pair only when both target pixels are valid.
Its mean is normalized by the total valid edge count across both directions.
It measures local depth changes, not image color edges. The weight 0.2 is an
initial experiment setting, not an established optimum. Invalid and padded
pixels do not contribute, including edges that touch them.

Both loss components and the blended loss are logged separately. Training logs
aggregate Smooth L1 by valid pixels and gradient error by valid edges. Validation
uses the same raw-output loss definitions, with global pixel/edge aggregation.
The depth metrics retain their existing prediction-clipping policy. As in the
baseline, checkpoint selection uses **validation Smooth L1**, keeping the
selection criterion fixed for the comparison. Component scales differ, so compare
MAE, RMSE, AbsRel, delta accuracy, gradient error, and fixed example panels rather
than comparing total losses from different weights.

The run starts fresh; do not resume the baseline to test this loss. A resume
restores its saved loss settings and rejects a conflicting `--gradient-weight`.
Old baseline checkpoints without this field use weight 0 and remain loadable.
To resume this experiment after an interruption:

```sh
.venv/bin/python -m script.CNN train --epochs 5 --output results/gradient_loss \
  --manifest results/gradient_loss/splits.json --resume results/gradient_loss/recovery.pt
```

First compare the best validation results and panels in `results/gradient_loss`
against `results/baseline`. Baseline test scores have already been observed;
keep loss selection/tuning on validation. A lower gradient error alone does not
establish better object boundaries or overall depth accuracy.

## Checks

```sh
python -m unittest discover -s tests -v
```

Tests cover hand-calculated metrics across unequal batches, invalid targets,
clipping, bit-rotation decoding, split reuse/drift, frozen encoder behavior,
corrupt-pair preflight/exclusions, and interrupted versus uninterrupted training.
No model accuracy or convergence claim is established by these checks.

## MiDaS Small comparison

```sh
.venv/bin/python -m script.compare_midas --split validation
```

Downloads the pinned official MiDaS v3.1 source and MiDaS_small checkpoint,
using official `small_transform`; also pins the EfficientNet source dependency.
The initial download requires internet access. Source and weights are cached
under `results/midas-cache`. The default run compares both custom checkpoints
on the same manifest, valid pixels, 320×384 RGB images, and target depths.
MiDaS uses its official internal resizing/normalization; output inverse depth
is resized back with bicubic interpolation. Inference uses batch size one.

**Two separate tables** are saved in `results/midas_comparison/validation_comparison.json`:

- `metric_depth_unaligned`: existing meter-depth performance for custom models.
- `oracle_aligned_relative_depth`: all models aligned to each image's ground
  truth with nonnegative scale and shift in inverse depth. Each custom model's
  depth is floored at the configured minimum before taking its reciprocal.
  Degenerate or negative-correlation fits use mean target inverse depth.
  Fitted inverse depths are clamped to [1/max_depth, 1/min_depth] and inverted.

The second table uses evaluation ground truth and **does not measure metric
prediction without ground truth**. Compare all models only within the same
protocol. MiDaS has no raw meter-depth score here. Alignment parameters,
checkpoint hashes, source revisions, image IDs, configuration and example panels
are saved. The test split and custom-model results remain unchanged.
MiDaS training overlap with SUN RGB-D source datasets is not ruled out; do not
claim independently verified zero-shot performance. No timing comparison is made.

For final test reporting using the same fixed protocol:

```sh
.venv/bin/python -m script.compare_midas --split test
```

For a quick diagnostic, add `--limit 2 --output results/midas_smoke`. Such results
are marked `smoke_test: true` and must not be reported as full-dataset scores.
Dependencies are included in `requirements.txt`. `--device cpu` is available if
an accelerator does not support a MiDaS operation.

## Predict depth for one photo

Run from the repository root:

```sh
.venv/bin/python -m script.predict data/rgb/test/08001.jpg
```

Replace the image argument with the path to your own JPG, PNG, HEIC/HEIF or other
Pillow-readable photo. HEIC/HEIF uses the `pillow-heif` dependency in requirements.txt;
the opener is registered automatically and the primary photo is decoded at full size. Paths with spaces must be quoted. The command needs only
the photo and a trained checkpoint; it does not need a dataset, manifest, depth
target, internet connection, or MiDaS. It uses `results/improvement_pass/best.pt` after a completed improvement pass,
otherwise `results/baseline/best.pt` (epoch 4 for the completed baseline
experiment). It selects CUDA/MPS/CPU. An explicit `--checkpoint` overrides this.

To use the gradient-loss checkpoint or choose an output directory:

```sh
.venv/bin/python -m script.predict data/rgb/test/08001.jpg \
  --checkpoint results/gradient_loss/best.pt --output results/predictions/gradient_example
```

The default output directory is `results/predictions/<photo stem>`. Each prediction
contains:

| File | Contents |
| --- | --- |
| `depth_m.npy` | Original-size float32 depth estimates in meters, clipped to the checkpoint's depth range |
| `raw_depth_m.npy` | Original-size float32 predictions before clipping, possibly including negative values |
| `depth_color.png` | Original-size RGB depth visualization, dark purple near and yellow far |
| `preview.png` | Input photo beside predicted depth, with a fixed meter color scale |
| `metadata.json` | Input/checkpoint paths and hashes, epoch, device, image sizes, range, clipping fraction and preprocessing policy |

Input EXIF orientation is applied; transparent pixels are composited on white.
The shared RGB transform uses the checkpoint's input size and ImageNet
normalization, matching training. Raw model predictions are bilinearly resized
back to the oriented photo's original dimensions, then clipped to its saved
range (normally 0.001–8 m). The numeric map has shape `(height, width)` and can
be loaded with `numpy.load(path, allow_pickle=False)`.

Existing prediction files are protected. Choose another `--output` directory or
pass `--overwrite` to replace a previous prediction. Inputs and checkpoints are
protected even when overwrite is requested. Use `--device cpu` if needed.

These are the custom model's unaligned learned depth estimates, with no use of
ground truth. They are distinct from the oracle-aligned MiDaS comparison.
Original-size output is interpolation of the trained 320×384 prediction, not
additional learned detail. Training used indoor SUN RGB-D images; performance
on arbitrary photos, especially outdoor scenes, has not been measured.

## Bounded improvement pass: longer training and encoder fine-tuning

```sh
.venv/bin/python -m script.improve --minutes 55 --epochs 3 --evaluate-test
```

Starts three candidates from the **same baseline best checkpoint** (epoch 4),
using the same saved training/validation split and original 320×384 resolution:

| Candidate | Encoder parameters trained | Gradient weight | Decoder LR | Encoder LR |
| --- | --- | ---: | ---: | ---: |
| `longer_decoder` | None | 0 | 0.0003 | — |
| `finetune_layer4` | Last ResNet block (`layer4`) | 0 | 0.0003 | 0.00003 |
| `finetune_gradient_01` | Last ResNet block (`layer4`) | 0.1 | 0.0003 | 0.00003 |

`--epochs 3` means at most **three additional epochs per candidate**, beyond the
source checkpoint's four training epochs. Each candidate begins with identical
weights and a fresh Adam optimizer; none continues from another candidate.
Training epoch numbers in the new checkpoints count this additional stage,
starting at 1, with source epoch and SHA-256 recorded under `initialization`.
Encoder BatchNorm running statistics remain fixed even for trainable blocks.
Only chosen block parameters are unfrozen; earlier encoder blocks stay frozen.

Candidates use a ReduceLROnPlateau scheduler (factor 0.5 after two validation
non-improvements, minimum LR 1e-6), early stopping after three non-improving
epochs, and recovery checkpoints every 100 batches. The source checkpoint is
retained as stage epoch 0 so a worsening continuation cannot replace it as best.
The same raw validation Smooth L1 criterion selects epochs and the final model.
MAE, RMSE, AbsRel, threshold accuracy and gradient error are also recorded.

The wall-time budget starts before data preflight and initial validation, and
remaining time is shared across remaining candidates. A candidate reaching its
time allowance saves incomplete recovery weights and retains the best completed
validation checkpoint. It does not rank incomplete evaluations. Validation and
checkpoint writes may finish slightly after the time allowance; optional final
test evaluation adds time. `--minutes 55` leaves some margin within an hour.

Results live in `results/improvement_pass`, with one directory per candidate,
`plan.json`, `progress.json`, `summary.json` and a copy of the validation-selected
weights at `best.pt`. With `--evaluate-test`, only that selected model is tested,
and its metrics/panels are saved in this directory. Existing baseline and
MiDaS comparison outputs are preserved. Test scores are not used for selection;
the test set has already been observed, so these are exploratory improvements,
not a newly untouched benchmark. Multiple seeds and scene-independent splits
remain unverified.

To continue an interrupted pass with another time allowance:

```sh
.venv/bin/python -m script.improve --minutes 55 --epochs 3 --evaluate-test --resume
```

Keep the original plan settings when resuming. Resume restores trainable groups,
optimizer, scheduler, epoch/batch position and loss accumulators. Use this workflow,
rather than `script.CNN train --resume`, for refinement checkpoints. CPU recovery
equivalence is tested; bit-identical accelerator behavior is not guaranteed.

To predict with the selected model:

```sh
.venv/bin/python -m script.predict data/rgb/test/08002.jpg \
  --checkpoint results/improvement_pass/best.pt --output results/predictions/improved_example
```

Once the improvement pass has selected a model, `script.predict` uses its
`results/improvement_pass/best.pt` by default. To use the original model instead,
pass `--checkpoint results/baseline/best.pt`.

## V2: aspect-preserving, augmented, higher-resolution training

```sh
.venv/bin/python -m script.train_v2 --epochs 30 --hours 7.5 --evaluate-test
```

This is a separate experiment under `results/highres_v2`. It starts from the
currently selected trained model, keeps prior checkpoints intact, and targets
up to 30 new epochs within a 7.5-hour training/preflight allowance. Validation,
reporting and optional selected-model test evaluation may add time; the allowance
leaves margin inside the approved eight-hour local run. Actual completed epochs
and runtime depend on measured throughput, validation stopping and available GPU
resources. The first GPU check uses batch size 2 at 480×640 with both late encoder
blocks trainable; a diagnostic run is not an accuracy result.

### Image geometry and augmentation

V2 preserves aspect ratio and centers the resized image on a 480×640 canvas.
RGB padding is zero **after ImageNet normalization**, representing the ImageNet
mean. Depth padding and invalid target pixels have a false validity mask; losses
and metrics ignore them and all touching gradient edges. Inference removes
padding before restoring the original photo size. Existing checkpoints missing
`resize_mode` retain their original stretched preprocessing.

Only training uses augmentation: paired RGB/depth horizontal flips (probability
0.5) and RGB-only brightness, contrast and saturation factors in [0.9, 1.1]. No
random geometric transform is applied independently to the two modalities.
Augmentation is seeded by record ID, epoch and the run seed, so skipping completed
batches on resume preserves each remaining sample's transformations. Validation
and standalone prediction have no random augmentation.

### Architecture and initialization

`architecture: v1` retains the original decoder and checkpoint keys. `v2` uses:

- Bilinear upsampling and learned 1×1 projections instead of transposed convolutions.
- Two 3×3 convolutions per decoder stage, with existing encoder skip connections.
- A full-resolution RGB feature branch and residual depth refinement before output.

When upgrading v1 to v2, encoder weights and BatchNorm buffers are copied exactly.
Matching decoder spatial/output convolutions are copied. Projection weights use
the spatial mean of the old transposed-convolution kernels, with axes transposed.
New stage convolutions start as identities, and the final correction branch
starts at zero. This is an **approximate warm start**, not identical predictions:
architecture and geometry have changed and need further training. The precise
transfer policy is recorded in `transfer.json`. Old checkpoints still load and
predict with v1 automatically; architecture is read from checkpoint configuration.

### Staged training and recovery

| New epochs | Trainable encoder blocks | Decoder LR | Encoder LR |
| --- | --- | ---: | ---: |
| 1–3 | None | 3e-4 initially | — |
| 4–6 | layer4 | At most 1e-4 | 1e-5 initially |
| 7–30 | layer3 and layer4 | At most 1e-4 | 1e-5 initially |

Encoder BatchNorm running statistics stay fixed throughout. The Adam parameter
groups include future blocks from the beginning; decoder optimizer state is
preserved when encoder blocks are unfrozen. ReduceLROnPlateau halves rates after
three non-improving validations (patience 2). Early stopping uses five
non-improving epochs and restarts its counter when entering a new encoder stage.
Loss remains masked Smooth L1 blended with gradient loss at weight 0.1.

Recovery checkpoints are written every 100 batches and before validation.
On timeout, the last fully evaluated best checkpoint is retained, and partial
training state is saved for recovery. Resume with:

```sh
.venv/bin/python -m script.train_v2 --resume --epochs 30 --hours 7.5 --evaluate-test
```

Resume restores saved configuration, architecture, augmentation, optimizer,
scheduler, batch position and RNG state. `--epochs` is the total target for this
stage and may be increased on resume; `--hours` grants a new allowance for that
invocation. Most other CLI settings are restored from the saved configuration.
CPU equivalence across encoder-stage transitions is tested. Accelerator
bit-identical recovery is not promised.

For a short GPU diagnostic of the most memory-intensive stage:

```sh
.venv/bin/python -m script.train_v2 --limit 4 --epochs 1 \
  --warmup-epochs 0 --layer4-epochs 0 --hours 0.1 --output results/v2_smoke
```

Diagnostics are marked as smoke tests and never become the prediction default.

### Comparable evaluation and boundaries

Different input resolutions are **not** compared using different target grids.
V2 validation restores each prediction to a fixed 320×384 evaluation grid. Ground
truth uses the original nearest-neighbor stretched resizing at that grid, exactly
matching the old model's scoring targets. This keeps sample membership, masks,
range, valid-pixel aggregation and units fixed. Old-model scores are reproduced
under this protocol; each model still uses its own input preprocessing. Canonical
scores measure output on that grid, not additional native-resolution detail.

Alongside MAE, RMSE, AbsRel, delta accuracy and gradient error, evaluation records
precision/recall/F1 for adjacent valid ground-truth depth jumps ≥0.1 m. Matches
are at the exact grid location. `boundary_gradient_mae_m` measures error on these
target edges. These are fixed diagnostics, not standardized boundary benchmarks;
they do not prove all chair legs or thin objects are resolved.

The lowest **canonical validation raw Smooth L1** selects the best v2 checkpoint.
Only a comparable full-data improvement over the source model is registered in
`results/selected_model.json`. `script.predict` checks that registry and its weight
hash before preferring the new model. A worse, partial-data, or unvalidated model
does not replace the previous default. If requested, test evaluation runs only
after selection and does not influence it. The test set was observed earlier;
these remain exploratory experiments on an image-level split.

Evaluate a saved v2 checkpoint with the existing command (it reads the recorded
architecture and automatically uses canonical evaluation):

```sh
.venv/bin/python -m script.CNN evaluate --checkpoint results/highres_v2/best.pt \
  --manifest results/highres_v2/splits.json --split validation --output results/highres_v2
```

### Outputs and controlled comparisons

The run saves configuration, transfer policy, source snapshots/hashes, package
versions, initial/reference validation, per-epoch history and example panels,
`progress.json`, `best.pt`, `latest.pt`, `recovery.pt`, `summary.json`,
`training_report.md`, `validation_curves.png` and optional selected-model test
results. `run_info.json` records the command, process ID and invocation start.
Keep the original model available explicitly with
`--checkpoint results/improvement_pass/best.pt`.

The full v2 run combines several changes. To isolate effects, use new output
folders with `--architecture v1`, `--resize-mode stretch`, `--no-augmentation`,
or `--size 320 384`. The same initial checkpoint, seed, manifest and fixed
validation protocol should be used across controlled comparisons. More training
or model capacity does not guarantee improvement on arbitrary internet photos.
Color scales stay fixed in meters; making a prediction more yellow is not an
optimization goal.

MiDaS comparison supports mixed input geometry and architecture:

```sh
.venv/bin/python -m script.compare_midas --split test \
  --checkpoints results/improvement_pass/best.pt results/highres_v2/best.pt \
  --output results/midas_v2
```

When v2, letterboxing or differing resolutions are present, the comparison uses
`canonical_original_inputs_v2`: each custom model receives the original RGB with
its own preprocessing; MiDaS uses its official transform on the original RGB.
All predictions are scored on the same canonical target grid. This differs from
`legacy_shared_resize_v1`, which first stretched every input to the same size.
Do not mix MiDaS tables from these protocols. Per-image oracle alignment and
pretraining-overlap limitations still apply.
