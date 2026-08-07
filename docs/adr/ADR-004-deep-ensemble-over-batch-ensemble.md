# ADR-004: Deep Ensemble Instead of True Batch Ensemble for Uncertainty Quantification

## Status
Accepted (Milestone 21)

## Context

README.txt's Module 7 describes "Batch Ensembles & Ranking Loss": *"Ensemble
forward passes are batched. The DeepHit ranking loss is computed strictly
within the same ensemble member (reshaped to `[Ensemble, Batch, Features]`)
to prevent entangled gradients."* Section 6's engineering-standards list also
names "Batch Ensemble gradient disentanglement" as one of the Red Team fixes
this project is expected to respect, alongside the project's general
GPU-latency consciousness (echoed in risk mitigation R4, "GPU latency:
mitigated by fixed grid and sparse masking," for the physics engine).

The literal "Batch Ensemble" technique (Wen et al., 2020, *BatchEnsemble: An
Alternative Approach to Efficient Ensemble and Lifelong Learning*) is a
specific, narrower thing than "an ensemble of models run in batched form." It
shares ONE base weight matrix across all `M` ensemble members. Each member's
distinct behavior comes entirely from a pair of cheap, per-member rank-1
perturbation vectors (`r_m`, `s_m`) that elementwise-scale the shared weight
matrix's inputs and outputs. This means `M` members cost barely more than a
single member's parameters and compute -- `M` extra rank-1 vectors per layer,
not `M` full extra weight matrices. That efficiency is the entire point of
the technique, and it is what would actually deliver on the GPU-latency
framing README's Module 7 and R4 gesture at.

## Decision

Milestone 21 implements `DeepEnsembleDeepHit`: **`M` (default 5) fully
independent `DeepHitSurvivalModel` instances**, each with its own complete,
separately and randomly initialized set of weights, trained and run
independently (see `production/src/models/deep_ensemble.py`). This is a
**Deep Ensemble** (Lakshminarayanan et al., 2017), not a Batch Ensemble.

This is a reasonable choice for a research platform at this stage: Deep
Ensembles are simpler to implement and reason about, have no shared-weight
coupling to get subtly wrong, and are a well-understood, statistically valid
method for epistemic uncertainty estimation via prediction disagreement
across independently-trained members.

It must NOT be silently presented as having solved the latency problem
README's Module 7 / R4 framing specifically calls out Batch Ensembles for
solving -- **that problem remains open.** A Deep Ensemble with `M=5` costs
approximately 5x the parameters and 5x the forward/backward compute of a
single MLP. If this ensemble's latency ever becomes a real constraint (most
plausibly once it needs to feed the live WebSocket API from Milestone 16, where
per-frame inference cost is directly on the critical path of real-time
streaming), that is the point to revisit true Batch Ensembles as an
optimization pass -- not before, and not by pretending this milestone already
did that work.

The class is deliberately named `DeepEnsembleDeepHit`, not
`BatchEnsembleDeepHit`, specifically so a future reader searching the
codebase for "Batch Ensemble" does not mistake this module for the technique
README actually describes.

## Consequences

- **Cost**: ~`M`x parameters and ~`M`x training/inference compute versus a
  single `DeepHitSurvivalModel`. This makes any Brier Score (or other metric)
  comparison against the single-MLP baseline (Milestone 14B) NOT a fair,
  equal-capacity comparison -- the same caveat already established for the
  MLP-vs-GNN comparison in Milestone 12 (different architectures/capacities,
  not an apples-to-apples ablation).
- **Benefit retained**: genuine epistemic uncertainty via independently
  initialized and independently trained members, with the gradient
  disentanglement property README's Module 7 requires (each member's ranking
  loss is computed only within that member's own predictions -- see
  `compute_disentangled_ensemble_loss` in `deep_ensemble.py`).
- **Benefit NOT retained**: the GPU-latency efficiency of true Batch
  Ensembles. This is explicitly left as future work, not solved here.
- **Future work candidate**: implement true Batch Ensemble layers (shared
  base weight + per-member rank-1 `r`/`s` vectors) if/when ensemble inference
  latency becomes a measured, real constraint on the live serving path
  (Milestones 16-19's WebSocket/REST API). Until then, the ~5x compute cost of
  the Deep Ensemble is accepted as a reasonable trade for implementation
  simplicity in a research platform.

## Update: True Batch Ensemble Implemented, Measured A/B, and a Real Correction to the Original Framing

This gap sat open since Milestone 21. It resurfaced during a training-
pipeline memory investigation (Aug 2026) that initially treated it as a
possible OOM-risk fix, not just the original compute-cost question. **That
premise was checked directly, before writing any code, and turned out not
to hold** -- reported here plainly, not glossed over, because the honest
finding matters more than confirming the assumption that motivated
building this.

### Step 0: the OOM-risk premise, measured and found not to apply

Direct measurement (`/usr/bin/time -v`, this project's own real
match-level training pipeline, not a synthetic benchmark):

| Measurement | Peak RSS |
|---|---|
| `build_training_data()` / `_load_and_split_dataset()` ALONE (dataset load, no ensemble) | **1.01GB** (1,013,636 KB) |
| Existing Deep Ensemble (M=5, fully independent) -- FULL pipeline (dataset load + training + eval + MLflow logging) | **1.15GB** (1,174,916 KB) |
| New Batch Ensemble -- FULL pipeline, same measurement | **1.18GB** (1,180,104 KB) |

The Deep Ensemble's OWN incremental cost above dataset loading is
**~157MB** (1.15GB − 1.01GB) -- nowhere near "~5x a single model's
footprint, held simultaneously" in any sense that matters at the process-
memory level, and the Batch Ensemble's own incremental cost (~166MB) is,
within measurement noise, the SAME. Both are two orders of magnitude
below the ~9.2GB level that caused this project's real, separate test-
suite OOM incidents the same night (a completely different problem --
many heavy test files' imports/allocations accumulating within one long-
lived `pytest` process -- see `context.md`'s own incident note; not this
training pipeline, which was never close to that level either before or
after this change).

**Why the premise didn't hold, once actually checked**: this project's
`DeepHitSurvivalModel` is a tiny `4 -> 32 -> 32 -> 12` MLP -- 1,612
parameters, ~6.4KB at float32. Even 5 fully independent copies (the
existing Deep Ensemble) total only ~8,060 parameters, ~32KB. No
parameter-sharing scheme was ever going to move the needle on a peak RSS
measured in gigabytes, dominated by PyTorch/MLflow's own import and
runtime footprint and by `build_training_data()`'s real work (fetching
and running `BiomechanicalPitchControl` across 8,074 samples). **The ~5x
compute-cost framing in this ADR's original Decision section was correct
and remains correct; the memory-footprint framing this Update set out to
check was not, for a model this small, and is corrected here rather than
left standing.**

### Step 1: `BatchEnsembleDeepHit` -- the real technique, finally implemented

`production/src/models/batch_ensemble.py` (new, additive; `deep_ensemble.py`
and `DeepEnsembleDeepHit` are UNCHANGED). A genuine Wen et al. (2020)
implementation: one shared `BatchEnsembleLinear` base weight matrix per
layer, plus, per member, a pair of rank-1 fast-weight vectors (`r`, `s`,
initialized to random ±1 signs -- NOT all-ones, which would collapse
every member to an identical effective weight at step 0) and a per-member
bias. Same `[M, B, num_bins]` forward-pass output shape and the same
`predict_with_uncertainty` interface as `DeepEnsembleDeepHit`, so it is a
genuine drop-in for comparison -- `compute_disentangled_ensemble_loss`
is reused UNCHANGED from `deep_ensemble.py` (it only operates on the
already-produced `[M, B, num_bins]` tensor, agnostic to which class built
it). Wired into `train.py` via new, additive `_train_and_log_batch_ensemble`/
`_run_batch_ensemble_stage` functions that closely mirror
`_train_and_log_deep_ensemble`/`_run_deep_ensemble_stage` (same
hyperparameters, same match-level split/seed, same ADR-010 four-signal
health gate) rather than generalizing those existing, already-validated
functions -- keeping the Deep Ensemble's own path completely untouched.

Confirmed directly (forward-pass shape/PMF-validity/parameter-count/
diversity smoke test, no training needed): `BatchEnsembleDeepHit` totals
**2,636 parameters (~1.64x a single model)** vs. `DeepEnsembleDeepHit`'s
**8,060 (~5.0x)** -- a real, correctly-implemented ~67% parameter
reduction. At this project's tiny model scale that reduction is a few
kilobytes, not a memory-relevant amount (see Step 0) -- but it is real,
and it is what the technique is supposed to do; the reduction would
become memory-relevant if this architecture ever scaled up substantially
(a much wider/deeper backbone), which is exactly the condition under
which this ADR's original "revisit if scale changes" framing anticipated
revisiting it.

### Step 2: the real A/B, same split (match_level, seed=42), both health-gated

| Model | Peak RSS (full pipeline) | Brier@15s | Brier@30s | Health gate |
|---|---|---|---|---|
| Deep Ensemble (M=5, existing, run_id `fc14567b02a645ba80aa939f908a2b56`) | 1.15GB | 0.1008 | 0.1882 | **PASSED** -- spike/cumulative_drift/saturation/frozen_val_loss all `False` |
| Batch Ensemble (M=5, new, run_id `0a46c25127d14ba8af4647275609bf9b`) | 1.18GB | 0.1010 | 0.1885 | **PASSED** -- spike/cumulative_drift/saturation/frozen_val_loss all `False` |

Both trained for the SAME 50 epochs on the SAME match-level train/val
split (seed=42), same `MLP_STABILIZED_LR`/`MLP_STABILIZED_WEIGHT_DECAY`,
same gradient clipping. Ensemble diversity (mean std of per-member
cumulative incidence @15s across the validation set) was comparable for
both: Deep Ensemble not separately re-quoted here (see its own MLflow
run), Batch Ensemble 0.0164 -- confirming the ±1 sign initialization
genuinely produces diverse members, not a collapsed ensemble.

**Memory**: statistically indistinguishable (1.18GB vs. 1.15GB -- a
~5MB difference on top of a shared ~1.01GB dataset-loading baseline both
runs pay identically; well within run-to-run noise, not a real
advantage either way, consistent with Step 0's own finding).

**Brier Score**: statistically indistinguishable (+0.0002 @15s, +0.0003
@30s, Batch Ensemble slightly higher/worse both times, but by an amount
far smaller than this project's own established noise floor for
architecturally-different comparisons -- RQ4's repeated-measurement
investigation found run-to-run Brier swings of 0.01-0.05 for a much more
different architecture (GNN vs. MLP) even under a FIXED split; a
0.0002-0.0003 gap here is not a real accuracy difference, in either
direction).

**Wall-clock training time**: Batch Ensemble 4:01 (241s) vs. Deep
Ensemble 4:41 (281s) -- a real, measured **~14% training-time reduction**,
the SAME shared-matmul-instead-of-M-separate-matmuls mechanism the
technique is actually designed around. This is the dimension README.txt's
Module 7 and this ADR's own R4 framing were originally about (GPU-latency
consciousness), not memory -- and it is the one dimension where a real,
if modest, measured benefit showed up.

### Step 3: honest conclusion

**Comparable predictive performance, no real accuracy tradeoff either
direction, no meaningful memory advantage at this project's current
model scale, and a real (if modest) compute/wall-clock win** -- consistent
with what the technique was always actually for. The Batch Ensemble is
now available as a genuine, additive, A/B-able alternative
(`BatchEnsembleDeepHit`) alongside the unmodified Deep Ensemble
(`DeepEnsembleDeepHit`); this ADR's original Decision (ship the simpler
Deep Ensemble for M21, defer Batch Ensemble) is not reversed -- both now
exist side by side, and a future caller can pick either with real,
measured numbers to decide from, rather than the framing being purely
theoretical.

**Does this close out the compute-cost gap named in this ADR's original
Consequences section and repeated in `docs/FULL_PROJECT_REPORT.md` §11.2 /
`README.md`'s roadmap table?** Partially, honestly stated: the
IMPLEMENTATION gap is closed (a true Batch Ensemble exists, is tested, is
health-gated, and is a real drop-in). The LATENCY motivation this ADR
named as the actual trigger for revisiting it ("once it needs to feed the
live WebSocket API... per-frame inference cost is directly on the
critical path") has still not been separately measured on that live
serving path specifically -- this Update measured TRAINING-time wall-
clock (a real, adjacent, but not identical question to live per-frame
INFERENCE latency under the WebSocket API's own request pattern). That
narrower, serving-path-specific latency measurement remains a legitimate,
smaller follow-up if ensemble inference ever becomes a real bottleneck
there -- named explicitly, not silently assumed already covered by this
Update's training-time number.

**Does this resolve the OOM risk that motivated re-opening this ADR
tonight?** **There was no training-pipeline OOM risk to resolve** -- Step
0's own direct measurement is the answer: neither ensemble approach ever
came close to a memory level that would matter (~1.15-1.18GB vs. the
~9.2GB level that caused this project's real OOM incidents, which were a
SEPARATE, already-fixed test-suite problem, not a training-pipeline one).
Stated plainly rather than let stand as an implied justification for this
work: this Update is worth having for closing a real implementation gap
and delivering a real, measured training-time speedup, not because it
fixed a memory problem that, on direct measurement, was never actually
there.
