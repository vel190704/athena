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
