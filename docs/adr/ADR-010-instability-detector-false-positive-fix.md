# ADR-010: Demote the Loss-Decrease Health Heuristic to a Non-Blocking Diagnostic

## Status
Accepted (this milestone, following a false positive discovered in Milestone 23)

## Context

`production/src/pipeline/train.py` has two logically distinct pieces of
health-checking machinery, and this ADR exists because they were being
treated as one:

1. **The four-signal core instability detector**, built across Milestones
   12B and 14B specifically in response to two real, silent training
   failures: the GNN's exploding-gradient blowup (Milestone 12, loss
   spiking and never recovering, caught only by eyeballing a printed log)
   and the MLP's frozen-softmax collapse at multi-competition scale
   (Milestone 14, train loss climbing steadily for 45 epochs with no
   single-epoch jump ever exceeding the spike threshold, while validation
   loss went bit-for-bit frozen). The four signals -- single-epoch spike,
   cumulative/windowed drift, output-saturation via batch variance AND
   entropy, and a frozen-val-loss backstop -- are each named-and-motivated
   by one of these real incidents and, put together, are this project's
   most principled, evidence-based line of defense against silent
   collapse.
2. **A separate, cruder addition**, added inside `train_and_evaluate()`'s
   `mlp_healthy` gate alongside Milestone 14B's "is the stabilized MLP
   genuinely learning well, not merely not-collapsed" check:
   `loss_decreased_meaningfully = (first_loss - last_loss) > 0.1 *
   first_loss`, requiring the final epoch's loss to be at least 10% lower
   than the first epoch's loss. This was a THIRD, additional criterion
   folded into the same `mlp_healthy` boolean that also required "no
   instability warnings" and "Brier Score in a sane range" -- but unlike
   those two, it was never independently motivated by a real historical
   failure the way the four core signals were.

**The false positive, and where it actually came from.** RESEARCH_FINDINGS.md
documents that "the codebase's own automated 'is the MLP genuinely healthy'
gate ... has declined to certify an RQ4 verdict on every subsequent re-run"
(Milestones 21 and 23), printing "loss decreased meaningfully=False ...
NOT issuing an RQ4 conclusion this run," even though the seed-42 MLP was
independently confirmed genuinely healthy by direct entropy/batch-variance
probing. Auditing `train.py` line by line (rather than assuming which
check was responsible) confirms this precisely: `habit_healthy` (the RQ2
gate, Milestone 23) depends ONLY on `instability_warning_fired` -- the
four-signal detector -- and that detector never fired. The only thing that
fired was `loss_decreased_meaningfully`, inside the separate `mlp_healthy`
gate that feeds the RQ4 conclusion. **The core four-signal detector is
exonerated by this audit; it is not the source of the false positive and
did not need its thresholds touched.**

**Why the crude check misfires.** `first_loss` is epoch 1's END-of-epoch
average loss (averaged across every mini-batch in that epoch), not the
loss at initialization. A small, well-conditioned MLP training at a low,
stabilized learning rate (1e-4, the Milestone 14B bundle) can converge to
very near its optimum WITHIN epoch 1's own mini-batches -- in which case
epoch 1's logged average is already close to the eventual floor. A model
that then correctly plateaus near that optimum for the remaining 49
epochs will show only a small further decrease by epoch 50, relative to
an already-mostly-converged epoch 1 value -- exactly the seed-42 MLP's
actual behavior in Milestone 23, and exactly what this check would
misread as "not learning." A fast-converging, genuinely healthy model and
a model that never learned anything can produce the same small
first-to-last delta; this single number cannot tell them apart, and
should never have been a gate on its own.

## Decision

1. **`loss_decreased_meaningfully` is removed from `mlp_healthy`'s gate.**
   `mlp_healthy` now depends on exactly two criteria: no instability
   warning fired (the four-signal detector) and Brier Score within the
   sane-range ceiling. The loss-decrease figure is still computed and
   printed -- clearly labeled `[diagnostic only, per ADR-010 NOT part of
   the health gate]` -- since a genuinely-zero further decrease across 50
   epochs is still useful context for a human glance; it simply never
   blocks a conclusion or requires manual override again.
2. **The four-signal detector's thresholds are UNCHANGED.** Per this
   ADR's own audit finding, it did not misfire, so this deliberately does
   NOT follow the "widen 1-of-4 to 2-of-4" contingency -- that would have
   been the right move only if the core detector itself had been the
   culprit, and it wasn't. Narrowing or widening a detector that already
   correctly classified every historical case it was asked to would risk
   overfitting to a single incident that this detector didn't even cause.
3. **The three signals that were previously inline in the epoch loop**
   (cumulative drift, saturation, frozen-val-loss) are factored out into
   standalone pure functions (`_check_cumulative_drift`, `_check_saturation`,
   `_check_frozen_val_loss`), mirroring the single-epoch spike check's
   existing `_check_for_instability` pattern. This has no effect on
   training behavior -- same thresholds, same print statements, same
   control flow -- but makes the detector regression-testable against
   synthetic sequences without an expensive real retraining run, and
   removes the verbatim duplication this logic previously had between
   `_train_and_log_model` and `_train_and_log_deep_ensemble`.
4. **`production/tests/test_instability_detector.py`** exercises these
   real production functions (not reimplementations) against four
   synthetic scenarios reproducing this project's actual historical
   patterns: Milestone 12's spike-and-never-recover shape (flags
   unstable, via the spike signal), Milestone 14's exact climb/frozen-val-loss
   numbers (3.30→5.07, frozen at 4.843820095062256 -- flags unstable, via
   saturation and frozen-val-loss, with the spike signal explicitly
   confirmed NOT to fire, reproducing the historical blind spot that
   motivated building those two signals), Milestone 23's fast-converge-
   then-plateau shape (does NOT flag -- the exact false positive this ADR
   fixes, with an explicit assertion that the OLD crude arithmetic would
   have misfired on this same scenario), and a slowly-but-continuously
   improving healthy control case (does NOT flag, confirming the fix
   didn't just narrow the detector to exclude Milestone 23's shape at the
   cost of missing other real problems).

The entropy/variance-based output-probing signal remains, as it always
has been, the primary trusted signal for resolving ambiguous cases in this
project's history (Milestones 12, 14, 23 were all ultimately settled by
direct probing, not by a loss-curve-shape heuristic) -- this ADR does not
change that; it removes a weaker, unprincipled heuristic that had been
sitting alongside it with equal gating authority it never earned.

## Consequences

- **RQ4's conclusion path** (`train_and_evaluate()` Step 5) will no
  longer withhold a conclusion purely because a healthy, fast-converging
  MLP shows a small first-to-last loss delta; it still correctly withholds
  one if any of the four principled signals fire, or if Brier Score is
  catastrophically bad.
- **No behavior change to the four-signal detector itself** -- every
  threshold (spike 50%, cumulative drift 30% over 20 epochs, saturation
  variance 1e-6 / entropy 0.1, frozen-val-loss exact equality) is
  untouched. Any future re-run of the Milestone 14B training path should
  reproduce the same spike/drift/saturation/frozen-val-loss verdicts as
  before this change -- only the separate `loss_decreased_meaningfully`
  criterion's role changed, from gate to diagnostic.
- **New regression-testing capability**: future changes to any of the
  four signals (or a fifth, if a new failure mode is ever discovered) can
  now be checked against this project's actual historical incidents
  cheaply, via `test_instability_detector.py`, without retraining a real
  model -- the same "verify against real/historical evidence, not
  assumption" discipline this project applies to loss curves, coordinate
  conventions, and CV homography math, now applied to the detector that
  watches all of them.
- **Open follow-up, not resolved here**: this detector has now correctly
  classified every historical case it has been asked to (three real
  incidents plus one healthy control), but it has still only ever been
  exercised against these specific reconstructed shapes and whatever real
  training runs happen to occur -- a genuinely novel failure mode with a
  different shape than M12/M14 could still, in principle, evade all four
  signals. This is the same category of caveat this project applies to
  every other empirically-tuned constant (Kalman Q/R, pitch-control
  radii, CV thresholds): validated against known cases, not proven
  complete against unknown ones.
