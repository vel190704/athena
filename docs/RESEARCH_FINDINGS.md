# Project Athena: Research Findings

**Status as of Milestone 35** (the RQ1-RQ5 findings below were produced under Milestone 23's methodology; §1's "Methodological update" paragraph documents the Milestone 35 match-level-split change and its one informational-only smoke test — see `docs/adr/ADR-011-match-level-train-val-split.md` for the full decision). All five Research Questions defined in `README.txt` have
been investigated at least once. This document synthesizes those findings, the
Architecture Decision Records that shaped the system, the recurring methodological
lessons this project's own history surfaced, and what remains open.

Every metric in this document was re-verified directly against either the project's
MLflow tracking store (`file:./mlruns`, experiment `project-athena-deephit`) or a fresh
re-run of the relevant test file, immediately before this document was written — not
transcribed from memory of prior conversation. Run IDs are cited explicitly wherever a
specific MLflow run backs a number, so any claim here can be re-checked directly.

---

## 1. Executive Summary

Project Athena is a research-grade tactical intelligence platform for football (soccer)
that treats a match as a differential game: physical fatigue, biomechanical limits, and
tactical positioning jointly determine how "dangerous" a given moment is. The system is
built on a strict architectural separation between **classical deterministic physics**
(a closed-form, analytical biomechanical pitch-control ODE; a causal Kalman filter for
latent ball friction) and **statistical ML inference** (a DeepHit discrete-time survival
model predicting near-term shot probability). Physics outputs are immutable feature
layers; ML models consume them but never modify them.

As of Milestone 23, the platform:
- Ingests real StatsBomb open-data event and 360 freeze-frame data across 12 competitions
  spanning men's and women's football (8,074 training samples from ~55 matches).
- Trains and compares three predictive architectures (a scalar-feature MLP, a GNN over
  player positions, and a 5-member Deep Ensemble) under a shared, strengthened
  instability-detection harness.
- Exposes live inference over a WebSocket API and an interactive counterfactual "what-if"
  REST endpoint, both backed by a Streamlit dashboard.
- Provides Integrated-Gradients feature attribution with a templated natural-language
  explanation layer, ensemble-based predictive uncertainty, Bayesian habit-memory
  blending (opt-in), and a real-substitution "oracle" validation pipeline.

All five RQs have working, honestly-reported answers below. None should be read as
permanently settled — RQ4 in particular has already reversed direction twice as data
scale and training stability changed, and is presented as a trajectory, not a single
number.

**Methodological update (Milestone 35), read alongside every finding below:** every RQ
finding in this document — including RQ2's null result and RQ4's full trajectory — was
produced under a train/validation split at the SAMPLE level (Milestone 7 through
Milestone 34). As of Milestone 35, this project's training code uses a MATCH-level split
instead (`production/src/pipeline/data_split.py`; see ADR-011), because sample-level
splitting let almost every match contribute samples to both sides, which is exactly what
starved RQ2's historical corpus down to 4 usable matches (see RQ2 below). **This does
not change or supersede any finding already reported below** — those remain accurate
statements about what was found under the methodology used at the time. A single
validation smoke test (the seed=42 stabilized MLP, Milestone 14B's exact
hyperparameters, retrained under the new match-level split) is logged in MLflow tagged
`split_type="match_level"` for the record, explicitly NOT as a new headline number; every
pre-existing run remains implicitly `split_type="sample_level"`. Re-running RQ2 and RQ4's
full comparisons under match-level splitting was, at the time this paragraph was
originally written, legitimate, separate future work. **That re-run has since been
performed** (full research scale, matched hyperparameters, health-gated) — see the
"Update" subsections appended to RQ2 and RQ4 below. This paragraph is left otherwise
unedited as a record of what was and wasn't known at the time; the new evidence is
appended, not backfilled into this paragraph's original wording.

---

## 2. Research Questions & Findings

### RQ1 — Does velocity-aware pitch control support well-calibrated short-term goal probability?

**Question (README):** *"Can velocity-aware pitch control improve short-term goal
probability calibration? (Success: Brier Score improvement ≥ X%)"*

**Methodology:** The `BiomechanicalPitchControl` engine (Module 1) computes, per frame,
each team's control probability over a sparse-masked pitch grid using the closed-form
analytical force-velocity ODE (validated in Milestone 2). Four scalar features derived
from this field (`attacking_control_near_ball`, `defending_control_near_ball`,
`attacking_control_final_third`, `space_behind_defending_line`) feed a DeepHit discrete-time
survival model, evaluated via time-dependent Brier Score at 15s and 30s horizons.

**Finding:** Across every genuinely healthy trained model, the pipeline produces
well-calibrated, sane Brier Scores, well inside the sanity ceiling used throughout this
project (2.5x the Milestone 12B reference floor):

| Model | Brier@15s | Brier@30s | MLflow run_id |
|---|---|---|---|
| Single MLP (Milestone 14B, seed=42) | 0.0942 | 0.1588 | `e2c42aeed7374c398643298a1580a08c` |
| Deep Ensemble (M=5, Milestone 21) | 0.0936 | 0.1582 | `5a006e396fa6468dbc5114804d9b2e35` |

**Caveats:** README's stated success criterion ("Brier Score improvement ≥ X%") implies a
comparison against a non-physics-informed baseline, but no such ablation (DeepHit trained
on non-physics-derived features, or a naive baseline) was ever run in this project's
history. The evidence for RQ1 is therefore that the physics-informed feature pipeline
supports well-calibrated prediction in absolute terms, not a measured relative
improvement over an alternative without velocity-aware pitch control. This is a real gap,
named explicitly rather than papered over with an implied percentage.

### RQ2 — Does Bayesian tactical memory improve prediction over purely live tracking?

**Question (README):** *"Does Bayesian tactical memory improve prediction over purely
live tracking? (Success: Calibration error reduction)"*

**Methodology (Milestone 22/23):** A historical positional heatmap Prior is built per
player from their own past events (binned into a 10x7 grid over the project's 100x68m
space), Bayesian-blended (Posterior ∝ Prior × Gaussian likelihood around the live
position) with the live-observed position, and substituted in before running
`BiomechanicalPitchControl`. Because StatsBomb's public 360 data exposes **no per-player
identity for ~21 of the 22 visible players** (verified directly against 21,273 real
freeze-frames across 6 cached matches — every entry carries only `location`, `teammate`,
`actor`, `keeper`), blending is scoped to the single acting player per event only. Heatmaps
were built match-aware and split-aware: the training-split match set (matches with zero
validation-split samples) fed a precomputed per-player-per-match bucket corpus, and each
sample's own match was additionally excluded from its own heatmap.

**Finding:** Habit blending made the Brier Score slightly *worse* at both horizons:

| Model | Brier@15s | Brier@30s | MLflow run_id |
|---|---|---|---|
| MLP, no blending (Milestone 14B) | 0.0942 | 0.1588 | `e2c42aeed7374c398643298a1580a08c` |
| MLP, habit blending (Milestone 23) | 0.0950 | 0.1601 | `671fb22ed0334f34a6a74059d1a17a4e` |

**Caveats (all apply, none is "the" explanation):**
- **Actor-only dilution**: the blended signal touches exactly 1 of ~22 visible players'
  positions, inside features that are themselves sums over ~11 players per side.
- **Inconsistent feature direction**: the chain's representative actor is not
  consistently the attacking-side actor (Milestone 7's frame-resolution logic can resolve
  to a defensive action), so blending doesn't consistently push the same feature
  direction across samples.
- **Corpus size, empirically decisive**: out of ~55 matches used, only **4 matches had
  zero validation-split samples** and were therefore training-bucket-eligible (a
  conservative partition: any match with even one validation sample is excluded from the
  training bucket corpus entirely, matching Milestone 7's training-split-only
  normalization discipline). This left **5,495 of 8,070 samples with a known actor
  (68%) falling back to the uniform cold-start prior** — for those samples, blending is
  close to a discretized identity transform on the live position, not a real historical
  signal.
- **Non-chronological history**: any other training-split match may contribute to a
  heatmap regardless of real-world date order relative to the match being predicted —
  this tests the blending *mechanism*, not a faithful live-deployment simulation.

**Verdict:** RQ2 is **not supported** by this run, but under conditions constrained
enough (68% cold-start, 1-of-22-player scope, non-chronological corpus) that this should
be read as a null result for this specific, heavily limited implementation — not a
general verdict on Bayesian habit memory. See Future Work §6 for what a fairer test would
require.

**Update — full re-validation under the match-level split (ADR-011), APPENDED, not
superseding the finding above.** Milestone 35 (ADR-011) fixed the sample-level-split
methodology that starved this RQ's original training-bucket corpus to 4 matches, but
deliberately did not re-run the comparison itself — "legitimate, separate future work,
not performed here." That re-run has now been executed, at full research scale (the same
~55-match, 12-competition, 8,074-sample dataset used above), reusing every already-
validated sub-function unmodified (`_load_and_split_dataset`, `_run_mlp_stabilization_and_robustness_check`,
`_run_gnn_stage`, `_build_habit_blended_features`, `_run_habit_blended_stage` — see
`run_match_level_rq2_rq4_full_revalidation()` in `train.py`). **This is one additional
data point under a corrected methodology — it does not retroactively invalidate the
result above, which remains an accurate statement about what was found under sample-level
splitting.**

*Match-level split (seed=42):* 42 training matches / 10 validation matches → 6,642 train
/ 1,432 validation samples (82.3% / 17.7%, not forced to a round percentage — see
`match_level_split`'s own report). The habit-memory training-bucket corpus itself is
**45 matches**, slightly larger than the 42 training-split matches: `_build_habit_blended_features`
also credits the 3 matches that were pooled but yielded zero 360-covered chains (no
training *samples*) with their real raw event data for other players' historical
heatmaps — a real, correct property of the existing code, not a discrepancy.

*Health gate (both models confirmed genuinely healthy before proceeding):* the
stabilized MLP (seed=42, matched Milestone 14B hyperparameters: lr=1e-4, grad clipping,
weight_decay=1e-4, 50 epochs) and the GNN both cleared all four ADR-010 instability
signals (spike, cumulative drift, saturation/entropy, frozen-val-loss — every one
`False`) and the MLP's Brier-sanity check. A seed=43 MLP robustness check
(Brier@15s/30s = 0.1002/0.1864) confirmed seed=42's result (0.1009/0.1873) was not a
one-off. Non-blended MLP baseline this run: **Brier@15s = 0.1009, Brier@30s = 0.1873**.

*Cold-start fallback rate, the figure this re-run exists to fix:* **2,164 of 8,070
samples with a known actor (26.8%)** fell back to the uniform cold-start prior — down
sharply from the original **68%** (5,495/8,070) at the 4-match corpus. The corpus-size
problem ADR-011 set out to fix is real and substantially fixed.

*Habit-blended MLP result (match-level, NEW):* **Brier@15s = 0.1017, Brier@30s =
0.1891** — all four instability signals `False` (genuinely healthy). Compared against
THIS run's own non-blended baseline (the correct apples-to-apples comparison, same split,
same hyperparameters): **+0.0008 @15s, +0.0018 @30s — both horizons WORSE**, the
identical direction as the original Milestone 23 result.

| Comparison | split_type | corpus | Brier@15s | Brier@30s | Δ vs. own non-blended baseline |
|---|---|---|---|---|---|
| MLP, habit blending (Milestone 23) | sample_level | 4 matches | 0.0950 | 0.1601 | +0.0008 / +0.0013 (worse) |
| MLP, habit blending (this run, NEW) | match_level | 45 matches | 0.1017 | 0.1891 | +0.0008 / +0.0018 (worse) |

**This is the notable finding this re-run was designed to surface either way.** The
delta habit blending imposes on Brier Score is **essentially unchanged** between the
original 4-match, 68%-cold-start corpus and this run's 45-match, 26.8%-cold-start
corpus — despite an 11x larger historical corpus and a cold-start rate cut by more than
half. **This argues against corpus size having been the primary driver of RQ2's original
null result.** The two scope limitations named alongside the original finding — the
actor-only blending constraint (only 1 of ~22 visible players' positions is ever
adjusted) and the chain's representative actor not consistently being the attacking-side
actor (diluting/scrambling the feature direction the blended signal pushes) — now read as
the more likely primary explanations, not merely co-listed caveats. This is a single run
at one corpus size increase, not a dose-response sweep across multiple corpus sizes, so
this conclusion is itself a hedged, not definitive, updated reading — but it is a real,
informative result, reported as such rather than assumed away.

**Updated verdict, hedged:** RQ2 remains **not supported**, now under BOTH the original
constrained-corpus implementation AND a corrected-methodology, ~11x-larger-corpus
re-run — a more robust null result than either run alone, and one that shifts the most
likely explanation away from "not enough historical data" toward the actor-only/
feature-direction limitations named above.

### RQ3 — Does latent friction estimation improve pass trajectory prediction?

**Question (README):** *"Does latent friction estimation improve pass trajectory
prediction? (Success: Pass landing error reduction ≤ X meters)"*

**Methodology (Milestone 1, ADR-008):** The causal Kalman filter (predict → observe →
correct, no look-ahead) estimates a synthetic ball's rolling friction coefficient `μ`
from noisy final-velocity observations, with `Cd = 0` (ground passes) so the synthetic
kinematics are exact modulo injected Gaussian measurement noise — isolating "is the
Kalman math correct" from "does the physics model match real aerodynamics" (ADR-008).
Gate: converge to within 2% of the true `μ` after 50 passes.

**Finding (re-run and verified, not recalled):** true `μ = 0.35`; deliberately
wrong prior `μ_0 = 0.5` (to prove correction, not trivial agreement); after 50 passes
the filter's posterior converged to **`μ = 0.35118`**, a relative error of
**0.336%** — over 5x tighter than the 2% gate (`production/tests/test_friction.py`).

**Caveats:** This validates the Kalman filter's *mathematical* correctness on synthetic
data only (ADR-008's explicit, deliberate scope). It does not prove real-world pass
landing error reduction: real StatsBomb passes have nonzero, unknown `Cd`, non-Gaussian
tracking noise, and the follow-up milestone ADR-008 calls for (extending synthetic
validation to nonzero-but-known `Cd`, then real data) has not yet been executed. RQ3's
literal success criterion ("pass landing error reduction ≤ X meters") has therefore not
yet been measured against real trajectories at all — only the filter's internal
convergence has been validated.

### RQ4 — Can graph-based team representations outperform handcrafted tactical features?

**Question (README):** *"Can graph-based team representations outperform handcrafted
tactical features? (Success: AUROC/Brier improvement over MLP)"*

This is the RQ whose answer has moved the most, and it is presented here as a full
trajectory, not a single number — treating any one snapshot as final would misrepresent
how this evidence actually developed.

**Stage 1 — single competition (World Cup 2022 only), 3,198 samples:**

| Stage | MLP Brier@15s/30s | GNN Brier@15s/30s | Note |
|---|---|---|---|
| Milestone 12 (initial) | 0.0846 / 0.1720 (`bcae3f79f6c04bbbbaa290f04098d929`) | 0.1070 / 0.2258 (`68d3ade44aea4f9e9259c7ef1a4c9ace`) | GNN training showed exploding-gradient instability |
| Milestone 12B (GNN stabilized: grad clipping, weight decay, lr 1e-3→1e-4) | 0.0846 / 0.1720 (unchanged — MLP was already stable) | 0.1051 / 0.2042 (`f3eeb1c8b6084acb841cb5f533e07145`) | GNN improved, but MLP still wins — hedged "not yet" |

**Stage 2 — multi-competition (12 competitions), 8,074 samples:**

| Stage | MLP Brier@15s/30s | GNN Brier@15s/30s | Note |
|---|---|---|---|
| Milestone 14 (scale-up) | 0.1263 / 0.2571 (`07f3ef56e13d434aa02a5b832de610c4`) | 0.1127 / 0.1905 (`8267bc29a3e54d9d92e146de9b4de145`) | **MLP silently collapsed** (softmax saturated to one time bin regardless of input) — GNN "wins" only because the comparison is invalid |
| Milestone 14B (MLP fixed: same stabilization bundle as GNN, strengthened detector, 2-seed robustness check) | 0.0942 / 0.1588 seed 42, 0.0948 / 0.1595 seed 43 (`e2c42aeed7374c398643298a1580a08c`, `dd757e27789345a69f23326c5cdf1891`) | 0.1141 / 0.1932 (`b8565e5b4b2c4512b998bccbb39d64db`) | MLP wins again, both models confirmed genuinely healthy |

**The Milestone 14 collapse was caught only by directly probing the trained model's
predictions** (finding `max_prob ≈ 1.0` on the last time bin for every input) — the
instability detector active *at that time* (single-epoch spike >50%) never fired,
because the collapse was a gradual multi-epoch drift with a frozen validation loss, not a
sudden spike. This directly motivated Milestone 14B's strengthened, multi-signal detector
(cumulative/windowed drift, dual-signal output-saturation, frozen-val-loss backstop).

**A live, current-code caveat worth stating plainly:** the codebase's own automated
"is the MLP genuinely healthy" gate (a >10% total loss decrease heuristic) has **declined
to certify** an RQ4 verdict on every subsequent re-run of this exact comparison
(Milestones 21 and 23 both re-executed this training path and both printed "loss
decreased meaningfully=False... NOT issuing an RQ4 conclusion this run"). The
Milestone 14B "MLP wins" conclusion rests on a manual override of that conservative
automated flag, backed by direct prediction-diversity probing (entropy and batch
variance far from collapse thresholds) — a legitimate, documented judgment call, but not
something the automation itself currently re-confirms on every run.

**Current-best-evidence conclusion:** at the current 8,074-sample scale, with matched
stabilized hyperparameters and both models confirmed genuinely healthy by direct
probing, **the handcrafted-feature MLP outperforms the GNN**. Given this exact
comparison has already reversed direction twice (Stage 1: MLP wins → Stage 2 as
measured: GNN "wins" via MLP's silent failure → Stage 2 corrected: MLP wins again), this
should be treated as the current data point, not a permanently settled architectural
verdict.

**Update — Stage 3: full re-validation under the match-level split (ADR-011), APPENDED,
not superseding the finding above.** Same re-run described in RQ2's update above (same
dataset, same split, same health gate) also retrained the GNN with matched
hyperparameters, giving RQ4 its first properly-conducted comparison under match-level
splitting (Milestone 35 itself ran only a single MLP-only smoke test, no GNN). Both
models cleared all four ADR-010 instability signals before this comparison was trusted.

| Model | split_type | Brier@15s | Brier@30s |
|---|---|---|---|
| MLP (Milestone 14B reference) | sample_level | 0.0942 | 0.1588 |
| GNN (Milestone 14B reference, `b8565e5b4b2c4512b998bccbb39d64db`) | sample_level | 0.1141 | 0.1932 |
| MLP (this run, NEW) | match_level | 0.1009 | 0.1873 |
| GNN (this run, NEW) | match_level | 0.1198 | 0.2437 |

**Within this run, the MLP again clearly outperforms the GNN at both horizons** (0.1009
vs. 0.1198 @15s; 0.1873 vs. 0.2437 @30s) — a third data point favoring the handcrafted
scalar features, using a cleaner train/val methodology than either prior comparison. A
second, independent observation worth naming: **both models score worse in absolute
terms under the match-level split than their sample-level-split references** (MLP +0.0067
@15s/+0.0285 @30s; GNN +0.0057 @15s/+0.0505 @30s, the GNN degrading noticeably more).
This is plausibly real signal, not noise: match-level splitting requires generalizing to
possession chains from MATCHES the model never saw any sample from, whereas sample-level
splitting let the validation set benefit from the model having already seen other chains
from the same match's tactical patterns — a harder, arguably more honest test of
generalization, and the GNN's larger degradation under it is itself a new, notable
observation about which architecture generalizes better across matches, not just within
them.

**Updated verdict, hedged (explicitly, per this project's own repeated caution about this
exact comparison):** the MLP wins again — now under three methodologically distinct
conditions (Stage 1 single-competition, Stage 2 corrected multi-competition, and this
match-level-split re-run) — but this comparison has already reversed direction twice
before in this project's own history, and a third confirming data point is still one
more data point, not a settled architectural verdict. The GNN's larger relative
degradation under match-level splitting is a new thread worth investigating on its own
(e.g., whether the GNN overfits to within-match structure more readily than the MLP),
not yet explained by anything in this run alone.

### RQ5 — Can counterfactual simulations predict the tactical effects of substitutions?

**Question (README):** *"Can counterfactual simulations predict the tactical effects of
substitutions? (Success: Predicts real-world xT shift within predefined bounds)"*

This RQ has **two distinct evidence threads**, kept separate deliberately — they test
different things and should not be merged into one finding.

**Thread A — general counterfactual tactical-alignment sanity checks (Milestones 13/14/14B).**
Hand-picked heuristic perturbations (`high_press`, `drop_deep`, `force_wide`) are applied
to a real match state's features/positions, and the resulting shift in predicted
cumulative incidence is checked against football intuition (e.g., does `high_press`
increase near-term threat?), reported as findings, never hard-asserted. Across the MLP
and GNN, results were mixed and match/model-dependent — sometimes aligned with intuition,
sometimes not — consistent with the explicit caveat that these perturbations are
hand-chosen heuristics, not empirically calibrated tactical shifts, and can produce
out-of-distribution feature combinations the model never saw in training.

**Thread B — Oracle Substitution Validation against real substitutions (Milestone 20).**
This is the RQ's actual literal question: for every real substitution in a match, compare
the model's predicted threat for the substituting team just before vs. just after,
holding team perspective fixed throughout (verified via an explicit `perspective_verified`
check on every result, since features are computed relative to whichever team is
acting/in-possession at a given frame). Tested on match 3857276 (Canada vs. Morocco,
World Cup 2022):

| sub_id | Team | Minute | Pre → Post threat@15s | Delta | Overlapping? |
|---|---|---|---|---|---|
| 0 | Canada | 59 | 0.3238 → 0.0758 | −0.2480 | Yes |
| 1 | Canada | 59 | 0.3238 → 0.0758 | −0.2480 | Yes |
| 2 | Canada | 60 | 0.0316 → 0.1245 | +0.0929 | Yes |
| 3 | Morocco | 64 | 0.0834 → 0.0287 | −0.0547 | Yes |
| 4 | Morocco | 64 | 0.0834 → 0.0287 | −0.0547 | Yes |
| 5 | Canada | 65 | 0.3917 → 0.0407 | −0.3510 | Yes |
| 6 | Canada | 75 | 0.1646 → 0.1322 | −0.0324 | Yes |
| 7 | Morocco | 75 | 0.0410 → 0.1293 | +0.0883 | Yes |
| 8 | Morocco | 75 | 0.0410 → 0.1293 | +0.0883 | Yes |
| 9 | Morocco | 84 | 0.0419 → 0.4653 | +0.4234 | **No** |

**9 of 10 substitutions had an overlapping ±2-minute window with at least one other
substitution** — this match had two clusters of near-simultaneous double/triple subs
(minute 59, 64, and 75). Since Oracle Validation is a purely observational before/after
snapshot, not a controlled comparison, overlapping-window deltas cannot be attributed to
any single substitution with confidence. **Only substitution #9** (Morocco, minute 84,
Hakimi → Jabrane) is isolated and interpretable on its own; it shows threat rising from
4.2% to 46.5%.

**Verdict:** Thread A provides mixed, model/scenario-dependent evidence about directional
tactical alignment. Thread B, on the single match tested, provides at most one genuinely
unconfounded real-substitution observation — nowhere near enough to assess whether the
model "predicts real-world xT shift within predefined bounds" as README's success
criterion requires. RQ5 remains, honestly, largely open on its literal question; Thread A
should not be read as bolstering Thread B's much thinner evidence base.

---

## 3. Architectural Decisions (ADRs)

All nine ADRs in `docs/adr/` are summarized below.

**ADR-001 — DeepHit over DeepSurv/Cox.** Football's tactical danger is a non-proportional,
time-varying hazard (a central overload means something different at minute 20 vs. 90);
Cox/DeepSurv's proportional-hazards assumption doesn't hold. DeepHit models the discrete-time
hazard shape directly and natively handles right-censoring (turnovers, fouls, out-of-play).
Cost: loses Cox's closed-form interpretability (mitigated by ADR-006's async explainability)
and requires the ranking loss to be computed strictly within-ensemble-member.

**ADR-002 — StatsBomb coordinate rescaling at the ingestion boundary.** StatsBomb's raw
120x80 unit grid is rescaled to this project's internal 100x68 grid exactly once, inside
`parse_360_frame`, via fixed `X_SCALE`/`Y_SCALE` factors. No downstream module may assume
or apply its own rescaling — every physical constant and feature threshold in the codebase
is defined in the 100x68 space on this guarantee.

**ADR-003 — Attacking-direction inference (superseded).** This ADR proposed inferring
each team's attacking direction per period from shot/touch locations, then forcing a
period-2 coordinate flip on the guaranteed rule that teams swap ends at half-time. It was
empirically tested during Milestone 10 and found to disagree with that guaranteed rule in
**37 of 40 team-periods** across 20 real matches. Three independent checks (goal-kick
clustering at x≈6-7 for every team in every period; exact 180° turnover-coordinate
mirroring; own-goal event-pair mirroring) showed why: StatsBomb already records
coordinates relative to the *acting team's own* attacking-left-to-right frame, in both
halves — there was no shared, physically-fixed frame to correct in the first place. This
sequence — build a fix for an assumed problem, validate empirically, discover the fix's
own premise was wrong, correct course — is one of the strongest illustrations of this
project's verify-before-trusting methodology, not a mistake to gloss over. ADR-003's
`direction.py` module was kept (not deleted) as correctly-implemented code answering a
different, currently-unneeded question; it remains available for a future data source
(e.g. Module 4's CV pipeline) that genuinely does use a shared coordinate frame.

**ADR-004 — Deep Ensemble instead of true Batch Ensemble.** README's Module 7 cites true
Batch Ensembles (Wen et al.): one shared base weight matrix, per-member cheap rank-1
perturbations, ~1x compute for M members. Milestone 21 implements a **Deep Ensemble**
instead — M=5 fully independent models, ~5x parameters/compute — a simpler, statistically
valid but heavier alternative, explicitly *not* solving the latency problem Batch
Ensembles exist for. Named `DeepEnsembleDeepHit`, deliberately not `BatchEnsembleDeepHit`,
to prevent exactly this confusion for future readers.

**ADR-005 — Sparse masked indexing over dense grids.** A dense `[100, 68, 22]` pitch-control
grid misses the sub-50ms latency budget; dynamic Adaptive Mesh Refinement was rejected
because variable tensor shapes cause Triton memory thrashing. Instead, indices where
`distance_to_ball <= 30m` (~2,000 of ~6,800 cells) are extracted via a **static-shape binary
mask**, not a reshaped/variable tensor — reducing compute from `O(100×68×22)` to `O(2000×22)`
while keeping shapes static for Triton kernel fusion.

**ADR-006 — Asynchronous SurvivalSHAP/explainability.** SHAP-style methods need hundreds
of forward passes per explanation, incompatible with the live hazard stream's frame-rate
cadence. Explanation generation runs in an async background worker on a fixed 5-second
cadence, pushing results to a **secondary** WebSocket channel so a slow or failed
explanation can never block or degrade the primary real-time signal. (Milestone 15's
Integrated-Gradients + mock-LLM explainer implements this "explain, don't predict,
decoupled from the real-time path" principle concretely, ahead of a real LLM integration.)

**ADR-007 — Reaction time fixed, acceleration fatigue-coupled (v1 asymmetry).** Fatigue
dynamically penalizes effective acceleration (`a_eff`, a continuous parameter inside the
already-closed-form ODE) but reaction time stays constant within a phase (a discrete
latency offset that would require re-deriving the ODE's initial conditions if made
dynamic). This asymmetry isolates ODE validation from a second, structurally different
dynamic in the same milestone; v2 must revisit reaction-time degradation explicitly.

**ADR-008 — Synthetic validation baseline for the Kalman filter.** Real StatsBomb data
conflates tracking noise, aerodynamic drag, spin, and genuine friction variation — a
real-data validation failure would be undiagnosable. The filter is validated first on
synthetic data with `Cd=0` (exactly known), isolating "is the Kalman math correct" from
"does the physics model match reality." See RQ3 above for the verified result. A
follow-up (nonzero-but-known `Cd`, then real data) is explicitly still open.

**ADR-009 — StatsBomb data is already per-actor oriented (supersedes ADR-003's flip).**
The direct successor to ADR-003's discovery: since raw coordinates are already recorded
relative to the acting team's own frame, no direction flip is needed at all — the ADR-002
rescale is the only transformation required. Verified independently for the 360
freeze-frame data specifically (not just event `location`): teammate-flagged goalkeepers
clustered at mean x≈10.7 and opponent-flagged keepers at mean x≈112 across ~20,000
observations, nearly identically in period 1 and period 2. This conclusion is explicitly
scoped to StatsBomb's data convention — a future computer-vision pipeline extracting raw
broadcast pixel coordinates will NOT arrive pre-normalized this way and will need real
per-half direction handling.

---

## 4. Methodological Lessons Learned

These are recurring patterns across this project's actual history, not abstract
principles — each names the specific incidents that taught them.

**(a) Training instability can be silent and pass surface-level checks.** Two separate
collapses were caught only by directly probing trained model outputs, not by the
instability detector active at the time: the GNN's exploding-gradient blowup (Milestone
12, loss spiking 3.07→4.58 around epoch 20) and — more insidiously — the MLP's frozen-softmax
collapse at multi-competition scale (Milestone 14: train loss climbed steadily from 3.30
to 5.07 over 45 epochs with no single-epoch jump ever exceeding the 50% spike threshold,
while validation loss went bit-for-bit frozen; only directly probing predictions revealed
`max_prob≈1.0` on the last time bin regardless of input). This motivated Milestone 14B's
strengthened, multi-signal detector (cumulative/windowed drift over 20 epochs, dual-signal
output-saturation via both batch variance and mean entropy, and a frozen-val-loss
backstop) — and even that detector's own conservative "10% loss decrease" health heuristic
has since needed manual override (see RQ4 above), underscoring that automated flags are a
floor, not a substitute for direct inspection.

**(b) External data schemas should never be assumed from memory.** StatsBomb's actual
event-type surprises (Milestone 5: zero-duration chains from 1-second timestamp
granularity), competition/matches JSON structure (Milestone 9: match validity is not
recoverable from a hardcoded list, only from the live competitions index), substitution
event field layout (Milestone 20: the incoming player is nested at
`event["substitution"]["replacement"]`, not top-level), and 360 freeze-frame
player-identity availability (Milestone 22: no per-player id/name for ~21 of 22 visible
players, verified across 21,273 real frames) all differed from what a reasonable prior
assumption would predict. Every one of these was caught by checking real cached data
directly before writing extraction code, not by reasoning from documentation or memory —
the single most repeated procedural step across this project's entire history.

**(c) Leakage boundaries need explicit design at every level a feature could depend on
other samples.** Milestone 7 established that normalization statistics (mean/std) must be
computed from the training split only, after the split, never from the full dataset.
Milestone 23 showed this same discipline extends further than per-sample correctness:
Bayesian habit-memory heatmaps depend on *other samples'* historical events, so a
per-sample-correct exclusion rule (never use a sample's own match) is not sufficient by
itself — the entire historical bucket corpus must also be restricted to matches that
contribute no validation-split samples at all. This is a stricter, match-level version of
the same underlying rule, and it was only surfaced because Milestone 22/23 introduced the
first feature genuinely dependent on cross-sample information.

---

## 5. System Capabilities

- **Physics core**: causal Kalman friction filter (predict→observe→correct); analytical
  closed-form biomechanical pitch-control ODE with sparse masked-index evaluation.
- **Ingestion**: StatsBomb open-data events + 360 freeze-frames across 12 verified
  360-covered competitions, both match periods, disk-cached.
- **Prediction models**: single-risk DeepHit MLP; GNN (SAGEConv) over player-position
  graphs; 5-member Deep Ensemble with per-member gradient-disentangled loss.
- **Evaluation**: time-dependent, censoring-aware Brier Score at 15s/30s horizons;
  strengthened 4-signal training-instability detector.
- **Experiment tracking**: full MLflow logging (params, per-epoch metrics, models,
  normalization artifacts) across every training run since Milestone 8, with a
  deterministic (tag-filter + lowest-Brier) model-selection rule replacing "most recent
  run" ambiguity.
- **Explainability**: Integrated Gradients feature attribution against the exact
  cumulative-incidence quantity reported, structured LLM prompt construction, async mock
  LLM executor (ADR-006-consistent).
- **Uncertainty quantification**: Deep Ensemble mean prediction + per-member spread;
  diversity sanity-checked, not assumed.
- **Digital twin**: hand-tuned counterfactual tactical-action simulator (scalar and
  graph representations); real-substitution Oracle Validation with verified fixed-team
  perspective and overlapping-window detection.
- **Bayesian habit memory**: opt-in, actor-scoped historical heatmap blending with
  cold-start fallback and match/split-aware leakage guards — off by default, every
  existing baseline unaffected.
- **Live serving**: FastAPI WebSocket tactical-threat stream (per-connection state,
  background alert pipeline) and a REST `/simulate` counterfactual endpoint, both backed
  by a single deterministically-loaded model at startup.
- **UI**: Streamlit dashboard with a live threat monitor and an interactive What-If
  simulator panel (single-blocking-loop architecture, documented trade-offs).

---

## 6. Future Work

Prioritized, not exhaustive:

1. **~~Match-level (not sample-level) train/val split~~ — DONE (Milestone 35, ADR-011).**
   The split had been at the sample level since Milestone 7, which was fine while no
   feature depended on cross-sample information, until Milestone 23 showed this becomes
   a real constraint the moment a feature (habit memory) depends on *other samples'
   matches*. `data_split.match_level_split` now guarantees no match straddles both
   splits, by construction. What remains open, and is NOT done: (a) chronological
   ordering within the training-history corpus is still not enforced (a separate,
   independent limitation from the split level itself — see RQ2's Thread scope notes
   above and ADR-011); and (b) actually re-running RQ2's habit-blended MLP and RQ4's
   MLP-vs-GNN comparisons under the new split — Milestone 35 only ran a single MLP smoke
   test, explicitly not a re-validation campaign. **Update: both re-runs have since been
   performed** (full research scale, matched Milestone 14B hyperparameters, health-gated
   before proceeding) — see the "Update" subsections appended to RQ2 and RQ4 above. The
   real corpus turned out to be 45 matches, not the ~42 estimated here (see RQ2's update
   for why). Item (a) above (chronological ordering) remains open and untouched by this
   re-run.
2. ~~**Larger-scale dataset**~~ **— substantially addressed for RQ2 by the match-level
   re-run above** (training-bucket corpus 4 → 45 matches, cold-start rate 68% → 26.8%),
   though notably this did NOT resolve RQ2's null result the way this item originally
   speculated it might (see RQ2's update — the null result persisted at nearly the same
   magnitude, arguing against corpus size as the primary driver). RQ4's fairer
   re-evaluation at scale remains only partially addressed: this was still a single run,
   not a repeated/multi-seed campaign at this larger scale.
3. **True Batch Ensembles** (per ADR-004): the current Deep Ensemble is a valid but ~5x
   parameter/compute alternative. If ensemble inference latency becomes a real
   constraint — most plausibly once uncertainty quantification needs to feed the live
   WebSocket API — implementing genuine shared-weight, rank-1-perturbation Batch
   Ensembles is the direct way to recover that efficiency without giving up the
   uncertainty signal.
4. **Computer Vision pipeline (Module 4)**: broadcast-video ingestion (YOLOv9 + ByteTrack
   + SoccerNet calibration) is fully decoupled from the ML launch date by design, but
   remains entirely unimplemented. Critically, per ADR-009's explicit scope note, this
   pipeline will need genuine per-half/per-team direction handling — CV-extracted pixel
   coordinates will NOT arrive pre-normalized to an acting-team frame the way
   StatsBomb's data does.
5. **RQ3's real-data validation gap**: extend the Kalman filter's synthetic validation to
   nonzero-but-known `Cd` (per ADR-008's own follow-up requirement), then finally to real
   StatsBomb pass trajectories, before RQ3's literal "pass landing error" success
   criterion can be assessed at all.
6. **RQ5 Thread B's single-match sample size**: Oracle Substitution Validation has only
   ever been run on one match, yielding one unconfounded observation. Running it across
   many matches — and explicitly stratifying by whether a substitution's window overlaps
   another — is necessary before Thread B can support any real conclusion.
