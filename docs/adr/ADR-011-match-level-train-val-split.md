# ADR-011: Match-Level Train/Validation Splitting

## Status
Accepted (Milestone 35)

## Context

Every training run in this project, from Milestone 7 through Milestone 34,
split samples into train/validation at the SAMPLE level
(`torch.utils.data.random_split` over the flat list of possession-chain
samples). This was a reasonable default while no feature depended on
information from other samples -- normalization statistics (Milestone 7)
only ever needed "the training split," not "the training split, grouped
by match."

Milestone 23 (Bayesian habit memory) introduced the first feature that
genuinely does depend on cross-sample information: a player's historical
positional heatmap is built from that player's OTHER events across OTHER
samples. This surfaced a real problem sample-level splitting was never
designed to handle. Because chains from the same match are scattered
randomly across a sample-level split, almost every non-trivial match ends
up contributing samples to BOTH the training and validation sets. Treating
such a straddling match as "training-eligible" for its train-split samples
while excluding it for its val-split samples would require per-sample
bucket exclusion finer than "the match" -- not how historical positional
data is naturally grouped. The only safe reading of "training-split only"
(Milestone 7's rule) was to exclude any straddling match from the
training-bucket corpus ENTIRELY, at the cost of shrinking that corpus
drastically: of ~55 matches, only **4** had zero validation-split samples
and were therefore training-bucket-eligible. RESEARCH_FINDINGS.md's RQ2
verdict names this directly as likely explaining much of that null result
-- a 68% cold-start fallback rate is not a fair test of the blending
mechanism.

## Decision

Introduce `production/src/pipeline/data_split.py`'s `match_level_split`:
partitions unique MATCH IDs into train/val groups (not sample indices),
then assigns every sample from a given match to whichever group that
match landed in. A match can no longer straddle both splits **by
construction** -- not by a downstream exclusion rule working around the
fact that it still could.

Consequences of splitting at the match level rather than the sample
level:
- The resulting train/val **sample**-count ratio is no longer forced to
  match `val_fraction` exactly, since matches contribute different
  numbers of samples (some chain-heavy, some light). `match_level_split`
  reports both the match-count split (what `val_fraction` actually
  controls) and the resulting sample-count ratio explicitly, rather than
  silently rounding to a clean percentage.
- A single unusually chain-heavy match could, in principle, dominate a
  small validation group. `match_level_split` checks for and reports this
  explicitly (a configurable share-of-group threshold, currently 30%,
  itself an unvalidated judgment call like every other hand-tuned
  constant in this project) rather than silently accepting a skewed
  split.
- `train.py`'s `_build_habit_blended_features` (Milestone 23) needed no
  logic change at all -- its existing set-difference computation
  (`training_match_ids = all_match_ids - val_match_ids`) was already
  correct in general; it simply no longer has anything to compensate for,
  since no match can straddle both groups anymore. The elaborate
  "conservative partition" documentation this function carried was
  simplified to reflect that reality, not left describing a workaround
  that no longer applies.

**Every MLflow run logged before this milestone used sample-level
splitting and is NOT retroactively re-tagged.** MLflow params are
immutable once logged, and rewriting history to claim those runs used a
methodology they didn't would misrepresent what actually produced those
numbers -- the same discipline this project applied to M12's unstable GNN
run and M14's collapsed MLP run (kept, never overwritten, never quietly
re-labeled). Every run logged FROM this milestone onward carries an
explicit `split_type` MLflow param (`"match_level"` for anything using the
new split; historical runs are implicitly `"sample_level"`, a convention
documented in a comment in `train.py`, not enforced retroactively).

**This milestone deliberately does NOT re-run the full comparison suite
under the new split.** Only a single validation smoke test was performed:
retraining the seed=42 stabilized MLP (identical hyperparameters to
Milestone 14B) under match-level splitting, logged and explicitly tagged
`split_type="match_level"` with a `comparability_note` pointing back to
this ADR. **Its Brier Scores are informational only and must never be
read as correcting, superseding, or invalidating any RQ1-RQ5 finding
already reported in RESEARCH_FINDINGS.md.** Those findings are accurate
statements about what was observed under the methodology used at the
time; a methodology change does not retroactively change what was found.
Re-running the GNN, Deep Ensemble, and habit-blended MLP comparisons
under match-level splitting -- and any resulting re-evaluation of RQ1-RQ5
-- is legitimate, separate future work, not performed here.

## Consequences

- **New utility, independently unit-tested**: `data_split.py` /
  `test_data_split.py` -- no-straddling guarantee, determinism given a
  seed, honest sample-ratio reporting under uneven match sizes, and the
  dominant-match imbalance check, all verified against constructed
  scenarios rather than assumed correct.
- **Habit-memory corpus size**: the smoke test's own match-level split
  (42 training matches, 10 validation matches, out of 52 matches that
  contributed at least one sample) demonstrates the training-bucket
  corpus available to a future habit-blending re-run would jump from
  Milestone 23's 4 matches to up to 42 -- over a 10x increase -- purely
  as a side effect of this refactor. This number is reported because it
  falls directly out of the split itself, not because RQ2 was re-run;
  actually re-training the habit-blended MLP against this larger corpus,
  and re-evaluating RQ2, is explicitly out of scope for this milestone.
- **Non-comparability is now a standing rule, not a one-time caveat**:
  any future run using `match_level_split` must be read alongside its
  `split_type` MLflow param, and never directly diffed against a
  `sample_level` run's Brier Score as if the same experiment were
  repeated -- the validation SET ITSELF is different, not just the model
  or hyperparameters.
- **Open follow-up, not resolved here**: re-running the MLP-vs-GNN
  (RQ4), Deep Ensemble (uncertainty), and habit-blended MLP (RQ2)
  comparisons under match-level splitting, at the larger available habit
  corpus, is the natural next step -- but a deliberate, separate
  undertaking, not a silent extension of this milestone.
