# ADR-012: `track_id` vs. `player_id` — Scope Boundary for CV-Context Habit Memory

## Status
Accepted

## Context
Milestone 22 built Bayesian habit-memory blending
(`production/src/pipeline/habit_memory.py`) keyed on StatsBomb's
`player_id` — a persistent, named identity valid across an entire
competition, recovered from `event["player"]["id"]`. This enables a
genuine CROSS-MATCH historical prior: `generate_player_heatmap` /
`build_player_match_buckets` / `heatmap_from_buckets` answer "how has
THIS SPECIFIC NAMED PLAYER tended to position himself, aggregated across
OTHER matches in the training corpus" — a question that only makes sense
because `player_id` means the same real person in match 1 and match 55.
(Milestone 22's own module docstring already scopes this to the single
*acting* player per event, since StatsBomb's 360 data exposes no
per-player identity for the other ~21 visible players — a separate,
already-documented limitation, not the one this ADR is about.)

A future extension aims to apply the same kind of positional-prior
blending to ALL players visible in CV-tracked video (Milestones 25-34,
34B), not just a single per-frame actor — using ByteTrack's `track_id`
(`production/src/cv/tracker.py`, Milestone 26) in place of `player_id`.
On the surface, both look like "an integer that identifies a specific
player across multiple observations," making it tempting to reuse
Milestone 22's exact mechanism with `track_id` substituted for
`player_id`. **This ADR exists to block that substitution before it
happens**, by documenting why the two identities are not the same kind of
thing.

## The Distinction

`track_id` is fundamentally NOT the same kind of identity as `player_id`.
`player_id` is guaranteed persistent and correct across an entire
competition by StatsBomb's own data pipeline; `track_id` carries no such
guarantee beyond a narrow, explicitly-scoped window, and this project's
own code already treats it that way:

- **Does not survive a gap larger than the pipeline trusts.** Milestone
  32's orchestrator (`production/src/cv/pipeline.py`) tracks each
  `track_id`'s last-observed FRAME INDEX and computes a `stale_gap_frames_threshold`
  (default 5): once the real elapsed gap since a track's last observation
  exceeds that threshold, the pipeline explicitly refuses to trust the
  prior observation for a velocity computation and falls back to `[0, 0]`
  rather than silently assuming continuity. This is a direct, load-bearing
  acknowledgment, already baked into shipped code, that `track_id`
  continuity is not guaranteed past an arbitrary gap — not a hypothetical
  concern this ADR is inventing.
- **Does not survive a camera cut.** Milestone 31's shot classifier
  (`shot_classifier.py`) exists specifically to skip non-tactical frames
  (replays, close-ups) before the rest of the pipeline runs; nothing in
  Milestone 32's orchestrator attempts to re-link a `track_id` across such
  a skip, and `tracker.py`'s own module docstring is explicit that a
  `track_id` is only meaningful within one continuous tracking segment.
- **Does not survive occlusion reliably.** `CV_PIPELINE_FINDINGS.md`
  documents the real, measured consequence of this on the one real clip
  processed so far (Milestone 34B): **152 unique `track_id`s observed
  against a real roster of only ~22-25 people** over a single 970-frame
  clip — a real, honestly-reported signal of heavy ID churn/fragmentation,
  not a hypothetical risk.
- **Obviously does not survive a transition to a different match or video
  at all.** `tracker.py`'s `run_tracking` and `pipeline.py`'s `CVPipeline`
  each construct their own ByteTrack state per video; `track_id` numbering
  is local to one `model.track()` invocation over one video source. Match
  A's `track_id=7` and match B's `track_id=7` are two unrelated integers
  that happen to collide by coincidence of counting — there is no
  mechanism anywhere in this codebase that relates them to each other, or
  to any real, named individual.
- **There is no code anywhere in this project that resolves a `track_id`
  to a real, named, cross-match player identity** (verified directly —
  `grep`ing `production/src/` for jersey-number OCR, roster lookup, or any
  cross-match identity-resolution logic returns nothing). Building that
  resolution — jersey-number OCR plus team-roster lookup, referred to
  elsewhere in this project's planning as "Track B" — is separate, harder,
  and explicitly not yet started work.

## Decision

Any future extension of habit-memory blending to CV-tracked players MUST
be built as a **different, more limited mechanism** than Milestone 22's —
an **in-match-only positional prior**, scoped strictly to the lifetime of
a single `track_id` within one continuous video, answering only: *"where
has this specific tracked player tended to be positioned earlier in THIS
SAME match's footage"* — never *"how has this named player tended to
position himself across other matches,"* which is a question only
`player_id`-keyed data can currently answer.

This new mechanism must be named distinctly from Milestone 22's —
`track_local_positional_prior` (not a reused or overloaded
`generate_player_heatmap`) — so the two are never confused in code, tests,
or documentation. Milestone 22's `habit_memory.py` module and its function
names are unchanged and untouched by this ADR; this is a boundary-setting
decision for future work, not a code change.

## Consequences

- CV-context "habit memory," once built, will answer a strictly
  **narrower** question than the StatsBomb-context one: no cross-match
  learning, no named-player identity, no historical corpus spanning a
  whole competition — only a within-video, within-single-track-lifetime
  positional tendency. A future reader must not assume feature parity
  between `generate_player_heatmap`-based blending and a future
  `track_local_positional_prior`-based mechanism; they are different
  capabilities answering different questions, not two implementations of
  the same feature.
- Any future findings document that reports on both must keep them as
  distinct as `CV_PIPELINE_FINDINGS.md` already keeps "validated on a real
  photo" separate from "validated on real broadcast footage with ground
  truth" — never claim CV habit memory "works the same as" the StatsBomb
  one.
- If Track B (jersey-OCR + roster resolution) is ever completed and can
  reliably resolve a `track_id` to a real cross-match identity, this
  boundary could in principle be revisited — at that point, and only
  then, a genuinely cross-match CV-derived prior comparable in kind to
  Milestone 22's becomes possible. That is unscoped future work and is
  neither started nor promised here.

## Explicitly Out of Scope for This ADR

- **Implementing `track_local_positional_prior`.** Deferred, and
  specifically gated behind Track A's camera-motion compensation being
  validated first: `tracker.py`'s own module docstring already establishes
  that this pipeline measures apparent PIXEL velocity, not true player
  velocity, because raw pixel motion conflates player movement with camera
  pan/zoom. The identical confound applies to raw tracked POSITION over
  time within a single shot if the camera moves during that shot — a
  positional prior built from motion-confounded coordinates would silently
  encode camera movement as if it were player tendency. This mechanism is
  not meaningful to build until motion compensation exists to correct for
  that, and motion compensation is itself not yet built.
- **Track B in general** (jersey OCR, roster resolution, any cross-match
  CV-derived identity). Deferred, not started, not solved by this ADR.

## Alternatives Considered

- **Reuse `generate_player_heatmap`/`heatmap_from_buckets` directly,
  substituting `track_id` for `player_id`**: rejected — would silently
  misrepresent an in-match-only, single-tracking-segment signal as if it
  carried the same cross-match historical grounding StatsBomb's
  `player_id`-keyed version has, with no way for a future reader (or the
  model's own feature semantics) to tell the two apart from the function
  name alone.
- **Block all CV habit-memory work until Track B is complete**: rejected
  as the only path forward — an in-match-only positional prior is a real,
  independently useful signal (e.g., "this player has drifted wider in the
  second half of this match") that does not require solving cross-match
  identity first. Gating all CV habit-memory work behind Track B would
  delay a genuinely separable, simpler capability for no necessary reason.
- **Approximate cross-match identity via jersey-color similarity or
  nearest-position matching across matches**: rejected — Milestone 22's
  own module docstring already names this exact category of shortcut as a
  false premise this project refuses to build ("this module does NOT
  invent or guess identity for non-actor players... that would silently
  build on a false premise"). The same principle applies here more
  strongly still: jersey color alone cannot even distinguish two
  same-team players, let alone establish real cross-match identity.
