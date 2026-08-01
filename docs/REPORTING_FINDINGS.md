# Project Athena: Reporting Track Findings

**Status as of Milestone 44** (Milestone 40's original report-generation
layer, Milestone 42's dashboard visualization layer, Milestone 43's
natural-language zone-explanation layer, and Milestone 44's validation
sweep across varied real player/team profiles, all covered below). This
document covers the new Historical Player & Team Analysis reporting layer
(`production/src/reporting/`) -- the first entirely new capability added
since the `RESEARCH_FINDINGS.md`/`CV_PIPELINE_FINDINGS.md` split, and
deliberately given its own home rather than being folded into either: it
is not an RQ1-RQ5 finding (`RESEARCH_FINDINGS.md`'s scope) and has no
CV/video dependency at all (`CV_PIPELINE_FINDINGS.md`'s scope) -- it is a
reporting layer built entirely on top of the StatsBomb track's
ALREADY-VALIDATED data and models (Milestones 1-36).

**This milestone opens no new validation gap.** Every number this layer
produces is derived from real StatsBomb event/360 data and the same
`BiomechanicalPitchControl` engine / deterministically-selected trained
DeepHit MLP already validated in `RESEARCH_FINDINGS.md` -- nothing here is
retrained, and nothing in `production/src/physics`, `spatial`, `models`,
`pipeline`'s training/serving code, or `serving/` was modified to build it.

---

## 1. What was built

`production/src/reporting/`:
- `player_report.py` -- `generate_player_report(player_id, match_ids)`:
  positional distribution, an aggregate positional heatmap (reusing
  `habit_memory.generate_player_heatmap` exactly as built for Milestone
  22), and summary stats (total minutes, primary position, formation
  frequency).
- `team_report.py` -- `generate_team_report(team_name, match_ids)`: an
  aggregate team-level pitch-control heatmap via the existing
  `BiomechanicalPitchControl` engine, and an aggregate DeepHit threat
  pattern by pitch zone and 15-minute game phase.
- `zone_explainer.py` -- `compute_zone_attributions` /
  `aggregate_zone_attributions`: extends Milestone 15's Integrated-
  Gradients explainer from 4 scalar features to per-PITCH-CONTROL-GRID-
  CELL attribution (Step 3, see §4).

## 2. Player Profile Report -- real-data validation

Run against Lionel Messi (StatsBomb `player_id=5503`) across 8 real
matches spanning both club (Barcelona) and international (Argentina) play:

| Positional distribution (% of tagged events) |
|---|
| Right Center Forward: 52.5% |
| Right Wing: 18.6% |
| Center Attacking Midfield: 15.5% |
| Center Forward: 13.4% |

**Methodology note, stated plainly rather than implied:** this is an
EVENT-COUNT share, not a time-weighted one. StatsBomb's `position` field
records a player's current on-pitch role at the moment of each tagged
action; there is no continuous minute-by-minute position ledger in this
data, so an event-count share is the honest, available
operationalization.

Total minutes across the 8 matches: **837.4** (a plausible ~104.7 min/match
average once stoppage time is included). Formation exposure:
`{4231: 94.95, 433: 236.23, 343: 17.2, 442: 258.47, 352: 230.55}` minutes,
primary formation **4-4-2**.

**Formation-field existence, verified rather than assumed** (per this
project's standing "don't fabricate a placeholder" discipline): StatsBomb's
`Starting XI` and `Tactical Shift` events both carry a real
`tactics.formation` int (e.g. 4141, 4231), confirmed by direct inspection
of cached event JSON before this field was relied on.

The positional heatmap reuses the `habit_memory.generate_player_heatmap`
10x7 grid over the verified 100x68m (ADR-002) pitch space -- the SAME
binning code Milestone 22 already validated, not a re-implementation.

## 3. Team Profile Report -- real-data validation

Run against Argentina across 4 real matches (316 qualifying 360-covered
possession-chain frames where Argentina was the acting/attacking side):

**Weakest control zones** (mean pitch-control probability, 10x7 grid):
consistently concentrated in column 0 (Argentina's own defensive third,
x=0-10m) -- e.g. `(col=0, row=6): 0.041`, `(col=0, row=0): 0.044`. This is
football-plausible: a team's own control is naturally lowest deep in its
own defensive corners, where it is rarely the side actually contesting
the ball.

**Threat by pitch zone** (predicted cumulative incidence at the 15s
horizon, `predict_cumulative_incidence` reused exactly from Milestone 13):

| Zone | Mean predicted threat |
|---|---|
| Defensive third | 6.1% |
| Middle third | 12.8% |
| Attacking third | 41.7% |

**A real, checkable football-sanity result**: attacking-third threat is
~7x defensive-third threat, and this direction (attacking > defensive) is
now asserted as a regression test (`test_generate_team_report_real_data`),
not just eyeballed once.

**Threat by 15-minute game phase** ranged 12.4%-20.8% across buckets, with
the highest value in the `90-105'` (stoppage/extra-time) bucket -- plausible
given both small late-bucket sample size and real late-match variance; not
treated as a strong finding on 4 matches, just reported as observed.

## 4. Step 3 — Zone-Level (Grid-Cell) Attribution: Tractable, Built, Validated

**Conclusion, stated up front per the explicit instruction to report this
plainly: Step 3 was tractable within this session's effort, not deferred
to a separate milestone.**

**Why it turned out tractable:** `feature_extractor.extract_features`'s 4
scalar features are themselves just MASKED SUMS over the per-cell control-
probability tensors `BiomechanicalPitchControl` already produces (e.g.
`attacking_control_near_ball = att_control[near_ball_mask].sum()`) — a
computation that is differentiable in principle all the way from grid
cells to the MLP's output. The only thing that breaks this in the
ORIGINAL code is `extract_features` calling `.item()` at the end, which
detaches the result from autograd. `zone_explainer._zone_features_from_grid`
reimplements the exact same four formulas (verified against
`feature_extractor.py`'s source) as a pure, differentiable function of two
new leaf tensors (per-cell attacking/defending control,
`requires_grad=True`) instead of calling the original function — nothing
in `feature_extractor.py`, `control.py`, or the trained MLP was modified;
`zone_explainer.py` only adds a differentiable REPLAY of the existing
aggregation step for one already-computed frame.

**Validated, not just "it ran":** the Integrated Gradients completeness
axiom (sum of per-cell attributions should equal `F(actual input) -
F(baseline)`) was checked against a real match frame and held to within
**0.0008** (tolerance 0.01, matching `test_explainer.py`'s existing
tolerance for the 4-scalar-feature version) — strong evidence the
differentiable reimplementation is mathematically equivalent to the
original, not merely producing plausible-looking numbers.

**Aggregated across many frames** (`aggregate_zone_attributions`, run on
Argentina's 316 qualifying frames): mean attacking-control attribution was
consistently NEGATIVE in the defensive-half columns (0-5) and consistently
POSITIVE in the attacking-third columns (6-9) — a genuine, multi-frame,
evidence-backed spatial pattern of the kind "the right half-space is
systematically open" statements would be built from, not a single-frame
anecdote.

**Scope, stated plainly (what this is NOT yet):**
- Attribution runs back to per-cell CONTROL VALUES for one frame's own
  active-cell set (`BiomechanicalPitchControl`'s existing sparse
  ball-distance mask, ADR-005) — it does NOT (yet) attribute further back
  through the physics ODE/Newton-Raphson solve to player positions or
  velocities.
- The baseline (zero control at every cell, "no team can reach anywhere")
  is a genuinely different KIND of reference point than
  `explainer.compute_attributions`'s "zero normalized feature space =
  average training match state" baseline — an absence-of-control
  reference, not an average-match-state one. The two explainers'
  attributions should not be read as directly comparable in magnitude.
- The multi-frame aggregate result above (Argentina, 4 matches) is a real,
  validated finding at this sample size, but has not been cross-checked
  against a larger match set or a genuinely left-vs-right-skewed team to
  confirm the mechanism recovers a KNOWN tactical asymmetry -- a natural
  next validation step, not performed in this session.

### Extension (Milestone 43) — Natural-Language Zone Explanation Layer

Closes the third layer of the original three-layer ask (state estimation /
threat estimation / tactical explanation) for the zone-level signal
specifically, the same way Milestone 15 already closed it for the 4 scalar
features. Added to `zone_explainer.py`: `identify_notable_zones`
(connected-component clustering of the per-cell attribution grid into
coherent positive/negative regions, not just individually-extreme cells)
and `build_zone_explanation_prompt`, feeding the REUSED, UNMODIFIED
Milestone 15 mock executor (`explainer.generate_explanation`) -- no real
LLM call, same async templated-mock convention as everywhere else in this
project.

**A real bug was found and fixed while building this, on real data, not
synthetic.** The first version of `identify_notable_zones` used one
threshold (`0.5 * max(|grid|)`) shared across both signs. On Argentina's
real aggregate grid, the positive extreme (~0.045) is over 2x the negative
extreme's magnitude (~-0.019) -- a single shared threshold pegged to the
larger side silently suppressed EVERY negative zone, which would have
erased the "systematically weak in the defensive half" half of the pattern
this same document already reported above. Fixed by thresholding each
sign against its OWN extreme value, independently.

**Real generated output, Argentina (4 matches, 316 sampled attacking
frames):**

> Tactical Analysis: The threat is 20.2%. Key drivers:
> attacking_third_central_channel. Mitigating factors:
> middle_third_left_side, defensive_third_central_channel.

Every claim in this traces to a real computed quantity -- the 20.2% is the
mean of `team_report`'s own `threat_by_pitch_zone` values; each named zone
and its magnitude comes from the real, connected-component-clustered
attribution grid (`attacking_third_central_channel`: +0.387, 13 cells;
`middle_third_left_side`: -0.107, 9 cells; `defensive_third_central_channel`:
-0.101, 9 cells) -- checked by an executable test (`test_zone_explainer.py`)
that fails if the explanation mentions any zone not actually computed that
run, not just asserted by eye.

**Two gaps between this output and the original reference example's
style, kept DISTINCT deliberately because they have different causes and
different fixes:**

1. **Prose flatness -- a mechanical consequence of literal executor reuse,
   fixable by writing new generation code.** Reusing `generate_explanation`
   UNMODIFIED (per this milestone's explicit instruction) means the output
   is that function's existing terse "Key drivers: X. Mitigating factors:
   Y." template with underscored zone identifiers, not flowing prose like
   "the right half-space is open." A real LLM call (still not built
   anywhere in this project -- ADR-006 scoped this as future work from
   Milestone 15 onward) or a purpose-built zone-specific templated
   generator would close this gap. This is a presentation-layer
   limitation: the DATA behind it is already real and sufficient; only the
   text-generation step is mechanical.

2. **Duration/movement claims (e.g. "open for 3-5 seconds", "the
   center-back is recovering backward") -- a DATA limitation, NOT a
   prompting limitation, and this distinction matters enough to state
   plainly rather than let it flatten into "needs a better prompt" later.**
   `aggregate_zone_attributions` is a STATIC, per-frame-AVERAGED spatial
   pattern -- there is no frame-to-frame trajectory analysis, no
   transition-state model, no player-specific movement-direction
   computation ANYWHERE in this project, StatsBomb track or CV track. No
   prompt engineering, no smarter mock executor, and no real LLM call
   could honestly produce a duration or recovery-direction claim from this
   data, because the information a duration claim would need to be true
   (how a zone's openness evolves frame-to-frame, which specific player is
   moving and in which direction) was never computed in the first place --
   it isn't sitting in the data unused, it doesn't exist. Closing this gap
   requires building an actual temporal/transition-state layer (the
   "state estimation" layer's own unbuilt piece, not the "tactical
   explanation" layer this milestone addresses) -- a new, separate,
   unscoped piece of work, not a prompt-writing exercise. Both
   `build_zone_explanation_prompt`'s prompt text and
   `test_zone_explainer.py`'s honesty check enforce this directly (the
   prompt instructs against such claims explicitly; the test regex-checks
   the generated output for "second"/"minute"/"recovering"/etc. and fails
   if any appear) precisely so this constraint cannot be silently dropped
   by a future edit that only touches the prompt wording.

## 5. Output Format

Every function above returns a plain, JSON-serializable `dict` (nested
lists for grids, floats, strings) — structured data, per this milestone's
explicit Step 4 instruction, not a rendered visualization. A dashboard/
visualization layer matching any reference layout is explicit, separate
follow-up work.

## 6. Future Work

1. Validate the zone-explainer's aggregate attribution pattern against a
   team/match set with a KNOWN real tactical asymmetry (e.g. a team that
   demonstrably attacks down one flank far more than the other), to
   confirm the mechanism recovers a signal known independently of the
   model, not just an internally-consistent one.
2. Extend `compute_zone_attributions` further back through the physics ODE
   to player positions/velocities, if a "which PLAYER'S positioning drove
   this threat" statement (rather than "which ZONE") becomes a real
   reporting requirement.
3. A visualization/dashboard layer rendering these reports' grids as
   actual heatmap images (e.g. via the Streamlit dashboard's existing
   frontend), once the underlying numbers here have had further scrutiny.
4. Broaden the player/team validation beyond the 8-match Messi / 4-match
   Argentina samples used here, once a specific downstream use case
   defines what sample size that use case actually needs. **Partially
   addressed by the Milestone 44 validation sweep (§7 below)** -- that
   sweep tested VARIETY (a substitute, a multi-role player, a goalkeeper,
   a 1-match team, an 8-formation team) rather than raw sample SIZE; a
   larger-N validation for a specific downstream use case remains open.
5. Build an actual temporal/transition-state layer -- a genuinely new
   piece of "state estimation" work (frame-to-frame trajectory analysis,
   a model of how a zone's openness evolves over time, player-specific
   movement-direction computation), NOT a Milestone 43 prompt-tuning task.
   This is the one, and only, thing standing between the current
   zone-explanation output and the original reference example's full
   style ("open for 3-5 seconds", "recovering backward") -- see Milestone
   43's extension entry in §4 for why no amount of better prompting or a
   smarter mock/real LLM executor can substitute for this: the underlying
   temporal information doesn't exist in any dataset or computation this
   project currently has, so there is nothing for a prompt to surface.

## 7. Validation Sweep (Milestone 44)

Everything through Milestone 43 was validated on exactly ONE player
(Messi, 8 matches) and ONE team (Argentina, 4 matches/316 frames) --
real data, but a narrow slice of the real variety this reporting layer
will actually see. This sweep deliberately selected VARIED, real cached
StatsBomb profiles most likely to stress assumptions Messi/Argentina
happened not to exercise, ran the report generators against each, and
reports the result honestly -- this was a validation pass, not a
rewrite; the report-generation logic was changed only where a genuine,
demonstrated gap was found.

**Test cases run:**

| # | Case | Player/Team (real id) | Data | Result |
|---|---|---|---|---|
| 1a | Zero-event substitute | Kristijan Jakić (32602), match 3869684 | Came on at 94.9', 0 tagged events | No crash. `positional_distribution={}`, `primary_position=None` -- correctly empty, not fabricated. `total_minutes_played=1.28` -- correctly non-zero (derived from substitution timing, independent of event tagging). Heatmap fell back to `habit_memory`'s own uniform cold-start prior (< 20 qualifying events), with a console warning already printed by that existing code. |
| 1b | Near-zero-event substitute | Yu-Min Cho (99479), match 3857262 | 1 tagged event | No crash, but a REAL GAP: `positional_distribution` returned `{"Right Center Back": 1.0}` -- a "100%" figure structurally indistinguishable from Messi's or a goalkeeper's genuinely well-supported 100%/52.5% figures. **This is the finding that drove the fix below.** |
| 2 | Multi-position player | Amine Adli (33401), 8 Bayer Leverkusen matches | 5 matches appeared in, 4 distinct tagged positions | No crash. `positional_distribution` correctly spread across 4 real roles (Left Attacking Midfield 73.0%, Left Wing 20.1%, Right Midfield 5.6%, Right Wing Back 1.3%), summing to 1.0. `formation_minutes` summed to within rounding of `total_minutes_played` across 5 different formations. Handled sensibly, no special-casing needed. |
| 3 | Goalkeeper | Lukáš Hrádecký (8667), 8 Bayer Leverkusen matches | 8/8 matches appeared in, 2346+ events | No crash. `positional_distribution={"Goalkeeper": 1.0}` (correctly, a real goalkeeper). Heatmap correctly concentrated entirely in columns 0-4 (near his own goal), zero density beyond -- a genuinely different, structurally sane spatial profile handled with no goalkeeper-specific code. `formation_minutes` summed correctly across 7 formations. |
| 4 | Team with 1 cached match | Barcelona, match 3773386 | 1 match, the only one cached with both events+360 for this team | No crash. `matches_used=1` (honestly reported, already an existing field). All 70 grid cells populated; weak-zone pattern and threat-by-zone/phase all structurally identical in shape/confidence to the 4-match Argentina report, with `matches_used` as the only differentiator -- see finding below. |
| 5 | Team with unusually varied formation history | Bayer Leverkusen, 8 of 34 cached matches (8 distinct formations used across the full 34: 343/352/442/3412/3421/3511/4231/4411, vs. Argentina's 4) | 8/8 matches used | No crash in `team_report.py` (which has no formation logic at all -- that lives entirely in `player_report.py`, already exercised by cases 2/3 above, both under this same team's formation history, both summing correctly). `team_report.py`'s own weak-zone/threat pattern was structurally sane and consistent with the Argentina/Barcelona results. |

**Real problems found: one, in Case 1b (and by extension, Case 1a's
heatmap path). No other case revealed a genuine bug.**

`generate_player_report`'s `positional_distribution` had NO sample-size
indicator anywhere in its output -- a percentage computed from 1 event
and a percentage computed from 1,838 events (Messi's real count) were
returned in an identically-shaped, identically-"confident-looking" dict,
with no way for a caller (or `player_visualizer.py`) to distinguish them.
`heatmap_grid` had a PARTIAL version of the same gap: `habit_memory.
generate_player_heatmap` already degrades gracefully below its own
`MIN_HISTORICAL_EVENTS=20` cold-start threshold (falling back to a
uniform grid), but that fallback was only ever announced via a `print()`
-- invisible to any caller that isn't watching the console, including
every API/renderer consumer of the returned dict.

**Fix applied (minimal, additive -- no existing field, key, or behavior
changed):** `generate_player_report` now also returns
`positional_distribution_event_count` (the real count backing those
percentages) and `heatmap_event_count` / `heatmap_used_uniform_fallback`
(whether `habit_memory`'s own real, existing `MIN_HISTORICAL_EVENTS`
threshold was cleared -- reusing that constant directly, not a
re-invented number). `player_visualizer.py` was extended to consume
these: the positional-distribution panel's title now always states the
real event count and, below the threshold, draws an explicit red "LOW
SAMPLE" banner; the heatmap panel's title states its own event count and
draws a red "UNIFORM FALLBACK" banner when triggered; the summary-text
panel lists both counts plainly. Verified visually against both a
1-event case (banners render, clearly) and Messi's well-supported case
(no false-positive banners). Two new regression tests
(`test_generate_player_report_low_sample_transparency_real_data`,
`test_generate_player_report_zero_events_no_crash`) encode this finding
against the real Cho/Jakić data directly, plus new assertions on the
existing Messi test confirming the fields read as confident there.
144 tests pass total, no regressions.

**Case 4 (Barcelona, 1 match) — a related but DISTINCT observation, NOT
treated as a bug requiring a fix.** `team_report.py`'s per-match
`matches_used` field already existed and already gets surfaced in
`team_visualizer.py`'s caption -- unlike `positional_distribution`, this
report type already had a transparency mechanism before this sweep. The
sweep confirms it works (Barcelona's report honestly says
`matches_used=1`) but does NOT close the same gap `team_report.py`'s
`team_visualizer.py` caption already flagged on its own: `matches_used`
is a MATCH-level count, not the FRAME-level count the underlying
pitch-control aggregation actually averages over (a large match with many
360-covered chains and a match with few contribute unequally per cell,
and `matches_used` alone can't distinguish that) -- see `team_visualizer.
py`'s own caption text for that already-documented caveat. Not fixed
here, for the same reason it wasn't fixed when first found: doing so
would require changing `generate_team_report`'s return contract, which
this validation sweep's own scope (fix genuine bugs minimally, don't
preemptively rewrite) does not call for absent a second, independent
finding that it's actively misleading -- it currently is not, since
`matches_used` is honestly labeled for what it is.

## 8. New Data Source: Season-by-Season Team Trend Data (football-data.co.uk)

A new, SEPARATE report type (`production/src/reporting/team_trend_data.py`)
built on top of a new, non-StatsBomb data source: football-data.co.uk's
match-results/team-stats CSVs for the top flight of the "big five"
European leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1),
through the 2025/26 season.

### 8.1 Scope boundary (read before touching this code)

football-data.co.uk provides **match results and team-level stats
only** — goals, shots, corners, cards, fouls, home/away record. It
carries **no event-level data and no pitch coordinates**. This means it
can **never** feed `BiomechanicalPitchControl`, and this module produces
**no heatmap and no pitch-control weak-zone analysis** — that remains
`team_report.py`'s job, entirely unmodified and never imported here.
`team_trend_data.py` answers a genuinely different question ("how has
this team's results/output trended year over year") from
`team_report.py`'s ("where is this team spatially strong/weak, aggregated
from historical StatsBomb match footage"). The two are never combined
into one number anywhere in this codebase, and nothing in
`production/src/reporting/team_report.py` or the physics/spatial/models
stack it depends on was imported or modified to build this feature.

### 8.2 Data source & license basis (verified directly, not assumed)

- football-data.co.uk's own `notes.txt` page states **no explicit
  license or redistribution terms** for the data.
- The main site frames the data as free, but its own stated scope is
  narrower than a general research-use grant: **"All data provided by
  Football-Data are made available for the purposes of league match
  prediction only"** (site's own wording), plus a disclaimer of
  responsibility for data accuracy.
- The PDDL-licensed GitHub mirror (`footballcsv/england` and siblings)
  was checked directly and found **stale — stuck at the 2020-21
  season**, more than four seasons behind. It cannot be used for a
  report claiming 2025/26 coverage, independent of the licensing
  question.
- **Conclusion:** this module fetches directly from football-data.co.uk's
  own CSV endpoints (confirmed live and current — a complete, played-out
  2025/26 Premier League season, 380 matches, last match dated
  24/05/2026, was downloaded and read directly while building this
  module), not the stale mirror.

**Compliance scope, stated plainly (this is a real, unresolved licensing
ambiguity, not a cleanly-licensed source):** no clean license (MIT, PDDL,
CC-BY, or similar) was found anywhere for these files, and the site's own
stated scope is narrower than general research use. This is handled the
same conservative way **ADR-014** handles the AGPL-derived pitch-keypoint
model: scoped explicitly to strictly **personal, non-distributed research
use only**. The cached CSVs and any report this module produces are for
local analysis, not for republishing or redistributing the underlying
data; nothing here is wired into `production/src/serving/api.py`'s live
WebSocket/REST layer or any other network-served endpoint. If this
feature is ever extended toward a served/distributed use case, this
licensing question must be revisited and resolved first — this is the
current conservative stance, not a permanent green light.

### 8.3 Schema verification (checked across leagues, not assumed uniform)

England's CSVs (`E0`) include a `Referee` column that Spain/Germany/
Italy/France's (`SP1`/`D1`/`I1`/`F1`) do **not** — confirmed by directly
downloading and diffing headers across all five current-season files, not
assumed from one league's schema. This module never reads `Referee`, so
the difference doesn't affect it, but it is real — exactly the class of
assumption this project's own history (§4b above) has repeatedly found
costly to skip checking. The columns this module *does* depend on —
`Date`, `HomeTeam`, `AwayTeam`, full-time goals/result, shots,
shots-on-target, fouls, corners, yellow/red cards — were confirmed
present under identical names across all five leagues' current-season
files.

### 8.4 Real-data validation

**Man City, 2019/20–2025/26 (a team expected to have zero gaps):** all 7
requested seasons found, all year-over-year deltas correctly flagged
`consecutive: true`. The numbers match known real history: 93 points
(2021/22 title), 89 (2022/23 treble season), 91 (2023/24 title), then a
real, correctly-captured drop to 71 points in 2024/25 and partial
recovery to 78 in 2025/26.

**Norwich, 2018/19–2025/26 (a team expected to have relegation gaps):**
only 2 of 8 requested seasons found in the top-flight data — 2019-20 (21
points, relegated) and 2021-22 (22 points, relegated again). The other 6
seasons (`2018-19, 2020-21, 2022-23, 2023-24, 2024-25, 2025-26`) are
explicitly listed in the report's `gap_seasons` field, never silently
dropped. The one computed year-over-year delta (2019-20 → 2021-22) is
correctly flagged `"consecutive": false`, since a Championship season
sits between the two found top-flight seasons — never presented as an
adjacent-year comparison it isn't.

Both runs used real, live-fetched data (no synthetic/mocked input), and
both were checked against known real football outcomes, not just
"it ran without crashing."

## 9. General-Purpose Team-Season Style Comparison (StatsBomb-based)

A new, general-purpose tool (`production/src/reporting/team_comparison.py`)
answers a different question from §8's football-data.co.uk trend
reports: not "how have this team's results trended" (results/stats only,
no coordinates), but "how does this team's real playing-style/activity
distribution compare between two team-seasons" — built entirely on
StatsBomb event data via `compare_team_seasons(team_a, season_a, team_b,
season_b)`. It is fully general (any two `(team, season)` pairs), not
hardcoded to the examples validated below.

### 9.1 Design: two analysis modes, auto-detected, never mixed

**`pitch_control_360`** — used ONLY when BOTH team-seasons have at least
one 360-covered match. Reuses `team_report.generate_team_report`
EXACTLY (the existing, unmodified `BiomechanicalPitchControl` weak/
strong-zone analysis) for both sides — nothing about that function was
touched to build this tool.

**`event_location_activity_map`** — the fallback, and by far the common
case. Aggregates each team's own raw event `location` fields (no 360, no
pitch-control physics) into the same 10x7 grid `habit_memory`/
`player_report`/`team_report` already use, as a density (each grid sums
to 1.0, so two team-seasons with very different total event counts stay
directly comparable cell-by-cell). Relies on ADR-009's established
finding that StatsBomb records locations already relative to the ACTING
team's own attacking-left-to-right frame — without that guarantee,
aggregating "defensive third" activity across many different matches
would not mean the same real part of the pitch each time.

**The mode choice is per-comparison, automatic, and never blended**: if
either side lacks 360 data, BOTH sides run in location-only mode, so a
360-based result is never diffed against a location-only result for the
other team. Verified directly against the LIVE `competitions.json` (not
assumed from memory): **only 8 season/club combinations in the entire
open-data catalog have any 360 coverage at all** — Barcelona 2020/21,
Bayer Leverkusen 2023/24, PSG 2021/22, PSG 2022/23, and a handful of
international tournaments (World Cup 2022, Euro 2020/2024, Women's Euro
2022/2025, Women's World Cup 2023) — and even those are per-match, not
guaranteed season-wide (a season's `match_available_360` flag on
`competitions.json` can be set while only a fraction of its individual
matches actually carry 360 data, which is why this tool checks each
match's own `match_status_360` field rather than trusting the
season-level flag). **Every other team-season, including both examples
validated below, runs in location-only mode.**

### 9.2 Data-richness discipline (Milestone 44's pattern, reused exactly)

`LOW_SAMPLE_MATCH_THRESHOLD = 10`, named and stated openly the same way
`habit_memory.MIN_HISTORICAL_EVENTS` is. A team-season built from fewer
matches than this is flagged `"LOW SAMPLE -- too thin for season-level
claims"` in the output's `data_richness` field, AND — when either side is
thin — a plain-language `reliability_caveat` string is attached directly
to the top-level comparison output (not buried in a nested per-side
field), naming both match counts and the ratio between them.

### 9.3 Validation Run 1 — Barcelona 2008 vs. Barcelona 2015 (both well-supported)

**Mode:** `event_location_activity_map` (neither side has 360 data).
**Data richness:** 2008 → 32 matches (31 La Liga + 1 Champions League,
resolved automatically across BOTH competitions sharing that season —
not hardcoded to La Liga alone), 2015 → 38 matches — both
`well-supported`, `reliability_caveat: None`.

**Real, traceable style diff:** *"The largest activity difference is in
the middle third: Barcelona 2008 concentrated 61.4% of its located
events there vs. Barcelona 2015's 56.8% (1.1x)."* Full zone breakdown:
2008 (`defensive_third=14.2%, middle_third=61.4%, attacking_third=24.4%`)
vs. 2015 (`defensive_third=18.5%, middle_third=56.8%,
attacking_third=24.7%`) — a modest but real, numerically-grounded shift
toward more own-defensive-third activity by 2015, consistent with the
two different real tactical eras these seasons represent.

**A real bug was found and fixed during this run, not hypothesized in
advance.** Since `team_a == team_b` here (comparing one club across two
eras — an expected, common use case, not an edge case), the
auto-generated summary sentence originally read "Barcelona concentrated
X%... vs. Barcelona's Y%" — genuinely ambiguous about which season was
which. Fixed by season-qualifying every label used in summary/diff text
(`"Barcelona 2008"` vs. `"Barcelona 2015"`) throughout both analysis
modes, not just patched in the one function where it was first noticed.

### 9.4 Validation Run 2 — Real Madrid 2016 vs. Barcelona 2008 (deliberately mismatched sample sizes)

**Data richness:** Real Madrid 2016 → **3 matches** (2 La Liga + the
2016/17 Champions League final vs. Juventus) → **`LOW SAMPLE`**;
Barcelona 2008 → 32 matches → `well-supported`.

**Reliability caveat, visible in the comparison output itself:**
> *"CAVEAT: Real Madrid 2016's side of this comparison is built from
> only 3 match(es), while Barcelona 2008's side has 32 (11x more). This
> comparison is NOT equally reliable on both sides — treat Real Madrid
> 2016's numbers as illustrative at best, not a real season-level
> characterization."*

Both Step-2 requirements confirmed on real data: the low-sample flag
fires correctly for the thin side, and the caveat is a first-class,
visible field on the comparison itself — never just a code-level flag a
caller could silently ignore.

## 10. Individual Player-Era Style Comparison — Enabled by the Data-Fallback Coverage Expansion

### 10.1 What made this possible: `data_fallback.py`

Every reporting tool through §9 required the caller to already know
which real StatsBomb `match_id`s to pass in. A new module,
`production/src/reporting/data_fallback.py`
(`find_or_fetch_team_matches`/`find_or_fetch_player_matches`), removes
that requirement: given a team name, or a player_id (optionally
narrowed by candidate team names), it searches every competition/season
in the LIVE `competitions.json` index — never a hardcoded list — and
fetches/caches whatever match-lists or event files aren't already
local. This had been discussed/planned earlier in this project's
history but never actually written into any file; confirmed by an
exhaustive codebase search before it was built.

Run against Lionel Messi's full real career (`player_id=5503`, searched
across 48 real competition-seasons — all 18 StatsBomb-covered La Liga
seasons, both his Ligue 1 PSG seasons, 3 Champions League seasons, Copa
América 2024, and the 2018/2022 World Cups): **596 total real matches
found, 588 of them new** beyond the 8 previously cached. The local
StatsBomb cache grew from 339 to 1,011 files (224 to 830 event files)
as a direct, measured result — see the standalone coverage-expansion
report for the full before/after breakdown, including the honest
negative findings along the way (Bayern Munich's real 2015/16 Bundesliga
release only covers 2 matches, not a full season; West Germany does not
appear in the World Cup 1990 release at all — only Argentina v. Brazil
does).

### 10.2 `player_comparison.py`: design

A new tool, `production/src/reporting/player_comparison.py`
(`compare_player_seasons`/`compare_player_across_eras`), mirrors
`team_comparison.py`'s proven design for individual players instead of
teams — reusing `generate_player_report` (unmodified), `data_fallback.py`
for match resolution, and `team_comparison.py`'s own `_zone_shares`/
`_season_start_year` helpers directly rather than reimplementing them a
second time.

**One real architectural difference from `team_comparison.py`, stated
plainly:** `team_comparison.py` has two genuinely different analysis
modes because `generate_team_report` itself branches on 360
availability. `generate_player_report` does not — it never calls
`fetch_match_360` or `BiomechanicalPitchControl`, only
`fetch_match_events` and `habit_memory.generate_player_heatmap`. So
`player_comparison.py` has exactly ONE real mode (event-location/heatmap
based). It still checks 360 availability per season and reports a
`pitch_control_diagnostic` — but honestly labeled as informational only
("would a future pitch-control-level comparison be possible in
principle"), never presented as a real second mode the underlying reused
function doesn't actually have.

**Positional role diff, kept as its own explicit section, separate from
the spatial zone diff:** the one thing genuinely specific to comparing a
player (not a team) across eras — their tagged on-pitch role can itself
change. `generate_player_report`'s existing `positional_distribution`
field is diffed directly for this.

**Data richness:** reuses `habit_memory.MIN_HISTORICAL_EVENTS` directly
as `LOW_SAMPLE_EVENT_COUNT_THRESHOLD` — the same threshold
`player_report.py`/`player_visualizer.py` already use for this exact
signal (Milestone 44's discipline), not a new number.

### 10.3 Real findings — Messi, three real career eras

All three eras (2006-07: 26 matches/4,665 events; 2014-15: 39
matches/9,988 events; 2022-23: 39 matches/8,864 events) confirmed
well-supported before being compared — no low-sample flag fired
anywhere.

**2006-07 (early Barcelona) vs. 2014-15 (peak Barcelona).** Positional
distribution, 2006-07: `Right Wing 73.7%, Left Wing 17.6%, Center
Forward 6.4%, Right Center Forward 1.6%, Right Midfield 0.7%`. 2014-15:
`Right Wing 68.7%, Center Forward 28.7%, Right Center Forward 1.8%, Left
Center Forward 0.8%`. **Not the clean "winger → false-9" story a prior
assumption might have predicted**: Right Wing remained his single
largest tagged role in both eras. What actually shifted, and by the
largest margin (+22.3 percentage points), was Center Forward involvement
nearly quadrupling (6.4% → 28.7%) — a real role-broadening, not a full
positional switch. Spatial activity moved modestly toward the attacking
third (40.0% → 45.6%, 1.1x) and the left half (30.9% → 35.4%).

**2014-15 (peak Barcelona) vs. 2022-23 (PSG).** Positional distribution,
2014-15: as above. 2022-23: `Right Center Forward 53.9%, Right Attacking
Midfield 20.8%, Right Wing 10.8%, Center Forward 9.4%, Center Attacking
Midfield 5.1%`. **The largest single shift across either comparison:**
Right Wing collapsed 68.7% → 10.8% (−57.9 percentage points), and an
entirely new role — Right Attacking Midfield (20.8%) — appears with zero
presence in 2014-15. **A genuine cross-check, not a repeated signal:**
the independent spatial heatmap diff (computed from raw event
coordinates, never from the position tags) tells the identical real
story — middle-third share rose 53.2% → 59.3% while attacking-third
share *fell* 45.6% → 37.9%, and left-half share rose 35.4% → 45.4%,
consistent with a deeper, more central, withdrawn creative role at PSG
versus peak-Barcelona's advanced wide/central threat. Two independently
computed signals agreeing on the same conclusion is real, meaningful
confirmation, not one metric assumed to match the other.

**One honest, correctly-handled edge case:** the 2022-23 season's 39
matches are ALL 360-covered (32 Ligue 1 + 7 World Cup 2022 matches, both
fully 360-covered competitions) — yet `pitch_control_possible_in_principle`
still correctly reports `false` for the 2014-15-vs-2022-23 comparison,
since 2014-15 has zero 360 matches and the diagnostic requires both
sides. Exactly the informational-only behavior the design calls for, not
a missed opportunity to do more than `generate_player_report` actually
supports.

### 10.4 Why this is worth naming plainly

This is the first genuine multi-era "how has an individual's style
evolved" analysis in this project, made possible specifically by §10.1's
coverage expansion — not a mock, not a single hardcoded example, but
real evidence, from real event data, of how one player's actual tagged
role and spatial behavior changed across three real, well-supported
points in a real career, with two independently computed signals
agreeing on the underlying story rather than one being assumed to
confirm the other.
