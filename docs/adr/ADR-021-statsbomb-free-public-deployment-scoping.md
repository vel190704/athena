# ADR-021: Scoping a Free, Non-Monetized Public Deployment Under the StatsBomb Open Data License

## Status
Accepted

## Context

ADR-020 established that the physics-ML track's RQ1-RQ5 findings
(DeepHit, GNN, Kalman) cannot be **commercially** deployed under
StatsBomb's Public Data User Agreement, because clause 1.2.2's ban on
"commercially exploit[ing] the data or any analysis derived from the use
of the Service" plainly reaches trained-model outputs, not just the raw
data.

That leaves an unanswered, narrower question: the actual goal here is a
**free, publicly-deployed website — not sold, no ads, no monetization —
built on this project's current StatsBomb-derived research findings and
reporting layer** (the player/team report and shot-map features). Is
that specific use case inside or outside clause 1.2.2's prohibition, and
is it meaningfully different from the "personal, non-commercial research
use" this project has operated under so far?

### Re-reading clause 1.2.2 and its surrounding context directly

The full Agreement (`LICENSE.pdf`, read in full for ADR-020, re-checked
here specifically for this question) contains **no definitions section**
anywhere in its 5 pages. "Service" and "User" are defined inline in the
preamble; "commercially exploit" is used in 1.2.2 but **never defined**.
There is no clause anywhere that distinguishes "public" from "private"
use, no mention of "website," "deployment," "hosting," "monetization,"
"advertising," or "product" at all. The Agreement was evidently not
drafted with this specific scenario in mind either way — it is silent by
omission, not by a considered distinction.

The only two textual anchors available to reason from:

1. **The stated purpose (1.1 and the preamble)**: the Service is "to be
   used for analysis, research and to facilitate the shared ideas &
   understanding of the data," and the preamble adds: "Any analysis or
   conclusions that are created as a result of using this data, **may be
   shared publicly**." This explicitly contemplates public sharing of
   *analysis/conclusions* — the exact activity a public reporting-layer
   website would be doing. This cuts toward "in-bounds."

2. **Clause 1.2.2's actual word, "commercially"**: on an ordinary reading,
   "commercially exploit" implies some commercial element — a sale, a
   fee, advertising revenue, lead generation, or business use — not mere
   public visibility. A site with zero monetization of any kind has a
   real textual argument that nothing "commercial" is occurring. This
   also cuts toward "in-bounds," provided monetization is genuinely zero.

**However, a separate, independent restriction complicates this — and
must not be overlooked just because the question was framed around
1.2.2.** Clause **1.2.1** bans, with **no commercial qualifier at all**:

> "edit, distort, distribute, reproduce, sell or in any way provide the
> data to any external or third party"

This clause doesn't care whether money changes hands. A public website
inherently makes its content available to an unbounded set of "external
third parts" — every visitor. If the site exposes the underlying
StatsBomb event-level data itself (raw shot coordinates, a queryable
event feed, a downloadable JSON export, or anything a visitor could use
to reconstruct the original dataset), that risks 1.2.1 regardless of
whether the site is free. This project's own shot-map feature, for
example, plots real per-shot `location` and `statsbomb_xg` values — that
is closer to visually redistributing the underlying data than it is to
"analysis" in the abstract sense the preamble describes (a written
conclusion, a trend, a percentile). Aggregated statistics, summaries, and
model-derived findings sit more comfortably under "analysis... shared
publicly"; a literal plot of the raw per-event values sits closer to
"providing the data."

### Searching for public precedent

Searched directly for any StatsBomb statement or documented
football-analytics-community consensus specifically addressing free,
non-monetized public deployments (dashboards, demo sites, portfolios)
under this license.

**Found no authoritative resolution either way.** The one directly
relevant result is
[statsbomb/open-data issue #47](https://github.com/statsbomb/open-data/issues/47),
"Clarification regarding license requirements for publications," opened
27 September 2024 by a user asking StatsBomb for exactly this kind of
guidance. **It has no maintainer or StatsBomb-team response and remains
open** as of this investigation. Secondary sources (Medium write-ups,
academic wiki pages) restate the attribution requirement but do not
address the commercial-vs-public-non-monetized distinction at all — they
were written by third parties, not StatsBomb, and are not authoritative.
No case of StatsBomb taking enforcement action against a free public demo
was found either, but absence of a known enforcement case is not
evidence of permission — StatsBomb explicitly "reserves the right to...
take any such measures it deems necessary" (clause preceding 1.3), and
silence from an unanswered GitHub issue is not a green light.

**Conclusion: this is genuinely unresolved by the text and unresolved by
any available external precedent — not a case where careful reading
yields a confident answer.** Per this project's established discipline
(ADR-014: don't resolve genuine ambiguity by assumption, scope
conservatively instead), the finding is reported as such rather than
softened into false confidence in either direction.

## Decision

**A free, zero-monetization public deployment of this project's
StatsBomb-derived reporting/findings is treated as CONDITIONALLY
IN-BOUNDS, under an explicit, conservative scope — mirroring ADR-014's
resolution of the AGPL ambiguity via constrained deployment rather than
by assumption.** All of the following constraints apply simultaneously;
none is optional:

1. **Zero monetization of any kind, permanently, not just at launch.**
   No ads, no donation/tip links, no premium tier, no email capture for
   resale or marketing, no affiliate links, nothing that generates or
   could plausibly be read as generating revenue or business value from
   the site. This is the fact load-bearing the entire "not commercial"
   argument under 1.2.2 — if it stops being true, this decision no longer
   applies and must be revisited.
2. **No raw StatsBomb data exposed to site visitors, in any form** — no
   public API endpoint returning event-level records, no downloadable
   JSON/CSV export of the underlying data, no interactive table of raw
   events. Only pre-aggregated, derived analysis and static visualizations
   (the existing player/team reports and shot-map renders, as produced —
   not the raw `shots` list or event feed itself) may be public-facing.
   This directly avoids clause 1.2.1, which has no commercial exception
   and is the harder, unconditional restriction identified above.
3. **Explicit "research demo, not a product" framing on the site
   itself** — a visible, permanent disclaimer (footer or about page)
   stating this is a free, non-commercial research/portfolio
   demonstration, not a commercial product or service.
4. **StatsBomb brand-logo attribution displayed on the site**, per
   clause 1.4 — already flagged as an existing compliance gap in ADR-020,
   now a hard requirement for any public deployment specifically, not
   just publications.
5. **This scoping decision is deliberately conservative, not a resolved
   legal conclusion.** It reduces risk to a level consistent with this
   project's existing research/portfolio framing; it does not constitute
   confirmation from StatsBomb that this use case is permitted.

## Consequences

- The current player/team report and shot-map reporting layer can be
  deployed publicly under the constraints above without waiting on
  StatsBomb's own unanswered issue #47.
- ADR-020's underlying conclusion is unchanged: this remains a
  **non-commercial** deployment, not a resolution of whether commercial
  deployment is permitted — a paid tier, ads, or any monetization
  mechanism added later would immediately fall back under ADR-020's
  existing prohibition and require its own separate resolution (a
  commercial StatsBomb license, or a different dataset).
- Any future feature that would expose raw event-level data publicly
  (a public API, a data-export button, an interactive raw-event browser)
  must be treated as a new licensing question under clause 1.2.1, not
  assumed covered by this ADR's scoping — this ADR's "no raw data
  exposed" constraint is a condition of its own applicability, not a
  general clearance.
- The genuine ambiguity around clause 1.2.2 for the free/public case
  remains formally unresolved with StatsBomb. Reaching out to StatsBomb
  directly (via their resource-center registration channel, referenced in
  the Agreement's own preamble) to seek explicit written confirmation
  for this specific use case is a real, available next step — named here
  as an option, not undertaken as part of this ADR.

## Alternatives Considered

- **Treat the absence of an explicit prohibition on "free public
  deployment" as implicit permission**: rejected — the same "don't
  resolve ambiguity by favorable assumption" discipline ADR-014 and
  ADR-020 both apply. Silence in a contract is not consent, and clause
  1.2.1's unconditional third-party-provision restriction means "free"
  alone does not clear every risk in this document.
- **Treat the deployment as prohibited outright, on the theory that any
  public-facing site is inherently "exploiting" StatsBomb's brand/data
  for this project's visibility/reputation**: considered, but rejected as
  overcorrecting past what the text supports — the preamble explicitly
  anticipates public sharing of analysis/conclusions, and "commercially"
  in 1.2.2 is a real, load-bearing word that a genuinely zero-monetization
  site has a legitimate argument against being caught by. Conservative
  scoping (this ADR's actual decision) was preferred over an outright ban
  that the text doesn't clearly require either.
- **Wait for a response to GitHub issue #47 (or file a new, more specific
  inquiry) before deploying anything**: a legitimate, more risk-averse
  path — not chosen as the default here because the issue has sat
  unanswered for over a year with no indication of a response timeline,
  and the conservative constraints in the Decision section substantively
  reduce risk in the meantime without an indefinite wait. Still named as
  the single action that would actually convert this from "conservatively
  scoped" to "confirmed," for a future contributor who wants certainty
  before expanding beyond this ADR's constraints.

## Update: Condition 2 Enforced for the Shot Map via a Public/Private Mode Switch

A real, dedicated compliance audit of every user-facing feature against
condition 2 ("no raw StatsBomb data exposed to site visitors, in any
form") was run after this ADR was accepted. It found exactly one
violation: the shot map (`generate_player_shot_map` /
`render_shot_map`), added after this ADR was originally written, returns
and renders each shot's exact, individually-located `(x, y)` coordinate
and real `statsbomb_xg` — directly traceable to one specific StatsBomb
Shot event, precisely the thing condition 2 exists to prevent. Every
other reporting feature audited (positional distribution, the aggregate
heatmap, the team-report pitch-control heatmap, team comparison, the
What-If simulator, the StatsBomb-sourced live tactical stream) was
already condition-2-compliant by construction — collapsed into counts,
shares, or per-cell means with no single event individually recoverable.

**Resolution, following this project's own ADR-014 precedent exactly**:
scope the constraint, do not remove the capability. The shot map's raw,
individually-plotted form remains fully available for LOCAL/private
research use, completely unchanged — nothing about it was deleted or
degraded for that use case. A second, ADR-021-condition-2-compliant
variant now exists alongside it:

- `player_report.generate_player_shot_map_aggregated` bins shots into the
  same `GRID_COLS x GRID_ROWS` grid `habit_memory`'s own positional
  heatmap already uses, returning shot-density and mean-`statsbomb_xg`
  PER CELL — no per-shot list, no individually-recoverable location.
- `player_visualizer.render_shot_map_aggregated` renders that as two
  binned heatmaps (density, mean xG), styled consistently with the
  existing aggregate positional-heatmap panel — no individual shot marker
  anywhere.
- A new, explicit, visible `PUBLIC_DEPLOYMENT` environment-variable flag
  (`api.py`, checked once at startup, documented in `README.md`) decides
  which variant `/reports/player/{player_id}/shot-map` serves: unset
  (default) serves the real per-shot data, byte-for-byte unchanged from
  before this fix; `true` serves ONLY the aggregated variant — the raw
  per-shot list is never even computed on that path, not merely withheld
  from an already-built response.
- `dashboard.py` mirrors the same flag client-side, with an additional
  defense-in-depth check: it also inspects whether the actual API
  response it received still carries a raw `shots` field, and refuses to
  render or display anything (fails closed with a visible configuration
  error) if its own flag says public but the response says otherwise —
  a compliance boundary should not rely on a single unverified signal.

Not chosen: deleting the raw shot-map feature outright. Rejected for the
same reason ADR-014 rejected deleting the CV pitch-keypoint model over
its own AGPL ambiguity — the licensing constraint is specific to PUBLIC
exposure of individually-located events; it says nothing against a
researcher, running this project locally for their own analysis, plotting
real shot locations for their own use. Removing the capability entirely
would over-correct past what condition 2 actually requires.

## Update: Condition 2 Applied to the Pass Network (Step 0 Decision)

A new reporting feature — a per-match pass network (`production/src/
reporting/pass_network.py`) — was scoped BEFORE being built, this time,
rather than being audited for condition-2 compliance after the fact (the
shot map's own history above). The question: does a pass network's raw
form count as RAW, individually-attributable data under condition 2, the
same as the shot map, or is it already condition-2-compliant by
construction, the same as the aggregate positional heatmap (which the
audit above found fine specifically because it is "collapsed into counts,
shares, or per-cell means with no single event individually recoverable")?

**On its face, a pass network's raw form looks structurally close to
BOTH precedents, not cleanly either one:**

- Like the shot map: it names real individual players and, for the
  network's NODES, plots something location-shaped (a Starting XI
  player's own average `(x, y)` for this match).
- Like the heatmap: a node's average location is a genuine aggregate
  (the mean of many individual pass-start locations for one player, not
  any single event's raw coordinate) — by the SAME test the audit above
  applied to the heatmap ("no single event individually recoverable"), an
  average alone does not let a viewer reconstruct any one specific pass's
  exact location.

**The deciding factor, on closer inspection, is the EDGES, not the
nodes, and it is a real difference from the heatmap precedent, not a
superficial one:** the heatmap aggregates over a whole SEASON's worth of
events for one player (large N, genuinely non-recoverable). A pass
network is inherently SINGLE-MATCH scope (see `pass_network.py`'s own
module docstring for why aggregating it across matches the way player/
team reports do would not mean anything). At single-match granularity,
many real player-pairs complete only 1-3 passes to each other over 90
minutes — verified directly against match 3857276's real cached data
(the validation match used throughout this addendum): 171 real directed
edges among 22 Starting XI players, several with `completed_passes == 1`.
An edge weight that low is not a meaningfully "aggregated" count in the
same sense a season total is — it is, in practice, a direct restatement
of one specific real pass event (two named players, StatsBomb's own
`pass.recipient` link), just relabeled as a "count." Combined with both
players' own average locations, a low-weight edge lets a viewer infer an
approximate real start/end location for that one specific pass — exactly
the kind of individual-event reconstruction condition 2 exists to
prevent, even though nothing on the response dict is literally named
`location` the way the shot map's per-shot field is.

**Decision: treat the raw pass network (`generate_pass_network`) as RAW,
individually-attributable data under condition 2 — the SAME treatment as
the shot map, not an exception, and not assumed automatically compliant
just because it resembles the (already-cleared) heatmap on the surface.**
This was decided BEFORE writing `api.py`'s endpoint or `dashboard.py`'s
panel, not discovered afterward by a separate compliance audit — the
explicit goal of scoping this up front was to not repeat the shot map's
own history (built first, found non-compliant only by a later dedicated
audit).

**Resolution, following the shot map's own established pattern exactly**
(itself following ADR-014's precedent: scope the constraint, do not
remove the capability):

- `pass_network.generate_pass_network` — the raw variant (real per-player
  average location, real pairwise completed-pass edge weights) — remains
  fully available for LOCAL/private research use, unchanged.
- `pass_network.generate_pass_network_aggregated` is the condition-2-
  compliant counterpart: real per-player TOTALS only (completed passes
  sent/received, distinct-partner count) and network-level summary stats
  (player/edge counts, density) — no player's average location and no
  PAIRWISE edge weight appears anywhere in this variant's output. A
  per-player total (e.g. "34 completed passes sent") is the same class of
  aggregate as the shot map's own already-compliant `shots_by_body_part`/
  `total_shots` scalars — real, but not individually recoverable back to
  any one specific pass.
- `pass_network_visualizer.render_pass_network_aggregated` renders that
  as a per-team bar chart of completed passes sent, styled consistently
  with the shot map's own aggregated-heatmap panel — no node position, no
  edge line, anywhere.
- The SAME `PUBLIC_DEPLOYMENT` environment-variable flag (`api.py`,
  already checked once at startup) decides which variant
  `/reports/pass-network/{match_id}` serves — unset (default) serves the
  real raw network, byte-for-byte the same shape this module always
  produces; `true` serves ONLY the aggregated variant, and the raw
  `nodes`/`edges` lists are never even computed on that path.
- `dashboard.py`'s Pass Network panel mirrors the shot map panel's exact
  defense-in-depth check: it inspects whether the actual API response it
  received still carries a raw `nodes` field, and fails closed (a visible
  configuration error, nothing rendered) if its own flag says public but
  the response says otherwise.

Not chosen: treating the raw pass network as automatically condition-2-
compliant on the theory that it is "just like the heatmap." Rejected
because the single-match/low-edge-weight reasoning above is a real,
substantive difference from the heatmap's season-long, large-N
aggregation, not a surface-level distinction — and this project's own
standing discipline (ADR-014, ADR-020, this ADR's own original Decision
section) is to resolve a genuine ambiguity conservatively, not by
whichever reading is more convenient to build.

## Update: Condition 2 Applied to the Player Dashboard's Match-Level Views (Step 0 Decision)

A further reporting extension — a Player Dashboard adding match-level
views on top of the existing (already-compliant) season/multi-match
player report (`production/src/reporting/player_report.py`) — was scoped
under the same discipline the Pass Network established above: resolve the
condition-2 question explicitly, in writing, BEFORE implementing, for
every NEW view, rather than assuming a "standard dashboard feature" is
exempt just because it feels familiar. Three views were proposed; each
was evaluated separately, since (unlike the Pass Network's single
raw/aggregated question) they do not all resolve the same way.

**1. Match-by-match summary table (minutes played, event-TYPE counts per
match) — EXEMPT, no gating.** This is a per-match TOTAL (e.g., "67 minutes,
2 shots, 45 passes"), never a location and never an individually-
enumerated event. It is the exact same class of data
`generate_player_report` already serves unconditionally today
(`positional_distribution`, `total_minutes_played`) and the shot map's own
already-compliant summary scalars (`total_shots`, `shots_by_body_part`) —
a count broken out PER MATCH is not more sensitive than the SUM of those
same counts across several matches, which this module already serves with
no gate. `generate_player_match_summary` needed no raw/aggregated split at
all.

**2. Match-level touch map — SPLIT DECISION, resolved by which of two
existing precedents actually applies, not by reflexively copying the Pass
Network's stricter treatment.** On inspection, a touch map has NO
pairwise relationship between two named individuals to reconstruct — it
is one player's own touches, structurally the SAME shape as the season
heatmap (already compliant) and the shot map's own grid-binned aggregated
variant (already compliant, and — per that variant's own established
`shot_map_used_low_sample_flag` precedent — accepted as compliant even at
low sample sizes, flagged rather than blocked). The genuinely new risk is
therefore not "single-match scope" in the abstract (the Pass Network's
own precedent doesn't generalize that broadly either — see that section's
real deciding factor, PAIRWISE edges, which does not exist here) but
specifically whether INDIVIDUAL touch points are plotted:
- A GRID-BINNED touch-density view (no individual point, same
  `GRID_COLS x GRID_ROWS` convention as the season heatmap and shot map)
  is compliant by construction, matching the existing precedent directly
  — `generate_player_match_touch_map_aggregated` needed no additional
  restriction beyond the grid-binning itself already established elsewhere.
- A RAW individual-touch scatter (exact `(x, y)` per touch, like the shot
  map's raw scatter) is the same risk class as the shot map's raw variant
  and is gated identically: `generate_player_match_touch_map` is
  LOCAL/PRIVATE ONLY.

**3. Key-event timeline (a chronological, per-event listing for one
player, one match) — RAW, gated, no split-decision ambiguity.** Condition
2's own text is explicit and does not hinge on location: "no public API
endpoint returning event-level records... no interactive table of raw
events." A per-minute, per-event-type listing IS an event-level record on
its own, regardless of whether a coordinate is attached.
`generate_player_match_timeline` (LOCAL/PRIVATE ONLY) returns exactly
that; `generate_player_match_timeline_aggregated` (public-safe) collapses
it to event-TYPE counts per coarse (`TIMELINE_BUCKET_MINUTES`=15) time
bucket — no individual event, exact minute, or outcome/body-part detail
is ever enumerated in that variant's output.

**A second, genuinely separate leak found and closed during
implementation, not by the Step 0 process above but by directly reading
real event JSON before trusting any field (the same rigor this project's
outcome/recipient verification already applied for the Pass Network):**
a real StatsBomb `Shot` event's own sub-dict carries a `freeze_frame` list
of roughly 15 OTHER real, named, individually-located players (teammates
and opponents alike) at the moment of that shot — verified directly
against Messi's real match 3857264 data. A naive "dump this event's own
type-specific sub-dict" implementation of the timeline's per-event detail
field would have leaked far MORE individually-located, individually-
attributable real player data than the touch location it was already
being gated for. `player_report._event_detail` uses an explicit
ALLOWLIST (`outcome.name`/`body_part.name`/`technique.name`, plus
`statsbomb_xg` for a `Shot` specifically) rather than a wholesale sub-dict
dump, so `freeze_frame` (and any other individually-located sub-field a
future StatsBomb event type might add) is never read or returned by this
function at all.

**Resolution, following the shot map's/Pass Network's own established
pattern exactly** (ADR-014's precedent: scope the constraint, do not
remove the capability):
- `generate_player_match_touch_map`/`generate_player_match_timeline`
  (raw) remain fully available for LOCAL/private research use, unchanged.
- `generate_player_match_touch_map_aggregated`/
  `generate_player_match_timeline_aggregated` are the condition-2-
  compliant counterparts, following this section's own reasoning above
  (grid-binning for touches; time-bucketed type-counts for the timeline).
- The SAME `PUBLIC_DEPLOYMENT` flag (`api.py`, already checked once at
  startup) decides which variant each of
  `/reports/player/{id}/match/{match_id}/touch-map` and
  `/reports/player/{id}/match/{match_id}/timeline` serves.
  `/reports/player/{id}/match-summary` is NOT gated by this flag at all,
  consistent with view 1's exemption above.
- `dashboard.py`'s Player Dashboard panel mirrors the shot map/Pass
  Network panels' exact defense-in-depth check for both gated views:
  inspects whether the actual API response still carries a raw
  `touches`/`timeline` field, and fails closed (a visible configuration
  error, nothing rendered) if its own flag says public but the response
  says otherwise.

Not chosen: gating the match summary table (view 1) "to be safe" despite
it clearing the same test the shot map's own summary scalars already
clear today. Rejected as over-correction — this project's discipline is
to resolve genuine ambiguity conservatively, not to gate data that
demonstrably is NOT raw under condition 2's own stated test, which would
only make the working feature set unnecessarily inconsistent with what it
already serves unconditionally elsewhere.

## Addendum: Press Resistance Index (per-player, season/multi-match
aggregate rate of "successful action while under pressure")

Resolved EXEMPT from condition 2 — not gated by `PUBLIC_DEPLOYMENT` —
reasoned through explicitly rather than assumed, per this project's own
standing discipline that every new view gets this check even when the
answer looks obvious (the Pass Network section above is the reason this
discipline exists at all: an "obviously fine, it's just a count" view
turned out to hide a real leak on closer inspection).

`generate_player_press_resistance_index`'s return value is, in full:
per-event-type (Pass/Dribble/Shot) `under_pressure_attempts` /
`successful_under_pressure` counts and a derived `success_rate`, plus one
`overall` combined count/rate, plus a `matches_requested`/
`matches_with_data` count and the `press_resistance_index_used_low_sample_flag`
boolean. Checked directly against condition 2's own test (no individual
event, no location, no timestamp, no way to reconstruct which specific
action(s) drove a given number):

- **No `location`** is read or returned anywhere in this function —
  `event_is_under_pressure` and the three per-type success checks
  (`_is_successful_pass_under_pressure`, `_is_successful_dribble_under_pressure`,
  `_is_successful_shot_under_pressure`) each read only a boolean/outcome-name
  field, never `event["location"]`.
- **No `minute`/timestamp** is read or returned — unlike the timeline
  view above, there is no per-event chronological ordering in this
  feature's output at all.
- **No individual event is ever enumerated** — every event this function
  touches is immediately folded into one of six running integer counters
  (three types × attempts/successes); no per-event dict, id, or record
  survives into the return value the way the timeline's raw `timeline`
  list or the shot map's raw `shots` list do.
- **Aggregated across an entire requested match set** (season-scale, by
  this feature's own design), not scoped to one match the way the touch
  map/timeline are — if anything a HIGHER floor of aggregation than the
  touch map's already-compliant grid-binned variant, not a lower one.

**Why this is a different shape of aggregate than the Pass Network edges
that motivated gating in the first place, not merely a shorter list of
fields:** a Pass Network edge pairs a count/weight with the two
endpoints' average LOCATIONS (via the graph's nodes), which is what made
a low-weight edge individually reconstructible into an approximate real
passing event — the leak was never "it's a count," it was "the count is
attached to spatial context that narrows down which real event(s)
produced it." A Press Resistance Index rate carries no spatial or
temporal context at all, at any sample size — a rate of `1/1` (successes/
attempts) still only says "one pass happened under pressure and it was
complete," with no way to know WHICH pass, WHEN, or WHERE. This places it
in the same class as view 1 above (the match summary table) and the
season heatmap/positional distribution, not in the Pass Network's or the
raw touch/timeline views' class.

**The adjacent, genuinely real concern this exemption does NOT wave away:**
a rate computed from a very small N (e.g. a player with exactly one
real under-pressure pass in the requested match set) can look identical
in shape to a well-supported one and silently overstate confidence to a
viewer. This is a STATISTICAL confidence problem, not a condition-2 raw-
data-exposure problem, and this project already has an established
mechanism for exactly this class of concern (Milestone 44's low-sample
flagging, reused verbatim by the shot map's `shot_map_used_low_sample_flag`
and the touch map's `touch_map_used_low_sample_flag`): transparency via a
flag, not gating or hiding the underlying number. `generate_player_press_resistance_index`
follows this same convention (`press_resistance_index_used_low_sample_flag`,
threshold `MIN_UNDER_PRESSURE_EVENTS_FOR_CONFIDENT_PRI = MIN_HISTORICAL_EVENTS`,
same value habit_memory.py and `MIN_SHOTS_FOR_CONFIDENT_SHOT_MAP` already
use) rather than inventing a new threshold or route.

**Resolution:** `generate_player_press_resistance_index` and its
`/reports/player/{player_id}/press-resistance` endpoint are served
UNCONDITIONALLY (not behind the `PUBLIC_DEPLOYMENT` flag), consistent
with view 1's match-summary exemption above — there is no raw/aggregated
split for this feature the way the touch map/timeline/shot map each have,
because there is no raw variant of this feature to begin with (its output
is a rate by construction, not a downsampling of a richer raw view).

## Addendum: Tactical Entropy (per-team, season/multi-match Shannon
conditional entropy over pass-DIRECTION transitions)

Resolved EXEMPT from condition 2 — not gated by `PUBLIC_DEPLOYMENT` —
checked explicitly against condition 2's own test rather than assumed
because it "sounds like" the Press Resistance Index precedent above; the
Pass Network section further above is exactly the cautionary example for
why an "obviously just a count" view can't be waved through without
looking.

**1. Scope, confirmed:** `generate_team_pass_entropy(team_name, match_ids)`
is a per-TEAM, season/multi-match AGGREGATE, the same shape as the Press
Resistance Index — one result per call, computed by pooling every
requested match's transitions together, never a per-match or per-pass
return value. There is no finer granularity anywhere in this feature
(unlike the Pass Network, which is inherently single-match and needed the
raw/aggregated split specifically because of that).

**2. Output, checked field-by-field against condition 2's own test (no
individual pass's location, player, or minute; nothing individually
recoverable):**
- `transition_counts`/`transition_probabilities`: a 3×3 matrix of INTEGER
  COUNTS / row-normalized PROBABILITIES between three abstract CATEGORY
  labels (`Forward`/`Backward`/`Sideways`) — never a player name, a
  location, or a minute. This is a step FURTHER removed from any single
  event than the Press Resistance Index's own per-event-type counts:
  there, each count bucket corresponded to one StatsBomb event TYPE
  (Pass/Dribble/Shot); here, each of the 9 matrix cells is itself the sum
  of potentially hundreds of individual real passes that all happened to
  land in the same category pair, pooled across every requested match.
  Knowing "Forward→Sideways occurred 214 times this season" gives no way
  to identify which 214 specific passes, on which pitch coordinates, by
  which players, in which matches — the categorization is many-to-one by
  construction (every real pass with a `end_location[0]-location[0]`
  delta anywhere in, e.g., the open interval (-5m, +5m) collapses into
  the same "Sideways" count), which is a stronger, not weaker, guarantee
  of non-recoverability than an average location (a mean CAN in principle
  be inverted with enough side information about N; a many-to-one integer
  bucket count cannot be inverted into its constituent events at all).
- `total_transitions`/`total_pass_attempts_considered`/
  `completed_pass_attempts_considered`: plain aggregate integers, the same
  class of data `generate_player_match_summary`'s per-match event-type
  counts (already exempt, view 1 above) already serve unconditionally.
- `conditional_entropy_bits`/`normalized_entropy`/
  `max_possible_entropy_bits`: derived scalars computed FROM the count
  matrix above — strictly less information than the matrix itself (an
  entropy value cannot be inverted back into the matrix that produced it,
  let alone into any individual pass), so if the matrix clears condition
  2 (it does, per the above), the entropy scalars trivially do too.
- `pass_entropy_used_low_sample_flag`: a boolean, no data content beyond
  itself.

Nothing else is returned. No `location`, no `player`/`player_id`, no
`minute`, and no per-match breakdown appears anywhere in this function's
return value.

**3. Resolution:** `generate_team_pass_entropy` and its
`/reports/team/{team_name}/pass-entropy` endpoint are served
UNCONDITIONALLY (not behind `PUBLIC_DEPLOYMENT`), consistent with the
Press Resistance Index and match-summary exemptions above. As with the
Press Resistance Index, there is no raw variant of this feature to begin
with — its output is a many-to-one category-transition tally by
construction, not a downsampling of a richer per-pass view — so there is
no raw/aggregated split to design here, unlike the Pass Network or the
Player Dashboard's touch map/timeline.

## Addendum: Session/Match Comparison (`compare_team_matches`, a
finer-granularity extension of `compare_team_seasons`)

Resolved EXEMPT from condition 2 — not gated by `PUBLIC_DEPLOYMENT` —
checked explicitly at this NEW, FINER granularity rather than assumed to
inherit `compare_team_seasons`'s own already-settled exemption
automatically. The concern raised before building this (Step 0, this
project's own standing discipline): a single match has meaningfully fewer
located events than a full season, so the SAME 10×7 zone-share grid
`compare_team_seasons`'s `event_location_activity_map` mode already uses
is now built from far fewer underlying events per comparison — is this
still safely aggregate, or does the smaller N start resembling the Pass
Network's individually-recoverable-edge problem rather than the safely-
aggregate season case?

**1. Real-data check, not assumed.** Every real cached `(team, match)`
combination's located-event count was checked before answering this (a
full scan of this project's whole `data/raw/` cache, 1,868 real
`(team, match)` pairs): minimum 896, 5th percentile 1,147, median 1,788,
mean 1,851, maximum 3,472. Spread across a 70-cell (10×7) grid, even the
THINNEST real match in this entire cache still averages ~12.8 located
events per cell. There is no genuinely sparse real case in this project's
cache at match granularity — the honest finding is that the concern
motivating this check (a low-event match producing near-empty, more
"individually traceable" cells) does not materialize in practice here,
though the reasoning below holds independently of that empirical
comfort margin.

**2. The actual deciding factor (reasoned from first principles, the
same test applied to every prior gating decision), independent of how
comfortable the real numbers above are:** Pass Network's raw edges were
gated because a count was paired with a NAMED, individually-attributable
player's own precise average `(x, y)` location — that combination is what
made a specific real pass event's approximate reconstruction possible.
`compare_team_matches`'s grid carries neither of those two ingredients,
at ANY sample size: no player name or ID anywhere in its output (this is
a TEAM-level aggregate, exactly like the season-level grid), and no
location finer than one ~10m×9.7m cell. A cell holding exactly 1 event
(hypothetically, in some future thinner match this cache doesn't
currently contain) still reveals nothing about WHICH real event produced
it — not its exact coordinate, not the player who made it, not its
minute, not even its StatsBomb event TYPE (`_build_location_activity_grid`
pools every located event type into one count, a coarser aggregation
than even Pass Network's own Pass-only edges). Sparsity changes
CONFIDENCE (how much a viewer should trust the resulting picture), not
RECOVERABILITY (whether a specific real event can be identified from the
output) — and confidence is exactly what Step 1's low-sample flag below
is built to signal transparently, the same "flag, don't gate" resolution
this project already applied to Press Resistance Index's and Tactical
Entropy's own small-N concerns, not a reason to gate this feature the way
Pass Network's edges were.

**3. Per-match 360 detection carries over correctly.** `_match_360_available`
checks each of the two SPECIFIC `match_id`s directly (a real
`fetch_match_360` call, treating a `None` result as unavailable) — the
SAME verify-via-a-real-fetch discipline `team_report.py`'s own chain-frame
builder already uses, not the season-level `match_available_360` flag on
`competitions.json` (which `compare_team_seasons`'s own
`_resolve_team_season_matches` already documented as unreliable at
anything finer than "some matches in this season have it"). Both
match_ids must independently have real 360 coverage for
`pitch_control_360` mode to be used; either one missing it falls back to
`event_location_activity_map` for BOTH sides, mirroring
`compare_team_seasons`'s own "never compare the two sides on different
footings" rule exactly.

**4. Resolution:** `compare_team_matches` reuses `_compare_360`/
`_compare_location` UNCHANGED (parameterized with single-match_id lists
instead of a season's full list) and is served UNCONDITIONALLY, via a
NEW, dedicated `/reports/team-comparison/match` endpoint rather than
optional parameters bolted onto `/reports/team-comparison` — chosen
because every other feature added this session (Press Resistance Index,
Tactical Entropy, the Player Dashboard's touch-map/timeline endpoints)
got its own dedicated endpoint with an unambiguous, fully-required
parameter contract rather than being folded into an existing endpoint via
optional/mutually-exclusive query parameters; `compare_team_seasons` and
`compare_team_matches` take genuinely different, non-overlapping
parameter shapes (`team_a`/`season_a`/`team_b`/`season_b` vs.
`team_name`/`match_id_a`/`match_id_b`), and ADR-018's own established
pattern is one endpoint per distinct report SHAPE, not one per UI tab —
Player Reports alone already spans half a dozen separate endpoints under
one conceptual feature area. No raw/aggregated split exists for this
feature, for the same reason none exists for Press Resistance Index or
Tactical Entropy: its output is a many-to-one grid aggregate by
construction, not a downsampling of a richer per-event view.

## Addendum: Passing Lane Visualizer (`generate_team_passing_lanes` /
`generate_team_passing_lanes_aggregated`)

**This is NOT simply exempt "because it feels like Session/Match
Comparison" -- checked explicitly, and the two features turn out to need
OPPOSITE treatments for different parts of the same output.** Session/
Match Comparison's 10x7 grid carries NO player identity anywhere; Passing
Lane's own `lanes` field (a named passer/recipient PAIR with a scalar
openness score) is structurally similar to that precedent, but this
feature's `nodes` field (each named player's own precise AVERAGE
LOCATION, needed to actually draw a lane on a pitch diagram) reintroduces
the EXACT ingredient combination — a named individual + a precise average
location + a real, non-trivial score attached to it — that already got
Pass Network's raw edges gated. Verified this directly by re-reading
Pass Network's own gating reasoning above rather than assuming the newer
feature automatically inherits either precedent.

**1. `lanes` (passer_id/name, recipient_id/name, `mean_lane_openness`,
`n_pass_samples`) — EXEMPT, same reasoning as Press Resistance Index.**
No location anywhere in this field. A named pair with a scalar (here,
averaged across real per-pass openness scores rather than a rate over
event-type counts) is the same shape already found exempt for a single
named entity — extending it to a named PAIR does not introduce location
or timing precision, the actual things condition 2 cares about. Knowing
"Piqué → Lenglet: 0.845 mean openness across 22 real passes" reveals no
single pass's exact trajectory, minute, or outcome.

**2. `nodes` (player_id, name, `avg_location`) — NOT exempt, same
reasoning as Pass Network's raw edges, gated the SAME way.** This is a
real per-player AVERAGE location (the mean of each player's own
`event.location`/`pass.end_location` across every real pass sample used),
individually attributable to a named player — structurally identical to
Pass Network's own node convention, which the existing "Condition 2
Applied to the Pass Network" Update section above already resolved must
be LOCAL/PRIVATE ONLY. There is no reason a location that was risky when
attached to a completed-pass COUNT becomes safe when attached to an
openness SCORE instead — the risky ingredient (a named individual's own
precise average position) is identical either way.

**3. Resolution, mirroring Pass Network's own raw/aggregated split
exactly (ADR-014's precedent: scope the constraint, do not remove the
capability):**
- `generate_team_passing_lanes` (raw, `nodes` + `lanes` both present) is
  LOCAL/PRIVATE USE ONLY.
- `generate_team_passing_lanes_aggregated` pops `nodes` before returning
  — `lanes` (the condition-2-EXEMPT field, per point 1) passes through
  UNCHANGED, keeping player names on each pair (matching
  `generate_pass_network_aggregated`'s own precedent of keeping
  per-player names in `player_summary` while stripping only location and
  pairwise edges).
- `/reports/team/{team_name}/passing-lanes` follows the SAME
  `PUBLIC_DEPLOYMENT` branching pattern the shot map / Pass Network /
  Player Dashboard touch-map endpoints already established: the raw
  variant is served only when `PUBLIC_DEPLOYMENT` is unset; the
  aggregated variant (no `nodes`) otherwise. `render_passing_lanes`
  (needs `nodes` to plot lines at real coordinates) is therefore also
  LOCAL/PRIVATE ONLY, exactly like `render_pass_network`; a
  location-free bar-chart renderer
  (`render_passing_lanes_aggregated`, ranking named pairs by openness
  with no pitch/location involved) is the public-safe counterpart,
  mirroring `render_pass_network_aggregated`'s own bar-chart fallback.

## Addendum: Opposition Analysis (`generate_team_opposition_analysis`)

Resolved EXEMPT from condition 2, checked per-metric rather than assumed
as a bundle, since this feature deliberately combines one REUSED
pre-existing field with two genuinely NEW ones.

**1. Weak-zone pitch control** — not a new gating question at all. This
is `generate_team_report`'s own EXISTING `weakest_control_zones` field,
unmodified, un-recomputed — already covered by the original compliance
audit's own finding above ("the aggregate heatmap... already
condition-2-compliant by construction"). Re-labeling it "opposition
scouting: where to attack this team" in the UI changes nothing about
the underlying data or its computation, so nothing new to resolve here.

**2. `build_up_tendency`** (`total_buildup_passes`, `long_passes`,
`long_pass_share`) — a plain aggregate rate/count pair, the same shape
already found exempt for Press Resistance Index/Tactical Entropy. No
location (only a pass's LENGTH is used, a derived scalar distance, never
the pass's own coordinates), no player, no minute anywhere in the
output.

**3. `set_piece_reliance`** (`total_shots`, `set_piece_shots`,
`set_piece_shot_share`) — same shape, same reasoning: an aggregate
count/rate over real `play_pattern` categories, no individual shot's
location, player, or minute exposed.

**Resolution:** `generate_team_opposition_analysis` and its
`/reports/team/{team_name}/opposition-analysis` endpoint are served
UNCONDITIONALLY, not gated by `PUBLIC_DEPLOYMENT` — consistent with
Press Resistance Index and Tactical Entropy's own exemptions, for the
same reason: pure aggregate counts/rates, nothing individually
recoverable at any sample size.

## Addendum: Player Similarity Search (`find_similar_players`)

Resolved EXEMPT from condition 2 — checked explicitly (not assumed
exempt just because it "feels like" the existing per-player scalar
aggregates), since this is the first feature in this project to combine
TWO named individuals' data into one derived output, a genuinely new
shape worth its own full pass rather than a pattern-match to precedent.

**The question, stated plainly:** does "player X is similar to player Y,
similarity score 0.87, driven by press resistance and shot volume"
expose anything individually-recoverable beyond what
`generate_player_press_resistance_index`/`generate_player_shot_map_aggregated`/
`generate_player_report`'s own `positional_distribution` already safely
expose on their own, today, unconditionally?

**No — and the reasoning holds regardless of how the score was
computed, not just because the inputs happen to already be exempt.** The
15-dimension feature vector a similarity score is derived from is built
ENTIRELY from scalars this project has ALREADY resolved condition-2-exempt
individually: `positional_distribution` shares (season-aggregate,
unconditional today), Press Resistance Index's overall/per-event-type
rates (EXEMPT, own addendum above), and the shot map's own AGGREGATED
summary scalars (`total_shots`/`goals`/`xg_per_shot`/`shots_by_body_part`
— `generate_player_shot_map_aggregated`'s own docstring: "ADR-021's own
reasoning already treats a count/sum/share as condition-2-compliant").
A cosine similarity score is a further reduction of these already-exempt
scalars — like Tactical Entropy's own entropy value relative to its
transition matrix, a derived scalar computed FROM an aggregate cannot
carry MORE individually-recoverable information than the aggregate
itself; it can only carry less (the raw feature values of either player
are not recoverable from the similarity score alone, and the score
itself is bounded to [-1, 1] with no location, minute, or event-level
content at any point in its computation).

**The one genuinely new ingredient — pairing two named individuals in a
single output — does not change this.** Both players' own underlying
aggregate profiles are ALREADY independently, unconditionally public
today, via their own existing endpoints; computing a distance between
two already-public vectors reveals nothing about either player's own
individual raw events (no location, no minute, no specific match) that
querying each player's own existing report wouldn't already reveal
separately. The `matched_features` explanation field returned alongside
each result is even more restrictive: only coarse GROUP LABEL STRINGS
(e.g. `"press resistance"`, `"shot volume"`), never a raw feature value,
a z-score, or either player's own underlying number.

**Resolution:** `find_similar_players` and its
`GET /reports/player/{player_id}/similar` endpoint are served
UNCONDITIONALLY, not gated by `PUBLIC_DEPLOYMENT` — the same class of
exemption as Press Resistance Index/Tactical Entropy/Opposition
Analysis, extended (for the first time) to a two-player comparison
rather than a single-player aggregate, for the reasoning above. The
offline precompute index (`data/app_state/player_similarity_index.json`)
does store each searchable player's own RAW feature values (needed to
compute `matched_features` at query time) — this file is locally-generated
application state on the SERVER, never returned to a caller in full; only
the query player's own already-exempt raw values and the coarse group
labels described above ever leave `find_similar_players`'s own return
value.

## Update: Dedicated Post-Hoc Audit of Automatic Match Report / Coach Mode /
AI Tactical Chat (verification-only pass, real HTTP tests both flag states)

Each of these three new endpoints (plus `/coach-mode`, which needed no new
compliance reasoning at all) was reasoned about individually during its own
build earlier this session, but never re-checked against condition 2 as a
dedicated, focused pass across all of them together. Run explicitly for
that reason, with real `TestClient` requests under BOTH `PUBLIC_DEPLOYMENT`
states and a raw-response-text search (the same method already established
for the shot map/pass network audits above), not code review alone.

**1. `/reports/match/{match_id}` (Automatic Match Report) — COMPLIANT,
confirmed by real test.** This aggregator calls `generate_team_report`
(already condition-2-compliant by construction) and
`generate_team_opposition_analysis` (already EXEMPT, own addendum above)
directly — neither needed a gate to begin with. For the pass network
sub-component, the endpoint correctly selects
`generate_pass_network_aggregated` vs. `generate_pass_network` based on
`PUBLIC_DEPLOYMENT` BEFORE calling `generate_automatic_match_report`,
mirroring the standalone `/reports/pass-network/{match_id}` endpoint's own
selection exactly. Verified against the real raw response body: `"nodes"`,
`"edges"`, and `avg_location` are present with `PUBLIC_DEPLOYMENT` unset and
genuinely absent with it set to true.

**2. `/coach-mode` — no gating needed, confirmed by real test, not
assumed.** Reuses `/simulate`'s own pipeline, which itself has zero
`PUBLIC_DEPLOYMENT`-relevant behavior (its response is two threat scalars
and a delta, nothing individually-located or player-attributable). Checked
`/simulate`'s source directly for any `PUBLIC_DEPLOYMENT` reference (none)
and confirmed `/coach-mode`'s own real raw response body carries no
`nodes`/`edges`/`avg_location`/`location`/`player_id`/`player_name` under
either flag state — identical response shape both ways, as expected for a
derived-scalar-only endpoint (the same class already established exempt
for Tactical Momentum, Press Resistance Index, Tactical Entropy, etc.).

**3. `POST /chat/tactical` — A REAL GAP WAS FOUND AND FIXED.** This
endpoint's context package is built by calling
`generate_automatic_match_report` directly (not by re-deriving the
match-report endpoint's own gating decision), and that call site omitted
the `pass_network_fn` argument entirely — silently defaulting to the RAW
`generate_pass_network` regardless of `PUBLIC_DEPLOYMENT`, even though that
same function's own docstring already documents that its caller is
responsible for passing the deployment-appropriate variant (exactly what
`/reports/match/{match_id}` does correctly). This did NOT produce an
externally-observable leak today — confirmed by real request/response
inspection, not assumed: `tactical_chat.format_context_package_text` never
reads the `pass_network` field into the prompt or the HTTP response at all
under either flag state. But the raw, gated variant (real per-player
average location + real pairwise completed-pass edge weights) was still
being COMPUTED, unconditionally, on every single chat turn under
`PUBLIC_DEPLOYMENT=true` — exactly the thing this project's own established
discipline treats as the actual compliance requirement throughout this ADR
("the raw ... list is never even computed on that path, not merely
withheld from an already-built response"), and a real latent risk: any
future extension of the chat context to summarize pass-network activity
(a plausible, natural addition) would have silently inherited an
already-computed raw result with no gate anywhere in its own code path.

**Fix applied** (minimal, no feature logic changed beyond this): the
`/chat/tactical` endpoint now selects `pass_network_fn` the same way
`/reports/match/{match_id}` already does
(`generate_pass_network_aggregated if PUBLIC_DEPLOYMENT else
generate_pass_network`) before calling `generate_automatic_match_report`.
Verified directly, not just re-read: monkeypatch-counted real calls to each
variant confirm `generate_pass_network` (raw) is called exactly once with
`PUBLIC_DEPLOYMENT` unset and ZERO times with it set to true (only
`generate_pass_network_aggregated` runs in that mode) —
`test_tactical_chat.py::test_chat_endpoint_public_deployment_never_computes_raw_pass_network`.
A second, belt-and-suspenders test
(`test_chat_endpoint_response_never_leaks_raw_pass_network_under_public_deployment`)
locks in the raw-HTTP-response-text guarantee the same way the shot
map/pass network's own tests already do.

**4. Chat's out-of-scope refusal instruction — sanity-checked, no gap, as
expected.** `tactical_chat._CHAT_GROUNDING_INSTRUCTIONS` is static text,
not conditioned on `PUBLIC_DEPLOYMENT` in any way — it instructs the model
to answer ONLY from whatever the context package actually contains and to
decline otherwise, which is correctly conservative regardless of how much
or how little that package holds under either deployment mode. There is no
scenario where a public deployment should get a MORE permissive refusal
policy than a private one, and none exists.

**Net finding:** one real, previously-unverified gap (item 3), now fixed
and regression-tested; the other three items were already correct,
confirmed by real test rather than by re-reading the code that was
originally reasoned about individually at build time.
