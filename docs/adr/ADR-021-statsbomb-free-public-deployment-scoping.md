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
