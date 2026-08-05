# ADR-020: Licensing Scope for StatsBomb Open Data (RQ1-RQ5, DeepHit, GNN, Kalman Filter)

## Status
Accepted

## Context

StatsBomb's open data underlies the entire physics-ML track of this
project — every RQ1-RQ5 finding, the DeepHit cumulative-incidence model,
the GNN degradation work, and the Kalman-filter validation baseline are
all trained and/or validated against data pulled from
`github.com/statsbomb/open-data`, which this project has downloaded from
since Milestone 3. Unlike every other data/model source in this project
(ADR-014's CV pretrained-model licensing scope, the football-data.co.uk
usage restriction, SoccerNet's NDA gate), this data source has never had
its own real licensing review. Given the growing amount of commercially-
framed language accumulating around the physics-ML track's results, this
gap needed to be closed before any commercialization claim, following the
same "verify, don't assume" discipline this project has applied
everywhere else (Milestone 3's own StatsBomb schema checks, ADR-014's
AGPL lineage tracing).

**The actual governing document** was located and read directly, not
summarized from memory or secondary sources: `LICENSE.pdf`, committed at
the root of `statsbomb/open-data` (fetched directly from GitHub — not
assumed present from a README description). It is titled the
**"StatsBomb Public Data User Agreement"**, and its footer is dated
`[StatsBomb Data: User Agreement Standard Terms - last updated 8
September 2023]`.

The Agreement is explicitly scoped to what it calls "the **Service**":

> "The provision of Github access to specific leagues of StatsBomb Data
> (the 'Service') is provided by StatsBomb Services Ltd ('StatsBomb') to
> any user (the 'User') who agrees to use the data according to the Terms
> and Conditions of this Agreement."

and, in Section 2.1:

> "The Service will be provided via github
> (https://github.com/statsbomb/open-data) of which is fully controlled
> by StatsBomb."

This confirms the document governs exactly the open-data GitHub
repository this project has been pulling from — **it does not describe
StatsBomb's separate commercial product's terms**, which are a distinct,
separately-negotiated paid agreement not represented anywhere in this
document.

### The operative restriction (Section 1.2)

> "1.1. Subject to the terms of this Agreement, StatsBomb will provide
> the User with access to the Service to be used for analysis, research
> and to facilitate the shared ideas & understanding of the data
>
> 1.2. The User may not:
> 1.2.1. edit, distort, distribute, reproduce, sell or in any way provide
> the data to any external or third party;
> **1.2.2. commercially exploit the data or any analysis derived from the
> use of the Service;**
> 1.2.3. use the Service for any activity of an illegal or fraudulent
> nature, to violate any laws;
> 1.2.4. use the Service to produce, transfer, distribute or publish any
> material that might be defamatory or damaging to any individual or
> organisation
> 1.2.5. decompile, reverse engineer, or otherwise attempt to obtain the
> source code of the Services;"

Section 7 (Intellectual Property Rights) reinforces this:

> "The User acknowledges and agrees that all data provided through the
> Service, is the property of StatsBomb. The User shall, except as
> expressly permitted herein, shall not modify, translate, transfer,
> distribute, license, sell or otherwise exploit for any purposes
> whatsoever any data, content or third party submissions or other
> proprietary rights not owned by the User..."

### Attribution requirement (Section 1.4)

> "1.4. The User is required to accredit any publication of analysis
> formed from StatsBomb Data with the StatsBomb brand logo."

The Agreement's preamble additionally frames the entire dataset's
purpose narrowly:

> "StatsBomb have made this data freely available and accessible to
> encourage and facilitate research and the shared analytical
> understanding of the game of Football. This is aimed to be a research
> tool, and is intended to be used as such."

### Answering the four specific questions this review was tasked with

**(a) Is commercial use of the raw data permitted, prohibited, or
ambiguous?** **Explicitly PROHIBITED**, not ambiguous. Clause 1.2.2 is a
direct, unqualified ban on commercially exploiting "the data."

**(b) Is commercial use of MODELS TRAINED ON this data addressed?** Not
by name — the Agreement never uses the words "model," "weights," or
"derivative work." But clause 1.2.2 does not stop at the raw data: it
also bans commercial exploitation of **"any analysis derived from the
use of the Service."** A trained model's predictions — the DeepHit
cumulative-incidence outputs, the GNN degradation findings, the
Kalman-filter results, i.e., literally RQ1-RQ5 — are analysis derived
from the use of this Service by any ordinary reading of that phrase.
There is no narrow, technical reading available here (e.g., "the model
weights are just numbers, not 'analysis'") that survives scrutiny: the
weights exist only because they were fit to this data, and everything
the model produces when applied is downstream analysis of that fitting
process. This is the same discipline ADR-014 held for the AGPL network
clause — not treating an unaddressed specific term ("model weights") as
license to assume the broader, plainly-worded restriction doesn't apply.
**Conclusion: commercial use of models trained on this data is not
"silent" — it falls squarely inside an existing, explicit prohibition.**

**(c) What attribution is required?** The StatsBomb brand logo (via their
Media Pack) must accompany any published analysis formed from the data
(clause 1.4) — required regardless of commercial/non-commercial status,
and independent of (not a substitute for) the commercial-exploitation
ban.

**(d) Difference between "open data" and the commercial product?** This
Agreement governs only the free GitHub "Service." It says nothing about
commercial-product terms because those are a separate agreement outside
this document's scope entirely — by clear implication, since clause
1.2.2 explicitly bans commercial exploitation under this Agreement, a
different, separately-negotiated commercial license from StatsBomb would
be the mechanism by which commercial use becomes permitted.

## Decision

**The physics-ML track's current findings (RQ1-RQ5, DeepHit, the GNN
degradation work, the Kalman-filter baseline) — everything trained
and/or validated against `statsbomb/open-data` — MAY NOT be commercially
deployed, sold, licensed, or offered as a paid product or service in
their current form.** This is not a cautious reading of an ambiguous
clause; it is a direct application of an explicit prohibition (1.2.2)
that the text itself extends beyond raw data to "any analysis derived
from the use of the Service."

What **remains fully permitted** under the current open-data license,
and requires no further action beyond compliance with 1.4:
- Continued non-commercial research use of the data and all
  StatsBomb-derived findings (which is this project's current framing).
- Publishing analysis, write-ups, demos, or a portfolio presentation of
  RQ1-RQ5 findings, DeepHit/GNN/Kalman results — **provided the StatsBomb
  brand logo accompanies any such publication**, per 1.4. This project
  has not been doing this consistently and should start.
- Sharing this project's code, methodology, and non-commercial analytical
  conclusions publicly (the preamble explicitly anticipates this: "Any
  analysis or conclusions that are created as a result of using this
  data, may be shared publicly").

What is **NOT permitted** without a separate commercial license from
StatsBomb:
- Selling access to the app, its predictions, or any derived analysis.
- Offering the trained DeepHit/GNN/Kalman models (or their outputs) as a
  paid API, product, or service to any external party.
- Any pricing, subscription, or revenue-generating deployment built
  around StatsBomb-open-data-derived findings, regardless of how
  indirect ("the app uses insights validated on this data" still
  qualifies as commercially exploiting analysis derived from the
  Service).

## Consequences

- **No commercialization claim can currently be made for the physics-ML
  track as it stands.** Any future business-facing pitch, pricing page,
  or paid-tier plan built around RQ1-RQ5/DeepHit/GNN/Kalman results would
  need one of the Alternatives below resolved first — this ADR is the
  formal blocker, the same role ADR-014 plays for the CV track's live
  serving of the AGPL-lineage keypoint model.
- **This does not block current work.** Continued research, model
  development, validation, local demos, and public (non-commercial)
  sharing of findings all remain fully in scope — nothing about the
  physics-ML track's day-to-day development is restricted by this
  finding, only its eventual commercial framing.
- **UPDATE — this specific gap is now CLOSED.** The clause 1.4
  attribution requirement is now satisfied in both places a user
  actually encounters this project: `production/frontend/dashboard.py`
  (a prominent, unconditional notice at the TOP of the persistent
  sidebar — "Data provided by StatsBomb", linked to statsbomb.com and
  this ADR — visible on every page load, regardless of which tab is
  open, with no click required) and `README.md` (a citation immediately
  below the project's own opening description, before any other
  content). Deliberately NOT gated behind `PUBLIC_DEPLOYMENT` — unlike
  the raw-data-exposure fixes ADR-021 documents, this is a licensing
  obligation that applies to ANY use of StatsBomb data, local or public,
  so it is unconditional. **One honest scoping note**: both citations
  are a clear TEXT credit ("Data provided by StatsBomb" + a link), not
  an embedded copy of StatsBomb's actual brand-logo image file — no such
  asset exists anywhere in this repository, and fetching/embedding their
  copyrighted logo image without going through their own Media Pack
  (referenced in the Agreement's own preamble) was judged out of scope
  for this pass rather than done informally. Any future real public
  deployment should pull the actual logo asset from that Media Pack
  before launch, to satisfy clause 1.4's literal "brand logo" wording,
  not just its underlying attribution intent.
- **The CV track is not automatically a cleaner commercial fallback.**
  ADR-014 already found the CV track's own pretrained pitch-keypoint
  model carries its own unresolved AGPL-3.0 (Ultralytics YOLOv8-Pose
  weights) and unverified Kaggle-training-data licensing lineage, and is
  restricted to local/non-networked research use for that reason. A
  commercial product scoped around the CV/live-video track alone would
  need ADR-014's own gaps resolved first — it is not a ready substitute
  path, just a differently-blocked one.
- Combined with ADR-014 (CV track) and the existing SoccerNet NDA gate
  and football-data.co.uk restriction, **every current data/model source
  in this project now has an explicit, documented commercial-use
  disposition** — none of them currently support unrestricted commercial
  deployment as-is.

## Alternatives Considered

- **Treat clause 1.2.2's "analysis derived from the use of the Service"
  as not covering trained model weights/predictions, since the word
  "model" never appears**: rejected. This is exactly the kind of
  motivated narrow reading ADR-014 explicitly rejected for AGPL's network
  clause — the fact that a term isn't used verbatim does not mean the
  broader, plainly-worded restriction doesn't apply. RQ1-RQ5's findings
  are, definitionally, analysis derived from the use of this Service.
- **Continue without a formal ADR, relying on the project's current
  non-commercial framing as sufficient**: rejected — this was the exact
  gap this review was commissioned to close; an implicit assumption is
  not a substitute for a documented, explicit scoping decision, and any
  future contributor or reviewer needs this decision written down before
  any commercialization conversation happens, not discovered mid-pitch.
- **Immediately pursue a commercial StatsBomb data license**: not
  pursued as part of this ADR — no concrete commercial scope, pricing
  model, or customer exists yet to justify the cost or negotiation
  effort. This remains a legitimate future path if/when this project's
  commercial ambitions become concrete enough to justify it; it is named
  here as the option, not exercised.
- **Source a different, fully commercially-cleared event dataset and
  retrain the physics-ML track against it**: named as a real alternative
  path, but **not yet vetted** — no specific replacement dataset has been
  identified or licensing-reviewed by this project. Pursuing this would
  require its own dedicated ADR with the same real-document-reading
  discipline applied here, not an assumption that a commercially-clean
  alternative is easy to find.
- **Scope any commercial product around the CV/live-video track instead,
  on the assumption its licensing is cleaner**: investigated and
  rejected as a clean fallback — per ADR-014, the CV track has its own
  separate, still-unresolved AGPL/Kaggle licensing lineage. It is a
  different blocker, not an escape from this one.
