# ADR-014: Licensing Scope for Pretrained CV Models (Pitch-Keypoint Detection and Beyond)

## Status
Accepted

## Context

Following ADR-013's conclusion that frame-to-frame optical-flow composition
is ruled out as a general camera-motion-compensation solution, and that
anchor-based re-calibration against known, fixed pitch geometry is the
viable path forward, a pretrained pitch-keypoint-detection model was
investigated as a faster alternative to building the classical Hough-
line-based anchor detector ADR-013 left as unbuilt future work: Roboflow's
`football-field-detection-f07vi` (YOLOv8-Pose, 32 keypoints, with a
documented `ViewTransformer` workflow for exactly this kind of
re-calibration), plus its open-source reference implementation
(`roboflow/sports`, forked as `DesusLove/Football-PitchVision`).

Verifying this before trusting it (the same discipline this project has
applied to every external claim since Milestone 3's StatsBomb schema
checks) surfaced two STACKED, unresolved licensing questions, not one:

1. **The model weights.** The wrapper code (`roboflow/sports` and its
   forks) is MIT-licensed, but the underlying YOLOv8-Pose model weights
   are AGPL-3.0 per Ultralytics' own stated licensing policy — custom-
   trained YOLO models remain AGPL-3.0 regardless of the surrounding
   wrapper's license, unless the trainer holds a paid Ultralytics
   Enterprise license (which this project does not). AGPL-3.0 carries a
   network-use clause: serving AGPL-derived functionality through a live,
   networked application would obligate releasing the full corresponding
   source of the ENTIRE connected application under AGPL-compatible
   terms — not just the CV component that happens to use the model. This
   project has a live, networked application: the FastAPI/WebSocket
   serving layer built since Milestone 16 (`production/src/serving/api.py`,
   `@app.websocket("/ws/tactical-stream")`), extended in Milestone 33 with
   a `source="cv"` path that streams real CV-pipeline output over that
   same connection.
2. **The training data.** The model was trained on 317 frames sourced
   from Kaggle's "DFL Bundesliga Data Shootout" competition. Its actual
   license/usage terms could not be directly verified — Kaggle's rules
   page requires authentication and was not fetchable during
   investigation. Widespread community reuse of Kaggle-competition-
   derived models is common practice, but common practice is not
   confirmation of compliant licensing, and this project's own discipline
   (verify, don't assume) does not treat "other people do it" as
   equivalent to "the terms permit it."

A candidate alternative, "No Bells, Just Whistles" (CVPR 2024), was
investigated specifically to sidestep the Ultralytics/AGPL lineage, but
was found to be trained directly on SoccerNet distribution data —
reintroducing the exact NDA-gated access problem this project has
correctly deferred to since Milestone 25, just one hop removed (relying on
a third party's compliance with SoccerNet's terms, itself unverifiable)
rather than actually resolved.

## Decision

**Any pretrained CV model with an unresolved or AGPL-derived licensing
lineage — including the Roboflow `football-field-detection` model — may
be used ONLY as a strictly LOCAL, NON-DISTRIBUTED, NON-NETWORK-SERVED
research prototype.** Specifically:

- The model MAY be downloaded, run, and evaluated locally for research
  and validation purposes: testing keypoint-detection quality against
  real footage, prototyping the anchor-based re-calibration approach
  ADR-013 called for, generating local demo output.
- The model MUST NOT be wired into `production/src/serving/api.py`'s
  FastAPI/WebSocket layer, or into any other live, network-accessible
  endpoint, until the licensing questions above are either RESOLVED
  (e.g., a definitively-cleared license is found for both the weights and
  the training data) or SUPERSEDED (e.g., an Ultralytics Enterprise
  License is obtained, or this project's overall licensing posture is
  deliberately and separately decided in a future ADR — not implicitly
  via a CV model choice).
- Any output produced using this model (annotated demo videos,
  screenshots, evaluation reports) is a locally-generated research
  artifact, not a feature of the live-served application, and must not be
  described or documented as one.

## Consequences

- Milestone 38's overlay renderer, and any future work building on a
  pitch-keypoint model under this scope, can proceed in an offline/batch/
  local capacity — producing recorded video output — consistent with
  framing this as a demo video, not a live public endpoint.
- **Real-time, publicly-served tactical-map functionality (Track A's
  original ambition of a live overlay alongside a broadcast feed) remains
  explicitly NOT delivered by this decision.** It is deferred, not solved,
  pending a cleaner licensing resolution for whatever keypoint-detection
  approach is ultimately adopted.
- This does NOT retroactively change anything about the existing CV
  track's use of YOLOv8m for detection/tracking/ball-detection
  (Milestones 25-30): that code was already local/batch/test-only and has
  never been live-served with real inference either (Milestone 33's
  `source="cv"` WebSocket path has only ever been exercised with mocked
  pipelines and one private, non-distributed local clip — see
  `CV_PIPELINE_FINDINGS.md`). This ADR formalizes an existing de facto
  practice for that code, and extends the same explicit discipline to the
  new keypoint model BEFORE it is ever run, rather than discovering the
  same question again later.
- Any future contributor proposing to wire ANY pretrained model into the
  live serving layer must first confirm its weight license and training-
  data provenance are both clean, or that this ADR has been formally
  superseded — this is now a standing check, not a one-off judgment call
  specific to this one model.

## Alternatives Considered

- **Commit the whole project to AGPL-3.0 licensing to permit live
  serving of AGPL-derived models**: rejected for now — this is a real,
  consequential decision about the entire project's licensing posture
  (affecting every future user/deployment, not just this one CV
  component) and should not be made implicitly as a side effect of one
  model choice. If desired later, it deserves its own deliberate ADR
  weighing that tradeoff directly.
- **Purchase an Ultralytics Enterprise License**: rejected as
  inappropriate for a personal, non-commercial research project — the
  cost and commercial framing don't match this project's actual context.
- **Keep searching for a cleanly-licensed alternative keypoint model**:
  attempted (see Context — "No Bells, Just Whistles" was specifically
  evaluated for this reason) and abandoned after finding that the two
  most relevant candidates both carry the same class of unresolved
  provenance issue. Pitch-keypoint research is concentrated enough around
  the Ultralytics/Roboflow and SoccerNet ecosystems specifically that this
  is not a problem a little more searching was expected to route around.
