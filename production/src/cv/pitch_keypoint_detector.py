"""ADR-014-scoped prototype (Module 4): pretrained pitch-keypoint detection
and per-frame anchor-based homography solving.

STANDALONE, part of the same isolated `production/src/cv/` tree introduced
in Milestone 25 -- nothing in `production/src/models`, `production/src/
pipeline`, `production/src/spatial`, `production/src/physics`, or
`production/src/serving` is imported by or imports from this module.

CRITICAL, PER ADR-014 (`docs/adr/ADR-014-cv-pretrained-model-licensing-scope.md`),
NON-NEGOTIABLE: this module uses a pretrained model (Roboflow's
`football-field-detection-f07vi`) whose weight license (AGPL-3.0-derived
per Ultralytics' stated policy) and training-data provenance are only
partially resolved. It MUST NOT be imported by, or wired into,
`production/src/serving/api.py`, any other FastAPI endpoint, or any other
live/network-accessible surface. This is a standalone, local, non-served
research prototype ONLY.

WHY THIS MODULE CALLS A HOSTED API, NOT LOCAL CACHED WEIGHTS (read before
assuming this is "just like camera_motion.py" in terms of runtime cost):
Roboflow's `inference`/`inference-gpu` package -- the ONLY package that
supports the intended "download weights once, cache locally, run fully
offline forever after" pattern -- has NO version compatible with this
environment's Python (3.14; every `inference` release requires <3.13, per
direct verification via `pip install`). Additionally, this specific public
project has no per-account-downloadable trained-weights export available
through the plain `roboflow` package's public API (`.model`/`.models()`/
`.trainings()` require project-OWNER permissions this API key does not
have; `.download()`/`.export()` only produce the annotated DATASET, not
trained weights). The only verified-working path to real predictions from
THIS specific pretrained model, in THIS environment, is Roboflow's classic
hosted inference endpoint (`https://serverless.roboflow.com/...`) via the
`roboflow` package's `KeypointDetectionModel.predict()` -- a genuine,
per-call NETWORK ROUND TRIP to Roboflow's servers, not local compute. Every
`detect_pitch_keypoints` call in this module therefore has REAL network
latency baked into its cost, which is NOT representative of what a truly
local deployment would cost. This is stated here explicitly so a future
reader does not mistake this prototype's timing numbers for local-
inference timing.

REQUIRES a `ROBOFLOW_API_KEY` (loaded via `python-dotenv` from a
git-ignored `.env` file, or already present in the environment) -- raises
`RuntimeError` immediately if absent, rather than silently failing later
or attempting to create/prompt for credentials.

KEYPOINT GEOMETRY, SOURCE AND VERIFICATION: the 32 real-world pitch
landmark positions and the `class_id` -> vertex-number label mapping below
are taken directly from the reference implementation
(`roboflow/sports`'s `sports/configs/soccer.py`, fetched and verified
against its raw source, not a paraphrase) and CROSS-CHECKED against this
model's own real prediction output on a real frame from this project's
Milestone 34B clip (confirmed: `class_id=13` -> `"15"`, `class_id=30` ->
`"14"`, `class_id=31` -> `"19"`, exactly matching the label-order list
below) -- not assumed from the reference repo alone. The reference
schema's native pitch dimensions (120m x 70m, i.e. 12000cm x 7000cm) are
DIFFERENT from this project's own established 100m x 68m grid (ADR-002);
`PITCH_KEYPOINTS_METERS` below is rescaled into ADR-002's space using the
exact same "rescale once, at the ingestion boundary" discipline ADR-002
itself established for StatsBomb's 120x80 grid.

ADR-015 / ADR-016, OUTLIER HANDLING -- READ BEFORE CHOOSING HOW TO CALL
`solve_homography_from_keypoints` (three approaches exist in this file,
only one is actually recommended): ADR-015 found six specific vertices
(`ADR015_KNOWN_UNRELIABLE_VERTICES` below) are consistently mislocalized
despite high confidence on this project's one real camera framing, and
that excluding them (`excluded_vertices=ADR015_KNOWN_UNRELIABLE_VERTICES`)
gets LOOCV median error to ~6.2m. ADR-016 then attempted to REPLACE that
fixed list with two runtime, data-driven mechanisms that would generalize
to an unseen camera angle by construction rather than by luck: (a) plain
per-frame iterative outlier rejection (no `excluded_vertices`, no
`reliability_tracker`) measured 35-39m median -- worse than the fixed
list, because 10-15 points per frame with ~30-40% contamination isn't
enough for within-frame statistics to cleanly separate bad from
mediocre-but-real; (b) adding cross-frame rolling reliability tracking
(`reliability_tracker=KeypointReliabilityTracker(...)`) measured 32-38m
median and, worse, sometimes flagged known-GOOD vertices (30, 15, 16, 29)
as the most unreliable ones -- accumulating evidence against each frame's
own noisy per-frame fit compounds that fit's bias rather than averaging
it out. Both (a) and (b) are kept in this file, exactly as tested, as an
honest record of what was tried and why it under-performed -- NEITHER is
the recommended call. **`excluded_vertices=ADR015_KNOWN_UNRELIABLE_VERTICES`
remains the only approach in this file validated to ~6.2m and is what any
real usage (e.g. Milestone 38's overlay renderer) should pass.**
"""

import os
import tempfile

import cv2
import numpy as np

MODEL_WORKSPACE = "roboflow-jvuqo"
MODEL_PROJECT = "football-field-detection-f07vi"
# Version 14 specifically -- confirmed, via direct query against Roboflow's
# own project-version metadata API, to have a `status: "finished"` training
# with real deployed `modelIds` (i.e. an actually-callable hosted model).
# The LATEST version at investigation time (18) was checked FIRST and found
# to have an EMPTY `models`/`trainings` metadata -- no hosted model at all
# for that version, despite being a valid, downloadable dataset version.
# Do not assume "latest version" implies "has a deployed model" for any
# Roboflow Universe project -- verified per-version here, not assumed.
MODEL_VERSION = "14"

# Native Roboflow schema dimensions (roboflow/sports' SoccerPitchConfiguration,
# centimeters).
_ROBOFLOW_PITCH_LENGTH_CM = 12000
_ROBOFLOW_PITCH_WIDTH_CM = 7000
_ROBOFLOW_PENALTY_BOX_WIDTH_CM = 4100
_ROBOFLOW_PENALTY_BOX_LENGTH_CM = 2015
_ROBOFLOW_GOAL_BOX_WIDTH_CM = 1832
_ROBOFLOW_GOAL_BOX_LENGTH_CM = 550
_ROBOFLOW_CENTRE_CIRCLE_RADIUS_CM = 915
_ROBOFLOW_PENALTY_SPOT_DISTANCE_CM = 1100


def _roboflow_vertices_cm() -> list[tuple[float, float]]:
    """The 32 real-world vertex positions in the Roboflow schema's own
    120x70m (12000x7000cm) space, indexed 0..31 for vertex numbers 1..32 --
    reproduces `roboflow/sports`' `SoccerPitchConfiguration.vertices`
    property exactly (fetched from its raw source and transcribed here,
    not re-derived independently), so this module has no runtime
    dependency on that external repo.
    """
    w = _ROBOFLOW_PITCH_WIDTH_CM
    length = _ROBOFLOW_PITCH_LENGTH_CM
    pbw = _ROBOFLOW_PENALTY_BOX_WIDTH_CM
    pbl = _ROBOFLOW_PENALTY_BOX_LENGTH_CM
    gbw = _ROBOFLOW_GOAL_BOX_WIDTH_CM
    gbl = _ROBOFLOW_GOAL_BOX_LENGTH_CM
    ccr = _ROBOFLOW_CENTRE_CIRCLE_RADIUS_CM
    psd = _ROBOFLOW_PENALTY_SPOT_DISTANCE_CM

    return [
        (0, 0),  # 1
        (0, (w - pbw) / 2),  # 2
        (0, (w - gbw) / 2),  # 3
        (0, (w + gbw) / 2),  # 4
        (0, (w + pbw) / 2),  # 5
        (0, w),  # 6
        (gbl, (w - gbw) / 2),  # 7
        (gbl, (w + gbw) / 2),  # 8
        (psd, w / 2),  # 9
        (pbl, (w - pbw) / 2),  # 10
        (pbl, (w - gbw) / 2),  # 11
        (pbl, (w + gbw) / 2),  # 12
        (pbl, (w + pbw) / 2),  # 13
        (length / 2, 0),  # 14
        (length / 2, w / 2 - ccr),  # 15
        (length / 2, w / 2 + ccr),  # 16
        (length / 2, w),  # 17
        (length - pbl, (w - pbw) / 2),  # 18
        (length - pbl, (w - gbw) / 2),  # 19
        (length - pbl, (w + gbw) / 2),  # 20
        (length - pbl, (w + pbw) / 2),  # 21
        (length - psd, w / 2),  # 22
        (length - gbl, (w - gbw) / 2),  # 23
        (length - gbl, (w + gbw) / 2),  # 24
        (length, 0),  # 25
        (length, (w - pbw) / 2),  # 26
        (length, (w - gbw) / 2),  # 27
        (length, (w + gbw) / 2),  # 28
        (length, (w + pbw) / 2),  # 29
        (length, w),  # 30
        (length / 2 - ccr, w / 2),  # 31
        (length / 2 + ccr, w / 2),  # 32
    ]


# ADR-002-style rescale: Roboflow's native 120x70m schema -> this project's
# established 100x68m internal grid. Applied ONCE, here, at the point this
# external schema enters the codebase -- exactly ADR-002's own rule for
# StatsBomb's 120x80 grid, extended to a second external provider as that
# ADR's own "Alternatives Considered" anticipated a future provider would
# need to do.
_KP_X_SCALE = 100.0 / 120.0
_KP_Y_SCALE = 68.0 / 70.0

# vertex number (1-32) -> (x_meters, y_meters) in THIS project's 100x68 grid.
PITCH_KEYPOINTS_METERS: dict[int, tuple[float, float]] = {
    vertex_number: (
        (x_cm / 100.0) * _KP_X_SCALE,
        (y_cm / 100.0) * _KP_Y_SCALE,
    )
    for vertex_number, (x_cm, y_cm) in enumerate(_roboflow_vertices_cm(), start=1)
}

# class_id (0-31, the model's own output index) -> vertex number (1-32).
# Source: roboflow/sports' SoccerPitchConfiguration.labels, transcribed
# exactly -- CROSS-VERIFIED against this model's real prediction output on
# a real Milestone 34B frame (class_id 13/30/31 confirmed to output class
# strings "15"/"14"/"19" respectively, matching this list's positions
# exactly), not trusted from the reference repo alone.
KEYPOINT_LABEL_ORDER: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17,
    18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 14, 19,
]

DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MIN_POINTS_FOR_HOMOGRAPHY = 4

# ADR-015: vertices found, via aggregate analysis across many real frames,
# to be consistently mislocalized despite high reported confidence -- all
# far-side/background landmarks under this project's one real camera
# framing (heavily foreshortened, e.g. vertex 25 at `(length, 0)`: ~53m
# error, vs. its near-side mirror vertex 30 at `(length, w)`: ~1.6m error).
# THIS IS A FIXED LIST DERIVED FROM ONE CAMERA ANGLE, NOT A GENERAL LAW --
# ADR-016 tried and failed to replace it with a runtime, data-driven
# mechanism (see the module docstring); this list remains the only
# approach in this file validated to ~6.2m median error and is what real
# usage should pass as `excluded_vertices`.
ADR015_KNOWN_UNRELIABLE_VERTICES: frozenset[int] = frozenset({19, 22, 23, 24, 25, 26})

# ADR-016: iterative outlier-rejection homography solving -- the exact
# masking-aware "refit, then re-evaluate every original point" template
# `team_classifier.classify_teams` built for Milestone 28's jersey-color
# clustering, reused here for correspondence points instead of color
# clusters. `OUTLIER_RESIDUAL_MULTIPLE` is that same "2x the MEDIAN
# residual among the currently-fit points" convention -- a std-dev over a
# set that still includes the outliers would itself be inflated by the
# very points it's meant to catch (the identical masking failure mode
# `classify_teams`'s docstring names), so the threshold is always computed
# from whichever point set was just used to fit, never from the full,
# possibly-contaminated set.
OUTLIER_RESIDUAL_MULTIPLE = 2.0
# M28's tentative-removal fraction (`classify_teams`'s
# `outlier_candidate_fraction`), reused verbatim -- but here applied on
# EVERY trimming round, not as a one-off bootstrap: a single round was
# empirically insufficient on this project's own real frames (some run
# ~30-40% contaminated, and one round of thresholding leaves a fit "smoothly
# compromised" rather than cleanly split -- see
# `solve_homography_from_keypoints`'s docstring for the confirmed trace).
OUTLIER_CANDIDATE_FRACTION = 0.2
MAX_OUTLIER_REJECTION_ITERATIONS = 5
# A small redundancy margin above the bare 4-point mathematical floor --
# the same reasoning `camera_motion.py`'s `MIN_BACKGROUND_FEATURE_POINTS`
# (20, vs. the bare homography minimum of 4) applies: a fit resting on
# exactly the minimum has no slack left to have actually rejected anything
# and be trustworthy. Expressed as an offset from `min_points` (not a flat
# constant) so a caller who raises/lowers `min_points` gets a floor that
# still tracks it.
MIN_RELIABLE_INLIERS_MARGIN = 2

# --- Multi-frame rolling reliability (see KeypointReliabilityTracker) ---

# EWMA decay: each new observation contributes weight (1 - decay) = 0.1;
# prior history decays by `decay` per frame. Effective memory length is
# ~1 / (1 - decay) = 10 frames -- chosen to land inside this same
# mechanism's own `MIN_OBSERVATIONS_BEFORE_TRUSTED` (5-10) window, so a
# vertex's rolling score is dominated by roughly the same span of recent
# frames that also gates whether it's trusted at all.
ROLLING_RESIDUAL_EWMA_DECAY = 0.9
MIN_OBSERVATIONS_BEFORE_TRUSTED = 5
# Same 2x-median convention as the per-frame mechanism (`OUTLIER_RESIDUAL_
# MULTIPLE`), applied across VERTICES' rolling scores instead of across
# one frame's residuals -- reused rather than re-derived, per this
# project's established "same convention, same reasoning" discipline.
ROLLING_RELIABILITY_EXCLUSION_MULTIPLE = 2.0
# Cross-vertex median (see `KeypointReliabilityTracker.exclusions`) is only
# meaningful with enough trusted vertices to compute one from -- reuses the
# bare homography floor rather than inventing a separate number.
MIN_TRUSTED_VERTICES_FOR_ROLLING_EXCLUSION = DEFAULT_MIN_POINTS_FOR_HOMOGRAPHY


class KeypointReliabilityTracker:
    """ADR-016 (multi-frame extension): an EWMA of each vertex's
    reprojection residual, accumulated ACROSS frames, so a vertex that is
    consistently poorly localized (even if no single frame's own residual
    distribution is spread out enough to flag it -- the diagnosed failure
    mode of the PER-FRAME-ONLY mechanism in `solve_homography_from_keypoints`
    when called without a tracker) still gets excluded once enough evidence
    has accumulated.

    NO GROUND TRUTH REQUIRED: `update()` is fed each frame's OWN residual
    against that SAME frame's own best-fit homography (internal
    consistency over time), never a ground-truth position -- this tracker
    has no access to, and does not need, real-world validation data.

    STATEFUL AND SEQUENTIAL: unlike `solve_homography_from_keypoints`
    called without a tracker (a pure per-frame function, safe to call in
    any order on any frame), an instance of this class must be updated in
    real frame order for its EWMA to mean what its name says -- create one
    instance per video/clip, not a fresh one per frame.
    """

    def __init__(
        self,
        decay: float = ROLLING_RESIDUAL_EWMA_DECAY,
        min_observations: int = MIN_OBSERVATIONS_BEFORE_TRUSTED,
    ):
        self.decay = decay
        self.min_observations = min_observations
        self._ewma_residual_m: dict[int, float] = {}
        self._observation_count: dict[int, int] = {}

    def update(self, vertex_number: int, residual_m: float) -> None:
        """Records one more frame's residual observation for
        `vertex_number`. Call this for every vertex confidently detected
        in a frame, even ones excluded from that frame's fit -- an
        excluded vertex still needs its score updated, or it can never
        earn its way back if conditions change (e.g. a camera cut to a
        different angle)."""
        prior = self._ewma_residual_m.get(vertex_number)
        self._ewma_residual_m[vertex_number] = (
            residual_m if prior is None else self.decay * prior + (1.0 - self.decay) * residual_m
        )
        self._observation_count[vertex_number] = self._observation_count.get(vertex_number, 0) + 1

    def is_trusted(self, vertex_number: int) -> bool:
        """`True` once `vertex_number` has at least `min_observations`
        recorded observations -- before that, there isn't enough history
        for its rolling score to mean anything, and callers should fall
        back to the per-frame-only mechanism for it (Step 1.3's explicit
        early-clip/post-scene-cut fallback)."""
        return self._observation_count.get(vertex_number, 0) >= self.min_observations

    def exclusions(self, vertex_numbers: list[int]) -> set[int]:
        """Given the vertex numbers confidently detected in the CURRENT
        frame, returns the subset that should be excluded from this
        frame's fit on ROLLING evidence alone -- vertices whose
        accumulated EWMA residual exceeds `ROLLING_RELIABILITY_EXCLUSION_
        MULTIPLE` (2x) times the MEDIAN rolling residual among the
        currently-TRUSTED vertices in this frame (the same "compute the
        threshold from a clean/comparable set, not a contaminated one"
        discipline `OUTLIER_RESIDUAL_MULTIPLE` already applies within one
        frame -- here applied across vertices instead of across points).

        Returns an empty set (defers entirely to the per-frame mechanism)
        if fewer than `MIN_TRUSTED_VERTICES_FOR_ROLLING_EXCLUSION` of the
        given vertices are trusted yet -- not enough trusted vertices to
        compute a meaningful median from.
        """
        trusted = [v for v in vertex_numbers if self.is_trusted(v)]
        if len(trusted) < MIN_TRUSTED_VERTICES_FOR_ROLLING_EXCLUSION:
            return set()
        scores = {v: self._ewma_residual_m[v] for v in trusted}
        threshold = ROLLING_RELIABILITY_EXCLUSION_MULTIPLE * float(np.median(list(scores.values())))
        return {v for v, s in scores.items() if s > threshold}


_model = None  # lazy-loaded singleton, see _get_model()


def _get_model():
    """Lazily constructs the hosted `KeypointDetectionModel` client. Raises
    `RuntimeError` immediately if `ROBOFLOW_API_KEY` is not set -- this
    module never attempts to create an account or proceed without one.
    """
    global _model
    if _model is not None:
        return _model

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. This prototype requires a Roboflow API key "
            "(see ADR-014) -- it will not attempt to proceed without one. Set it in the "
            "environment, or in a git-ignored .env file loaded via python-dotenv."
        )

    from roboflow.models.keypoint_detection import KeypointDetectionModel

    _model = KeypointDetectionModel(
        api_key=api_key,
        id=f"{MODEL_WORKSPACE}/{MODEL_PROJECT}/{MODEL_VERSION}",
        version=MODEL_VERSION,
        confidence=1,  # request ALL candidate keypoints, including low-confidence ones --
        # filtering happens in solve_homography_from_keypoints, not here, so callers can
        # inspect the full confidence distribution rather than a pre-filtered subset.
    )
    return _model


def detect_pitch_keypoints(frame: np.ndarray) -> dict:
    """Runs the pretrained pitch-keypoint model on `frame` via Roboflow's
    hosted inference endpoint (see module docstring for why this is a
    network call, not local compute).

    Returns `{"keypoints": [{"vertex_number": int, "x_px": float,
    "y_px": float, "confidence": float}, ...], "pitch_confidence": float}`
    -- `pitch_confidence` is the model's own confidence that a pitch is
    present in the frame at all (its single object-detection class);
    `keypoints` always has exactly 32 entries (one per vertex_number
    1-32), since the model reports a confidence for every keypoint
    regardless of visibility -- near-zero confidence is how it signals
    "not visible in this frame," not omission.
    """
    model = _get_model()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        cv2.imwrite(tmp_path, frame)
        result = model.predict(tmp_path).json()
    finally:
        os.remove(tmp_path)

    predictions = result.get("predictions", [])
    if not predictions:
        return {"keypoints": [], "pitch_confidence": 0.0}

    best = max(predictions, key=lambda p: p["confidence"])
    keypoints = [
        {
            "vertex_number": KEYPOINT_LABEL_ORDER[kp["class_id"]],
            "x_px": kp["x"],
            "y_px": kp["y"],
            "confidence": kp["confidence"],
        }
        for kp in best["keypoints"]
    ]
    return {"keypoints": keypoints, "pitch_confidence": best["confidence"]}


def _fit_homography(points: list[dict]) -> np.ndarray | None:
    """Plain-DLT pixel->meter fit on `points` -- no internal RANSAC. Outlier
    rejection is handled explicitly by `solve_homography_from_keypoints`'s
    iterative median-threshold loop; running RANSAC underneath as well
    would just be a second, redundant outlier-rejection mechanism acting on
    the same points under a different, uncoordinated rule (ADR-016
    replaces the earlier RANSAC-based fit with this iterative mechanism
    entirely, rather than layering both).

    Returns `None` if fewer than `DEFAULT_MIN_POINTS_FOR_HOMOGRAPHY` points
    are given, or if `cv2.findHomography` cannot find a solution (e.g.
    near-collinear points).
    """
    if len(points) < DEFAULT_MIN_POINTS_FOR_HOMOGRAPHY:
        return None
    src = np.array([[p["x_px"], p["y_px"]] for p in points], dtype=np.float32)
    dst = np.array([PITCH_KEYPOINTS_METERS[p["vertex_number"]] for p in points], dtype=np.float32)
    homography, _ = cv2.findHomography(src, dst, method=0)
    return homography


def _residuals_meters(homography: np.ndarray, points: list[dict]) -> np.ndarray:
    """Forward (pixel->meter) reprojection residual, in METERS, for each of
    `points` against `homography`.

    DIRECTION CHOSEN FOR NUMERICAL STABILITY: `homography` already maps
    pixel -> meter (the direction `_fit_homography`/`calibration.
    compute_homography` both produce), so projecting each point's detected
    PIXEL position forward through it needs no extra step. Measuring
    residuals in pixel space instead would require inverting `homography`
    first (`np.linalg.inv`) purely to get back to pixel space to compare --
    an unnecessary extra numerical operation (and inversion is more
    error-sensitive than forward application) for no benefit, since a
    meter-space residual is exactly what this project's own accuracy bar
    (ADR-015's "~6m median error", `calibration.py`'s "100x68m pitch
    space") is already expressed in.
    """
    src = np.array([[p["x_px"], p["y_px"]] for p in points], dtype=np.float32)
    dst = np.array([PITCH_KEYPOINTS_METERS[p["vertex_number"]] for p in points], dtype=np.float32)
    src_h = np.hstack([src, np.ones((len(src), 1), dtype=np.float32)])
    proj = (homography @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return np.linalg.norm(proj - dst, axis=1)


def solve_homography_from_keypoints(
    detected_keypoints: list[dict],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_points: int = DEFAULT_MIN_POINTS_FOR_HOMOGRAPHY,
    reliability_tracker: "KeypointReliabilityTracker | None" = None,
    excluded_vertices: frozenset[int] = frozenset(),
) -> dict:
    """Solves a FRESH pixel->meter homography from ONE frame's detected
    keypoints (the `ViewTransformer` pattern) via ITERATIVE, MASKING-AWARE
    OUTLIER REJECTION (ADR-016) -- no composition with any prior frame's
    estimate, no dependency on frame ordering or history.

    ADR-016 REPLACES a fixed 6-vertex exclusion list (ADR-015's
    19/22/23/24/25/26) with this runtime mechanism: the physical cause
    behind that fixed list (far-side/foreshortened landmarks localize
    poorly despite high reported confidence) is real, but a fixed list
    only helps if the NEXT camera framing happens to foreshorten the same
    vertices -- unverified, and unlikely to hold exactly for a different
    angle. This mechanism instead discovers whichever points are
    unreliable IN THIS FRAME, from the data itself, every time it runs:

      1. Fit an initial homography on every keypoint clearing
         `min_confidence` (`_fit_homography`).
      2. Compute each fit point's reprojection residual against that fit
         (`_residuals_meters`, meters, see its docstring for the forward-
         direction choice), and check `classify_teams`'s Milestone 28
         threshold convention: does any point exceed `OUTLIER_RESIDUAL_MULTIPLE`
         (2x) times the MEDIAN residual among the CURRENTLY-FIT points? If
         not, this fit is already clean -- converged, stop here.
      3. **If not yet converged, TRIM a fixed fraction, not just whatever
         crosses the threshold**: remove the worst `OUTLIER_CANDIDATE_FRACTION`
         (0.2) of the currently-fit points BY RAW RESIDUAL MAGNITUDE (M28's
         `outlier_candidate_fraction`, reused verbatim), REFIT on the
         remainder, and repeat step 2's convergence check against the NEW
         fit. **Fixed-fraction trimming every round, not just a one-off
         bootstrap pass, was found necessary while building this on real
         data**: this project's own real frames sometimes have ~30-40% of
         their confidently-detected points genuinely bad (Milestone 39's
         finding), and a single round of either a 2x-median THRESHOLD or a
         single 20% BOOTSTRAP still leaves enough contamination that the
         next fit's own residual spread doesn't cleanly separate good from
         bad -- the exact masking failure `classify_teams`'s docstring
         names, generalized here to "one round isn't always enough,"
         confirmed empirically rather than assumed. Repeating the trim
         until step 2's threshold check actually passes handles this.
      4. Iterate steps 2-3 until: the threshold check passes (converged, a
         clean fit), `MAX_OUTLIER_REJECTION_ITERATIONS` (5) trimming rounds
         have run, or trimming further would drop the fit set below
         `min_points` -- in the last two cases, the LAST fit that still had
         >= `min_points` support is kept, rather than forcing a solve on
         too few points or trimming forever.
      5. Re-evaluate EVERY originally-confident point (including every one
         trimmed away along the way) against the FINAL fit, using a
         threshold from the final fit set's own (by now clean) residual
         distribution -- mirrors M28's "refit centroids, then re-evaluate
         every original point" pattern, giving a trimmed point a fair
         chance to be re-admitted if the final fit explains it well after
         all (this is where a genuinely-good point removed by an early,
         still-imprecise trimming round gets its recourse).
      6. STALENESS FALLBACK: if the final inlier count is below
         `min_points + MIN_RELIABLE_INLIERS_MARGIN` (a small redundancy
         margin above the bare mathematical floor -- see that constant's
         own comment), the frame is flagged `is_stale`, reusing
         `CameraMotionTracker`'s exact naming convention for "don't
         silently trust an unreliable result." UNLIKE
         `CameraMotionTracker.get_corrected_homography` (which always
         returns its best estimate even when stale, since a continuously-
         composed correction has no better fallback), this function
         returns `homography: None` when `is_stale` -- a per-frame anchor
         solve has a real alternative to a bad estimate (skip this
         frame's overlay), so failing closed is preferred here.

    `detected_keypoints`: `detect_pitch_keypoints(frame)["keypoints"]`.
    `min_confidence`: keypoints below this are excluded before any fitting.
    `min_points`: the bare mathematical floor (4, matching this project's
    other homography code) for attempting a solve, or continuing to
    refit, at all.
    `reliability_tracker`: optional `KeypointReliabilityTracker` (ADR-016's
    multi-frame extension). When given, any vertex its `exclusions()`
    flags on ACCUMULATED cross-frame evidence is dropped from the
    candidate set BEFORE this frame's own per-frame trimming runs (steps
    1-6 below then proceed on whatever remains) -- this is how a vertex
    that's consistently mediocre-but-not-obviously-bad in any single frame
    (the diagnosed limitation of running this function without a tracker)
    still gets excluded once enough cross-frame evidence exists. If
    dropping the rolling-flagged vertices would leave fewer than
    `min_points` candidates, the rolling exclusion is SKIPPED entirely for
    this frame (falls back to the plain per-frame mechanism on the full
    `confident` set) rather than starving the solve. Regardless of
    whether rolling exclusion changed anything, every confidently-detected
    vertex's residual against the FINAL homography is fed back into the
    tracker via `update()` before returning -- including vertices the
    rolling mechanism itself excluded, so their score can still recover if
    conditions change (e.g. a camera cut).
    `excluded_vertices`: vertex numbers dropped BEFORE any fitting, no
    matter what this frame's own data looks like -- e.g.
    `ADR015_KNOWN_UNRELIABLE_VERTICES`. **This is the only mechanism in
    this file validated to ~6.2m median error (ADR-015); passing it is the
    current recommendation for any real usage.** `reliability_tracker` and
    `excluded_vertices` are independent and may be combined, though ADR-016
    found the tracker adds no benefit on its own.

    Returns a dict: `{"homography": np.ndarray | None, "is_stale": bool,
    "inlier_vertex_numbers": list[int], "outlier_vertex_numbers":
    list[int], "iterations": int}`. `homography` follows the same
    pixel->meter convention as `calibration.compute_homography`.
    """
    confident = [
        kp
        for kp in detected_keypoints
        if kp["confidence"] >= min_confidence and kp["vertex_number"] not in excluded_vertices
    ]

    def _stale_result(outliers: list[dict], iterations: int = 0) -> dict:
        return {
            "homography": None,
            "is_stale": True,
            "inlier_vertex_numbers": [],
            "outlier_vertex_numbers": [p["vertex_number"] for p in outliers],
            "iterations": iterations,
        }

    if len(confident) < min_points:
        return _stale_result(confident)

    candidates = confident
    if reliability_tracker is not None:
        rolling_excluded = reliability_tracker.exclusions([p["vertex_number"] for p in confident])
        narrowed = [p for p in confident if p["vertex_number"] not in rolling_excluded]
        if len(narrowed) >= min_points:
            candidates = narrowed
        # else: rolling exclusion would starve the solve -- skip it this
        # frame, fall back to the full `confident` set.

    homography = _fit_homography(candidates)
    if homography is None:
        return _stale_result(confident)
    fit_points = candidates

    # Steps 2-4: trim a fixed fraction per round (not a one-off bootstrap,
    # and not threshold-gated removal -- see docstring for why: a single
    # round of either was insufficient against this project's own real
    # frames, where the true contaminated fraction sometimes exceeds what
    # one round of 2x-median thresholding can safely separate).
    #
    # SKIPPED WHEN `excluded_vertices` IS GIVEN: `excluded_vertices` and
    # this iterative trimming are ALTERNATIVES, not composable by default
    # -- measured directly while building this: running the trimming loop
    # ON TOP OF an already ADR-015-filtered candidate set made the
    # high-motion-window LOOCV median WORSE (10.3m vs. 7.0m with a single
    # plain fit), because trimming a small, already-mostly-clean set still
    # finds SOME point above the 2x-median threshold from ordinary
    # per-frame variance and removes it needlessly. `excluded_vertices`
    # therefore takes the single-fit path ADR-015 was actually validated
    # with, rather than being silently degraded by ADR-016's mechanism.
    iterations_run = 0
    for iterations_run in (
        range(1, MAX_OUTLIER_REJECTION_ITERATIONS + 1) if not excluded_vertices else range(0)
    ):
        residuals = _residuals_meters(homography, fit_points)
        threshold = OUTLIER_RESIDUAL_MULTIPLE * float(np.median(residuals))

        if not bool((residuals > threshold).any()):
            break  # converged: this fit is already clean

        num_to_trim = max(1, round(OUTLIER_CANDIDATE_FRACTION * len(fit_points)))
        if len(fit_points) - num_to_trim < min_points:
            break  # trimming further would drop below the well-conditioned floor

        ascending_order = np.argsort(residuals)
        trimmed = [fit_points[i] for i in ascending_order[: len(fit_points) - num_to_trim]]
        refit = _fit_homography(trimmed)
        if refit is None:
            break
        homography = refit
        fit_points = trimmed

    # Step 5: re-evaluate EVERY point in `candidates` (including every one
    # trimmed away along the way) against the FINAL fit, using a threshold
    # from the final fit set's own (clean, converged) residual
    # distribution -- a point trimmed by an early, still-imprecise round
    # gets a fair chance to be re-admitted here. Deliberately NOT re-
    # admitting rolling-excluded vertices here: they were excluded on
    # ACCUMULATED cross-frame evidence, which one good-looking frame's
    # residual should not be allowed to override within a single solve.
    candidate_residuals = _residuals_meters(homography, candidates)
    final_threshold = OUTLIER_RESIDUAL_MULTIPLE * float(
        np.median(_residuals_meters(homography, fit_points))
    )
    inliers = [p for p, r in zip(candidates, candidate_residuals) if r <= final_threshold]
    per_frame_outliers = [p for p, r in zip(candidates, candidate_residuals) if r > final_threshold]
    rolling_excluded_points = [p for p in confident if p not in candidates]
    outliers = per_frame_outliers + rolling_excluded_points

    # Inlier/outlier reporting is unconditional (useful diagnostic
    # information regardless of outcome); only the homography ITSELF is
    # withheld when stale, per Step 6's fail-closed choice above.
    reliable_floor = min_points + MIN_RELIABLE_INLIERS_MARGIN
    is_stale = len(inliers) < reliable_floor

    # Feed every confidently-detected vertex's residual against this FINAL
    # homography back into the tracker (Step 1.2) -- including vertices
    # the rolling mechanism itself excluded, so their score can still
    # recover if conditions change (e.g. a camera cut). Skipped when
    # `is_stale`: there's no trustworthy fit to measure residuals against.
    if reliability_tracker is not None and not is_stale:
        all_confident_residuals = _residuals_meters(homography, confident)
        for p, r in zip(confident, all_confident_residuals):
            reliability_tracker.update(p["vertex_number"], float(r))

    return {
        "homography": None if is_stale else homography,
        "is_stale": is_stale,
        "inlier_vertex_numbers": [p["vertex_number"] for p in inliers],
        "outlier_vertex_numbers": [p["vertex_number"] for p in outliers],
        "iterations": iterations_run,
    }
