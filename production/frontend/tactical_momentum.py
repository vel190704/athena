"""Tactical Momentum (additive new feature): a rolling-window smoothing +
trend indicator over the already-streamed `threat_15s` signal that
dashboard.py's Live CV Monitor tab receives over `/ws/tactical-stream`.

Deliberately a PLAIN Python module -- no `streamlit` import, no side
effects at import time -- kept separate from `dashboard.py` (a Streamlit
SCRIPT, not a safely importable module: it executes top-level Streamlit
calls that assume a live ScriptRunContext) specifically so this pure
computation can be unit-tested directly via a normal `import`, without
going through `streamlit.testing.v1.AppTest`. `dashboard.py` imports
`_compute_tactical_momentum` and the constants below and calls them over
the SAME `threat_buffer` list it already accumulates from incoming
WebSocket messages -- this module holds no buffer of its own and performs
no I/O; all buffering/streaming/rendering stays in dashboard.py.

ADR-021 condition 2 (checked explicitly for BOTH additions in this module,
Tactical Momentum and Match Segmentation below, not assumed): both inherit
the compliance audit's existing "StatsBomb-sourced live tactical
stream... already condition-2-compliant by construction" finding (ADR-021's
Update section), for BOTH `source=statsbomb` and `source=cv` -- each reads
ONLY the already-compliant `threat_15s` scalar out of each message (never
`players`/`ball`, which `source=cv` messages also carry) and reduces it
FURTHER (Match Segmentation reduces Tactical Momentum's own already-derived
`smoothed_now`/`classification` output into an even coarser discrete
category label). A derived value computed FROM an already-exempt aggregate
cannot carry MORE individually-recoverable information than that aggregate
-- the same reasoning ADR-021's own Tactical Entropy/Player Similarity
addenda established -- so both stay unconditionally EXEMPT, not gated
behind `PUBLIC_DEPLOYMENT`, for the same reason and via the same "no new
data extraction" mechanism.
"""

# Chosen INDEPENDENTLY of dashboard.py's MAX_THREAT_BUFFER_LEN (60) -- that
# buffer controls how much history the LINE CHART retains for visual
# inspection, a separate concern from how many points denoise one momentum
# reading. 10 was chosen so several independent windows (6, at the
# buffer's full 60-message length) fit inside the retained buffer, and so
# momentum updates on roughly a 10-message lag (~10s at this stream's
# default delay=1.0s pacing, per simulator.py's `live_match_stream`) --
# responsive enough to be useful live, without letting single-message
# noise dominate it.
MOMENTUM_SMOOTHING_WINDOW_MESSAGES = 10

# The discrete-derivative lookback: trend = smoothed_now - smoothed_then,
# where smoothed_then is the same MOMENTUM_SMOOTHING_WINDOW_MESSAGES-wide
# average as of this many messages ago. Kept as a SEPARATE named constant
# from the smoothing window above (even though both currently equal 10)
# since they are conceptually distinct: one is "how many points denoise a
# single reading," the other is "how far back to compare against."
MOMENTUM_TREND_LOOKBACK_MESSAGES = 10

MOMENTUM_MIN_MESSAGES_FOR_TREND = MOMENTUM_SMOOTHING_WINDOW_MESSAGES + MOMENTUM_TREND_LOOKBACK_MESSAGES

# Half of api.py's own SPIKE_THRESHOLD (0.05) -- see _compute_tactical_
# momentum's docstring for the full reasoning.
MOMENTUM_TREND_THRESHOLD = 0.025


def _compute_tactical_momentum(threat_buffer: list[float]) -> dict:
    """Tactical Momentum: a rolling-window smoothing + trend indicator
    over the already-streamed `threat_15s` signal. Pure, side-effect-free
    computation over a plain list of floats -- reads the SAME
    `threat_buffer` dashboard.py's existing line chart already
    accumulates, does not duplicate any buffering logic, and never
    touches the WebSocket message itself (works identically regardless of
    which field(s) a given message carries beyond `threat_15s` -- see
    dashboard.py's module docstring for why that matters for source=cv).

    STEP 0 DEFINITIONS -- a genuine judgment call, documented explicitly
    here rather than left implicit (same discipline player_report.py's
    shot-map "on-target" choice already established for this project):

    1. SMOOTHING: a simple moving average over the most recent
       MOMENTUM_SMOOTHING_WINDOW_MESSAGES (10) raw `threat_15s` readings.
       See the module-level constant's own comment for why 10, and why
       independent of dashboard.py's MAX_THREAT_BUFFER_LEN (60).
    2. TREND: a discrete derivative, NOT a linear regression -- the
       smoothed value now MINUS the smoothed value from
       MOMENTUM_TREND_LOOKBACK_MESSAGES (10) messages ago. A full
       regression would fit a line to samples that are event-paced, not
       evenly time-spaced (see `live_match_stream`'s own docstring: one
       message per 360-tagged event, not one per fixed real-world
       second), for marginal benefit over a simple two-point difference
       of two already-smoothed values -- not used here for that reason.
    3. CLASSIFICATION: `trend > +MOMENTUM_TREND_THRESHOLD` (0.025, i.e.
       2.5 percentage points of threat probability) -> "Building";
       `trend < -MOMENTUM_TREND_THRESHOLD` -> "Fading"; otherwise ->
       "Stable". 0.025 is deliberately HALF of api.py's own
       SPIKE_THRESHOLD (0.05): SPIKE_THRESHOLD is calibrated to flag a
       single, noisy message-to-message jump, whereas this trend value is
       already a difference of two SMOOTHED (10-point-averaged) readings
       -- a real, sustained shift half that size already reflects genuine
       directional movement rather than raw single-message noise, and
       reasonably deserves a lower bar than the raw spike-alert
       threshold. A reasonable, stated choice, not a provably optimal
       one -- validated against a real match stream (see
       test_dashboard.py) rather than picked and left unchecked.
    4. INSUFFICIENT DATA: fewer than MOMENTUM_MIN_MESSAGES_FOR_TREND (20
       = 10 + 10) raw `threat_15s` readings in the buffer -> returns a
       `{"status": "warming_up", ...}` dict instead of a fabricated or
       crashing trend/classification. Once enough data exists, returns
       `{"status": "ready", "smoothed_now": ..., "trend": ...,
       "classification": ...}`.
    """
    n = len(threat_buffer)
    if n < MOMENTUM_MIN_MESSAGES_FOR_TREND:
        return {
            "status": "warming_up",
            "messages_so_far": n,
            "messages_needed": MOMENTUM_MIN_MESSAGES_FOR_TREND,
        }

    smoothed_now = (
        sum(threat_buffer[-MOMENTUM_SMOOTHING_WINDOW_MESSAGES:]) / MOMENTUM_SMOOTHING_WINDOW_MESSAGES
    )
    lookback_window = threat_buffer[
        -(MOMENTUM_TREND_LOOKBACK_MESSAGES + MOMENTUM_SMOOTHING_WINDOW_MESSAGES):
        -MOMENTUM_TREND_LOOKBACK_MESSAGES
    ]
    smoothed_then = sum(lookback_window) / MOMENTUM_SMOOTHING_WINDOW_MESSAGES

    trend = smoothed_now - smoothed_then
    if trend > MOMENTUM_TREND_THRESHOLD:
        classification = "Building"
    elif trend < -MOMENTUM_TREND_THRESHOLD:
        classification = "Fading"
    else:
        classification = "Stable"

    return {
        "status": "ready",
        "smoothed_now": smoothed_now,
        "trend": trend,
        "classification": classification,
    }


# =============================================================================
# Match Segmentation (additive new feature): classifies the current period
# of live play into a discrete game-phase label, derived ENTIRELY from
# `_compute_tactical_momentum`'s own already-computed output
# (`smoothed_now`, `classification`) -- no new data extraction, no change to
# Tactical Momentum's own logic above. Kept in this SAME plain-Python
# module for the SAME testability reason Tactical Momentum itself lives
# here (no `streamlit` import, unit-testable via a normal `import`).
# =============================================================================

from collections import Counter

# STEP 0.1: ELEVATED THREAT LEVEL -- a genuine, disclosed judgment call,
# NOT api.py's SPIKE_THRESHOLD (0.05) reused blindly: SPIKE_THRESHOLD is a
# DELTA (how big a single message-to-message jump must be to alert), a
# fundamentally different quantity from this, an absolute LEVEL threshold
# on the SMOOTHED signal (is the CURRENT state elevated at all, regardless
# of whether it just changed). Reusing a delta constant as a level cutoff
# would be a category error, not a shortcut.
#
# Chosen by direct inspection of a real streamed match (3857276, 351 real
# `threat_15s` messages, `test_dashboard.py`'s own validation match),
# smoothed the SAME way this module's own `MOMENTUM_SMOOTHING_WINDOW_MESSAGES`
# already does (332 ready `smoothed_now` readings once warmed up): real
# smoothed values ranged [0.034, 0.495], median 0.105, mean 0.149. 0.15
# (15 percentage points of predicted 15s shot probability) sits just above
# the median/mean and marks the top ~30% of real smoothed readings from
# this match as "elevated" -- a meaningful, non-trivial minority (neither
# an always-true nor an always-false cutoff against real data), consistent
# with `generate_team_report`'s own real observed split between
# defensive/middle-third threat (~0.06-0.11) and attacking-third threat
# (~0.46-0.50) elsewhere in this project.
ELEVATED_THREAT_LEVEL = 0.15

# STEP 0.1 (continued): the DWELL/HYSTERESIS window. A phase label must not
# flip on a single noisy message, the SAME "don't flip on noise" reasoning
# Tactical Momentum's own smoothing/lookback windows already apply -- but
# implemented here as a MAJORITY-VOTE PERSISTENCE FILTER over the last
# SEGMENT_DWELL_MESSAGES RAW (pre-hysteresis) classifications, not a
# stateful "N consecutive agreements" state machine: this keeps
# `classify_match_phase` a PURE function of `threat_buffer` alone (like
# `_compute_tactical_momentum` above), needing no externally-threaded
# state object, and it is straightforward to unit-test directly against a
# synthetic noisy sequence (see test_dashboard.py). 5 was chosen as half
# of MOMENTUM_SMOOTHING_WINDOW_MESSAGES (10) -- big enough that a single
# one-message blip (1 vote out of 5) can never win a majority against a
# genuinely persistent alternative, small enough that a real, sustained
# phase change is still reflected within ~5 messages (~5s at this stream's
# default delay=1.0s pacing), not many tens of messages later.
SEGMENT_DWELL_MESSAGES = 5

_PHASE_BUILDING_ATTACK = "Building Attack"
_PHASE_DEFENSIVE_CONSOLIDATION = "Defensive Consolidation"
_PHASE_TRANSITION = "Transition"
_PHASE_STABLE = "Stable"


def _raw_phase_classification(smoothed_now: float, momentum_classification: str) -> str:
    """STEP 0.1: the RAW (pre-hysteresis) phase label for one instant --
    a genuine, disclosed judgment call, a plain 2x3 decision table over
    (threat LEVEL: elevated/low) x (momentum: Building/Stable/Fading),
    collapsed into 4 categories:

      - "Building Attack": threat is ELEVATED and momentum is Building or
        Stable (NOT fading) -- the attacking side currently has a live,
        sustained-or-escalating threat. Covers both the clearest case
        (elevated AND still rising) and the "elevated and holding" case
        (Stable momentum at an already-elevated level still means a real,
        currently-live threat, not a calm phase).
      - "Defensive Consolidation": threat is LOW and momentum is Fading --
        the clearest "defense has shut things down and it is still
        receding" signal.
      - "Stable": threat is LOW and momentum is Stable -- genuinely calm,
        uneventful play, the natural default/normal state.
      - "Transition": the remaining two cells, where LEVEL and TREND
        actively DISAGREE about where things are headed -- (elevated but
        Fading: threat was high and is now receding, not yet fully
        resolved into "consolidated") or (low but Building: threat is
        just starting to rise from a calm base, not yet meaningfully
        dangerous). Both are genuinely in-between, not-yet-resolved
        states, which is exactly what "Transition" should mean.
    """
    elevated = smoothed_now >= ELEVATED_THREAT_LEVEL

    if elevated and momentum_classification in ("Building", "Stable"):
        return _PHASE_BUILDING_ATTACK
    if not elevated and momentum_classification == "Fading":
        return _PHASE_DEFENSIVE_CONSOLIDATION
    if not elevated and momentum_classification == "Stable":
        return _PHASE_STABLE
    return _PHASE_TRANSITION  # (elevated, Fading) or (not elevated, Building)


def classify_match_phase(threat_buffer: list[float]) -> dict:
    """Match Segmentation: classifies the CURRENT period of live play into
    a discrete game-phase label. Pure, side-effect-free -- reads the SAME
    `threat_buffer` `_compute_tactical_momentum` above already reads, adds
    no new buffering or streaming logic.

    STEP 0.2, INSUFFICIENT DATA: segmentation cannot exist before momentum
    itself does (the phase table needs `smoothed_now`, which only exists
    once `_compute_tactical_momentum` reports `"status": "ready"`) -- reuses
    that SAME `MOMENTUM_MIN_MESSAGES_FOR_TREND` gate rather than inventing
    a second, independent warm-up threshold. Returns
    `{"status": "warming_up", ...}` (same shape as momentum's own) until
    then.

    HYSTERESIS: recomputes the RAW classification (`_raw_phase_classification`)
    for the current buffer AND for each of the `SEGMENT_DWELL_MESSAGES - 1`
    prior buffer states (i.e. the buffer truncated by 1, 2, ... messages),
    then reports the MAJORITY-VOTE label across however many of those raw
    classifications are actually available (fewer than
    `SEGMENT_DWELL_MESSAGES` only in the first few messages right after
    warm-up, when going further back would cross the warm-up boundary).
    Ties favor the MORE RECENT raw classification (Python's `Counter.
    most_common` preserves first-seen order among equal counts, and the
    CURRENT classification is always inserted first) -- a deliberate,
    disclosed choice, not an arbitrary one: on a genuine split vote,
    trusting the freshest read is more useful live than trusting the
    stalest one.

    Returns `{"status": "ready", "phase": ..., "raw_phase": ...,
    "smoothed_now": ..., "momentum_classification": ...}` once ready --
    `raw_phase` (this instant's own unsmoothed label, before the majority
    vote) is included alongside the hysteresis-smoothed `phase` so a
    caller/UI can show both if useful, the same way momentum exposes both
    `trend` (a number) and `classification` (its own smoothed label).
    """
    momentum = _compute_tactical_momentum(threat_buffer)
    if momentum["status"] == "warming_up":
        return {
            "status": "warming_up",
            "messages_so_far": momentum["messages_so_far"],
            "messages_needed": momentum["messages_needed"],
        }

    raw_labels = []
    for lag in range(SEGMENT_DWELL_MESSAGES):
        if lag == 0:
            truncated_buffer = threat_buffer
        else:
            truncated_buffer = threat_buffer[:-lag]
        lagged_momentum = _compute_tactical_momentum(truncated_buffer)
        if lagged_momentum["status"] == "warming_up":
            break  # crossed the warm-up boundary -- no more history to vote with
        raw_labels.append(
            _raw_phase_classification(lagged_momentum["smoothed_now"], lagged_momentum["classification"])
        )

    phase = Counter(raw_labels).most_common(1)[0][0]

    return {
        "status": "ready",
        "phase": phase,
        "raw_phase": raw_labels[0],
        "smoothed_now": momentum["smoothed_now"],
        "momentum_classification": momentum["classification"],
    }
