"""Milestone 22 validation: Bayesian Habit Blending (Module 6).

STEP 0 FINDING (see habit_memory.py's module docstring for the full
writeup): StatsBomb's public 360 freeze-frame data exposes NO per-player
identity for ~21 of the 22 visible players -- only the acting player's
identity is known (via the parent event's own `player` field). This
re-scopes habit blending to the acting player only; this file's real-data
test (Step 5.5) exercises exactly that scope, not a fabricated multi-player
version.
"""

import math
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import torch

from production.src.ingestion.statsbomb_io import (
    fetch_match_360,
    fetch_match_events,
    parse_360_frame,
)
from production.src.pipeline.data_split import match_level_split
from production.src.pipeline.feature_extractor import extract_features
from production.src.pipeline.habit_memory import (
    GRID_COLS,
    GRID_ROWS,
    MIN_HISTORICAL_EVENTS,
    bayesian_blend_habit,
    build_player_match_buckets,
    generate_mock_heatmap,
    generate_player_heatmap,
    heatmap_from_buckets,
)

MATCH_ID = 3857276


def _fetch_first_qualifying_event_and_frame():
    """First real period-1 Pass event with an associated 360 frame --
    same event-selection convention as every prior milestone's tests."""
    events = fetch_match_events(MATCH_ID)
    frames = fetch_match_360(MATCH_ID)
    frames_by_event_uuid = {f["event_uuid"]: f for f in frames}

    for event in events:
        if event["period"] != 1:
            continue
        if event["type"]["name"] != "Pass":
            continue
        if "location" not in event:
            continue
        frame_data = frames_by_event_uuid.get(event["id"])
        if frame_data is None:
            continue
        return event, frame_data, events

    raise RuntimeError(f"no period-1 Pass event with a 360 frame found in match {MATCH_ID}")


def test_bayesian_blend_pulls_expectation_toward_prior():
    """Step 5.1: a synthetic Prior heavily weighted to the left wing
    (x=20, y=60), blended with a live position at pitch center (x=50,
    y=34), should pull the Posterior's expectation toward the Prior --
    lower x, higher y than the live position alone.
    """
    prior = generate_mock_heatmap((20.0, 60.0), sigma_meters=15.0)
    assert prior.shape == (GRID_COLS, GRID_ROWS)
    assert math.isclose(prior.sum(), 1.0, abs_tol=1e-9)

    expected_x, expected_y = bayesian_blend_habit((50.0, 34.0), prior)

    assert math.isfinite(expected_x) and math.isfinite(expected_y)
    assert expected_x < 50.0, f"expected_x={expected_x} should be pulled below the live x=50"
    assert expected_y > 34.0, f"expected_y={expected_y} should be pulled above the live y=34"


def test_cold_start_falls_back_to_uniform_prior():
    """Step 5.2: fewer than MIN_HISTORICAL_EVENTS qualifying events for a
    player must produce the UNIFORM fallback grid, not a sparse/noisy one.
    """
    player_id = 999999  # synthetic id, guaranteed not to appear in real data
    sparse_event_count = MIN_HISTORICAL_EVENTS - 1
    assert sparse_event_count >= 0

    synthetic_events = [
        {"player": {"id": player_id, "name": "Synthetic Player"}, "location": [60.0, 40.0]}
        for _ in range(sparse_event_count)
    ]
    events_by_match = {1: synthetic_events}

    heatmap = generate_player_heatmap(player_id, events_by_match, exclude_match_id=None)

    assert heatmap.shape == (GRID_COLS, GRID_ROWS)
    assert np.all(np.isfinite(heatmap))
    uniform_value = 1.0 / (GRID_COLS * GRID_ROWS)
    assert np.allclose(heatmap, uniform_value), (
        f"expected a uniform fallback ({uniform_value} everywhere) with only "
        f"{sparse_event_count} qualifying events (< {MIN_HISTORICAL_EVENTS}), got {heatmap}"
    )


def test_generate_player_heatmap_excludes_target_match_leakage():
    """The data-leakage guard: events from `exclude_match_id` must never
    contribute to the Prior, even if that match alone would otherwise push
    the player over the cold-start threshold.
    """
    player_id = 42424242
    target_match_events = [
        {"player": {"id": player_id, "name": "X"}, "location": [10.0, 10.0]}
        for _ in range(MIN_HISTORICAL_EVENTS + 5)
    ]
    other_match_events = [
        {"player": {"id": player_id, "name": "X"}, "location": [10.0, 10.0]}
        for _ in range(3)
    ]
    events_by_match = {123: target_match_events, 456: other_match_events}

    heatmap = generate_player_heatmap(player_id, events_by_match, exclude_match_id=123)

    # Excluding match 123 leaves only 3 qualifying events -- below the
    # cold-start threshold -- so this MUST be the uniform fallback, proving
    # match 123's events were genuinely excluded (if they'd leaked in, the
    # combined count would clear the threshold and produce a sharply
    # spiked, non-uniform prior instead).
    uniform_value = 1.0 / (GRID_COLS * GRID_ROWS)
    assert np.allclose(heatmap, uniform_value)


def test_bayesian_blend_numerical_stability_under_minimal_overlap():
    """Step 5.3: an adversarial case where the Prior and the live-position
    Gaussian have essentially no overlap (a tight Prior in one corner, a
    tight live-position likelihood in the opposite corner) must still
    produce a FINITE result -- this is exactly what the epsilon floor
    (habit_memory.bayesian_blend_habit) exists to guarantee.
    """
    sparse_prior = generate_mock_heatmap((2.0, 2.0), sigma_meters=1.0)

    expected_x, expected_y = bayesian_blend_habit((98.0, 66.0), sparse_prior, sigma_meters=0.5)

    assert math.isfinite(expected_x), f"expected_x is not finite: {expected_x}"
    assert math.isfinite(expected_y), f"expected_y is not finite: {expected_y}"


def test_extract_features_default_behavior_unchanged():
    """Step 5.4, CRITICAL regression test: with habit_heatmaps=None (the
    default), extract_features must behave EXACTLY as it did before
    Milestone 22 -- verified here by calling it both with the parameter
    completely omitted and with it explicitly passed as None, on the same
    real match state, and asserting byte-identical results. This is what
    makes the opt-in boundary real rather than aspirational: every prior
    MLflow baseline (MLP, GNN, Deep Ensemble) was trained on this exact
    unblended path.
    """
    event, frame_data, _ = _fetch_first_qualifying_event_and_frame()
    parsed_frame = parse_360_frame(event, frame_data)

    features_default_omitted = extract_features(parsed_frame)
    features_default_explicit = extract_features(parsed_frame, habit_heatmaps=None)

    assert features_default_omitted == features_default_explicit, (
        "extract_features must produce IDENTICAL output whether habit_heatmaps is omitted or "
        "explicitly None -- any difference here would mean the opt-in default isn't truly off."
    )
    for key, value in features_default_omitted.items():
        assert math.isfinite(value), f"feature {key!r} is not finite: {value}"


def test_extract_features_with_habit_blending_enabled_differs_and_is_finite():
    """Step 5.5: with habit blending explicitly enabled for the frame's
    real acting player (the only player identity Step 0 confirmed is
    available), the resulting features must differ from the unblended
    baseline and contain no NaNs/shape errors.
    """
    event, frame_data, _ = _fetch_first_qualifying_event_and_frame()
    parsed_frame = parse_360_frame(event, frame_data)

    actor_player_id = parsed_frame["actor_player_id"]
    assert actor_player_id is not None, "expected a known actor_player_id for a real Pass event"

    # A synthetic prior deliberately far from the actor's actual live
    # position, so the blend has a visible effect worth asserting on.
    actor_pos = parsed_frame["player_pos"][parsed_frame["is_actor"]][0].tolist()
    far_position = (
        (actor_pos[0] + 40.0) % 100.0,
        (actor_pos[1] + 30.0) % 68.0,
    )
    habit_heatmaps = {actor_player_id: generate_mock_heatmap(far_position, sigma_meters=10.0)}

    features_unblended = extract_features(parsed_frame)
    features_blended = extract_features(parsed_frame, habit_heatmaps=habit_heatmaps)

    assert set(features_unblended.keys()) == set(features_blended.keys())
    for key, value in features_blended.items():
        assert math.isfinite(value), f"blended feature {key!r} is not finite: {value}"

    assert features_unblended != features_blended, (
        "habit blending with a heatmap far from the actor's live position should change at "
        "least one feature -- got identical output to the unblended baseline"
    )

    # And the ORIGINAL parsed_frame's player_pos must be untouched (never
    # mutated in place by extract_features).
    assert torch.equal(
        parsed_frame["player_pos"][parsed_frame["is_actor"]][0],
        torch.tensor(actor_pos, dtype=parsed_frame["player_pos"].dtype),
    )


def test_heatmap_from_buckets_leakage_guard_holds_under_match_level_partition():
    """Milestone 35 / ADR-011 Step 2.2: confirms the leakage guard (a
    sample's own match is never in its own heatmap corpus) still holds
    now that `train.py` feeds `heatmap_from_buckets` a match-level-clean
    `included_match_ids` set (via `match_level_split`) instead of the
    Milestone 23 sample-level "any match with a validation sample is
    excluded entirely" workaround.

    Under match-level splitting, a sample's own match is either (a) in
    the training group -- in which case it's still explicitly excluded
    via `exclude_match_id`, exactly as Milestone 22's base rule always
    required -- or (b) in the validation group, and therefore was never
    in `included_match_ids` (the caller's training-match set) to begin
    with. Both paths are exercised here directly against the real bucket
    functions, not a reimplementation.
    """
    player_id = 999
    # Three matches: 100 and 200 each have MIN_HISTORICAL_EVENTS qualifying
    # events (enough to clear the cold-start threshold on their own), 300
    # has only 3 (deliberately below it on its own).
    events_by_match = {
        100: [{"player": {"id": player_id}, "location": [10.0, 10.0]} for _ in range(MIN_HISTORICAL_EVENTS)],
        200: [{"player": {"id": player_id}, "location": [50.0, 30.0]} for _ in range(MIN_HISTORICAL_EVENTS)],
        300: [{"player": {"id": player_id}, "location": [90.0, 60.0]} for _ in range(3)],
    }
    buckets = build_player_match_buckets(events_by_match)

    # Simulate match_level_split having assigned match 300 to validation,
    # and matches 100/200 to training -- a clean partition, no straddling.
    training_match_ids = [100, 200]

    # Case (a): a TRAINING-split sample whose own match is 100 -- must
    # exclude match 100's events via exclude_match_id, leaving only
    # match 200's events (still >= MIN_HISTORICAL_EVENTS on its own).
    heatmap_a, num_qualifying_a, is_cold_start_a = heatmap_from_buckets(
        player_id, buckets, training_match_ids, exclude_match_id=100
    )
    assert num_qualifying_a == MIN_HISTORICAL_EVENTS, (
        f"expected exactly match 200's {MIN_HISTORICAL_EVENTS} events after excluding match 100, "
        f"got {num_qualifying_a}"
    )
    assert is_cold_start_a is False

    # Case (b): match 300 (the validation-split match) was NEVER passed
    # in `training_match_ids` at all -- so its events can't leak into any
    # training-split sample's heatmap regardless of exclude_match_id.
    heatmap_b, num_qualifying_b, _is_cold_start_b = heatmap_from_buckets(
        player_id, buckets, training_match_ids, exclude_match_id=100
    )
    assert num_qualifying_b == MIN_HISTORICAL_EVENTS  # identical to (a) -- match 300 never reachable

    # Sanity: if exclude_match_id excluded NOTHING (e.g. a genuinely
    # different sample's own match), both training matches' events count,
    # proving the guard is doing real exclusion work above, not a no-op.
    heatmap_c, num_qualifying_c, _is_cold_start_c = heatmap_from_buckets(
        player_id, buckets, training_match_ids, exclude_match_id=999999
    )
    assert num_qualifying_c == 2 * MIN_HISTORICAL_EVENTS
    assert not np.allclose(heatmap_a, heatmap_c)


def test_match_level_split_training_matches_are_fully_eligible_for_habit_corpus():
    """Milestone 35 / ADR-011 Step 2: with match-level splitting, a match
    assigned to the training group is FULLY eligible for the training-
    bucket corpus (no partial/conservative exclusion needed) -- verified
    by confirming `match_level_split`'s train/val match partition, when
    fed directly into `heatmap_from_buckets` as `included_match_ids`, is
    already clean by construction (every training match contributes,
    none held back "just in case").
    """
    # 20 matches, 10 samples each -- mirrors test_data_split.py's
    # even-sized scenario so the match-count split is exact and easy to
    # reason about here too.
    sample_match_ids = [i // 10 for i in range(200)]
    train_indices, val_indices = match_level_split(sample_match_ids, val_fraction=0.2, seed=42)

    training_match_ids = sorted({sample_match_ids[i] for i in train_indices})
    validation_match_ids = sorted({sample_match_ids[i] for i in val_indices})

    # No conservative exclusion: training + validation match sets are a
    # complete, non-overlapping partition of ALL matches -- unlike
    # Milestone 23's sample-level rule, where a straddling match was
    # dropped from BOTH the training corpus's eligibility and effectively
    # wasted.
    assert set(training_match_ids).isdisjoint(validation_match_ids)
    assert set(training_match_ids) | set(validation_match_ids) == set(sample_match_ids)
    assert len(training_match_ids) == 16  # 20 - round(20 * 0.2)
    assert len(validation_match_ids) == 4
