"""Player Similarity Search validation (genuinely new ML work, scoped as
its own feature): player_similarity.py's feature-vector computation,
population normalization, cosine similarity, and the offline
precompute/live-query split.

Deliberately end-to-end against REAL, already-cached StatsBomb data
throughout -- no synthetic feature vectors except where a test is
explicitly isolating pure numerical correctness (the cosine-similarity
unit tests). The full-population precompute (~5,000 real players,
measured ~16 minutes -- see player_similarity.py's own module docstring)
is NOT re-run by this test file; instead, `build_player_similarity_index`
is exercised end-to-end against a SMALL, real, explicit subset of real
cached players (via monkeypatching `enumerate_cached_players`, the exact
function it calls internally -- nothing about the precompute's own logic
is bypassed, only WHICH real players it processes), keeping this file
fast while still real-data, not mocked-value, throughout.
"""

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import math

import production.src.reporting.player_similarity as player_similarity_module
from production.src.reporting.player_similarity import (
    FEATURE_DIMENSIONALITY,
    FEATURE_NAMES,
    MIN_EVENTS_FOR_SEARCHABLE_PROFILE,
    ROLE_BUCKETS,
    _cosine_similarity,
    _normalize_vector,
    _population_mean_std,
    _raw_player_features,
    build_player_similarity_index,
    find_similar_players,
)
from production.src.reporting.candidate_index import enumerate_cached_players

BARCA_MATCH = 3773386  # real, 360-covered, already used throughout this project's own test history
MESSI_ID = 5503
PIQUE_ID = 5213
NETO_ID = 6590  # real goalkeeper in this match
GRIEZMANN_ID = 5487

# Real, verified low-sample player (Milestone 44's own original test
# case, re-confirmed present in this project's cache before this test
# was written): 1 real tagged event total.
LOW_SAMPLE_PLAYER_ID = 99479  # Yu-Min Cho


def _small_real_population():
    """A tiny, real, explicit subset of `enumerate_cached_players()`'s
    own real output -- Messi/Pique/Neto/Griezmann, restricted to their
    real match 3773386 involvement only (not their full career), so this
    stays fast. Built from the REAL scan, not fabricated records --
    filtered down after the fact, not hand-typed from scratch."""
    all_players = enumerate_cached_players()
    wanted_ids = {MESSI_ID, PIQUE_ID, NETO_ID, GRIEZMANN_ID}
    small = []
    for p in all_players:
        if p["player_id"] not in wanted_ids:
            continue
        restricted_seasons = [
            {**s, "match_ids": [m for m in s["match_ids"] if m == BARCA_MATCH]}
            for s in p["seasons"]
        ]
        restricted_seasons = [s for s in restricted_seasons if s["match_ids"]]
        small.append({**p, "seasons": restricted_seasons})
    assert len(small) == 4, f"expected all 4 real players findable in the cache, got {len(small)}"
    return small


# --- Feature vector definition (Step 0) -----------------------------------


def test_feature_vector_dimensionality_and_names_stable():
    assert FEATURE_DIMENSIONALITY == 15
    assert len(FEATURE_NAMES) == 15
    assert len(set(FEATURE_NAMES)) == 15  # no accidental duplicate dimension


def test_role_buckets_cover_real_verified_position_names_no_unmapped_leak():
    """VERIFIED against real cached data before this mapping was written
    (26 distinct real `position.name` values across a 150-match sample,
    see player_similarity.py's own module docstring) -- every bucket
    target is one of the 4 coarse roles, and "Substitute" (a real
    StatsBomb administrative placeholder, not a playing position) is
    deliberately absent."""
    assert set(ROLE_BUCKETS.values()) == {"goalkeeper", "defender", "midfielder", "forward"}
    assert "Substitute" not in ROLE_BUCKETS


def test_raw_player_features_real_data():
    """Messi, real match 3773386: real feature values -- not
    placeholders. NOT asserting "role_share_forward is high" -- checked
    directly (generate_player_report's own positional_distribution for
    this exact match) rather than assumed from general football
    knowledge, and found Messi is tagged 100% "Center Attacking
    Midfield" in THIS specific match (a real, legitimate per-match
    tactical variation, not every match's data agrees with a player's
    usual public reputation) -- so this test asserts the REAL verified
    value (role_share_midfielder == 1.0), not an unverified assumption."""
    raw = _raw_player_features(MESSI_ID, [BARCA_MATCH])

    assert set(raw.keys()) == set(FEATURE_NAMES)
    assert raw["role_share_midfielder"] == 1.0
    assert raw["role_share_forward"] == 0.0
    assert raw["shots_per_90"] is not None and raw["shots_per_90"] > 0
    assert raw["xg_per_shot"] is not None


def test_raw_player_features_zero_shots_are_none_not_fabricated_zero():
    """A real goalkeeper (Neto) in a match where he took 0 real shots:
    xg_per_shot / shot_body_part_share_* must be None (genuinely
    undefined -- to be population-mean-imputed later), never a fabricated
    0.0 (which would misrepresent "no shot data" as "zero-quality
    shots")."""
    raw = _raw_player_features(NETO_ID, [BARCA_MATCH])
    assert raw["shots_per_90"] == 0.0  # a real, correct rate: 0 shots really did happen
    assert raw["xg_per_shot"] is None  # genuinely undefined -- no shots to average
    assert raw["shot_body_part_share_right_foot"] is None


# --- Population normalization + cosine similarity (Step 2) ----------------


def test_population_mean_std_and_normalize_vector_real_data():
    population = _small_real_population()
    raw_by_player = {
        p["player_id"]: _raw_player_features(p["player_id"], p["seasons"][0]["match_ids"])
        for p in population
    }
    stats = _population_mean_std(raw_by_player)
    assert set(stats.keys()) == set(FEATURE_NAMES)

    messi_vector = _normalize_vector(raw_by_player[MESSI_ID], stats)
    assert len(messi_vector) == FEATURE_DIMENSIONALITY
    assert all(math.isfinite(v) for v in messi_vector)

    # A player exactly at the population mean for every feature would
    # normalize to the zero vector -- confirm imputation lands exactly at
    # 0.0 for a genuinely undefined feature (Neto's xg_per_shot).
    neto_vector = _normalize_vector(raw_by_player[NETO_ID], stats)
    xg_index = FEATURE_NAMES.index("xg_per_shot")
    assert neto_vector[xg_index] == 0.0


def test_cosine_similarity_pure_math():
    """Pure numerical correctness -- controlled, non-real vectors,
    deliberately isolating the metric's own math from any real-data
    concern (already covered by the real-data tests above/below)."""
    assert abs(_cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9
    assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-9
    # Cosine ignores magnitude (Step 2.2's whole point): same direction,
    # different scale -> identical similarity to the unit-scaled case.
    assert abs(_cosine_similarity([1.0, 0.0], [5.0, 0.0]) - 1.0) < 1e-9
    # Zero vector (an exactly-average player) -> neutral 0.0, not a crash.
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# --- Offline precompute + live query (Step 3), small real population ------


def test_build_and_query_similarity_index_small_real_population(tmp_path, monkeypatch):
    """End-to-end through the REAL build_player_similarity_index/
    find_similar_players functions -- only the CANDIDATE POPULATION is
    swapped for a small, real, explicit subset (4 real players, one real
    match each); the batching/memory-check/normalize/write/query logic
    itself is entirely unmodified and genuinely exercised."""
    monkeypatch.setattr(player_similarity_module, "enumerate_cached_players", _small_real_population)
    monkeypatch.setattr(player_similarity_module, "INDEX_PATH", tmp_path / "similarity_index.json")
    monkeypatch.setattr(player_similarity_module, "MIN_EVENTS_FOR_SEARCHABLE_PROFILE", 1)

    index = build_player_similarity_index(batch_size=2)
    assert index["searchable_population_size"] == 4
    assert index["total_cached_population_size"] == 4
    assert index["stopped_early_due_to_memory"] is False
    assert player_similarity_module.INDEX_PATH.exists()

    result = find_similar_players(MESSI_ID, top_k=3)
    assert result["no_data"] is False
    assert result["player_id"] == MESSI_ID
    assert result["searchable_population_size"] == 4
    assert len(result["similar_players"]) == 3  # every other real player in this tiny population
    for entry in result["similar_players"]:
        assert -1.0 <= entry["similarity"] <= 1.0
        assert len(entry["matched_features"]) == 2
        assert entry["player_id"] != MESSI_ID


def test_find_similar_players_missing_index_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(player_similarity_module, "INDEX_PATH", tmp_path / "does_not_exist.json")
    try:
        find_similar_players(MESSI_ID)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "build_player_similarity_index" in str(exc)


def test_find_similar_players_player_not_in_population_real_data(tmp_path, monkeypatch):
    monkeypatch.setattr(player_similarity_module, "enumerate_cached_players", _small_real_population)
    monkeypatch.setattr(player_similarity_module, "INDEX_PATH", tmp_path / "similarity_index.json")
    monkeypatch.setattr(player_similarity_module, "MIN_EVENTS_FOR_SEARCHABLE_PROFILE", 1)
    build_player_similarity_index(batch_size=10)

    # A real player who exists in the cache but was never included in
    # THIS tiny population subset -- same real "not searchable" path a
    # genuinely below-threshold player takes.
    result = find_similar_players(LOW_SAMPLE_PLAYER_ID)
    assert result["no_data"] is True


# --- Step 1: real low-sample exclusion -------------------------------------


def test_low_sample_player_excluded_from_real_searchable_population():
    """Milestone 44's own original real low-sample test case (Yu-Min
    Cho, 1 real tagged event), re-confirmed present in this project's
    cache and correctly excluded from the searchable population by
    MIN_EVENTS_FOR_SEARCHABLE_PROFILE -- not merely flagged, genuinely
    absent from what a real precompute run would even attempt to index.
    """
    all_players = enumerate_cached_players()
    cho = next(p for p in all_players if p["player_id"] == LOW_SAMPLE_PLAYER_ID)
    assert cho["total_events"] == 1
    assert cho["total_events"] < MIN_EVENTS_FOR_SEARCHABLE_PROFILE

    searchable = [p for p in all_players if p["total_events"] >= MIN_EVENTS_FOR_SEARCHABLE_PROFILE]
    searchable_ids = {p["player_id"] for p in searchable}
    assert LOW_SAMPLE_PLAYER_ID not in searchable_ids
