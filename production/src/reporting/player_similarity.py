"""Player Similarity Search (genuinely new ML work, scoped as its own
feature -- not folded into any of tonight's other extension-style
reporting features).

Given a player, finds the top-K most similar OTHER real cached players
by cosine similarity over a fixed, documented, 15-dimensional feature
vector built ENTIRELY from ALREADY-COMPUTED scalar aggregates
`player_report.py`'s own existing functions already produce (
`generate_player_report`, `generate_player_press_resistance_index`,
`generate_player_shot_map`) -- no new event parsing, no new data source,
no fine-grained spatial data (the heatmap grid is deliberately excluded
-- see FEATURE_NAMES's own comment below).

Mirrors `candidate_index.py`'s OWN precompute-once/query-cheap pattern
directly: a slow, explicit, MANUALLY-triggered offline batch precompute
(`build_player_similarity_index`) writes a JSON index to
`data/app_state/player_similarity_index.json` (locally-generated
application state, NOT `data/raw/` -- the SAME boundary `alert_store.py`
already established, for the same reason: `data/raw/` is documented
throughout this project as a cache of EXTERNAL data, not somewhere this
project's own derived/computed artifacts belong). The LIVE query path
(`find_similar_players`) only ever reads that already-built index --
never recomputes the population, and there is no automatic TTL/staleness
check that could trigger an expensive rebuild mid-session unexpectedly
(same discipline `candidate_index.py`'s own "Refresh cache list" button
already established).
"""

import json
import logging
import time
from pathlib import Path

import psutil

from production.src.pipeline.habit_memory import MIN_HISTORICAL_EVENTS
from production.src.reporting.candidate_index import enumerate_cached_players
from production.src.reporting.player_report import (
    generate_player_press_resistance_index,
    generate_player_report,
    generate_player_shot_map,
)

logger = logging.getLogger(__name__)

INDEX_DIR = Path("data/app_state")
INDEX_PATH = INDEX_DIR / "player_similarity_index.json"

# Step 1: reuses candidate_index.py's OWN population-level low-sample gate
# directly (the exact same threshold enumerate_cached_players()'s own
# `low_sample` field already applies: total_events <
# LOW_SAMPLE_EVENT_THRESHOLD == habit_memory.MIN_HISTORICAL_EVENTS) rather
# than re-deriving a separate, similar-but-different threshold. A player
# below this bar is EXCLUDED from the searchable population entirely --
# not merely flagged -- since there is no meaningful profile to search
# FOR or match AGAINST from a 1-event "profile."
MIN_EVENTS_FOR_SEARCHABLE_PROFILE = MIN_HISTORICAL_EVENTS  # candidate_index.LOW_SAMPLE_EVENT_THRESHOLD

# Step 3: explicit batch size + a real memory-safety floor, given
# tonight's own repeated real OOM history. VERIFIED directly (not
# assumed) before picking these: a 100-real-player random sample (see
# this feature's own task report for the full numbers) measured
# ~193ms/player and, critically, showed `psutil.virtual_memory().available`
# essentially FLAT before vs. after -- this workload is pure Python
# dict/JSON processing over already-cached files, never touches
# PyTorch/MLflow/YOLO (the actual residency drivers behind tonight's real
# OOM incident). The real risk here measured much lower than that
# incident's, but batching + an explicit floor check is still applied
# regardless, not skipped just because one measurement looked safe.
PRECOMPUTE_BATCH_SIZE = 500
MIN_AVAILABLE_MEMORY_GB_TO_CONTINUE = 1.0

# ============================================================================
# Step 0: THE exact, documented, ORDER-STABLE 15-dimension feature vector.
# Every dimension is an ALREADY-COMPUTED scalar aggregate reached via
# player_report.py's own existing functions -- this module's own
# per-player code never reads a raw StatsBomb event directly.
#
# INCLUDED, with justification per dimension:
#   0-3  role_share_{goalkeeper,defender,midfielder,forward} -- coarse
#        positional-role shares (sum to 1.0), derived from
#        generate_player_report's `positional_distribution` (StatsBomb's
#        own real, granular `position.name` -- 26 distinct real values
#        VERIFIED across a 150-match real sample) via an explicit,
#        checked mapping into 4 coarse buckets (ROLE_BUCKETS below) --
#        the granular taxonomy itself is too high-cardinality to use as
#        fixed-length numeric features directly.
#   4-7  press_resistance_{overall,pass,dribble,shot}_rate -- reused
#        DIRECTLY from generate_player_press_resistance_index's own
#        `overall`/`event_types.{pass,dribble,shot}` success rates.
#   8    shots_per_90 -- total_shots (generate_player_shot_map) normalized
#        by total_minutes_played/90 (generate_player_report) -- NOT a raw
#        count, so a lower-minutes player isn't penalized/boosted just
#        for playing less.
#   9    goals_per_90 -- same normalization, using the shot map's own
#        `goals`. A genuinely relevant addition beyond the roadmap's own
#        explicit list: shots_per_90 alone conflates a high-volume,
#        low-conversion player with a clinical, low-volume finisher;
#        goals_per_90 gives the vector a second, complementary
#        attacking-OUTPUT signal, not just volume.
#   10   xg_per_shot -- reused directly from generate_player_shot_map
#        (StatsBomb's own real per-shot xG, averaged -- see that
#        function's own docstring for why this is NOT this project's
#        DeepHit model's output).
#   11-14 shot_body_part_share_{right_foot,left_foot,head,other} -- the
#        shot map's own VERIFIED, already-fixed 4-value real
#        `body_part.name` set (see generate_player_shot_map's own
#        docstring) -- no new bucket-mapping judgment call needed here,
#        unlike positional role, since this categorical field is already
#        small and fixed.
#
# DELIBERATELY EXCLUDED, with reasons:
#   - The full heatmap_grid (70 cells) -- too high-dimensional for a
#     15-feature "style" vector, and per this feature's own explicit
#     scoping: closer to the granular spatial-data class ADR-021 has been
#     careful about elsewhere (Pass Network's raw edges, the touch
#     map/shot map's raw scatter) than to the safe scalar-rate category
#     the rest of this vector is built from.
#   - Any RAW, unnormalized count (total_shots, positional_distribution_
#     event_count, under_pressure_attempts, ...) -- would bias similarity
#     toward players with more OVERALL minutes/involvement regardless of
#     their actual style, exactly this feature's own stated concern.
#   - total_minutes_played itself -- a proxy for playing TIME, not style;
#     used only as shots_per_90/goals_per_90's normalizer, never as a
#     feature in its own right (including it directly would reintroduce
#     the same activity-level bias per-90 normalization exists to remove).
#   - primary_formation / primary_position (categorical, not naturally
#     numeric) -- surfaced in the dashboard's own explanation text, not
#     as a similarity-search DIMENSION.
#   - A separate "goal conversion rate" (goals/shots) -- fully determined
#     by already including both shots_per_90 and goals_per_90; adding it
#     would be a 16th, non-independent dimension for no real benefit.
# ============================================================================
FEATURE_NAMES: tuple[str, ...] = (
    "role_share_goalkeeper",
    "role_share_defender",
    "role_share_midfielder",
    "role_share_forward",
    "press_resistance_overall_rate",
    "press_resistance_pass_rate",
    "press_resistance_dribble_rate",
    "press_resistance_shot_rate",
    "shots_per_90",
    "goals_per_90",
    "xg_per_shot",
    "shot_body_part_share_right_foot",
    "shot_body_part_share_left_foot",
    "shot_body_part_share_head",
    "shot_body_part_share_other",
)
FEATURE_DIMENSIONALITY = len(FEATURE_NAMES)  # 15

# Step 0's role-bucket mapping -- VERIFIED against real cached data (26
# distinct real `position.name` values checked across a 150-match sample)
# before writing this mapping, not assumed exhaustive. StatsBomb's real
# "Substitute" position tag (3 of ~490,000 real position-tagged events
# checked) is an administrative placeholder, not a real playing position
# -- deliberately left OUT of every bucket (skipped, not force-mapped).
ROLE_BUCKETS: dict[str, str] = {
    "Goalkeeper": "goalkeeper",
    "Left Center Back": "defender",
    "Right Center Back": "defender",
    "Center Back": "defender",
    "Left Back": "defender",
    "Right Back": "defender",
    "Left Wing Back": "defender",
    "Right Wing Back": "defender",
    "Left Center Midfield": "midfielder",
    "Right Center Midfield": "midfielder",
    "Center Defensive Midfield": "midfielder",
    "Left Defensive Midfield": "midfielder",
    "Right Defensive Midfield": "midfielder",
    "Center Attacking Midfield": "midfielder",
    "Left Attacking Midfield": "midfielder",
    "Right Attacking Midfield": "midfielder",
    "Left Midfield": "midfielder",
    "Right Midfield": "midfielder",
    "Center Midfield": "midfielder",
    "Center Forward": "forward",
    "Left Center Forward": "forward",
    "Right Center Forward": "forward",
    "Left Wing": "forward",
    "Right Wing": "forward",
    "Secondary Striker": "forward",
}
ROLE_BUCKET_NAMES = ("goalkeeper", "defender", "midfielder", "forward")

SHOT_BODY_PARTS = ("Right Foot", "Left Foot", "Head", "Other")

# Coarse group label per feature -- used only for the "which features
# drove this match" explanation (Step 5's explicit requirement: a brief
# note, not a bare score). Deliberately GROUPED (e.g. shots_per_90 and
# goals_per_90 both -> "shot volume") so the explanation reports one real
# signal once, not the same underlying tendency twice under two names.
_FEATURE_GROUP_LABELS: dict[str, str] = {
    "role_share_goalkeeper": "positional role",
    "role_share_defender": "positional role",
    "role_share_midfielder": "positional role",
    "role_share_forward": "positional role",
    "press_resistance_overall_rate": "press resistance",
    "press_resistance_pass_rate": "press resistance",
    "press_resistance_dribble_rate": "press resistance",
    "press_resistance_shot_rate": "press resistance",
    "shots_per_90": "shot volume",
    "goals_per_90": "shot volume",
    "xg_per_shot": "shot quality",
    "shot_body_part_share_right_foot": "shot technique",
    "shot_body_part_share_left_foot": "shot technique",
    "shot_body_part_share_head": "shot technique",
    "shot_body_part_share_other": "shot technique",
}


def _raw_player_features(player_id: int, match_ids: list[int]) -> dict[str, float | None]:
    """One player's own RAW (not yet population-normalized) feature
    values, keyed by FEATURE_NAMES -- reuses generate_player_report /
    generate_player_press_resistance_index / generate_player_shot_map
    UNCHANGED (calls them, does not reimplement any of their logic).

    A ratio/share feature that is genuinely undefined for this player
    (e.g. `xg_per_shot` with 0 real shots, or a per-90 rate with 0 known
    minutes) is `None` here -- imputed with the SEARCHABLE POPULATION's
    own mean for that feature later (see `_normalize_vector`), never
    silently defaulted to 0.0 (which would misrepresent "no shot data"
    as "zero-quality shots" or "zero shot rate").
    """
    report = generate_player_report(player_id, match_ids)
    pri = generate_player_press_resistance_index(player_id, match_ids)
    shot_map = generate_player_shot_map(player_id, match_ids)

    role_counts = {bucket: 0.0 for bucket in ROLE_BUCKET_NAMES}
    for position_name, share in report["positional_distribution"].items():
        bucket = ROLE_BUCKETS.get(position_name)
        if bucket is not None:
            role_counts[bucket] += share
    role_total = sum(role_counts.values())
    role_shares = (
        {bucket: value / role_total for bucket, value in role_counts.items()}
        if role_total > 0 else {bucket: None for bucket in ROLE_BUCKET_NAMES}
    )

    total_minutes = report["total_minutes_played"]
    total_shots = shot_map["total_shots"]
    goals = shot_map["goals"]
    minutes_90s = total_minutes / 90.0 if total_minutes > 0 else None
    shots_per_90 = (total_shots / minutes_90s) if minutes_90s is not None and minutes_90s > 0 else None
    goals_per_90 = (goals / minutes_90s) if minutes_90s is not None and minutes_90s > 0 else None

    body_part_shares = {part: None for part in SHOT_BODY_PARTS}
    if total_shots > 0:
        for part in SHOT_BODY_PARTS:
            body_part_shares[part] = shot_map["shots_by_body_part"].get(part, 0) / total_shots

    return {
        "role_share_goalkeeper": role_shares["goalkeeper"],
        "role_share_defender": role_shares["defender"],
        "role_share_midfielder": role_shares["midfielder"],
        "role_share_forward": role_shares["forward"],
        "press_resistance_overall_rate": pri["overall"]["success_rate"],
        "press_resistance_pass_rate": pri["event_types"]["pass"]["success_rate"],
        "press_resistance_dribble_rate": pri["event_types"]["dribble"]["success_rate"],
        "press_resistance_shot_rate": pri["event_types"]["shot"]["success_rate"],
        "shots_per_90": shots_per_90,
        "goals_per_90": goals_per_90,
        "xg_per_shot": shot_map["xg_per_shot"],
        "shot_body_part_share_right_foot": body_part_shares["Right Foot"],
        "shot_body_part_share_left_foot": body_part_shares["Left Foot"],
        "shot_body_part_share_head": body_part_shares["Head"],
        "shot_body_part_share_other": body_part_shares["Other"],
    }


def _population_mean_std(raw_features_by_player: dict[int, dict[str, float | None]]) -> dict[str, tuple[float, float]]:
    """Per-feature `(mean, std)` computed ACROSS the searchable population
    (Step 2.1: population-level, never per-player) -- stored explicitly so
    a NEW query player can be normalized consistently against the SAME
    baseline later, without recomputing the whole population's statistics.
    Only players with a DEFINED (non-None) value for a given feature
    contribute to that feature's own mean/std, so an already-imputed
    value never influences the very mean it would be imputed with.
    """
    stats: dict[str, tuple[float, float]] = {}
    for name in FEATURE_NAMES:
        values = [f[name] for f in raw_features_by_player.values() if f[name] is not None]
        if not values:
            stats[name] = (0.0, 1.0)  # no searchable player has this defined -- guarded, not crashed
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        stats[name] = (mean, std if std > 1e-9 else 1.0)  # guard a zero-variance feature against divide-by-zero
    return stats


def _normalize_vector(raw: dict[str, float | None], stats: dict[str, tuple[float, float]]) -> list[float]:
    """Mean-impute (undefined -> population mean, i.e. a z-score of 0
    after normalization) then z-score every feature, in FEATURE_NAMES'
    own fixed order -- the exact order every stored/query vector uses."""
    vector = []
    for name in FEATURE_NAMES:
        mean, std = stats[name]
        value = raw[name] if raw[name] is not None else mean
        vector.append((value - mean) / std)
    return vector


def build_player_similarity_index(batch_size: int = PRECOMPUTE_BATCH_SIZE) -> dict:
    """Step 3's OFFLINE precompute: scans every real cached player
    (`candidate_index.enumerate_cached_players` -- the SAME lightweight
    population enumeration the Player Reports tab's own dropdown already
    uses, not a new scan), excludes anyone below
    `MIN_EVENTS_FOR_SEARCHABLE_PROFILE` (Step 1 -- excluded from the
    population entirely, not merely flagged), computes each remaining
    player's RAW 15-feature vector in explicit BATCHES of `batch_size`,
    checking real available memory (`psutil`) BETWEEN batches -- never
    attempted as one giant in-memory pass -- then population-normalizes
    (Step 2) and writes the result to `INDEX_PATH`.

    MANUALLY triggered (Step 3.3): no automatic TTL/staleness check that
    could trigger an expensive rebuild mid-session unexpectedly --
    `dashboard.py`'s own Player Similarity panel gets an explicit
    "Rebuild similarity index" button, mirroring `candidate_index.py`'s
    own "Refresh cache list" precedent exactly, not an automatic one.
    """
    start = time.time()
    players = enumerate_cached_players()
    searchable_players = [p for p in players if p["total_events"] >= MIN_EVENTS_FOR_SEARCHABLE_PROFILE]
    logger.info(
        f"Player Similarity precompute: {len(searchable_players)} of {len(players)} cached players "
        f"clear MIN_EVENTS_FOR_SEARCHABLE_PROFILE={MIN_EVENTS_FOR_SEARCHABLE_PROFILE}."
    )

    raw_features_by_player: dict[int, dict[str, float | None]] = {}
    player_names: dict[int, str] = {}
    stopped_early = False

    for batch_start in range(0, len(searchable_players), batch_size):
        batch = searchable_players[batch_start:batch_start + batch_size]
        for p in batch:
            match_ids = sorted({mid for s in p["seasons"] for mid in s["match_ids"]})
            raw_features_by_player[p["player_id"]] = _raw_player_features(p["player_id"], match_ids)
            player_names[p["player_id"]] = p["name"]

        processed = batch_start + len(batch)
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        logger.info(
            f"Player Similarity precompute: batch {batch_start // batch_size + 1} done "
            f"({processed} of {len(searchable_players)} players), {available_gb:.2f}GB memory available."
        )
        if available_gb < MIN_AVAILABLE_MEMORY_GB_TO_CONTINUE:
            logger.warning(
                f"Player Similarity precompute: available memory ({available_gb:.2f}GB) fell below "
                f"MIN_AVAILABLE_MEMORY_GB_TO_CONTINUE={MIN_AVAILABLE_MEMORY_GB_TO_CONTINUE}GB -- "
                f"stopping early after {processed} of {len(searchable_players)} players rather than "
                "risking an OOM (tonight's own repeated real incident)."
            )
            stopped_early = True
            break

    stats = _population_mean_std(raw_features_by_player)
    vectors = {
        str(player_id): _normalize_vector(raw, stats)
        for player_id, raw in raw_features_by_player.items()
    }

    index = {
        "feature_names": list(FEATURE_NAMES),
        "normalization_stats": {name: list(stats[name]) for name in FEATURE_NAMES},
        "player_names": {str(pid): name for pid, name in player_names.items()},
        "raw_features": {str(pid): raw_features_by_player[pid] for pid in raw_features_by_player},
        "vectors": vectors,
        "searchable_population_size": len(vectors),
        "total_cached_population_size": len(players),
        "stopped_early_due_to_memory": stopped_early,
        "built_at_unix": time.time(),
        "build_duration_seconds": time.time() - start,
    }

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index))
    logger.info(
        f"Player Similarity precompute done: {len(vectors)} players indexed in "
        f"{index['build_duration_seconds']:.1f}s, written to {INDEX_PATH}."
    )
    return index


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Step 2.2's chosen metric -- see `find_similar_players`'s own
    docstring for the full justification (cosine over Euclidean:
    "similar TYPE of player" over "similar overall activity level")."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0  # an exactly-average player (every feature == population mean) has a zero vector; cosine is undefined there, 0.0 is a neutral fallback, not a fabricated score
    return dot / (norm_a * norm_b)


def _top_matching_features(query_raw: dict, other_raw: dict, stats: dict, n: int = 2) -> list[str]:
    """Which FEATURE GROUPS (not raw dimension names) most drove this
    match -- the `n` groups where the two players' own z-scored values
    land CLOSEST together, deduplicated by group label (e.g. "shot
    volume" covers both shots_per_90 and goals_per_90; reporting both
    separately would double-count one real signal). A brief,
    human-readable explanation, not a bare numeric score, per this
    feature's own explicit requirement (Step 5).
    """
    closeness_by_group: dict[str, float] = {}
    for name in FEATURE_NAMES:
        mean, std = stats[name]
        q = query_raw[name] if query_raw[name] is not None else mean
        o = other_raw[name] if other_raw[name] is not None else mean
        z_diff = abs((q - mean) / std - (o - mean) / std)
        group = _FEATURE_GROUP_LABELS[name]
        closeness_by_group[group] = min(closeness_by_group.get(group, float("inf")), z_diff)

    ranked = sorted(closeness_by_group.items(), key=lambda kv: kv[1])
    return [group for group, _diff in ranked[:n]]


def find_similar_players(player_id: int, top_k: int = 5) -> dict:
    """Step 3.2's LIVE query path: a nearest-neighbor lookup against the
    ALREADY-PRECOMPUTED, disk-cached index (`INDEX_PATH`) -- NEVER
    recomputes the population live. Raises `FileNotFoundError` with a
    clear, actionable message if the index hasn't been built yet (Step
    3.3: no auto-rebuild).

    SIMILARITY METRIC (Step 2.2, a real judgment call): COSINE similarity
    over the population-normalized 15-dimension vector, not Euclidean
    distance. Cosine ignores a vector's own MAGNITUDE and compares only
    its DIRECTION/shape across the 15 z-scored dimensions -- two players
    who deviate from the population average in the SAME relative pattern
    (e.g. both above-average on press resistance and shot volume, below
    on aerial share) score highly similar even if one deviates much more
    STRONGLY than the other (a genuine attacking talent vs. a milder
    version of the same style). Euclidean, even on already-z-scored
    features, would penalize that magnitude difference directly -- it
    answers "how far apart are these two profiles in absolute terms,"
    closer to "similar OVERALL intensity of involvement" than to "similar
    TYPE of player," which is this feature's actual intent (the roadmap's
    own framing: "two wingers with different overall activity levels but
    similar relative tendencies would still match"). Cosine is the
    better fit for that intent and is what this function uses.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Player Similarity index not found at {INDEX_PATH} -- run "
            "build_player_similarity_index() (or click 'Rebuild similarity index' "
            "in the dashboard) first."
        )
    index = json.loads(INDEX_PATH.read_text())

    query_key = str(player_id)
    if query_key not in index["vectors"]:
        return {
            "player_id": player_id,
            "no_data": True,
            "reason": (
                f"player_id={player_id} is not in the searchable population (either not cached, or "
                f"below MIN_EVENTS_FOR_SEARCHABLE_PROFILE={MIN_EVENTS_FOR_SEARCHABLE_PROFILE} real "
                "tagged events)."
            ),
        }

    query_vector = index["vectors"][query_key]
    query_raw = index["raw_features"][query_key]
    stats = {name: tuple(pair) for name, pair in index["normalization_stats"].items()}

    scored = [
        (_cosine_similarity(query_vector, other_vector), other_key)
        for other_key, other_vector in index["vectors"].items()
        if other_key != query_key
    ]
    scored.sort(key=lambda pair: -pair[0])

    similar_players = [
        {
            "player_id": int(other_key),
            "name": index["player_names"][other_key],
            "similarity": similarity,
            "matched_features": _top_matching_features(query_raw, index["raw_features"][other_key], stats),
        }
        for similarity, other_key in scored[:top_k]
    ]

    return {
        "player_id": player_id,
        "name": index["player_names"][query_key],
        "no_data": False,
        "top_k": top_k,
        "similar_players": similar_players,
        "searchable_population_size": index["searchable_population_size"],
        "index_built_at_unix": index["built_at_unix"],
    }
