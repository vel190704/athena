"""StatsBomb open-data 360 fetcher and freeze-frame parser (Module 3 ingestion).

Pulls events and 360 freeze-frame JSON from the public statsbomb/open-data
GitHub repo, caching each fetched JSON to data/raw/ (gitignored) keyed by
match_id so repeated runs/tests read from disk instead of re-hitting GitHub
and risking rate-limiting.
"""

import json
import time
from pathlib import Path

import requests
import torch
from tqdm import tqdm

EVENTS_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
)
THREE_SIXTY_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data/three-sixty/{match_id}.json"
)
MATCHES_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/"
    "{competition_id}/{season_id}.json"
)
COMPETITIONS_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
)

CACHE_DIR = Path("data/raw")

# StatsBomb uses a fixed 120x80 unit coordinate grid for every match
# regardless of the real stadium's pitch dimensions. Rescaling into the
# 100x68 meter-space grid used by BiomechanicalPitchControl is therefore an
# approximation (real pitches vary in size, typically ~100-110m x 64-75m),
# not an exact meter conversion.
X_SCALE = 100.0 / 120.0
Y_SCALE = 68.0 / 80.0

# Applied after any real network fetch (never after a cache hit) to reduce
# the risk of GitHub rate-limiting across a run.
REQUEST_DELAY_SECONDS = 0.5


def _cache_path(match_id: int, kind: str) -> Path:
    return CACHE_DIR / f"{match_id}_{kind}.json"


def _fetch_json(url: str, cache_path: Path):
    """GET `url` as JSON, transparently caching to/reading from `cache_path`.

    Returns the parsed JSON, or None if the endpoint returned a non-200
    status (e.g. 404 for a match_id with no data at this endpoint).
    """
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    response = requests.get(url, timeout=30)
    time.sleep(REQUEST_DELAY_SECONDS)
    if response.status_code != 200:
        return None

    data = response.json()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def fetch_match_events(match_id: int):
    """Fetch (and cache) the events JSON for `match_id`, or None if absent."""
    url = EVENTS_URL_TEMPLATE.format(match_id=match_id)
    return _fetch_json(url, _cache_path(match_id, "events"))


def fetch_match_360(match_id: int):
    """Fetch (and cache) the 360 freeze-frame JSON for `match_id`, or None if absent."""
    url = THREE_SIXTY_URL_TEMPLATE.format(match_id=match_id)
    return _fetch_json(url, _cache_path(match_id, "360"))


def _matches_cache_path(competition_id: int, season_id: int) -> Path:
    return CACHE_DIR / f"matches_{competition_id}_{season_id}.json"


def fetch_competition_matches(competition_id: int, season_id: int) -> list[dict]:
    """Fetch (and cache) the full list of match objects for a competition
    and season.

    Each element of the returned list is a full match OBJECT (with fields
    like `match_id`, `home_team`, `away_team`, ...) -- this endpoint does
    NOT return a bare list of match_id integers, which was verified against
    a live fetch before writing this function.
    """
    url = MATCHES_URL_TEMPLATE.format(competition_id=competition_id, season_id=season_id)
    matches = _fetch_json(url, _matches_cache_path(competition_id, season_id))
    if matches is None:
        raise ValueError(
            f"no match list found for competition_id={competition_id}, season_id={season_id}"
        )
    return matches


def _competitions_cache_path() -> Path:
    return CACHE_DIR / "competitions.json"


def fetch_competitions_index() -> list[dict]:
    """Fetch (and cache) StatsBomb's full competitions index."""
    data = _fetch_json(COMPETITIONS_URL, _competitions_cache_path())
    if data is None:
        raise ValueError("could not fetch the StatsBomb competitions index")
    return data


def find_360_competitions() -> list[dict]:
    """Competitions/seasons verified to have 360 freeze-frame data
    available, via the `match_available_360` field on the LIVE competitions
    index -- NOT a hardcoded list from memory (a competition "sounding like"
    it should have 360 data, e.g. an older World Cup or Euros, may or may
    not actually have public 360 coverage; this is the same lesson as
    Milestone 3's match-id verification and Milestone 9's competition-
    matches endpoint shape -- verify against the live source, don't assume).

    Returns a list of dicts with competition_id, season_id, competition_name,
    and season_name for each qualifying (competition, season) pair.
    """
    index = fetch_competitions_index()
    return [
        {
            "competition_id": c["competition_id"],
            "season_id": c["season_id"],
            "competition_name": c["competition_name"],
            "season_name": c["season_name"],
        }
        for c in index
        if c.get("match_available_360")
    ]


def batch_extract_valid_matches(
    competition_season_pairs: list[tuple[int, int]], num_matches: int
) -> list[int]:
    """Return up to `num_matches` match_ids with BOTH events and 360 data
    present, drawn from ACROSS the given (competition_id, season_id) pairs
    in the order given -- not just a single competition -- reusing the same
    validity check, disk cache, tqdm progress, and rate-limit handling as
    the original single-competition version.

    Stops as soon as `num_matches` valid matches are found in total, rather
    than exhaustively checking every match in every competition. If all
    given competitions combined have fewer than `num_matches` valid
    matches, returns however many were found and prints a warning instead
    of raising.
    """
    valid_ids: list[int] = []

    for competition_id, season_id in competition_season_pairs:
        if len(valid_ids) >= num_matches:
            break

        matches = fetch_competition_matches(competition_id, season_id)
        candidate_match_ids = [m["match_id"] for m in matches]

        progress = tqdm(
            candidate_match_ids,
            desc=f"Verifying match validity (competition_id={competition_id}, season_id={season_id})",
        )
        for match_id in progress:
            if len(valid_ids) >= num_matches:
                break

            events = fetch_match_events(match_id)
            if events is None:
                continue
            frames = fetch_match_360(match_id)
            if frames is None:
                continue

            valid_ids.append(match_id)
            progress.set_postfix(valid=len(valid_ids))

        progress.close()

    if len(valid_ids) < num_matches:
        print(
            f"WARNING: only found {len(valid_ids)} valid matches (events + 360 data both "
            f"present) across {len(competition_season_pairs)} competition/season pairs; "
            f"requested {num_matches}."
        )

    return valid_ids


def find_valid_match_id(candidate_ids: list[int]) -> int:
    """Return the first candidate match_id with valid events AND 360 data.

    Tries each candidate in order rather than gating the pipeline on a
    single unverified hardcoded match_id.
    """
    for match_id in candidate_ids:
        events = fetch_match_events(match_id)
        if events is None:
            continue
        frames = fetch_match_360(match_id)
        if frames is None:
            continue
        return match_id
    raise ValueError(f"No valid match_id found among candidates: {candidate_ids}")


def parse_360_frame(event_data: dict, frame_data: dict) -> dict:
    """Parse one event + its 360 freeze-frame into physics-engine tensors.

    ball_pos is taken from the event's own `location` field: for on-ball
    events (Pass, Shot, ...) StatsBomb records the event's location as the
    ball's position at that moment -- the 360 freeze-frame itself carries no
    separate ball coordinate.
    """
    ball_raw = event_data["location"]
    ball_pos = torch.tensor([ball_raw[0] * X_SCALE, ball_raw[1] * Y_SCALE], dtype=torch.float32)

    freeze_frame = frame_data["freeze_frame"]
    n_visible = len(freeze_frame)

    player_pos = torch.tensor(
        [[p["location"][0] * X_SCALE, p["location"][1] * Y_SCALE] for p in freeze_frame],
        dtype=torch.float32,
    ).reshape(n_visible, 2)  # do NOT zero-pad to 22; N varies per frame

    # StatsBomb 360 freeze-frames carry no velocity field. Zero velocity
    # isolates spatial geometry for this milestone; filling this in (e.g.
    # from tracking data or event-to-event finite differencing) is a known
    # v1 gap deferred to a later milestone.
    player_vel = torch.zeros((n_visible, 2), dtype=torch.float32)
    fatigue_mod = torch.ones(n_visible, dtype=torch.float32)

    # teammate=True means "shares a team with the event's acting player"
    # (the "attacking team" for an on-ball event like Pass/Shot). A single
    # 360 freeze-frame carries no absolute home/away team_id, so we use
    # attacking/defending (teammate/non-teammate) semantics throughout,
    # not absolute team identity.
    is_teammate = torch.tensor([bool(p["teammate"]) for p in freeze_frame], dtype=torch.bool)

    return {
        "ball_pos": ball_pos,
        "player_pos": player_pos,
        "player_vel": player_vel,
        "fatigue_mod": fatigue_mod,
        "is_teammate": is_teammate,
        "event_type": event_data["type"]["name"],
        "period": event_data["period"],
        # The acting player's team (same team `is_teammate` is relative to
        # above) -- Milestone 10's direction-normalization lookup keys on
        # team identity, not just period, since attacking direction is
        # inferred per team.
        "team": event_data["team"]["name"],
    }
