"""Possession chain builder and censoring classifier (Milestone 5, expanded
to both periods in Milestone 9).

Groups StatsBomb events into possession chains using StatsBomb's OWN
`possession` field (NOT re-derived from event-type transitions), then
classifies how each chain terminates -- shot ("event") vs one of several
censored-termination reasons -- for later survival-analysis duration
labeling. This module only produces (duration, event_flag, censor_reason)
labels; it does not touch survival_dataset.py's tensor plumbing.

Period-boundary finding (Milestone 9, verified against match 3857276):
StatsBomb's `possession` counter does NOT reset at half-time -- it keeps
incrementing across periods, AND the exact possession value active at the
half-time whistle (85, in this match) is reused for BOTH the trailing
"Pass" + "Half End" events of period 1 and the "Half Start" events that
open period 2. Grouping by `possession` alone would silently merge these
into one chain spanning the half-time boundary. Grouping by the tuple
`(period, possession)` instead makes that structurally impossible, rather
than relying on a special-case check to catch it after the fact.

Real-data findings (verified against StatsBomb open-data match 3857276,
period 1) that shaped the rules below:

- A possession's terminal event is almost never the Shot itself: StatsBomb
  keeps a shot's immediate aftermath (Block, Goal Keeper save, Clearance,
  rebound) inside the SAME possession id. Checking "does this chain contain
  a Shot anywhere" is what actually finds shot outcomes; checking "is the
  terminal event a Shot" finds essentially none.
- 'Throw-in' / 'Goal Kick' / 'Corner' are not standalone StatsBomb event
  *types* at all -- restarts after the ball leaves the field are signaled by
  the *next* chain's `play_pattern` (e.g. "From Throw In"). This is a more
  reliable out-of-bounds signal than trying to match a literal event name.
- The single most common turnover-terminal case by far was an incomplete
  'Ball Receipt*' (ball_receipt.outcome.name == 'Incomplete') -- 24 of 85
  chains in the sample match -- which was not in any naive candidate list
  assumed up front. Confirms the instruction to verify against real data
  rather than assume a fixed type list is exhaustive.
"""

from collections import defaultdict

# Terminal event types (verified against real data) that, combined with a
# team change on the following possession, indicate an open-play turnover.
# 'Ball Receipt*' and 'Pass' are handled separately below since they
# additionally require checking a nested outcome field == 'Incomplete'
# (a *completed* pass/receipt is essentially never a chain's terminal
# event in practice, but the check guards the edge case explicitly rather
# than assuming it).
TURNOVER_TERMINAL_TYPES = {
    "Interception",
    "Ball Recovery",
    "Dispossessed",
    "Duel",
    "Clearance",
    "Miscontrol",
    "Block",
}

FOUL_TERMINAL_TYPES = {"Foul Committed", "Foul Won"}

# The play_pattern of the NEXT chain's first event is StatsBomb's own
# signal that play restarted after the ball left the field. Verified
# against real data: every chain that clearly ended with the ball going out
# (terminal `out: True` on a Clearance/Miscontrol/Block, or a Duel that
# resulted in a throw-in) was followed by a chain whose first event carried
# one of these play_pattern values.
OUT_OF_BOUNDS_NEXT_PLAY_PATTERNS = {"From Throw In", "From Goal Kick", "From Corner"}


def _chain_contains_shot(chain_events: list) -> bool:
    return any(e["type"]["name"] == "Shot" for e in chain_events)


def _is_incomplete_reception(event: dict) -> bool:
    """True for a terminal 'Pass' or 'Ball Receipt*' event whose nested
    outcome is 'Incomplete' -- i.e. the ball changed hands mid-flight.

    Verified against real data: this was the single most common terminal
    pattern overall (accounting for more chains than every other turnover
    type combined), and was not in any naive candidate list assumed up
    front.
    """
    type_name = event["type"]["name"]
    if type_name == "Ball Receipt*":
        return event.get("ball_receipt", {}).get("outcome", {}).get("name") == "Incomplete"
    if type_name == "Pass":
        return event.get("pass", {}).get("outcome", {}).get("name") == "Incomplete"
    return False


def _classify_chain(
    chain_events: list, next_chain_events: list | None, is_last_chain_of_period: bool
) -> tuple[int, str]:
    """Return (event_flag, censor_reason) for one possession chain.

    Priority order (first match wins):
      1. Chain contains a Shot anywhere -> event (shot).
      2. Terminal event is a foul -> foul.
      3. Last chain of the period AND terminal event is Half End -> half_end.
      4. Next chain's first event's play_pattern signals a dead-ball restart
         caused by the ball leaving the field -> out_of_bounds.
      5. Next chain exists, its possession_team differs from this chain's,
         AND the terminal event is a recognized turnover type -> turnover.
      6. Otherwise -> other (counted for manual inspection, never forced).
    """
    terminal = chain_events[-1]
    terminal_type = terminal["type"]["name"]

    if _chain_contains_shot(chain_events):
        return 1, "shot"

    if terminal_type in FOUL_TERMINAL_TYPES:
        return 0, "foul"

    if is_last_chain_of_period and terminal_type == "Half End":
        return 0, "half_end"

    if next_chain_events is not None:
        next_play_pattern = next_chain_events[0]["play_pattern"]["name"]
        if next_play_pattern in OUT_OF_BOUNDS_NEXT_PLAY_PATTERNS:
            return 0, "out_of_bounds"

        this_team = chain_events[0]["possession_team"]["name"]
        next_team = next_chain_events[0]["possession_team"]["name"]
        team_changed = next_team != this_team

        is_turnover_type = terminal_type in TURNOVER_TERMINAL_TYPES or _is_incomplete_reception(
            terminal
        )
        if team_changed and is_turnover_type:
            return 0, "turnover"

    return 0, "other"


def build_possession_chains(events: list, periods: tuple = (1, 2)) -> list[dict]:
    """Group StatsBomb events (across `periods`) into possession chains and
    classify how each one terminates.

    Uses StatsBomb's own `possession` field, combined with `period`, as the
    grouping key -- this function classifies HOW a chain ends, it does not
    re-derive WHEN possession changes. `periods` defaults to `(1, 2)` (both
    halves); pass `periods=(1,)` to restore the original period-1-only scope.

    Each period is processed as its own independent, self-contained
    sequence: a period's last chain always gets `next_chain_events=None`
    for classification purposes, so a chain can never be compared against
    the following period's first chain (which would otherwise risk
    misclassifying period 1's final chain as a turnover against period 2's
    team, or letting a period's own `half_end` chain get missed). This also
    means `half_end` is applied per period independently -- with both
    periods processed, expect roughly two `half_end` chains per match, one
    per half.
    """
    relevant_events = [e for e in events if e["period"] in periods]

    groups = defaultdict(list)
    for e in relevant_events:
        groups[(e["period"], e["possession"])].append(e)
    for chain_events in groups.values():
        chain_events.sort(key=lambda e: e["index"])

    # Pre-kickoff/pre-second-half administrative bookkeeping (Starting XI,
    # Half Start, Tactical Shift) is grouped under its own possession id but
    # carries no on-pitch location data -- it is not a real possession of
    # play, so it's excluded here.
    valid_keys = {key for key in groups if any("location" in e for e in groups[key])}

    other_count = 0
    chains = []
    for period in periods:
        period_keys = sorted(key for key in valid_keys if key[0] == period)

        for i, key in enumerate(period_keys):
            chain_events = groups[key]
            is_last = i == len(period_keys) - 1
            next_chain_events = groups[period_keys[i + 1]] if not is_last else None

            event_flag, censor_reason = _classify_chain(chain_events, next_chain_events, is_last)
            if censor_reason == "other":
                other_count += 1

            start = chain_events[0]
            end = chain_events[-1]
            start_total_seconds = start["minute"] * 60 + start["second"]
            end_total_seconds = end["minute"] * 60 + end["second"]

            chains.append(
                {
                    "chain_id": key[1],  # the raw StatsBomb possession value
                    "period": key[0],
                    "team": chain_events[0]["possession_team"]["name"],
                    "start_minute": float(start["minute"]),
                    "end_minute": float(end["minute"]),
                    "duration_seconds": float(end_total_seconds - start_total_seconds),
                    "event_flag": event_flag,
                    "censor_reason": censor_reason,
                }
            )

    if other_count > 0:
        print(f"[chain_builder] {other_count} of {len(chains)} chains classified as 'other'")

    return chains
