"""Milestone 5/9 validation: possession chain grouping and censoring
classification on real StatsBomb data, now covering both periods.
"""

from collections import Counter, defaultdict

from production.src.ingestion.statsbomb_io import fetch_match_events
from production.src.pipeline.chain_builder import build_possession_chains

MATCH_ID = 3857276  # cached real StatsBomb open-data match (Milestone 3)

# StatsBomb's minute/second fields have 1-second granularity (mandated by
# Step 1.4: use minute/second, not the timestamp string, for consistency).
# A possession chain that is entirely contained within a single second
# (e.g. a shot immediately blocked and saved, or an instant miscontrol that
# goes straight out of play) legitimately computes to duration_seconds ==
# 0 -- this is a resolution artifact of the mandated timestamp fields, not
# a possession-grouping bug. Real chains 3 and 16 in this match hit exactly
# this case (verified by inspecting their event lists directly), so the
# sanity check below asserts durations are never *negative* (which WOULD
# indicate a grouping bug, e.g. two possessions merged out of order) rather
# than strictly positive.
MAX_PLAUSIBLE_DURATION_SECONDS = 180

MIN_SCALE_RATIO_VS_PERIOD1_ONLY = 1.5


def test_possession_chains_on_real_match():
    events = fetch_match_events(MATCH_ID)
    chains = build_possession_chains(events)  # default periods=(1, 2)

    assert len(chains) > 0

    assert any(c["event_flag"] == 1 for c in chains)
    assert any(c["censor_reason"] in {"turnover", "out_of_bounds"} for c in chains)

    for c in chains:
        assert c["duration_seconds"] >= 0, f"negative duration in chain {c['chain_id']}: {c}"
        assert c["duration_seconds"] <= MAX_PLAUSIBLE_DURATION_SECONDS, (
            f"implausibly long chain {c['chain_id']} "
            f"({c['duration_seconds']}s) likely indicates a possession-grouping bug: {c}"
        )

    reason_counts = Counter(c["censor_reason"] for c in chains)
    print(f"\nChains per censor_reason (n={len(chains)}): {dict(reason_counts)}")
    print(f"'other' count: {reason_counts.get('other', 0)} ({reason_counts.get('other', 0) / len(chains):.1%})")

    print("\nFirst 5 chains:")
    for c in chains[:5]:
        print(c)


def test_both_periods_roughly_double_the_period1_only_count():
    events = fetch_match_events(MATCH_ID)

    period1_only_chains = build_possession_chains(events, periods=(1,))
    both_periods_chains = build_possession_chains(events, periods=(1, 2))

    print(
        f"\nperiod-1-only: {len(period1_only_chains)} chains; "
        f"both periods: {len(both_periods_chains)} chains"
    )
    assert len(both_periods_chains) > MIN_SCALE_RATIO_VS_PERIOD1_ONLY * len(period1_only_chains)


def test_half_end_classified_once_per_period():
    events = fetch_match_events(MATCH_ID)
    chains = build_possession_chains(events, periods=(1, 2))

    half_end_chains = [c for c in chains if c["censor_reason"] == "half_end"]
    print(f"\nhalf_end chains: {half_end_chains}")

    assert len(half_end_chains) >= 2, (
        f"expected at least one half_end chain per period (2 total), got "
        f"{len(half_end_chains)}: {half_end_chains}"
    )
    # Each period's half_end chain should belong to a distinct period --
    # not two chains from the same period both mislabeled half_end.
    assert len({c["period"] for c in half_end_chains}) >= 2


def test_no_chain_spans_the_half_time_boundary():
    """Direct regression check for the real-data finding that motivated
    (period, possession) grouping: StatsBomb's possession counter does NOT
    reset at half-time in this match, and possession id 85 has real events
    in BOTH period 1 (a trailing Pass + Half End) and period 2 (Half Start
    bookkeeping). Grouping by `possession` alone would merge them into one
    chain spanning the half-time boundary.
    """
    events = fetch_match_events(MATCH_ID)

    # Independently reconstruct the NAIVE (possession-alone) grouping
    # directly from raw events to prove the bug precondition is real in
    # this data, not hypothetical.
    naive_groups_by_possession = defaultdict(set)
    for e in events:
        if e["period"] in (1, 2):
            naive_groups_by_possession[e["possession"]].add(e["period"])
    boundary_spanning_possession_ids = [
        pid for pid, periods in naive_groups_by_possession.items() if len(periods) > 1
    ]
    assert len(boundary_spanning_possession_ids) > 0, (
        "expected at least one possession id with events in both periods in this match "
        "(the real scenario (period, possession) grouping guards against)"
    )
    assert 85 in boundary_spanning_possession_ids

    # Now confirm build_possession_chains' actual output never merges
    # across that boundary: every returned chain has exactly one period
    # (structurally guaranteed by the schema), no (chain_id, period) pair
    # repeats, and the specific boundary-adjacent chain (id=85) has a
    # small, sane duration -- not one inflated by a cross-period merge.
    chains = build_possession_chains(events, periods=(1, 2))

    identity_keys = [(c["chain_id"], c["period"]) for c in chains]
    assert len(identity_keys) == len(set(identity_keys))

    boundary_chain = next(c for c in chains if c["chain_id"] == 85)
    print(f"\nboundary-adjacent chain (chain_id=85): {boundary_chain}")
    assert boundary_chain["period"] == 1
    assert boundary_chain["duration_seconds"] < MAX_PLAUSIBLE_DURATION_SECONDS
