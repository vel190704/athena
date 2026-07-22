"""Milestone 5 validation: possession chain grouping and censoring
classification on real StatsBomb data.
"""

from collections import Counter

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


def test_possession_chains_on_real_match():
    events = fetch_match_events(MATCH_ID)
    chains = build_possession_chains(events)

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
