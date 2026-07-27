"""Milestone 35 (methodological fix): match-level train/validation
splitting. See ADR-011 for the full history.

This project's train/validation split has been at the SAMPLE level since
Milestone 7 -- fine while no feature depended on cross-sample information.
Milestone 23 (Bayesian habit memory) showed this becomes a real problem
the moment a feature genuinely does: because possession chains from the
same match land all over both a random sample-level split, only 4 of ~55
matches ended up having ZERO validation-split samples (the training-bucket
corpus's conservative eligibility rule), starving the historical corpus
that feature actually needed.

`match_level_split` fixes this at the source: it partitions MATCH IDs
into train/val groups, then assigns every sample from a given match to
whichever group that match landed in. No match can ever straddle both
splits again, by construction -- not by a downstream exclusion rule
working around the fact that it still could.
"""

import numpy as np

# A single match contributing more than this fraction of its OWN group's
# total samples is flagged explicitly as a real, reportable imbalance
# (e.g. one unusually chain-heavy match dominating a small validation
# set) rather than silently accepted. This is a judgment call, not a
# validated statistical threshold -- same status as every other
# hand-tuned constant in this project (Kalman Q/R, pitch-control radii,
# CV thresholds) until checked against more data.
DOMINANT_MATCH_SAMPLE_SHARE_WARNING_THRESHOLD = 0.30


def match_level_split(
    match_ids_per_sample: list[int],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Splits sample indices into `(train_indices, val_indices)` at the
    MATCH level, not the sample level: every sample from a given
    `match_id` is assigned to whichever group that match_id landed in, so
    no match ever contributes samples to both splits.

    `match_ids_per_sample[i]` is the match_id that produced sample `i` --
    the same per-sample match tracking `train.py`'s `build_training_data`
    (Milestone 23) already builds as `all_sample_match_ids`, consumed here
    directly rather than reconstructed.

    The MATCH-count split is what `val_fraction` actually controls.
    Because matches contribute different numbers of samples (some
    possession-chain-heavy, some light), the resulting SAMPLE-count ratio
    will generally NOT equal `val_fraction` exactly -- this is expected
    and is reported explicitly below, not forced to a round number by any
    per-sample rebalancing (which would silently reintroduce sample-level
    mixing and defeat the entire point of this function).

    Uses a `numpy.random.Generator` seeded independently from any
    torch/global RNG state -- deterministic given `seed`, but not the same
    stream `torch.utils.data.random_split` would have produced (that
    function is not used at all here, unlike the sample-level splitting
    this replaces).

    Returns `(train_indices, val_indices)`, each a sorted list of sample
    indices.
    """
    unique_match_ids = sorted(set(match_ids_per_sample))
    num_matches = len(unique_match_ids)

    rng = np.random.default_rng(seed)
    shuffled_match_ids = rng.permutation(unique_match_ids)

    if num_matches <= 1:
        # Can't meaningfully split a single match across both groups
        # without violating match-level exclusivity -- the entire match
        # goes to train, val is empty. A real production dataset has
        # dozens of matches; this is a defensive edge case, not an
        # expected path.
        n_val_matches = 0
    else:
        n_val_matches = max(1, round(num_matches * val_fraction))
        n_val_matches = min(n_val_matches, num_matches - 1)  # keep train non-empty too

    val_match_ids = set(shuffled_match_ids[:n_val_matches].tolist())
    train_match_ids = set(shuffled_match_ids[n_val_matches:].tolist())

    train_indices = [i for i, m in enumerate(match_ids_per_sample) if m in train_match_ids]
    val_indices = [i for i, m in enumerate(match_ids_per_sample) if m in val_match_ids]

    n_train, n_val = len(train_indices), len(val_indices)
    total = n_train + n_val
    print(
        f"[match_level_split] {len(train_match_ids)} train matches / {len(val_match_ids)} val "
        f"matches (target val_fraction={val_fraction:.0%} of matches) -> {n_train} train samples "
        f"/ {n_val} val samples ({n_train / total:.1%} / {n_val / total:.1%} sample ratio -- NOT "
        "forced to match val_fraction exactly, since matches contribute different sample counts)."
    )

    _report_dominant_match_imbalance("train", train_indices, match_ids_per_sample)
    _report_dominant_match_imbalance("val", val_indices, match_ids_per_sample)

    return sorted(train_indices), sorted(val_indices)


def _report_dominant_match_imbalance(
    group_name: str, group_indices: list[int], match_ids_per_sample: list[int]
) -> None:
    """Prints an explicit WARNING if any single match contributes more
    than `DOMINANT_MATCH_SAMPLE_SHARE_WARNING_THRESHOLD` of `group_name`'s
    total samples -- e.g. one unusually chain-heavy match dominating a
    small validation set. Silent (no print at all) if the group is
    balanced or empty.
    """
    if not group_indices:
        return

    counts: dict[int, int] = {}
    for i in group_indices:
        match_id = match_ids_per_sample[i]
        counts[match_id] = counts.get(match_id, 0) + 1

    total = len(group_indices)
    dominant_match_id, dominant_count = max(counts.items(), key=lambda kv: kv[1])
    dominant_share = dominant_count / total

    if dominant_share > DOMINANT_MATCH_SAMPLE_SHARE_WARNING_THRESHOLD:
        print(
            f"[match_level_split] WARNING: match_id={dominant_match_id} alone contributes "
            f"{dominant_count}/{total} ({dominant_share:.1%}) of the {group_name} group's "
            f"samples -- exceeds the {DOMINANT_MATCH_SAMPLE_SHARE_WARNING_THRESHOLD:.0%} "
            "single-match imbalance threshold. This is a real, reportable skew (a chain-heavy "
            "match dominating this group), not silently accepted."
        )
