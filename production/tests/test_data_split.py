"""Milestone 35 / ADR-011: unit tests for `match_level_split`."""

from production.src.pipeline.data_split import match_level_split


def test_no_match_straddles_both_splits():
    """The core property this function exists to guarantee: every sample
    from a given match_id lands entirely in ONE group."""
    match_ids_per_sample = (
        [1] * 10 + [2] * 5 + [3] * 20 + [4] * 3 + [5] * 8 + [6] * 12 + [7] * 6 + [8] * 4 + [9] * 9 + [10] * 7
    )
    train_indices, val_indices = match_level_split(match_ids_per_sample, val_fraction=0.2, seed=42)

    train_matches = {match_ids_per_sample[i] for i in train_indices}
    val_matches = {match_ids_per_sample[i] for i in val_indices}

    assert train_matches.isdisjoint(val_matches), (
        f"a match straddled both splits: {train_matches & val_matches}"
    )
    assert set(train_indices).isdisjoint(val_indices)
    assert sorted(train_indices + val_indices) == list(range(len(match_ids_per_sample)))


def test_split_is_deterministic_given_seed():
    match_ids_per_sample = [i // 5 for i in range(200)]  # 40 matches, 5 samples each
    result_a = match_level_split(match_ids_per_sample, val_fraction=0.2, seed=42)
    result_b = match_level_split(match_ids_per_sample, val_fraction=0.2, seed=42)
    assert result_a == result_b


def test_different_seeds_can_produce_different_splits():
    match_ids_per_sample = [i // 5 for i in range(200)]
    train_42, val_42 = match_level_split(match_ids_per_sample, val_fraction=0.2, seed=42)
    train_7, val_7 = match_level_split(match_ids_per_sample, val_fraction=0.2, seed=7)
    assert (train_42, val_42) != (train_7, val_7)


def test_match_count_ratio_approximates_val_fraction():
    """20 equal-sized matches, val_fraction=0.2 -> should select ~4
    matches for validation (exact, since this is evenly divisible)."""
    match_ids_per_sample = [i // 10 for i in range(200)]  # 20 matches, 10 samples each
    train_indices, val_indices = match_level_split(match_ids_per_sample, val_fraction=0.2, seed=42)

    val_matches = {match_ids_per_sample[i] for i in val_indices}
    assert len(val_matches) == 4  # round(20 * 0.2)
    assert len(val_indices) == 40  # 4 matches * 10 samples, exact here since all matches are equal-sized


def test_uneven_match_sizes_produce_a_sample_ratio_that_need_not_match_val_fraction():
    """Directly demonstrates the documented expectation: when matches
    contribute very different sample counts, the resulting SAMPLE ratio
    can diverge meaningfully from val_fraction, even though the MATCH
    ratio is honored."""
    # 10 matches: one with 100 samples, nine with 5 samples each (145 total).
    match_ids_per_sample = [0] * 100 + sum(([m] * 5 for m in range(1, 10)), [])
    train_indices, val_indices = match_level_split(match_ids_per_sample, val_fraction=0.2, seed=1)

    val_matches = {match_ids_per_sample[i] for i in val_indices}
    assert len(val_matches) == 2  # round(10 * 0.2)
    # If match 0 (the 100-sample match) landed in val, the sample ratio is
    # wildly different from 20%; if it landed in train, val is tiny. Either
    # way, val's sample share should visibly NOT be a clean ~20%.
    sample_val_fraction = len(val_indices) / len(match_ids_per_sample)
    assert sample_val_fraction != 0.2


def test_single_match_dominating_validation_triggers_imbalance_warning(capsys):
    # 10 matches: one 100-sample match, nine tiny 2-sample matches. With
    # val_fraction=0.2 (2 of 10 matches), whichever 2 matches land in val,
    # if the dominant match is among them, it will visibly exceed the 30%
    # single-match share threshold within its own group.
    match_ids_per_sample = [0] * 100 + sum(([m] * 2 for m in range(1, 10)), [])
    # Seed chosen (by trial) so the dominant match (id=0) lands in the val
    # group, actually triggering the warning this test asserts on.
    found_triggering_seed = False
    for seed in range(50):
        capsys.readouterr()
        train_indices, val_indices = match_level_split(match_ids_per_sample, val_fraction=0.2, seed=seed)
        val_matches = {match_ids_per_sample[i] for i in val_indices}
        if 0 in val_matches:
            found_triggering_seed = True
            captured = capsys.readouterr()
            assert "WARNING" in captured.out
            assert "match_id=0" in captured.out
            break
    assert found_triggering_seed, "expected at least one seed in range(50) to put the dominant match in val"


def test_balanced_split_prints_no_imbalance_warning(capsys):
    match_ids_per_sample = [i // 10 for i in range(200)]  # 20 equal-sized matches
    capsys.readouterr()
    match_level_split(match_ids_per_sample, val_fraction=0.2, seed=42)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out
