"""Time-dependent Brier Score evaluation for the single-risk DeepHit model
(Milestone 7).

Uses the SAME inclusive-cumsum survival convention as DeepHitLoss:
S(t) = 1 - cumsum(PMF up to and including bin t).
"""

import torch


def calculate_brier_score(
    predictions: torch.Tensor,
    durations: torch.Tensor,
    censor_or_event_bin: torch.Tensor,
    events: torch.Tensor,
    time_bin: int,
) -> tuple[float, int]:
    """Time-dependent Brier Score at a fixed `time_bin`.

    Returns (brier_score, num_excluded): the excluded (bucket 3) count is
    returned alongside the score, not just printed, because it materially
    affects how much to trust the score (e.g. for MLflow tracking -- see
    Milestone 8) and shouldn't only be recoverable by parsing console output.

    `censor_or_event_bin` is each sample's actual duration_bin regardless of
    event/censored status -- i.e. the same value as `durations` (both
    parameters are kept, per this project's single-risk usage, so the call
    site can name explicitly which duration field is driving the
    event-vs-censoring-time comparisons below; a future competing-risks
    extension could legitimately let them diverge, so the equality is
    asserted rather than assumed).

    Every sample falls into exactly one of three buckets by comparing its
    duration bin against `time_bin`:

      1. Failed by time_bin (events==1 AND duration<=time_bin): true status
         at time_bin is "already failed." Contributes S(t)^2 (a good
         prediction has LOW S(t) here).
      2. Survived past time_bin (duration > time_bin, for EITHER events==1
         [had the event later than time_bin] OR events==0 [censored after
         time_bin, so known to have survived at least that long]).
         Contributes (S(t) - 1)^2 (a good prediction has HIGH S(t) here).
      3. Excluded (events==0 AND duration<=time_bin): censored at or before
         time_bin -- true status at time_bin is genuinely unknown (we
         stopped observing before we'd know), so these are dropped from
         both the sum and the count entirely, never defaulted to a label.
    """
    assert torch.equal(durations, censor_or_event_bin), (
        "durations and censor_or_event_bin must be identical in this project's "
        "single-risk setting -- see calculate_brier_score's docstring"
    )

    survival = 1.0 - torch.cumsum(predictions, dim=1)  # same convention as DeepHitLoss
    time_bin_idx = torch.full((predictions.shape[0], 1), time_bin, dtype=torch.long)
    survival_at_t = survival.gather(1, time_bin_idx).squeeze(1)  # [batch]

    events_bool = events.bool()

    # Bucket 2: survived past time_bin. Event-flag-independent by
    # construction -- (events==1 & dur>t) union (events==0 & dur>t)
    # collapses to just "dur > t" regardless of events.
    survived_past_mask = censor_or_event_bin > time_bin

    # Bucket 1: failed by time_bin (must have an observed event).
    failed_by_t_mask = events_bool & (censor_or_event_bin <= time_bin)

    # Bucket 3: censored at/before time_bin -- excluded entirely, no label
    # assumed.
    excluded_mask = (~events_bool) & (censor_or_event_bin <= time_bin)

    num_excluded = int(excluded_mask.sum().item())
    print(
        f"[Brier @ time_bin={time_bin}] excluded (bucket 3, censored at/before "
        f"this bin) samples: {num_excluded} / {predictions.shape[0]}"
    )

    bucket1_sq_err = survival_at_t[failed_by_t_mask] ** 2
    bucket2_sq_err = (survival_at_t[survived_past_mask] - 1.0) ** 2

    included_sq_err = torch.cat([bucket1_sq_err, bucket2_sq_err])
    if included_sq_err.numel() == 0:
        return float("nan"), num_excluded  # every sample was censored at/before time_bin

    return included_sq_err.mean().item(), num_excluded
