"""Time-dependent Brier Score evaluation for the single-risk DeepHit model
(Milestone 7), and single-sample cumulative incidence prediction for the
counterfactual simulator (Milestone 13).

Uses the SAME inclusive-cumsum survival convention as DeepHitLoss:
S(t) = 1 - cumsum(PMF up to and including bin t).
"""

import torch

from production.src.pipeline.survival_dataset import FEATURE_KEYS


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


def predict_cumulative_incidence(
    model: torch.nn.Module,
    features_dict: dict,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    time_bin: int = 3,
) -> float:
    """Cumulative incidence -- 1 - S(time_bin) -- for a single feature dict,
    under a trained single-risk DeepHit model.

    Named for what it actually computes: the cumulative incidence (CDF) of
    the event occurring by `time_bin`, NOT "hazard," which technically
    refers to the instantaneous hazard rate -- a different quantity DeepHit
    doesn't directly output. Keep this naming distinction in any code or
    printed output that consumes this function.

    Applies the SAME normalization convention used at training time --
    `(x - normalization_mean) / normalization_std` -- using TRAINING-split-
    derived stats the caller passes in. This function does NOT recompute
    normalization statistics from the single sample given here; doing so
    would be statistically meaningless (a single point has no spread) and
    would silently diverge from the scale the model was actually trained on.
    """
    features_tensor = torch.tensor(
        [features_dict[key] for key in FEATURE_KEYS], dtype=torch.float32
    ).unsqueeze(0)  # [1, num_features]

    normalized_features = (features_tensor - normalization_mean) / normalization_std

    model.eval()
    with torch.no_grad():
        predictions = model(normalized_features)  # [1, num_bins] PMF

    survival = 1.0 - torch.cumsum(predictions, dim=1)  # same convention as DeepHitLoss/Brier
    survival_at_t = survival[0, time_bin].item()

    return 1.0 - survival_at_t


def predict_cumulative_incidence_graph(
    model: torch.nn.Module,
    graph_data,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    time_bin: int = 3,
) -> float:
    """Graph-representation counterpart to `predict_cumulative_incidence`,
    for the GNN (Milestone 14). Same 1 - S(time_bin) definition and naming
    rationale -- see that function's docstring.

    `graph_data` is a single (ungrouped) PyG `Data` object, e.g. from
    `graph_builder.build_graph_from_frame`. Only the continuous node-feature
    columns ([x, y, dist_to_ball] -- indices 0, 1, 6) are standardized,
    using the SAME training-split-derived mean/std the GNN was trained
    with; vx/vy and the is_attacker/is_defender flags are left untouched,
    matching Milestone 12's normalization rule. `graph_data` itself is not
    mutated -- a modified clone is built and passed to the model.
    """
    normalized_data = graph_data.clone()
    x = normalized_data.x.clone()
    x[:, [0, 1, 6]] = (x[:, [0, 1, 6]] - normalization_mean) / normalization_std
    normalized_data.x = x

    model.eval()
    with torch.no_grad():
        predictions = model(normalized_data)  # [1, num_bins] PMF (single graph, no .batch set)

    survival = 1.0 - torch.cumsum(predictions, dim=1)
    survival_at_t = survival[0, time_bin].item()

    return 1.0 - survival_at_t
