"""ADR-010 regression tests: the training-instability detector
(`production/src/pipeline/train.py`) against SYNTHETIC epoch-loss /
output-distribution sequences reproducing this project's known historical
patterns (Milestones 12, 14, 23), rather than expensive real retraining.

Background (see ADR-010 and RESEARCH_FINDINGS.md's RQ4 section): the
codebase's four-signal core detector (single-epoch spike, cumulative/
windowed drift, output saturation via variance+entropy, frozen-val-loss
backstop -- all built in response to real Milestone 12/14 failures) never
misfired. The false positive came from a SEPARATE, cruder addition inside
`train_and_evaluate()`'s `mlp_healthy` gate: `loss_decreased_meaningfully =
(first_loss - last_loss) > 0.1 * first_loss`. This fired "unhealthy" on
Milestone 23's genuinely healthy, fast-converging-then-plateauing seed-42
MLP, because a model that converges within its first epoch and then
correctly plateaus shows exactly the same small further decrease this
crude check would misread as "not learning." The fix demotes this check to
a printed, non-blocking diagnostic and drops it from `mlp_healthy`'s gate,
which now depends only on (a) the four-signal detector and (b) a sane
Brier Score ceiling. These tests confirm the REMAINING (fixed) detector
logic -- the actual production functions, not reimplemented copies --
still classifies all four known historical shapes correctly.
"""

from production.src.pipeline.train import (
    CUMULATIVE_DRIFT_WINDOW_EPOCHS,
    VAL_LOSS_LOG_INTERVAL_EPOCHS,
    _check_cumulative_drift,
    _check_for_instability,
    _check_frozen_val_loss,
    _check_saturation,
)


def _run_detector_on_synthetic_run(
    epoch_losses: list[float],
    val_loss_by_checkpoint: dict[int, float],
    batch_variance_by_checkpoint: dict[int, float],
    mean_entropy_by_checkpoint: dict[int, float],
) -> dict:
    """Replays the REAL per-epoch checkpoint structure `_train_and_log_model`
    uses (checks at every `VAL_LOSS_LOG_INTERVAL_EPOCHS`-th epoch, plus the
    final epoch, with cumulative drift additionally gated on
    `epoch > CUMULATIVE_DRIFT_WINDOW_EPOCHS`), calling the exact same
    production detector functions train.py's training loop calls -- not a
    reimplementation of their logic.
    """
    num_epochs = len(epoch_losses)
    cumulative_drift_fired = False
    saturation_fired = False
    frozen_val_loss_fired = False
    val_loss_history: dict[int, float] = {}

    for epoch in range(1, num_epochs + 1):
        if epoch % VAL_LOSS_LOG_INTERVAL_EPOCHS == 0 and epoch > CUMULATIVE_DRIFT_WINDOW_EPOCHS:
            fired, _fraction = _check_cumulative_drift("Synthetic", epoch_losses, epoch)
            cumulative_drift_fired = cumulative_drift_fired or fired

        if epoch % VAL_LOSS_LOG_INTERVAL_EPOCHS == 0 or epoch == num_epochs:
            epoch_val_loss = val_loss_by_checkpoint[epoch]
            batch_variance = batch_variance_by_checkpoint[epoch]
            mean_entropy = mean_entropy_by_checkpoint[epoch]

            saturation_fired = saturation_fired or _check_saturation(
                "Synthetic", epoch, batch_variance, mean_entropy
            )
            frozen_val_loss_fired = frozen_val_loss_fired or _check_frozen_val_loss(
                "Synthetic", epoch, val_loss_history, epoch_val_loss
            )
            val_loss_history[epoch] = epoch_val_loss

    spike_fired = _check_for_instability("Synthetic", epoch_losses)
    instability_warning_fired = (
        spike_fired or cumulative_drift_fired or saturation_fired or frozen_val_loss_fired
    )
    return {
        "spike_fired": spike_fired,
        "cumulative_drift_fired": cumulative_drift_fired,
        "saturation_fired": saturation_fired,
        "frozen_val_loss_fired": frozen_val_loss_fired,
        "instability_warning_fired": instability_warning_fired,
    }


CHECKPOINT_EPOCHS = list(range(VAL_LOSS_LOG_INTERVAL_EPOCHS, 51, VAL_LOSS_LOG_INTERVAL_EPOCHS))  # 5..50


def test_m12_gnn_exploding_gradient_pattern_flags_unstable():
    """Milestone 12: GNN loss reported as 'spiking 3.07 -> 4.58 around
    epoch 20 and never recovering.' The historical one-line summary
    doesn't specify whether that climb was a single-epoch jump or spread
    across a short window, and literal adjacent-epoch values of exactly
    3.07/4.58 give a (4.58-3.07)/3.07 = 49.2% relative increase -- just
    under the 50% single-epoch spike threshold. This test reproduces the
    'sudden spike that never recovers' SHAPE the spike detector exists to
    catch (Signal 1 was built specifically because of this incident) using
    a slightly larger, unambiguous jump (3.07 -> 4.70, +53.1%) rather than
    silently rounding a threshold-adjacent number in the detector's favor.
    """
    epoch_losses = [3.07] * 19 + [4.70] * 31  # spike at epoch 20, holds elevated -- never recovers
    assert len(epoch_losses) == 50

    # Diverse, non-collapsed outputs and non-frozen val loss throughout --
    # this scenario should be caught by the SPIKE signal specifically, not
    # by saturation or frozen-val-loss coincidentally.
    val_loss_by_checkpoint = {e: 3.0 + 0.001 * i for i, e in enumerate(CHECKPOINT_EPOCHS)}
    batch_variance_by_checkpoint = {e: 0.02 for e in CHECKPOINT_EPOCHS}
    mean_entropy_by_checkpoint = {e: 1.5 for e in CHECKPOINT_EPOCHS}

    result = _run_detector_on_synthetic_run(
        epoch_losses, val_loss_by_checkpoint, batch_variance_by_checkpoint, mean_entropy_by_checkpoint
    )

    assert result["spike_fired"] is True
    assert result["instability_warning_fired"] is True


def test_m14_mlp_frozen_softmax_collapse_flags_unstable():
    """Milestone 14: MLP train loss 'climbed steadily from 3.30 to 5.07
    over epochs 5-50 -- NO single epoch-to-epoch jump ever exceeded 50% --
    while val_loss went bit-for-bit frozen at 4.843820095062256 for the
    final 25 epochs.' Direct probing confirmed softmax saturation to
    ~1.0 on the last time bin. Uses the exact historical constants.
    """
    epoch_losses = [3.30] * 4 + [
        3.30 + (5.07 - 3.30) * (e - 5) / 45 for e in range(5, 51)
    ]
    assert len(epoch_losses) == 50
    assert epoch_losses[4] == 3.30
    assert abs(epoch_losses[-1] - 5.07) < 1e-9

    # No single-epoch jump should exceed 50% -- confirms this reproduces
    # the historical "spike check has a real blind spot" property, not an
    # accidental spike.
    for i in range(1, len(epoch_losses)):
        assert (epoch_losses[i] - epoch_losses[i - 1]) / epoch_losses[i - 1] < 0.5

    FROZEN_VAL_LOSS = 4.843820095062256
    val_loss_by_checkpoint = {}
    batch_variance_by_checkpoint = {}
    mean_entropy_by_checkpoint = {}
    for e in CHECKPOINT_EPOCHS:
        if e >= 25:
            # Bit-for-bit frozen from epoch 25 onward (spans the final 25
            # epochs, 26-50, with the freeze already visible AT epoch 25).
            val_loss_by_checkpoint[e] = FROZEN_VAL_LOSS
            batch_variance_by_checkpoint[e] = 1e-8  # collapsed
            mean_entropy_by_checkpoint[e] = 0.001  # near one-hot
        else:
            val_loss_by_checkpoint[e] = 3.0 + 0.05 * (e // VAL_LOSS_LOG_INTERVAL_EPOCHS)
            batch_variance_by_checkpoint[e] = 0.02  # healthy, pre-collapse
            mean_entropy_by_checkpoint[e] = 1.5  # healthy, pre-collapse

    result = _run_detector_on_synthetic_run(
        epoch_losses, val_loss_by_checkpoint, batch_variance_by_checkpoint, mean_entropy_by_checkpoint
    )

    # The historical point of this incident: the SPIKE check misses it.
    assert result["spike_fired"] is False
    # But the signals built in response to it (Milestone 14B) catch it.
    assert result["saturation_fired"] is True
    assert result["frozen_val_loss_fired"] is True
    assert result["instability_warning_fired"] is True


def test_m23_fast_converging_then_plateauing_mlp_does_not_flag():
    """Milestone 23: the seed-42 MLP was genuinely healthy -- sharp initial
    convergence (within the first few epochs) followed by a legitimate,
    small further decrease as it holds near its optimum, with genuinely
    diverse outputs -- but the OLD `loss_decreased_meaningfully` gate
    (first-epoch loss vs. final-epoch loss, > 10% required) would have
    flagged it, because most of the improvement happens WITHIN the first
    epoch's many mini-batches, leaving only a small further delta by
    epoch 50. This is the exact false positive ADR-010 fixes. The detector
    itself (all four principled signals) must NOT flag this run, and no
    manual override should be needed.
    """
    # Sharp initial convergence (epochs 1-5), then a small, legitimate
    # plateau descent for the remaining 45 epochs.
    sharp_convergence = [1.10, 1.06, 1.03, 1.015, 1.008]
    plateau = [1.008 - 0.0002 * i for i in range(1, 46)]
    epoch_losses = sharp_convergence + plateau
    assert len(epoch_losses) == 50

    first_loss, last_loss = epoch_losses[0], epoch_losses[-1]
    loss_decreased_meaningfully = (first_loss - last_loss) > 0.1 * first_loss
    # Demonstrates the historical false positive: under the OLD (now-fixed)
    # crude gate, this genuinely healthy run would have been marked
    # "unhealthy" purely on this arithmetic.
    assert loss_decreased_meaningfully is False, (
        "this scenario is constructed to reproduce the exact arithmetic that used to "
        "misfire -- if this assertion fails, the scenario no longer demonstrates the "
        "historical false positive"
    )

    # Genuinely diverse outputs (well clear of both saturation thresholds),
    # with tiny per-checkpoint jitter so no two are bit-for-bit identical.
    val_loss_by_checkpoint = {e: 1.05 - 0.0009 * i for i, e in enumerate(CHECKPOINT_EPOCHS)}
    batch_variance_by_checkpoint = {e: 0.05 - 0.0001 * i for i, e in enumerate(CHECKPOINT_EPOCHS)}
    mean_entropy_by_checkpoint = {e: 2.0 - 0.001 * i for i, e in enumerate(CHECKPOINT_EPOCHS)}

    result = _run_detector_on_synthetic_run(
        epoch_losses, val_loss_by_checkpoint, batch_variance_by_checkpoint, mean_entropy_by_checkpoint
    )

    assert result["spike_fired"] is False
    assert result["cumulative_drift_fired"] is False
    assert result["saturation_fired"] is False
    assert result["frozen_val_loss_fired"] is False
    assert result["instability_warning_fired"] is False, (
        "the fixed detector must NOT flag a genuinely healthy, fast-converging-then-"
        "plateauing run -- no manual override should ever be needed for this case"
    )


def test_healthy_slowly_improving_model_does_not_flag():
    """Control case: a genuinely healthy model that is smoothly, gradually,
    still improving across all 50 epochs (never plateaus) -- should also
    NOT be flagged, and (unlike the M23 case) would also have passed the
    OLD crude loss-decrease gate, confirming the fix didn't just narrow
    the detector to exclude M23's shape at the cost of missing real
    problems elsewhere.
    """
    epoch_losses = [2.5 - (2.5 - 1.0) * (e - 1) / 49 for e in range(1, 51)]
    assert len(epoch_losses) == 50
    assert epoch_losses[0] == 2.5
    assert abs(epoch_losses[-1] - 1.0) < 1e-9

    first_loss, last_loss = epoch_losses[0], epoch_losses[-1]
    loss_decreased_meaningfully = (first_loss - last_loss) > 0.1 * first_loss
    assert loss_decreased_meaningfully is True  # this healthy case passes either gate

    val_loss_by_checkpoint = {e: 2.5 - (2.5 - 1.0) * (e - 1) / 49 + 0.01 for e in CHECKPOINT_EPOCHS}
    batch_variance_by_checkpoint = {e: 0.05 for e in CHECKPOINT_EPOCHS}
    mean_entropy_by_checkpoint = {e: 2.0 for e in CHECKPOINT_EPOCHS}

    result = _run_detector_on_synthetic_run(
        epoch_losses, val_loss_by_checkpoint, batch_variance_by_checkpoint, mean_entropy_by_checkpoint
    )

    assert result["instability_warning_fired"] is False
