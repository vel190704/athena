"""Milestone 6B validation: DeepHitLoss (NLL + ranking loss).

Covers: a training smoke test (finite loss, real gradients through the
model), and a hand-constructed directional sanity check that specifically
exercises the ranking loss's sign -- a shape/finiteness check alone would
never catch a sign error in the ranking formula.
"""

import torch

from production.src.models.deephit import DeepHitSurvivalModel
from production.src.models.deephit_loss import DeepHitLoss

SEED = 42
NUM_FEATURES = 4
NUM_BINS = 12
BATCH_SIZE = 32


def _make_pmf(weights: list[float]) -> torch.Tensor:
    """Normalize hand-picked, unnormalized bin weights into a valid PMF."""
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum()


def test_loss_is_finite_scalar_and_gradients_flow():
    torch.manual_seed(SEED)

    model = DeepHitSurvivalModel(num_features=NUM_FEATURES, num_bins=NUM_BINS)
    loss_fn = DeepHitLoss()

    features = torch.randn(BATCH_SIZE, NUM_FEATURES)
    durations = torch.randint(0, NUM_BINS, (BATCH_SIZE,))
    events = torch.randint(0, 2, (BATCH_SIZE,)).float()

    predictions = model(features)
    loss = loss_fn(predictions, durations, events)

    assert loss.shape == ()
    assert torch.isfinite(loss)

    loss.backward()

    param = next(model.parameters())
    assert param.grad is not None
    assert torch.isfinite(param.grad).all()


def test_ranking_loss_penalizes_violated_ranking_more_than_correct_ranking():
    """Hand-constructed, fixed PMFs (not model output): sample i has an
    observed event at bin 2, sample j is censored at bin 8. Scenario A gives
    i the early-concentrated PMF and j the late-concentrated PMF (correct
    ranking: i's survival at bin 2 is low, j's is high). Scenario B swaps
    which sample gets which PMF (violated ranking: i now looks like it
    survives past bin 2 better than j does, despite i being the one who
    actually failed there).
    """
    loss_fn = DeepHitLoss()

    durations = torch.tensor([2, 8])
    events = torch.tensor([1.0, 0.0])  # i: observed event, j: censored

    early_concentrated = _make_pmf([30, 30, 30, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # mass at/before bin 2
    late_concentrated = _make_pmf([1, 1, 1, 1, 1, 1, 1, 1, 1, 30, 30, 30])  # mass at bins 9-11

    # Scenario A: correct ranking -- i (failed at bin 2) gets the
    # early-concentrated PMF, j (censored at bin 8) gets the
    # late-concentrated PMF.
    predictions_a = torch.stack([early_concentrated, late_concentrated])
    ranking_loss_a = loss_fn.ranking_loss(predictions_a, durations, events)

    # Scenario B: violated ranking -- same durations/events, but PMFs swapped.
    predictions_b = torch.stack([late_concentrated, early_concentrated])
    ranking_loss_b = loss_fn.ranking_loss(predictions_b, durations, events)

    print(f"\nScenario A (correct ranking) loss:  {ranking_loss_a.item():.6f}")
    print(f"Scenario B (violated ranking) loss: {ranking_loss_b.item():.6f}")

    assert ranking_loss_b > ranking_loss_a
