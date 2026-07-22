"""Single-risk DeepHit network (Module 7 prep, Milestone 6A).

Architecture only. The DeepHit loss function (log-likelihood term +
ranking loss, per ADR-001) is a later milestone (6B) and is not implemented
here.
"""

import torch
import torch.nn as nn


class DeepHitSurvivalModel(nn.Module):
    """MLP producing a per-sample probability mass function (PMF) over
    discrete time bins, for a SINGLE risk (shot).

    Output shape is [batch_size, num_bins] -- NOT [batch_size, num_bins, 2].
    DeepHit's original formulation supports multiple *competing* risks
    (e.g. death from cause A vs cause B), each with its own predicted
    probability channel, and the README/ADR-001 describe DeepHit as
    handling "competing risks." It would therefore be an easy mistake to
    read that as license to add a second "censoring" output channel here --
    but that's a category error for this project. This project has exactly
    one real event type (a shot) and administrative right-censoring
    (turnover, foul, out-of-bounds, half-end, or duration exceeding the
    modeled horizon). Censoring is the ABSENCE of an observed event within
    the horizon, not a second competing outcome the network predicts a
    probability for: it's handled at the loss level in Milestone 6B, via
    the implicit survival function S(t) = 1 - cumsum(PMF up to t), not as
    an output channel of this network. This is therefore the single-risk
    DeepHit formulation, not the competing-risks one.
    """

    def __init__(self, num_features: int, num_bins: int, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins),
        )
        # Softmax across the num_bins dimension: each sample's output is a
        # valid PMF over time bins (sums to 1.0). This implies zero mass on
        # "the event never happens at all" -- an accepted v1 simplification.
        # The true "never within horizon" probability is only used at the
        # label/loss level, via Step 1.4's horizon-censoring rule in
        # survival_dataset.py, not modeled as network output mass here.
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.network(features)
        return self.softmax(logits)
