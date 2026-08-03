"""Milestone 21 (Module 7): Deep Ensemble uncertainty quantification.

NAMING CLARIFICATION -- read this before assuming "Batch Ensemble" and
"Deep Ensemble" are interchangeable (see ADR-004 for the full writeup):

README.txt's Module 7 cites "Batch Ensembles" (Wen et al., 2020): a SINGLE
shared base weight matrix across all M members, with each member's
distinct behavior coming from cheap per-member rank-1 perturbation vectors
multiplied elementwise into that shared matrix. This makes M members cost
barely more than one member's parameters/compute -- the entire point,
motivating README's Module 7 line "Ensemble forward passes are batched"
and the project's general GPU-latency consciousness (echoed in the R4
risk mitigation and the Red Team fixes list in README Section 6).

`DeepEnsembleDeepHit` below is NOT that. It is a DEEP ENSEMBLE: M fully
independent `DeepHitSurvivalModel` instances, each with its own complete,
separately-initialized set of weights. This is a simpler and statistically
well-understood technique for uncertainty estimation, but it is
~M times heavier in both parameters and compute than a single model --
the exact cost Batch Ensembles exist to avoid. This class is deliberately
NOT named `BatchEnsembleDeepHit` so nobody mistakes this module for having
solved that latency problem; it hasn't. True Batch Ensembles remain a
candidate future optimization (see ADR-004) if/when this needs to feed the
live WebSocket API (Milestone 16) under real latency constraints.
"""

import torch
from torch import nn

from production.src.models.deephit import DeepHitSurvivalModel
from production.src.models.deephit_loss import DeepHitLoss

DEFAULT_ENSEMBLE_SIZE = 5


class DeepEnsembleDeepHit(nn.Module):
    """M independent, identically-architected `DeepHitSurvivalModel`
    instances. Each member has its own separately (randomly) initialized
    weights -- `nn.Linear.reset_parameters()` draws from PyTorch's global
    RNG stream, which advances with every member constructed in the loop
    below, so no explicit per-member seeding is needed for the members to
    differ from each other.
    """

    def __init__(self, num_features: int, num_bins: int, M: int = DEFAULT_ENSEMBLE_SIZE, hidden_dim: int = 32):
        super().__init__()
        self.M = M
        self.num_bins = num_bins
        self.members = nn.ModuleList(
            [DeepHitSurvivalModel(num_features, num_bins, hidden_dim) for _ in range(M)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """BROADCAST, not partition: every one of the M members sees the
        SAME FULL batch `x` of shape [B, F]. The return value is
        [M, B, num_bins] -- output[m] is member m's full prediction for
        ALL B examples, not a prediction for some B/M-sized slice of them.

        This distinction matters because "reshape [B, F] to [M, B, F]" is
        genuinely ambiguous between broadcasting (what this does) and
        partitioning the batch across members (which would silently
        produce a meaningless ensemble: members would then disagree only
        because they saw different examples, not because they hold
        genuinely different learned beliefs about the SAME input).
        """
        return torch.stack([member(x) for member in self.members], dim=0)  # [M, B, num_bins]

    def predict_with_uncertainty(self, x: torch.Tensor, time_bin: int = 3):
        """Returns (mean_pmf, std_cumulative_incidence,
        per_member_cumulative_incidence):

          - mean_pmf: [B, num_bins] -- PMF averaged across the M members.
          - std_cumulative_incidence: [B] -- standard deviation, ACROSS
            THE M MEMBERS, of cumulative incidence (inclusive cumsum PMF)
            at `time_bin`. This is the ensemble's epistemic-uncertainty
            estimate for each of the B input samples.
          - per_member_cumulative_incidence: [M, B] -- each member's own
            cumulative incidence at `time_bin`, exposed directly (not just
            the summary std) so ensemble diversity can be inspected
            per-member, not only through one aggregate number.
        """
        pmf_per_member = self.forward(x)  # [M, B, num_bins]
        mean_pmf = pmf_per_member.mean(dim=0)  # [B, num_bins]

        cumulative_per_member = torch.cumsum(pmf_per_member, dim=2)  # [M, B, num_bins]
        per_member_cumulative_incidence = cumulative_per_member[:, :, time_bin]  # [M, B]
        std_cumulative_incidence = per_member_cumulative_incidence.std(dim=0)  # [B]

        return mean_pmf, std_cumulative_incidence, per_member_cumulative_incidence


def compute_disentangled_ensemble_loss(
    pmf_per_member: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
    loss_fn: DeepHitLoss,
) -> torch.Tensor:
    """For each of the M members INDEPENDENTLY, computes the FULL DeepHit
    loss (NLL + the existing vectorized pairwise ranking loss) using ONLY
    that member's own [B, num_bins] predictions, then returns the mean of
    the M per-member losses.

    Deliberately a Python loop over M (small and fixed, e.g. 5) -- NOT a
    reshape/flatten to [M*B, num_bins] fed through `DeepHitLoss` once.
    `DeepHitLoss.ranking_loss` forms pairwise comparisons across whatever
    batch dimension it's given; flattening M and B together would create
    spurious pairwise comparisons BETWEEN different ensemble members'
    predictions, entangling gradients across members and corrupting the
    cross-member independence the whole uncertainty estimate depends on.
    This is a small, fixed-size loop over ensemble members -- not the
    batch-dimension Python loop this project correctly avoids elsewhere.
    """
    M = pmf_per_member.shape[0]
    total_loss = pmf_per_member.new_zeros(())
    for m in range(M):
        total_loss = total_loss + loss_fn(pmf_per_member[m], durations, events)
    return total_loss / M
