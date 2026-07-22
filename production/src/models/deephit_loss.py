"""Single-risk DeepHit loss (Milestone 6B): NLL + pairwise ranking loss.

Index-convention note (the off-by-one risk this module is built to avoid):
F(t) is the INCLUSIVE cumulative sum of the PMF from bin 0 through bin t,
so F[:, t] is "probability the event has occurred by the end of bin t,
inclusive." S(t) = 1 - F(t) follows the same inclusive convention. Every
gather below indexes each sample's own F/S at its own duration bin using
`torch.gather`, never a Python indexing loop.
"""

import torch
import torch.nn as nn

# Applied immediately before every torch.log() call in this module. An
# untrained network can assign near-zero PMF mass to a bin; unclamped
# log(0) is -inf and silently corrupts gradients without necessarily
# crashing anything downstream.
EPS = 1e-8


class DeepHitLoss(nn.Module):
    """Single-risk DeepHit loss: NLL (event + censored survival terms) plus
    a fully vectorized pairwise concordance/ranking loss.

    No competing-risks term: this project has one real risk (shot) and
    administrative right-censoring (see deephit.py's docstring) -- there is
    no second predicted channel to weight a risk-specific NLL against.
    """

    def __init__(self, alpha: float = 1.0, sigma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def _survival_function(predictions: torch.Tensor) -> torch.Tensor:
        """S[:, t] = 1 - (inclusive cumulative PMF through bin t)."""
        cumulative = torch.cumsum(predictions, dim=1)
        return 1.0 - cumulative

    def nll_loss(
        self, predictions: torch.Tensor, durations: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        """Negative log-likelihood: PMF mass at the observed bin for events,
        inclusive survival at the censoring bin for censored samples.
        """
        survival = self._survival_function(predictions)  # [batch, num_bins]
        duration_idx = durations.unsqueeze(1)  # [batch, 1]

        pmf_at_duration = predictions.gather(1, duration_idx).squeeze(1)  # [batch]
        survival_at_duration = survival.gather(1, duration_idx).squeeze(1)  # [batch]

        pmf_at_duration = torch.clamp(pmf_at_duration, min=EPS)
        survival_at_duration = torch.clamp(survival_at_duration, min=EPS)

        event_term = -torch.log(pmf_at_duration)
        censored_term = -torch.log(survival_at_duration)

        per_sample = events * event_term + (1.0 - events) * censored_term
        return per_sample.mean()

    def ranking_loss(
        self, predictions: torch.Tensor, durations: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        """Fully vectorized pairwise ranking loss (no Python loops over
        batch pairs), per the README's engineering standard -- Module 7's
        batch ensembles will need this same broadcast-not-loop pattern
        reshaped to [Ensemble, Batch, Features] later.

        For every ordered pair (i, j): if sample i had an observed event
        and i's duration bin is strictly earlier than j's, the pair is
        valid and contributes exp((S_i(t_i) - S_j(t_i)) / sigma), i.e. both
        samples' survival evaluated at i's own duration bin t_i. A correctly
        ranked pair (i truly failed earlier: S_i(t_i) << S_j(t_i)) drives
        the exponent very negative -> near-zero loss; a violated ranking
        (model thinks i survives past t_i better than j does) drives it
        positive -> large loss.
        """
        batch_size = predictions.shape[0]
        survival = self._survival_function(predictions)  # [batch, num_bins]

        duration_idx = durations.unsqueeze(1)  # [batch, 1]
        survival_at_own_duration = torch.clamp(
            survival.gather(1, duration_idx).squeeze(1), min=EPS
        )  # [batch]; S_i(t_i)

        # M[i, j] = S_j(t_i): row i is sample j's survival curve evaluated at
        # sample i's own duration bin, for every j. survival.t() is
        # [num_bins, batch] indexed [t, j]; selecting rows by `durations`
        # (a LongTensor of shape [batch]) gives, for each i, the row
        # survival.t()[durations[i]] = survival[:, durations[i]] over j.
        survival_j_at_duration_i = torch.clamp(survival.t()[durations], min=EPS)  # [batch, batch]

        diff = survival_at_own_duration.unsqueeze(1) - survival_j_at_duration_i  # [batch, batch]

        events_i = events.unsqueeze(1).bool()  # [batch, 1]
        duration_i = durations.unsqueeze(1)  # [batch, 1]
        duration_j = durations.unsqueeze(0)  # [1, batch]
        valid_mask = events_i & (duration_i < duration_j)  # [batch, batch]
        valid_mask_f = valid_mask.to(predictions.dtype)

        pair_losses = torch.exp(diff / self.sigma) * valid_mask_f

        num_valid = valid_mask_f.sum()
        # If there are zero valid pairs (a real possibility with random
        # data -- no events, or no valid orderings), the numerator above is
        # already exactly 0 (every term is masked out), so clamping the
        # denominator to at least 1 yields 0/1 = 0 instead of 0/0 = NaN.
        return pair_losses.sum() / torch.clamp(num_valid, min=1.0)

    def forward(
        self, predictions: torch.Tensor, durations: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        return self.nll_loss(predictions, durations, events) + self.alpha * self.ranking_loss(
            predictions, durations, events
        )
