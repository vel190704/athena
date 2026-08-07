"""True Batch Ensemble (Wen et al., 2020) -- the technique README.txt's
Module 7 actually describes ("Ensemble forward passes are batched... a
single shared base weight matrix... M members cost barely more than one
member's parameters/compute"), implemented for real here for the first
time. See ADR-004 for the full history: Milestone 21 built
`DeepEnsembleDeepHit` (M fully independent models, ~Mx cost) instead,
explicitly named to avoid being mistaken for this technique, with true
Batch Ensembles deferred as future work.

NOT a modification of `deep_ensemble.py` -- a new, separate, additive
class. `DeepEnsembleDeepHit` is untouched; this is a genuine drop-in
ALTERNATIVE for A/B comparison, not a replacement. `compute_disentangled_ensemble_loss`
IS reused directly from `deep_ensemble.py` (imported, not duplicated) --
that function only operates on an already-produced `[M, B, num_bins]`
prediction tensor, so it is equally correct here regardless of which
architecture produced that tensor; re-deriving an identical copy would
violate this project's own DRY convention for zero benefit.

THE TECHNIQUE, per Wen et al. 2020: each `BatchEnsembleLinear` layer holds
ONE shared base weight matrix `W` (shape `[out, in]`, identical to a
plain `nn.Linear`'s), plus, per ensemble member `m`, a pair of cheap
RANK-1 "fast weight" vectors `r_m` (`[in]`) and `s_m` (`[out]`) and a
per-member bias `b_m` (`[out]`, same as a normal `nn.Linear`'s bias would
be, kept per-member since a bias is already cheap and per-member biases
are standard in the original technique). Member `m`'s effective weight
matrix is `W * outer(s_m, r_m)` -- but this is never materialized
per-member; it's computed via two elementwise multiplies around ONE
shared matmul:

    y_m = ((x * r_m) @ W^T) * s_m + b_m

For `M` members this costs `M` extra `[in]`+`[out]`+`[out]`-shaped
vectors per layer (a few hundred extra scalars for this project's tiny
4->32->32->12 architecture), NOT `M` extra `[out, in]` weight matrices --
the entire efficiency claim this technique is named for, and the literal
gap ADR-004 named as unresolved.
"""

import math

import torch
from torch import nn

from production.src.models.deep_ensemble import compute_disentangled_ensemble_loss  # noqa: F401 -- re-exported for callers

DEFAULT_ENSEMBLE_SIZE = 5


class BatchEnsembleLinear(nn.Module):
    """One BatchEnsemble layer: a single shared `weight` (`[out, in]`,
    same shape/init convention as `nn.Linear`), plus per-member rank-1
    `r`/`s` fast-weight vectors and a per-member `bias`.

    Accepts EITHER a plain `[B, in]` input (the first layer in a stack --
    broadcast to all `M` members) OR an already-per-member `[M, B, in]`
    input (every later layer, chained from a previous `BatchEnsembleLinear`'s
    own `[M, B, out]` output) -- so these layers compose directly into a
    multi-layer network without the caller needing to manually track which
    shape a given layer expects.
    """

    def __init__(self, in_features: int, out_features: int, M: int):
        super().__init__()
        self.M = M
        self.in_features = in_features
        self.out_features = out_features

        # Shared base weight -- ONE matrix for all M members, same
        # Kaiming-uniform init `nn.Linear` itself uses (a=sqrt(5)), so a
        # BatchEnsembleLinear's shared trunk starts from the exact same
        # distribution a plain nn.Linear layer of this shape would.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Per-member rank-1 fast weights and bias.
        self.r = nn.Parameter(torch.empty(M, in_features))
        self.s = nn.Parameter(torch.empty(M, out_features))
        self.bias = nn.Parameter(torch.empty(M, out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Random +-1 SIGN initialization for r/s -- the original
        # BatchEnsemble paper's own convention (Wen et al. 2020, Sec 3.2),
        # not an arbitrary choice: initializing r/s to all-ones would make
        # every member IDENTICAL at step 0 (since they'd all compute the
        # exact same effective weight W*outer(1,1)=W), defeating the
        # purpose of having M members at all -- diversity would have to
        # emerge from gradient noise alone, far slower and less reliable
        # than starting from M genuinely different effective weight
        # matrices. Uses PyTorch's global RNG stream (like `nn.Linear.
        # reset_parameters()` and `DeepEnsembleDeepHit`'s own member
        # construction loop already do -- see that class's docstring) so
        # M members constructed in sequence naturally differ, no explicit
        # per-member seeding required.
        r_signs = torch.randint(0, 2, self.r.shape, dtype=self.r.dtype) * 2 - 1
        s_signs = torch.randint(0, 2, self.s.shape, dtype=self.s.dtype) * 2 - 1
        with torch.no_grad():
            self.r.copy_(r_signs)
            self.s.copy_(s_signs)

        # Bias: same fan-in-based uniform bound `nn.Linear` itself uses,
        # applied per-member (biases are already cheap; keeping one per
        # member, rather than sharing a single bias, is the original
        # technique's own convention and costs nothing meaningful).
        fan_in = self.in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # First layer: [B, in] -- broadcast (not copy; expand is a
            # view) to every member.
            x = x.unsqueeze(0).expand(self.M, -1, -1)  # [M, B, in]
        # x: [M, B, in]
        x_scaled = x * self.r.unsqueeze(1)  # [M,1,in] broadcasts over B -> [M,B,in]

        M, B, _ = x_scaled.shape
        # THE efficiency-defining step: ONE shared matmul for all M*B rows
        # at once, against the ONE shared `weight` -- not M separate
        # matmuls against M separate weight matrices.
        y = x_scaled.reshape(M * B, self.in_features) @ self.weight.t()
        y = y.reshape(M, B, self.out_features)

        return y * self.s.unsqueeze(1) + self.bias.unsqueeze(1)  # [M,B,out]


class BatchEnsembleDeepHit(nn.Module):
    """True Batch Ensemble counterpart to `DeepEnsembleDeepHit` -- SAME
    architecture shape (`num_features -> hidden_dim -> hidden_dim ->
    num_bins`, ReLU between, softmax over `num_bins`, matching
    `DeepHitSurvivalModel` exactly) and SAME forward-pass output shape
    (`[M, B, num_bins]`, softmax already applied), so it is a genuine
    drop-in for `DeepEnsembleDeepHit` wherever that class is used --
    `compute_disentangled_ensemble_loss`, `calculate_brier_score`, and
    every ADR-010 health-check function this project already has all
    consume a `[M, B, num_bins]` tensor and do not care which class
    produced it.

    The ONLY thing that differs from `DeepEnsembleDeepHit` is HOW that
    tensor gets computed: one shared trunk of `BatchEnsembleLinear` layers
    instead of `M` fully independent `DeepHitSurvivalModel` instances --
    see this module's own docstring for the full technique explanation
    and ADR-004 for why this was deferred until now.
    """

    def __init__(self, num_features: int, num_bins: int, M: int = DEFAULT_ENSEMBLE_SIZE, hidden_dim: int = 32):
        super().__init__()
        self.M = M
        self.num_bins = num_bins
        self.fc1 = BatchEnsembleLinear(num_features, hidden_dim, M)
        self.fc2 = BatchEnsembleLinear(hidden_dim, hidden_dim, M)
        self.fc3 = BatchEnsembleLinear(hidden_dim, num_bins, M)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same BROADCAST semantics as `DeepEnsembleDeepHit.forward`'s own
        docstring: every one of the M members sees the SAME full batch
        `x` of shape `[B, F]`; the return value is `[M, B, num_bins]`,
        NOT a batch-partitioned reshape."""
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        logits = self.fc3(h)
        return self.softmax(logits)  # [M, B, num_bins]

    def predict_with_uncertainty(self, x: torch.Tensor, time_bin: int = 3):
        """Identical interface and semantics to
        `DeepEnsembleDeepHit.predict_with_uncertainty` -- see that
        method's own docstring for the full field-by-field explanation.
        Reproduced here (not imported) only because it is genuinely a
        method OF this class (needs `self.forward`); the computation
        itself is the same three-line reduction over an `[M, B, num_bins]`
        tensor either class can produce.
        """
        pmf_per_member = self.forward(x)  # [M, B, num_bins]
        mean_pmf = pmf_per_member.mean(dim=0)  # [B, num_bins]

        cumulative_per_member = torch.cumsum(pmf_per_member, dim=2)  # [M, B, num_bins]
        per_member_cumulative_incidence = cumulative_per_member[:, :, time_bin]  # [M, B]
        std_cumulative_incidence = per_member_cumulative_incidence.std(dim=0)  # [B]

        return mean_pmf, std_cumulative_incidence, per_member_cumulative_incidence
