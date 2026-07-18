"""Biomechanical, velocity-aware pitch control engine (Module 1).

Per ADR-005, physics is evaluated only over a sparse subset of the pitch
grid (cells within `mask_radius` of the ball), not a dense 100x68 grid.
Time-to-intercept is solved from the closed-form analytical ODE solution
specified in the README (NOT a constant-acceleration quadratic and NOT
numerical integration), using a fixed-iteration vectorized Newton-Raphson
root-find so the whole computation stays batched tensor ops with no native
Python loops over players or grid cells.
"""

import torch
import torch.nn as nn


class BiomechanicalPitchControl(nn.Module):
    """Velocity-aware pitch control via analytical time-to-intercept.

    Note on ADR-005: this reference implementation extracts active grid
    coordinates via boolean-mask indexing, which is variable-length
    (`N_active` depends on the ball position each call). The Triton
    production kernel path described in ADR-005 additionally carries a
    static-shape binary mask alongside the sparse coordinates to avoid
    kernel recompilation; that padding step is a production-kernel
    concern deferred to the Triton port and is out of scope for this
    vectorized PyTorch reference implementation.
    """

    def __init__(
        self,
        max_speed: float = 8.0,
        max_accel: float = 7.0,
        reaction_time: float = 0.2,
        k: float = 1.0,
        mask_radius: float = 30.0,
        newton_iters: int = 15,
    ):
        super().__init__()
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.reaction_time = reaction_time
        self.k = k
        self.mask_radius = mask_radius
        self.newton_iters = newton_iters

    def forward(
        self,
        player_pos: torch.Tensor,   # [P, 2]
        player_vel: torch.Tensor,   # [P, 2]
        fatigue_mod: torch.Tensor,  # [P]
        pitch_grid: torch.Tensor,   # [N_total, 2]
        ball_pos: torch.Tensor,     # [2]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # --- Sparse masking (ADR-005): only cells within mask_radius of the
        # ball are ever handed to the physics solver. ---
        dist_to_ball = torch.linalg.norm(pitch_grid - ball_pos.unsqueeze(0), dim=-1)  # [N_total]
        active_mask = dist_to_ball <= self.mask_radius
        active_grid_coords = pitch_grid[active_mask]  # [N_active, 2]

        # --- Biomechanical force-velocity curve: acceleration capacity decays
        # with CURRENT TOTAL SPEED MAGNITUDE (full L2 norm of velocity), not
        # any radial component. Fatigue further scales the result. ---
        speed = torch.linalg.norm(player_vel, dim=-1)  # [P]
        a_eff = self.max_accel * (1.0 - torch.clamp(speed / self.max_speed, 0.0, 1.0))
        a_eff = a_eff * fatigue_mod  # [P]
        # Guard against a_eff == 0 (player already at max speed / zero fatigue
        # capacity): the closed-form solution divides by a_max, and its exact
        # limit as a_max -> 0 is the constant-velocity case x(t) = v0*t. A tiny
        # floor keeps that limit numerically stable without materially
        # changing the physics.
        a_max = a_eff.clamp(min=1e-6).unsqueeze(1)  # [P, 1], broadcasts over N_active

        # --- Radial velocity projection: project each player's full 2D
        # velocity onto the unit direction from the player to each target
        # cell. This is a distinct quantity from `speed` above and is used
        # only as the ODE's initial condition v0. ---
        diff = active_grid_coords.unsqueeze(0) - player_pos.unsqueeze(1)  # [P, N_active, 2]
        dist = torch.linalg.norm(diff, dim=-1)
        # Guard d == 0 (target cell coincides with player position) before
        # it's used as a divisor for the unit direction vector and as the ODE
        # root-find target; the tiny epsilon makes the solved t collapse to
        # ~0, so the returned time-to-intercept below is ~reaction_time.
        dist_safe = dist.clamp(min=1e-6)
        direction = diff / dist_safe.unsqueeze(-1)  # unit vector, player -> target
        v_radial = torch.sum(player_vel.unsqueeze(1) * direction, dim=-1)  # [P, N_active]

        # A player moving away from the target cell (v_radial < 0) is clamped
        # to 0 rather than penalized further. This (a) keeps x(t) monotonic
        # in t, which the Newton root-find below relies on for a single
        # well-defined positive root, and (b) matches the standard
        # simplification in pitch-control literature: moving away is treated
        # as "starting from rest toward the target," not as a penalty beyond
        # that.
        v0 = torch.clamp(v_radial, min=0.0)  # [P, N_active]

        v_max = self.max_speed

        # --- Analytical ODE position equation, solved for t via vectorized
        # Newton-Raphson (fixed iteration count -> bounded, predictable
        # runtime; no dynamic while-loop, no native Python loop over
        # players/cells — every iteration below operates on the full
        # [P, N_active] tensor at once):
        #   x(t) = v_max*t - (v_max - v0)*(v_max/a_max)*(1 - exp(-a_max*t/v_max))
        #   x'(t) = v_max - (v_max - v0)*exp(-a_max*t/v_max)
        t = 2.0 * dist_safe / (v0 + v_max)  # initial guess
        for _ in range(self.newton_iters):
            exp_term = torch.exp(-a_max * t / v_max)
            x_t = v_max * t - (v_max - v0) * (v_max / a_max) * (1.0 - exp_term)
            dx_dt = v_max - (v_max - v0) * exp_term
            t = t - (x_t - dist_safe) / dx_dt.clamp(min=1e-6)
            t = torch.clamp(t, min=0.0)

        time_to_intercepts = t + self.reaction_time  # [P, N_active]

        # --- Probability conversion. t_ball = 0 is a placeholder: a real
        # ball-trajectory time estimate (from the friction/aero physics of
        # Module 2) will replace this constant in a later milestone. ---
        t_ball = 0.0
        control_probabilities = 1.0 / (1.0 + torch.exp(self.k * (time_to_intercepts - t_ball)))

        return active_grid_coords, control_probabilities, time_to_intercepts
