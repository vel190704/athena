"""Milestone 1 gate: validate the Causal Kalman Friction Filter on synthetic data.

Per Section 5 (Strict Build Order) and ADR-008, the Kalman filter must
back-calculate a known synthetic friction coefficient within a 2% margin of
error before any ML work is touched. Cd = 0 (ground passes only), so the
synthetic kinematics are exact modulo the injected measurement noise.
"""

import numpy as np

from production.src.physics.kalman_friction import KalmanFrictionFilter

# Hard requirement: fixed seed for full reproducibility, since this test
# gates Milestone 1.
np.random.seed(42)

TRUE_MU = 0.35
NUM_PASSES = 50
PASS_DISTANCE_M = 15.0
INITIAL_VELOCITY_MS = 15.0
MEASUREMENT_NOISE_STD_MS = 0.05  # fixed-magnitude Gaussian noise on final velocity
G = 9.81
ERROR_MARGIN = 0.02  # 2% margin of error on the final posterior estimate


def true_final_velocity(v_initial: float, mu: float, distance: float, g: float) -> float:
    """Noiseless final velocity from v^2 = u^2 - 2*mu*g*d (Cd = 0)."""
    v_squared = v_initial**2 - 2.0 * mu * g * distance
    return np.sqrt(v_squared)


def test_kalman_friction_filter_converges_within_2_percent():
    # Deliberately wrong prior guess (true_mu = 0.35, prior = 0.5) to prove
    # the filter actually corrects toward the truth rather than trivially
    # starting there.
    kf = KalmanFrictionFilter(
        initial_mu=0.5,
        initial_variance=0.1,
        process_noise_q=1e-5,
        measurement_noise_r=0.01,
    )

    noiseless_final_v = true_final_velocity(INITIAL_VELOCITY_MS, TRUE_MU, PASS_DISTANCE_M, G)

    for _ in range(NUM_PASSES):
        # Causal ordering: predict (time update) BEFORE the physics engine /
        # observation for this step is consumed, then correct (measurement
        # update) using the newly observed pass.
        kf.predict()

        noisy_final_v = noiseless_final_v + np.random.normal(0.0, MEASUREMENT_NOISE_STD_MS)
        observed_mu = kf.observe_mu_from_pass(
            v_initial=INITIAL_VELOCITY_MS,
            v_final=noisy_final_v,
            distance=PASS_DISTANCE_M,
        )

        kf.correct(observed_mu)

    relative_error = abs(kf.mu - TRUE_MU) / TRUE_MU
    assert relative_error < ERROR_MARGIN, (
        f"Final posterior mu={kf.mu:.5f} deviates from true_mu={TRUE_MU} "
        f"by {relative_error * 100:.3f}%, exceeding the {ERROR_MARGIN * 100:.2f}% gate"
    )
