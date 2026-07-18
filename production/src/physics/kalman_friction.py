"""Causal scalar Kalman filter for latent rolling friction coefficient mu(t).

Per ADR-008, this filter is validated on synthetic data with a known drag
coefficient (Cd = 0 for ground passes) before real-world aerodynamic noise is
introduced. Per Module 2 / System Assumption A2, mu(t) is assumed to vary
slowly compared to a single pass duration, so it is modeled as a scalar
random walk: mu_t = mu_{t-1} + w, w ~ N(0, process_noise_q).

Causal ordering (mandatory, see ADR-008 and Section 3 Module 2):
    predict() -> physics engine consumes self.mu -> correct()
The physics engine must never observe a mu that has already been updated
with the current step's own measurement.
"""


class KalmanFrictionFilter:
    """Scalar Kalman filter tracking the latent rolling friction coefficient mu(t).

    Q (process_noise_q) and R (measurement_noise_r) are tunable
    hyperparameters, not hardcoded constants: they will need sweeping once
    real StatsBomb data replaces synthetic data in later milestones.
    """

    def __init__(
        self,
        initial_mu: float,
        initial_variance: float,
        process_noise_q: float = 1e-5,
        measurement_noise_r: float = 0.01,
        g: float = 9.81,
    ):
        """
        Args:
            initial_mu: filter's starting belief about mu.
            initial_variance: filter's starting uncertainty about mu.
            process_noise_q: variance of the per-step process noise w in the
                random-walk transition mu_t = mu_{t-1} + w. Default is small
                (1e-5), reflecting the slow-drift assumption (A2).
            measurement_noise_r: variance of the observation noise on mu
                itself. Default 0.01. Note: the observed mu passed to
                correct() is derived from a ratio of two noisy velocity
                measurements (see observe_mu_from_pass), so its effective
                noise is not strictly Gaussian even if the underlying
                velocity measurement noise is. R may therefore need
                empirical retuning against real data rather than a
                closed-form derivation from velocity sensor noise.
            g: gravitational acceleration (m/s^2). Exposed as a constructor
                arg, not hardcoded in the update equations, so the filter
                can be reused for other surfaces/units later.
        """
        self.mu = initial_mu
        self.variance = initial_variance
        self.process_noise_q = process_noise_q
        self.measurement_noise_r = measurement_noise_r
        self.g = g

    def predict(self) -> None:
        """Time Update: project mu forward to time t.

        State transition is the identity mu_t = mu_{t-1} + w with
        E[w] = 0, so the mean estimate self.mu is unchanged; only the
        uncertainty grows by process_noise_q. Must be called before the
        physics engine consumes self.mu for the current step, and before
        correct() is called with the current step's observation.
        """
        self.variance = self.variance + self.process_noise_q

    def correct(self, observed_mu: float) -> None:
        """Measurement Update: incorporate a new observed mu.

        Computes the Kalman gain from measurement_noise_r and the current
        (post-predict) variance, then updates self.mu and self.variance.
        Must be called only after predict() and after the physics engine
        has already consumed the predicted (pre-correction) self.mu for
        this step.
        """
        kalman_gain = self.variance / (self.variance + self.measurement_noise_r)
        self.mu = self.mu + kalman_gain * (observed_mu - self.mu)
        self.variance = (1.0 - kalman_gain) * self.variance

    def observe_mu_from_pass(self, v_initial: float, v_final: float, distance: float) -> float:
        """Compute the observed rolling friction coefficient from a ground pass.

        Derived from v^2 = u^2 - 2*mu*g*d (constant deceleration kinematics),
        solved for mu: mu = (u^2 - v^2) / (2*g*d).

        Aerodynamic drag is ignored entirely for this v1.0 synthetic baseline
        (ground passes only, per ADR-008 / System Assumption A3).

        Args:
            v_initial: initial (struck) velocity, u, in m/s.
            v_final: observed final velocity, v, in m/s.
            distance: distance traveled, d, in meters.

        Returns:
            The observed friction coefficient mu implied by this single pass.
        """
        return (v_initial**2 - v_final**2) / (2.0 * self.g * distance)
