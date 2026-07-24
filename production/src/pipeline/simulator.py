"""Tactical action counterfactual perturbation engine (Milestone 13,
Module 8 / RQ5).

Perturbs a real baseline feature vector to approximate the spatial effect
of a hypothetical tactical action (forcing play wide, pressing high,
dropping deep), so the resulting shift can be fed through the trained
DeepHit MLP and compared against real football intuition (RQ5) -- an open
research question this module reports evidence toward, not an engineering
assertion it enforces.

IMPORTANT CAVEATS (read before trusting simulator output):
- The percentages below are hand-chosen heuristic approximations of
  tactical shifts, NOT empirically calibrated from real pressing/
  dropping-deep event data. If this simulator becomes a load-bearing part
  of the project (rather than an exploratory tool), calibrating these
  numbers against real tactical-change event data is a candidate for a
  future ADR.
- Each action perturbs only 1-2 of the 4 features independently. Real
  tactical shifts likely move several features together in correlated ways
  not modeled here (e.g. a high press plausibly also compresses
  attacking_control_final_third, not just attacking_control_near_ball and
  space_behind_defending_line). A perturbed feature vector may therefore
  represent a feature combination the model never saw during training --
  an OUT-OF-DISTRIBUTION input, not a realistic match state. This matters
  for how much weight to put on the resulting predictions.
"""

FEATURE_KEYS = (
    "attacking_control_near_ball",
    "defending_control_near_ball",
    "attacking_control_final_third",
    "space_behind_defending_line",
)

SUPPORTED_ACTIONS = ("no_change", "force_wide", "high_press", "drop_deep")


def perturb_features(base_features: dict, action: str) -> dict:
    """Return a NEW feature dict representing `base_features` after a
    hypothetical tactical action. `base_features` is never mutated.

    Supported actions:
      - 'no_change': baseline, unperturbed.
      - 'force_wide': attacking_control_near_ball x0.80,
        attacking_control_final_third x0.90, space_behind_defending_line x1.15.
      - 'high_press': attacking_control_near_ball x1.20,
        space_behind_defending_line x0.70.
      - 'drop_deep': space_behind_defending_line x1.40,
        defending_control_near_ball x1.20.
    """
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"unknown action {action!r}; supported: {SUPPORTED_ACTIONS}")

    features = dict(base_features)

    if action == "no_change":
        return features

    if action == "force_wide":
        features["attacking_control_near_ball"] *= 0.80
        features["attacking_control_final_third"] *= 0.90
        features["space_behind_defending_line"] *= 1.15
    elif action == "high_press":
        features["attacking_control_near_ball"] *= 1.20
        features["space_behind_defending_line"] *= 0.70
    elif action == "drop_deep":
        features["space_behind_defending_line"] *= 1.40
        features["defending_control_near_ball"] *= 1.20

    return features
