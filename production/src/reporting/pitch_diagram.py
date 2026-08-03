"""Milestone 42 (new reporting track, shared helper): matplotlib pitch
outline drawing, shared by `player_visualizer.py` and `team_visualizer.py`.

DELIBERATELY NOT REUSING `production/src/cv/tactical_map_renderer.py`'s
pitch-drawing code (which exists and does the same visual job): that
module pulls its geometry from `pitch_keypoint_detector.PITCH_KEYPOINTS_METERS`,
a CV-track (Module 4) artifact. Reporting-track code (Milestone 40/42) is
built entirely on the StatsBomb track and is documented, repeatedly, as
having NO CV dependency whatsoever (`REPORTING_FINDINGS.md`) -- importing
a CV-track module here to save a few lines of matplotlib primitives would
quietly cross that boundary for no real benefit. This module instead
draws standard, illustrative pitch markings (penalty box/goal box/center
circle proportions) directly with matplotlib patches, using only
`feature_extractor.PITCH_LENGTH`/`PITCH_WIDTH` (already an approved,
StatsBomb-track constant) for scale. These proportions are for VISUAL
REFERENCE ONLY, not calibration geometry -- nothing here feeds into any
computed report value.
"""

from matplotlib import patches

from production.src.pipeline.feature_extractor import PITCH_LENGTH, PITCH_WIDTH

PITCH_FACE_COLOR = "#2e8b3d"
PITCH_LINE_COLOR = "white"
PITCH_LINE_WIDTH = 1.2

# Standard (illustrative, not match-specific) penalty/goal box and center
# circle proportions in meters.
_PENALTY_BOX_DEPTH = 16.5
_PENALTY_BOX_WIDTH = 40.3
_GOAL_BOX_DEPTH = 5.5
_GOAL_BOX_WIDTH = 18.32
_CENTRE_CIRCLE_RADIUS = 9.15


def draw_pitch_outline(ax) -> None:
    """Draws a standard top-down pitch outline on `ax`, in the SAME
    100x68m space every StatsBomb-track module uses (ADR-002). Sets
    `ax`'s limits/aspect so callers can plot directly in pitch-meter
    coordinates afterward.
    """
    ax.set_facecolor(PITCH_FACE_COLOR)

    boundary = patches.Rectangle(
        (0, 0), PITCH_LENGTH, PITCH_WIDTH, fill=False, edgecolor=PITCH_LINE_COLOR, linewidth=PITCH_LINE_WIDTH
    )
    ax.add_patch(boundary)

    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH], color=PITCH_LINE_COLOR, linewidth=PITCH_LINE_WIDTH)
    centre_circle = patches.Circle(
        (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
        _CENTRE_CIRCLE_RADIUS,
        fill=False,
        edgecolor=PITCH_LINE_COLOR,
        linewidth=PITCH_LINE_WIDTH,
    )
    ax.add_patch(centre_circle)

    penalty_y = (PITCH_WIDTH - _PENALTY_BOX_WIDTH) / 2
    goal_y = (PITCH_WIDTH - _GOAL_BOX_WIDTH) / 2
    for x_edge, direction in ((0, 1), (PITCH_LENGTH, -1)):
        penalty_x = x_edge if direction == 1 else x_edge - _PENALTY_BOX_DEPTH
        goal_x = x_edge if direction == 1 else x_edge - _GOAL_BOX_DEPTH
        ax.add_patch(
            patches.Rectangle(
                (penalty_x, penalty_y),
                _PENALTY_BOX_DEPTH,
                _PENALTY_BOX_WIDTH,
                fill=False,
                edgecolor=PITCH_LINE_COLOR,
                linewidth=PITCH_LINE_WIDTH,
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (goal_x, goal_y),
                _GOAL_BOX_DEPTH,
                _GOAL_BOX_WIDTH,
                fill=False,
                edgecolor=PITCH_LINE_COLOR,
                linewidth=PITCH_LINE_WIDTH,
            )
        )

    ax.set_xlim(-2, PITCH_LENGTH + 2)
    ax.set_ylim(-2, PITCH_WIDTH + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
