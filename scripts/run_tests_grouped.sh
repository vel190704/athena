#!/usr/bin/env bash
# Runs production/tests/ as six SEPARATE `pytest` processes (one per
# logical group below) instead of one single `pytest production/tests/`
# invocation covering all ~44 files/220+ tests at once.
#
# WHY (Aug 2026 OOM incident): running the whole suite as one process hit
# a real OOM kill twice in one night at ~9.2GB peak RSS -- PyTorch, MLflow,
# a YOLO checkpoint, PyTorch Geometric, and matplotlib all end up loaded
# SIMULTANEOUSLY within that one process by the time it's worked through
# every test file, and none of that ever gets returned to the OS mid-run
# (Python's own allocator, and especially PyTorch's caching allocator,
# both hold onto freed memory for reuse rather than releasing it back).
#
# Investigation found TWO real, separately-measured contributors, in this
# order of impact:
# 1. (Secondary, ~18% of one file's own peak) api.py's `lifespan` was
#    being re-triggered up to 36 times within one pytest session by
#    repeated `with TestClient(app) as client:` blocks in test_api.py
#    alone, needlessly reloading the same deterministic model/checkpoint
#    every time -- now guarded (see api.py's `lifespan` and
#    `_yolo_checkpoint_warmed`). Measured: test_api.py's own peak RSS
#    dropped ~18% (1.65GB -> 1.34GB) from this fix alone. Real, but not
#    close to sufficient by itself.
# 2. (Dominant) `test_dashboard.py` is, ON ITS OWN, an ~8.5GB-peak file --
#    ~30 tests, most doing a full Streamlit AppTest script re-execution
#    (sometimes several `.run()` calls per test) plus real HTTP round
#    trips against a real uvicorn server (the `live_api_server` fixture)
#    that stays resident for the whole file. Isolating it into its own
#    group (below) did NOT meaningfully lower ITS OWN peak (measured
#    8.46GB alone vs. 8.27GB when combined with three other, much
#    lighter, serving-track files) -- what isolation DOES buy is removing
#    the OTHER files' overhead from stacking on top of that already-high
#    baseline (the combined-group run had 6 test failures, reproduced as
#    genuine resource-pressure flakes -- each passed cleanly re-run in
#    isolation; the isolated `dashboard` group run had zero failures).
#    Splitting `test_dashboard.py` into smaller sub-batches of its own
#    ~30 tests is a real, not-yet-taken next step if this file's own
#    ~8.5GB baseline ever becomes a problem again on a lower-memory
#    machine or under heavier concurrent desktop load -- named here
#    explicitly, not silently left as "solved."
#
# Splitting into separate OS PROCESSES is what actually bounds peak
# memory ACROSS groups: each group's process exits and the OS reclaims
# everything before the next group starts, the same way closing and
# reopening a terminal reclaims a shell's own accumulated state. It does
# NOT shrink any one file's own peak if that file is heavy on its own
# (see `dashboard`, above) -- know that limit before assuming this script
# makes memory pressure a fully solved problem.
#
# Usage: scripts/run_tests_grouped.sh [pytest args...]
#   scripts/run_tests_grouped.sh              # normal run, -q by default
#   scripts/run_tests_grouped.sh -v            # verbose, forwarded to every group
#
# Exits non-zero if ANY group fails (matching plain `pytest`'s own exit
# code convention), after running every group regardless of earlier
# failures (so one broken group doesn't hide results from the other four).

set -u
cd "$(dirname "$0")/.." || exit 1

EXTRA_ARGS=("$@")
if [ ${#EXTRA_ARGS[@]} -eq 0 ]; then
    EXTRA_ARGS=("-q")
fi

# Group membership: every file in production/tests/ appears in EXACTLY
# one group below (verified by diff against a full `ls production/tests/*.py`
# listing when this script was written -- 44 files, 6 groups, no gaps, no
# duplicates). Grouped by which `production/src/` subsystem each file's
# own imports are primarily about, per this project's own directory
# layout (cv/, models/, pipeline/, reporting/, serving/) -- not arbitrary.
#
# `serving` was originally ONE group (test_api.py, test_alert_store.py,
# test_dashboard.py, test_simulate_api.py together) -- split into
# `serving-api` and `dashboard` after the FIRST real validation run of
# this script measured that combined group peaking at 8.27GB (nearly the
# original whole-suite OOM's 9.2GB) and causing a real, reproduced-in-
# isolation-as-passing test flake under that memory pressure.
# test_dashboard.py alone is the dominant cost (many AppTest reruns, each
# a full Streamlit script re-execution, several against a REAL uvicorn
# server via the `live_api_server` fixture) -- isolated into its own
# group rather than combined with anything else.
GROUP_CV="test_adapter.py test_ball_detector.py test_calibration.py test_camera_motion.py test_cv_detector.py test_cv_pipeline.py test_cv_tracker.py test_shot_classifier.py test_tactical_map_renderer.py test_team_classifier.py"
GROUP_MODELS="test_deephit.py test_deephit_loss.py test_direction.py test_explainer.py test_friction.py test_gnn_model.py test_gnn_simulator.py test_graph_builder.py test_instability_detector.py test_spatial.py test_uncertainty.py test_dataset.py test_dataset_graph_integration.py"
GROUP_PIPELINE="test_chain_builder.py test_data_split.py test_feature_extraction.py test_habit_memory.py test_simulator.py test_mlflow.py test_oracle.py"
GROUP_REPORTING="test_candidate_index.py test_player_comparison.py test_player_dashboard.py test_report_visualizer.py test_reporting.py test_shot_map.py test_team_comparison.py test_team_trend_data.py test_team_trend_visualizer.py test_zone_explainer.py"
GROUP_SERVING_API="test_api.py test_alert_store.py test_simulate_api.py"
GROUP_DASHBOARD="test_dashboard.py"

GROUP_NAMES=("cv" "models" "pipeline" "reporting" "serving-api" "dashboard")
GROUP_FILES=("$GROUP_CV" "$GROUP_MODELS" "$GROUP_PIPELINE" "$GROUP_REPORTING" "$GROUP_SERVING_API" "$GROUP_DASHBOARD")

overall_exit=0
summary_lines=()

for i in "${!GROUP_NAMES[@]}"; do
    name="${GROUP_NAMES[$i]}"
    files="${GROUP_FILES[$i]}"
    # shellcheck disable=SC2206
    file_paths=()
    for f in $files; do
        file_paths+=("production/tests/$f")
    done

    echo ""
    echo "================================================================"
    echo "GROUP: $name (${#file_paths[@]} files)"
    echo "================================================================"
    echo "-- memory before this group's process starts --"
    free -h 2>/dev/null || true

    # /usr/bin/time -v (if present) reports this GROUP's own peak RSS --
    # the actual number that matters for confirming process isolation is
    # working (each group's process exiting and being reaped is what
    # returns this memory to the OS before the NEXT group starts, unlike
    # one single process covering every group).
    if command -v /usr/bin/time >/dev/null 2>&1; then
        /usr/bin/time -v python -m pytest "${file_paths[@]}" "${EXTRA_ARGS[@]}"
        group_exit=$?
    else
        python -m pytest "${file_paths[@]}" "${EXTRA_ARGS[@]}"
        group_exit=$?
    fi

    if [ $group_exit -ne 0 ]; then
        overall_exit=1
        summary_lines+=("  $name: FAILED (exit $group_exit)")
    else
        summary_lines+=("  $name: passed")
    fi
done

echo ""
echo "================================================================"
echo "OVERALL SUMMARY (6 separate processes -- each group's own pass/fail"
echo "counts are in its own section above; this is just pass/fail per group)"
echo "================================================================"
for line in "${summary_lines[@]}"; do
    echo "$line"
done

exit $overall_exit
