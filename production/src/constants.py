"""Project Athena -- shared, dependency-free constants used across several
otherwise-independent modules (`production/src/pipeline`, `production/src/
models`, `production/src/serving`).

Deliberately imports NOTHING from this project and NOTHING heavy (no
torch, no pandas, no mlflow) -- this is precisely why these values were
previously duplicated independently in each consuming module rather than
imported from one another. `explainer.py`'s own prior comment said it
plainly: `BIN_SIZE_SECONDS` was "kept local to avoid a training-pipeline
import." A plain-values-only module like this one can be imported from
anywhere with zero circular-import or heavy-dependency risk, which is the
whole reason to centralize here instead of having each module import from
whichever other module happened to define the value first.

Engineering-review action item: closes the `TIME_BIN`/`BIN_SIZE_SECONDS`/
`MLFLOW_EXPERIMENT_NAME` duplication found across `oracle_validator.py`,
`api.py`, `survival_dataset.py`, `explainer.py`, and `train.py`. Every
value here is UNCHANGED from what each of those files already
independently defined -- this is a pure de-duplication, not a value
change; no computed output or model behavior differs because of it.
"""

# The DeepHit discrete-time bin INDEX this project's headline "Brier@15s"
# convention (Milestones 8/13/14/15) has used since Milestone 8 -- combined
# with BIN_SIZE_SECONDS below, TIME_BIN * BIN_SIZE_SECONDS = 15 seconds.
# Previously redefined independently in `oracle_validator.py` and `api.py`.
TIME_BIN = 3

# Milestone 6A convention: each DeepHit discrete-time bin spans this many
# seconds. Previously redefined independently in `explainer.py` and
# `survival_dataset.py` (which additionally derives `NUM_BINS` from it --
# see that module; `NUM_BINS`/`MAX_DURATION_SECONDS` are not duplicated
# elsewhere, so they stay defined there, not here).
BIN_SIZE_SECONDS = 5.0

# The single MLflow experiment name every training/logging/lookup call in
# this project reads from or writes to. Previously redefined independently
# in `explainer.py` and `train.py`.
MLFLOW_EXPERIMENT_NAME = "project-athena-deephit"
