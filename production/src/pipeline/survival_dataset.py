"""Tensor plumbing for DeepHit survival-analysis training data (Module 7
prep), now serving BOTH the scalar-feature MLP and the graph-based GNN
(Milestone 12, RQ4) from the exact same underlying observation per sample.

Scope note: this module transforms an ALREADY-COMPUTED possession-chain
dataset -- scalar features (Milestone 3), raw 360 frame tensors (Milestone
3's parse_360_frame), and chain dicts (Milestone 5's
build_possession_chains) -- into discretized-time PyTorch/PyG tensors for a
DataLoader. It does not derive duration/event/censor_reason itself; that
possession-chain labeling logic lives in chain_builder.py. It also does not
resolve which 360 frame represents a chain -- that resolution (Milestone
7's "first event with an associated 360 freeze-frame" logic) happens once,
upstream in train.py, and its single result is what both the scalar
features and the raw frame tensors here must come from -- see
TacticalSurvivalDataset's docstring for why that matters.
"""

import torch
from torch.utils.data import Dataset

from production.src.models.graph_builder import build_graph_from_frame

# Fixed, documented key order for flattening a feature dict into a tensor.
# Must match the keys produced by
# production/src/pipeline/feature_extractor.py:extract_features.
FEATURE_KEYS = (
    "attacking_control_near_ball",
    "defending_control_near_ball",
    "attacking_control_final_third",
    "space_behind_defending_line",
)

# DeepHit (ADR-001) requires discretizing time into fixed-width bins. A
# 60-second horizon in 5-second bins gives 12 bins. These are tunable
# hyperparameters to be validated against real possession-duration
# statistics in a later milestone, not derived constants.
MAX_DURATION_SECONDS = 60.0
BIN_SIZE_SECONDS = 5.0
NUM_BINS = int(MAX_DURATION_SECONDS // BIN_SIZE_SECONDS)  # 12


class TacticalSurvivalDataset(Dataset):
    """Wraps (scalar features, raw 360 frame, chain) possession-level
    triples into discretized-time DeepHit training tensors, serving both a
    scalar-feature representation (for the MLP) and a graph representation
    (for the GNN) built from the SAME underlying frame.

    `features`, `frames`, and `chains` are three parallel, same-order,
    same-length lists: the i-th scalar feature dict, the i-th raw parsed
    360 frame dict (statsbomb_io.parse_360_frame's output -- must contain
    `player_pos`, `player_vel`, `is_teammate`, `ball_pos`), and the i-th
    chain dict (chain_builder.py's build_possession_chains output) must all
    describe the SAME possession chain, resolved from the SAME
    representative event.

    CRITICAL for RQ4 validity: `features[i]` and `frames[i]` MUST be
    derived from the exact same resolved 360-freeze-frame event, not
    independently re-looked-up -- if the scalar-feature extraction and the
    graph construction ended up pulling from different events for the same
    chain, the MLP-vs-GNN comparison would silently be comparing different
    observations under the same label. This class trusts its caller
    (train.py) to have resolved each chain's representative event ONCE and
    passed the SAME frame dict into both `extract_features` (producing
    `features[i]`) and here (`frames[i]`) -- it does not and cannot verify
    this itself, since by the time frames arrive here they're already
    independent dicts with no back-reference to a shared event id.
    """

    def __init__(self, features: list[dict], frames: list[dict], chains: list[dict]):
        if not (len(features) == len(frames) == len(chains)):
            raise ValueError(
                f"features ({len(features)}), frames ({len(frames)}), and chains "
                f"({len(chains)}) must all have equal length"
            )

        self.features = features
        self.frames = frames

        durations = []
        event_flags = []
        horizon_censored_count = 0

        for chain in chains:
            # Milestone 5 found real chains with duration_seconds == 0 (a
            # legitimate consequence of StatsBomb's 1-second minute/second
            # timestamp resolution for very fast phases, not a bug). Floor
            # to a tiny positive epsilon BEFORE validation so those chains
            # remain usable instead of being rejected outright.
            raw_duration = max(1.0, chain["duration_seconds"])

            event_flag = chain["event_flag"]
            # Horizon-censoring: the model only reasons within
            # MAX_DURATION_SECONDS (NUM_BINS bins). A chain whose real
            # duration exceeds that horizon cannot be credited with an
            # observed event even if `event_flag` says a shot occurred,
            # because the model is never shown a time bin past the
            # horizon to place that probability mass on. This is a
            # deliberate modeling decision, not an incidental clamp: such
            # samples are treated as right-censored at the horizon, same
            # as any other administrative censoring reason.
            if raw_duration >= MAX_DURATION_SECONDS:
                if event_flag == 1:
                    horizon_censored_count += 1
                event_flag = 0

            durations.append(raw_duration)
            event_flags.append(event_flag)

        if horizon_censored_count > 0:
            print(
                f"[TacticalSurvivalDataset] {horizon_censored_count} of {len(chains)} chains "
                f"had a real shot beyond the {MAX_DURATION_SECONDS}s horizon; event_flag "
                "forced to 0 (horizon-censored)."
            )

        if any(d <= 0 for d in durations):
            raise ValueError("all durations must be > 0")

        self.durations = durations
        self.events = event_flags
        self.duration_bins = [min(int(d // BIN_SIZE_SECONDS), NUM_BINS - 1) for d in durations]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        feature_dict = self.features[idx]
        features_tensor = torch.tensor(
            [feature_dict[key] for key in FEATURE_KEYS], dtype=torch.float32
        )  # [num_features]

        frame = self.frames[idx]
        graph_data = build_graph_from_frame(
            frame["player_pos"], frame["player_vel"], frame["is_teammate"], frame["ball_pos"]
        )

        duration_bin_tensor = torch.tensor(self.duration_bins[idx], dtype=torch.long)
        event_tensor = torch.tensor(self.events[idx], dtype=torch.float32)

        return features_tensor, graph_data, duration_bin_tensor, event_tensor
