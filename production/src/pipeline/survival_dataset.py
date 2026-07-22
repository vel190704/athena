"""Tensor plumbing for survival-analysis training data (Module 7 prep).

Scope note: this module only transforms an ALREADY-COMPUTED tabular
(features, duration, event) possession-level dataset into PyTorch tensors
for a DataLoader. It does NOT derive duration/event from raw StatsBomb
event chains (walking a possession forward to find its terminating shot or
turnover/out-of-bounds) -- that possession-chain labeling logic is a
distinct, harder problem deferred to a follow-up milestone.
"""

import torch
from torch.utils.data import Dataset

# Fixed, documented key order for flattening a feature dict into a tensor.
# Must match the keys produced by
# production/src/pipeline/feature_extractor.py:extract_features.
FEATURE_KEYS = (
    "attacking_control_near_ball",
    "defending_control_near_ball",
    "attacking_control_final_third",
    "space_behind_defending_line",
)


class TacticalSurvivalDataset(Dataset):
    """Wraps precomputed (features, duration, event) possession-level data.

    `durations` and `events` are assumed already derived upstream (e.g. by
    walking possession chains to their terminating shot or turnover -- not
    implemented here); this class only performs the list-of-dicts /
    list-of-scalars -> tensor conversion needed to feed a DataLoader.
    """

    def __init__(self, features: list[dict], durations, events):
        if not (len(features) == len(durations) == len(events)):
            raise ValueError(
                f"features ({len(features)}), durations ({len(durations)}), and "
                f"events ({len(events)}) must all have equal length"
            )
        if any(d <= 0 for d in durations):
            raise ValueError("all durations must be > 0")

        self.features = features
        self.durations = durations
        self.events = events

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        feature_dict = self.features[idx]
        features_tensor = torch.tensor(
            [feature_dict[key] for key in FEATURE_KEYS], dtype=torch.float32
        )  # [num_features]

        # 0-dimensional scalar tensors: the standard convention for most
        # survival-analysis libraries (including pycox's DeepHit), which
        # this project is expected to eventually use.
        duration_tensor = torch.tensor(self.durations[idx], dtype=torch.float32)
        event_tensor = torch.tensor(self.events[idx], dtype=torch.float32)

        return features_tensor, duration_tensor, event_tensor
