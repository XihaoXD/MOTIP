# Copyright (c) Ruopeng Gao. All Rights Reserved.
# FIFO Queue for managing historical trajectory features.
# Adapted from HATReID-MOT: https://github.com/HELLORPG/HATReID-MOT

import torch
from collections import deque


class FIFOQueue:
    """
    First-In-First-Out queue for storing historical ReID features
    of a single trajectory.

    Each trajectory maintains its own FIFOQueue. When fitting the
    FLD model, features from all trajectories' queues are collected
    together with their trajectory IDs.

    The queue supports temporal weight decay: older features get
    exponentially decayed weights, implementing the Temporal-Shifted
    Trajectory Centroid from the HAT paper (Section 3.3).

    Args:
        max_len: Maximum number of features to store per trajectory.
            Corresponds to the history window T in the paper.
        weight_decay_ratio: Decay factor applied to weights at each
            time step. A value of 0.9 means weights decay by 10%
            each step, giving more importance to recent features.
    """

    def __init__(self, max_len: int, weight_decay_ratio: float = 0.9):
        self.max_len = max_len
        self.weight_decay_ratio = weight_decay_ratio
        self.features = deque(maxlen=self.max_len)
        self.weights = deque(maxlen=self.max_len)

    def __len__(self):
        return len(self.features)

    def add(self, feature: torch.Tensor):
        """
        Add a new feature to the queue.

        All existing weights are decayed by weight_decay_ratio before
        the new feature (with weight 1.0) is appended.

        Args:
            feature: A feature vector (1D tensor) to add.
        """
        self.features.append(feature.detach())
        self.weights.append(torch.tensor(1.0))

        # Decay all weights (including the just-added one gets decayed
        # on the next add):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] * self.weight_decay_ratio

    def get(self):
        """
        Get all stored features and their weights.

        Returns:
            Tuple of (features_list, weights_list).
        """
        return list(self.features), list(self.weights)
