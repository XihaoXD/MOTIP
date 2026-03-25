# Copyright (c) Ruopeng Gao. All Rights Reserved.
"""
Learnable upsampling: LDA subspace (at most num_id_vocabulary-1 dims) -> DETR feature_dim.

Use with FLD: pad LDA output to fixed width, then Linear+norm so ID decoder sees 256-d trainable features.
"""

import torch
import torch.nn as nn


class FLDProjector(nn.Module):
    def __init__(self, lda_input_dim: int, feature_dim: int):
        """
        Args:
            lda_input_dim: Fixed max LDA width = num_id_vocabulary - 1 (pad shorter LDA outputs with zeros).
            feature_dim: Same as DETR / trajectory feature dim (e.g. 256).
        """
        super().__init__()
        self.lda_input_dim = lda_input_dim
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(lda_input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_lda_padded: torch.Tensor) -> torch.Tensor:
        """x_lda_padded: (*, lda_input_dim) -> (*, feature_dim)"""
        return self.net(x_lda_padded)


def pad_lda_to_fixed_dim(lda_out: torch.Tensor, fixed_dim: int) -> torch.Tensor:
    """Pad or truncate last dim of LDA output to fixed_dim."""
    d = lda_out.shape[-1]
    if d == fixed_dim:
        return lda_out
    if d > fixed_dim:
        return lda_out[..., :fixed_dim]
    return torch.nn.functional.pad(lda_out, (0, fixed_dim - d), value=0.0)
