# Copyright (c) Ruopeng Gao. All Rights Reserved.
"""
Training-time FLD: fit LDA per (batch, group) on visible trajectory points in the clip,
then map LDA outputs to feature_dim with FLDProjector (trainable).

LDA is computed under torch.no_grad (eigenproblem not differentiable); gradients flow through FLDProjector only.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from models.motip.fld_projector import FLDProjector, pad_lda_to_fixed_dim


def _make_lda(device, dtype, fld_use_standard_scaler, fld_direct_inter_class_diff, fld_use_weighted_class_mean):
    from models.hat.lda import LDA

    return LDA(
        use_shrinkage=True,
        dtype=dtype,
        use_standard_scaler=fld_use_standard_scaler,
        direct_inter_class_diff=fld_direct_inter_class_diff,
        use_weighted_class_mean=fld_use_weighted_class_mean,
        device=device,
    )


def apply_fld_to_seq_info_training(
    seq_info: dict,
    fld_projector: FLDProjector,
    num_id_vocabulary: int,
    fld_factor_thr: float,
    fld_use_standard_scaler: bool,
    fld_direct_inter_class_diff: bool,
    fld_use_weighted_class_mean: bool,
) -> Tuple[dict, Optional[torch.Tensor]]:
    """
    Replace trajectory_features / unknown_features with fld_projector(pad(LDA(x))) where LDA is fitted per (b,g).

    Returns:
        new_seq_info: dict (copy with modified feature tensors)
        fld_align_loss: 1 - mean cosine similarity between projected and original unknown features
            (where FLD applied); None if nothing was transformed.
    """
    seq_info = copy.deepcopy(seq_info)
    tf = seq_info["trajectory_features"]
    uf = seq_info["unknown_features"]
    B, G, T, N, D = tf.shape
    device = tf.device
    dtype = tf.dtype
    max_lda = num_id_vocabulary - 1

    traj_mask = seq_info["trajectory_masks"]
    traj_ids = seq_info["trajectory_id_labels"]
    unk_mask = seq_info["unknown_masks"]

    cos_sum = torch.tensor(0.0, device=device, dtype=dtype)
    cos_cnt = 0

    for b in range(B):
        for g in range(G):
            vis = ~traj_mask[b, g]
            coords = []
            feats_list = []
            ids_list = []
            for t in range(T):
                for n in range(N):
                    if not vis[t, n]:
                        continue
                    tid = int(traj_ids[b, g, t, n].item())
                    if tid < 0:
                        continue
                    coords.append((t, n))
                    feats_list.append(tf[b, g, t, n])
                    ids_list.append(tid)
            if len(feats_list) < 2:
                continue
            tfeats = torch.stack(feats_list, dim=0)
            tids = torch.tensor(ids_list, dtype=torch.long, device=device)
            n_tracks = torch.unique(tids).numel()
            if n_tracks < 2:
                continue
            if tfeats.shape[0] <= fld_factor_thr * n_tracks:
                continue

            lda = _make_lda(device, dtype, fld_use_standard_scaler, fld_direct_inter_class_diff, fld_use_weighted_class_mean)
            with torch.no_grad():
                lda.fit(
                    tfeats.detach(),
                    tids.detach(),
                    score=torch.ones(tfeats.shape[0], device=device, dtype=dtype),
                )
                lda_traj = lda.transform(tfeats.detach())
            lda_traj_pad = pad_lda_to_fixed_dim(lda_traj, max_lda)
            proj_traj = fld_projector(lda_traj_pad)
            for k, (t, n) in enumerate(coords):
                tf[b, g, t, n] = proj_traj[k]

            # Unknown: same LDA, transform unknown embeddings
            u_coords = []
            ufeats_list = []
            for t in range(T):
                for n in range(N):
                    if unk_mask[b, g, t, n]:
                        continue
                    u_coords.append((t, n))
                    ufeats_list.append(uf[b, g, t, n])
            if len(ufeats_list) == 0:
                continue
            ufeats = torch.stack(ufeats_list, dim=0)
            with torch.no_grad():
                lda_u = lda.transform(ufeats.detach())
            lda_u_pad = pad_lda_to_fixed_dim(lda_u, max_lda)
            proj_u = fld_projector(lda_u_pad)
            orig_u = ufeats.detach()
            cos = F.cosine_similarity(proj_u, orig_u, dim=-1)
            cos_sum = cos_sum + cos.sum()
            cos_cnt += cos.numel()
            for k, (t, n) in enumerate(u_coords):
                uf[b, g, t, n] = proj_u[k]

    seq_info["trajectory_features"] = tf
    seq_info["unknown_features"] = uf

    if cos_cnt == 0:
        return seq_info, None
    fld_align_loss = 1.0 - (cos_sum / float(cos_cnt))
    return seq_info, fld_align_loss
