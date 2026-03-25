# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn.functional as F
import einops
from scipy.optimize import linear_sum_assignment

from structures.instances import Instances
from structures.ordered_set import OrderedSet
from utils.misc import distributed_device
from utils.box_ops import box_cxcywh_to_xywh
from models.misc import get_model
from typing import Optional


def resolve_tracking_mode(
        tracking_mode: Optional[str],
        use_motip: bool,
        use_hat: bool,
        use_fld: bool,
        primary_id_source: str,
) -> tuple[bool, bool, bool, str]:
    """
    Ablation modes (mutually exclusive when ``tracking_mode`` is set):
      - ``motip``: pure MOTIP (ID Decoder), MOTIP trajectory, no FLD before decoder.
      - ``motip_fld``: DETR features → FLD(LDA) → ID Decoder, MOTIP trajectory.
      - ``hat``: DETR + HAT (LDA+match, no ID Decoder), HAT trajectory (id_queue_hat / id_label_to_id_hat).

    When ``tracking_mode`` is None, legacy flags ``use_motip`` / ``use_hat`` / ``use_fld`` apply (incl. dual-branch).
    """
    if tracking_mode is None or str(tracking_mode).strip() == "":
        return use_motip, use_hat, use_fld, primary_id_source

    mode = str(tracking_mode).strip().lower()
    if mode == "motip":
        return True, False, False, "motip"
    if mode in ("motip_fld", "motip-fld", "fld_motip"):
        return True, False, True, "motip"
    if mode == "hat":
        return False, True, False, "hat"
    raise ValueError(
        f"Unknown TRACKING_MODE={tracking_mode!r}. "
        "Use 'motip' | 'motip_fld' | 'hat', or omit for legacy USE_MOTIP/USE_HAT."
    )


class RuntimeTracker:
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "hungarian",
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
            # Ablation: mutually exclusive modes (overrides use_motip/use_hat/use_fld when set). See resolve_tracking_mode.
            tracking_mode: Optional[str] = None,
            # MOTIP vs HAT: which ID branch(es) to run (after DETR). Legacy if tracking_mode is None.
            use_motip: bool = False,
            use_hat: bool = True,
            primary_id_source: str = "motip",  # which branch updates trajectory when both are on
            # FLD settings (used inside MOTIP's _get_id_pred_labels when use_fld):
            use_fld: bool = True,
            fld_hist_len: int = 60,
            fld_factor_thr: float = 4.0,
            fld_similarity_alpha: float = 1.0,
            fld_use_standard_scaler: bool = False,
            fld_direct_inter_class_diff: bool = True,
            fld_use_weighted_class_mean: bool = True,
            fld_transfer_dtype: torch.dtype = torch.float32,
            # HAT (standalone LDA+matching) settings:
            hat_hist_len: int = 60,
            hat_factor_thr: float = 4.0,
            hat_similarity_alpha: float = 1.0,
            hat_use_standard_scaler: bool = False,
            hat_direct_inter_class_diff: bool = True,
            hat_use_weighted_class_mean: bool = True,
            hat_weight_decay: float = 0.9,
            hat_transfer_dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype

        use_motip, use_hat, use_fld, primary_id_source = resolve_tracking_mode(
            tracking_mode, use_motip, use_hat, use_fld, primary_id_source,
        )
        self.tracking_mode = (
            str(tracking_mode).strip().lower()
            if tracking_mode is not None and str(tracking_mode).strip()
            else None
        )

        # For FP16:
        if self.dtype != torch.float32:
            if self.dtype == torch.float16:
                self.model.half()
            else:
                raise NotImplementedError(f"Unsupported dtype {self.dtype}.")

        self.use_sigmoid = use_sigmoid
        self.assignment_protocol = assignment_protocol.lower()
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.area_thresh = area_thresh
        self.only_detr = only_detr
        self.num_id_vocabulary = get_model(model).num_id_vocabulary

        # MOTIP vs HAT:
        self.use_motip = use_motip
        self.use_hat = use_hat
        self.primary_id_source = primary_id_source.lower()
        assert self.primary_id_source in ("motip", "hat"), f"primary_id_source must be 'motip' or 'hat', got {primary_id_source}"
        if not self.use_motip and not self.use_hat and not self.only_detr:
            raise ValueError("At least one of use_motip or use_hat must be True when only_detr is False.")

        # FLD settings (MOTIP branch):
        self.use_fld = use_fld
        self.fld_hist_len = fld_hist_len
        self.fld_factor_thr = fld_factor_thr
        self.fld_similarity_alpha = fld_similarity_alpha
        self.fld_use_standard_scaler = fld_use_standard_scaler
        self.fld_direct_inter_class_diff = fld_direct_inter_class_diff
        self.fld_use_weighted_class_mean = fld_use_weighted_class_mean
        self.fld_transfer_dtype = fld_transfer_dtype

        # HAT (standalone LDA) settings:
        self.hat_hist_len = hat_hist_len
        self.hat_factor_thr = hat_factor_thr
        self.hat_similarity_alpha = hat_similarity_alpha
        self.hat_use_standard_scaler = hat_use_standard_scaler
        self.hat_direct_inter_class_diff = hat_direct_inter_class_diff
        self.hat_use_weighted_class_mean = hat_use_weighted_class_mean
        self.hat_weight_decay = hat_weight_decay
        self.hat_transfer_dtype = hat_transfer_dtype

        # Check for the legality of settings:
        assert self.assignment_protocol in ["hungarian", "id-max", "object-max", "object-priority", "id-priority"], \
            f"Assignment protocol {self.assignment_protocol} is not supported."

        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=dtype,
            device=distributed_device(),
        )

        # Trajectory fields:
        self.next_id = 0
        self.id_label_to_id = {}
        self.id_queue = OrderedSet()
        # Init id_queue:
        for i in range(self.num_id_vocabulary):
            self.id_queue.add(i)
        # All fields are in shape (T, N, ...)
        self.trajectory_features = torch.zeros(
            (0, 0, 256), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_boxes = torch.zeros(
            (0, 0, 4), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_id_labels = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_times = torch.zeros(
            (0, 0), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_masks = torch.zeros(
            (0, 0), dtype=torch.bool, device=distributed_device(),
        )

        # Shared history queues for FLD (MOTIP) and/or HAT. Updated by primary branch only when both are on.
        if self.use_fld or self.use_hat:
            from models.hat.fifo_queue import FIFOQueue
            self.fld_history_queues = {}

        # HAT-only ID state (used when both use_motip and use_hat, so HAT has its own global ID space).
        self.next_id_hat = 0
        self.id_label_to_id_hat = {}
        self.id_queue_hat = OrderedSet()
        for i in range(self.num_id_vocabulary):
            self.id_queue_hat.add(i)

        self.current_track_results = {}
        self.current_track_results_motip = {}
        self.current_track_results_hat = {}
        return

    @staticmethod
    def _pad_features_to_detr_dim(feat: torch.Tensor, detr_dim: int) -> torch.Tensor:
        """
        LDA projects to (num_classes - 1) dims, but trajectory_modeling / id_decoder expect DETR feature dim (e.g. 256).
        Zero-pad the last dimension so pretrained FFN/LayerNorm shapes match.
        """
        d = feat.shape[-1]
        if d == detr_dim:
            return feat
        if d > detr_dim:
            return feat[..., :detr_dim]
        return torch.nn.functional.pad(feat, (0, detr_dim - d), value=0.0)

    def _fld_project_or_pad(self, lda_out: torch.Tensor, detr_dim: int) -> torch.Tensor:
        """If model has trained FLDProjector, map LDA->feature_dim; else zero-pad to detr_dim."""
        m = get_model(self.model)
        proj = getattr(m, "fld_projector", None)
        if proj is not None:
            from models.motip.fld_projector import pad_lda_to_fixed_dim
            lda_in = proj.lda_input_dim
            p = pad_lda_to_fixed_dim(lda_out, lda_in)
            return proj(p)
        return self._pad_features_to_detr_dim(lda_out, detr_dim)

    def _filter_and_assign(
            self,
            scores, categories, boxes, output_embeds, id_pred_labels,
            id_queue, assign_fn,
    ):
        """Filter low-conf newborns, truncate if needed, assign newborn IDs; returns (scores, categories, boxes, output_embeds, id_labels)."""
        keep_idxs = (id_pred_labels != self.num_id_vocabulary) | (scores > self.newborn_thresh)
        scores = scores[keep_idxs]
        categories = categories[keep_idxs]
        boxes = boxes[keep_idxs]
        output_embeds = output_embeds[keep_idxs]
        id_pred_labels = id_pred_labels[keep_idxs]

        n_activate = 0
        n_newborn = 0
        for _ in range(len(id_pred_labels)):
            if id_pred_labels[_].item() != self.num_id_vocabulary:
                n_activate += 1
                id_queue.add(id_pred_labels[_].item())
            else:
                n_newborn += 1

        n_remaining = len(id_queue) - n_activate
        if n_newborn > n_remaining:
            keep_idxs = torch.ones(len(id_pred_labels), dtype=torch.bool, device=id_pred_labels.device)
            newborn_idxs = (id_pred_labels == self.num_id_vocabulary)
            newborn_keep = torch.ones(len(newborn_idxs), dtype=torch.bool, device=newborn_idxs.device)
            newborn_keep[n_remaining:] = False
            keep_idxs[newborn_idxs] = newborn_keep
            scores = scores[keep_idxs]
            categories = categories[keep_idxs]
            boxes = boxes[keep_idxs]
            output_embeds = output_embeds[keep_idxs]
            id_pred_labels = id_pred_labels[keep_idxs]

        id_labels = assign_fn(pred_id_labels=id_pred_labels)
        return scores, categories, boxes, output_embeds, id_labels

    @torch.no_grad()
    def update(self, image):
        detr_out = self.model(frames=image, part="detr")
        scores, categories, boxes, output_embeds = self._get_activate_detections(detr_out=detr_out)

        id_pred_motip = None
        id_pred_hat = None
        if self.only_detr:
            id_pred_single = self.num_id_vocabulary * torch.ones(
                boxes.shape[0], dtype=torch.int64, device=boxes.device
            )
        else:
            if self.use_motip:
                id_pred_motip = self._get_id_pred_labels(boxes=boxes, output_embeds=output_embeds)
            if self.use_hat:
                id_pred_hat = self._get_id_pred_labels_hat(boxes=boxes, output_embeds=output_embeds)

        # Single branch (only_detr, or only MOTIP, or only HAT)
        if self.only_detr:
            scores_f, categories_f, boxes_f, output_embeds_f, id_labels_f = self._filter_and_assign(
                scores, categories, boxes, output_embeds, id_pred_single,
                self.id_queue, self._assign_newborn_id_labels,
            )
            self.current_track_results = {
                "score": scores_f, "category": categories_f, "bbox": box_cxcywh_to_xywh(boxes_f) * self.bbox_unnorm,
                "id": torch.tensor([self.id_label_to_id[_] for _ in id_labels_f.tolist()], dtype=torch.int64),
            }
            for _ in range(len(id_labels_f)):
                self.id_queue.add(id_labels_f[_].item())
            self._update_trajectory_infos(boxes=boxes_f, output_embeds=output_embeds_f, id_labels=id_labels_f)
            self._filter_out_inactive_tracks()
            self.current_track_results_motip = {}
            self.current_track_results_hat = {}
            return

        if self.use_motip and not self.use_hat:
            scores_f, categories_f, boxes_f, output_embeds_f, id_labels_f = self._filter_and_assign(
                scores, categories, boxes, output_embeds, id_pred_motip,
                self.id_queue, self._assign_newborn_id_labels,
            )
            self.current_track_results = {
                "score": scores_f, "category": categories_f, "bbox": box_cxcywh_to_xywh(boxes_f) * self.bbox_unnorm,
                "id": torch.tensor([self.id_label_to_id[_] for _ in id_labels_f.tolist()], dtype=torch.int64),
            }
            self.current_track_results_motip = self.current_track_results
            self.current_track_results_hat = {}
            for _ in range(len(id_labels_f)):
                self.id_queue.add(id_labels_f[_].item())
            self._update_trajectory_infos(boxes=boxes_f, output_embeds=output_embeds_f, id_labels=id_labels_f)
            self._filter_out_inactive_tracks()
            return

        if self.use_hat and not self.use_motip:
            scores_f, categories_f, boxes_f, output_embeds_f, id_labels_f = self._filter_and_assign(
                scores, categories, boxes, output_embeds, id_pred_hat,
                self.id_queue_hat, self._assign_newborn_id_labels_hat,
            )
            for _ in range(len(id_labels_f)):
                self.id_queue_hat.add(id_labels_f[_].item())
            for _ in id_labels_f.tolist():
                if _ not in self.id_label_to_id_hat:
                    self.id_label_to_id_hat[_] = self.next_id_hat
                    self.next_id_hat += 1
            self.current_track_results = {
                "score": scores_f, "category": categories_f, "bbox": box_cxcywh_to_xywh(boxes_f) * self.bbox_unnorm,
                "id": torch.tensor([self.id_label_to_id_hat[_] for _ in id_labels_f.tolist()], dtype=torch.int64),
            }
            self.current_track_results_motip = {}
            self.current_track_results_hat = self.current_track_results
            self._update_trajectory_infos(boxes=boxes_f, output_embeds=output_embeds_f, id_labels=id_labels_f)
            self._filter_out_inactive_tracks()
            return

        # Both MOTIP and HAT
        scores_m, categories_m, boxes_m, output_embeds_m, id_labels_motip = self._filter_and_assign(
            scores, categories, boxes, output_embeds, id_pred_motip,
            self.id_queue, self._assign_newborn_id_labels,
        )
        for _ in range(len(id_labels_motip)):
            self.id_queue.add(id_labels_motip[_].item())

        scores_h, categories_h, boxes_h, output_embeds_h, id_labels_hat = self._filter_and_assign(
            scores, categories, boxes, output_embeds, id_pred_hat,
            self.id_queue_hat, self._assign_newborn_id_labels_hat,
        )
        for _ in range(len(id_labels_hat)):
            self.id_queue_hat.add(id_labels_hat[_].item())

        # Ensure every HAT id_label has a global ID (re-matched tracks were not in id_label_to_id_hat yet)
        for _ in id_labels_hat.tolist():
            if _ not in self.id_label_to_id_hat:
                self.id_label_to_id_hat[_] = self.next_id_hat
                self.next_id_hat += 1

        self.current_track_results_motip = {
            "score": scores_m, "category": categories_m, "bbox": box_cxcywh_to_xywh(boxes_m) * self.bbox_unnorm,
            "id": torch.tensor([self.id_label_to_id[_] for _ in id_labels_motip.tolist()], dtype=torch.int64),
        }
        self.current_track_results_hat = {
            "score": scores_h, "category": categories_h, "bbox": box_cxcywh_to_xywh(boxes_h) * self.bbox_unnorm,
            "id": torch.tensor([self.id_label_to_id_hat[_] for _ in id_labels_hat.tolist()], dtype=torch.int64),
        }
        self.current_track_results = self.current_track_results_motip if self.primary_id_source == "motip" else self.current_track_results_hat

        id_labels_primary = id_labels_motip if self.primary_id_source == "motip" else id_labels_hat
        boxes_primary = boxes_m if self.primary_id_source == "motip" else boxes_h
        output_embeds_primary = output_embeds_m if self.primary_id_source == "motip" else output_embeds_h
        self._update_trajectory_infos(boxes=boxes_primary, output_embeds=output_embeds_primary, id_labels=id_labels_primary)
        self._filter_out_inactive_tracks()
        return

    def get_track_results(self):
        """When both MOTIP and HAT are on, returns {\"motip\": {...}, \"hat\": {...}}. Otherwise returns the single branch result (backward compatible)."""
        if self.use_motip and self.use_hat:
            return {"motip": self.current_track_results_motip, "hat": self.current_track_results_hat}
        return self.current_track_results

    def _get_activate_detections(self, detr_out: dict):
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        output_embeds = detr_out["outputs"][0]
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        activate_indices = (scores > self.det_thresh) & (area > self.area_thresh)
        # Selecting:
        boxes = boxes[activate_indices]
        output_embeds = output_embeds[activate_indices]
        scores = scores[activate_indices]
        categories = categories[activate_indices]
        return scores, categories, boxes, output_embeds

    def _get_id_pred_labels(self, boxes: torch.Tensor, output_embeds: torch.Tensor):
        if self.trajectory_features.shape[0] == 0:
            return self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            # 1. prepare current infos:
            current_features = output_embeds[None, ...]     # (T, N, ...)
            current_boxes = boxes[None, ...]                # (T, N, 4)
            current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=distributed_device())
            current_times = self.trajectory_times.shape[0] * torch.ones(
                (1, output_embeds.shape[0]), dtype=torch.int64, device=distributed_device(),
            )
            
            # 2. Get trajectory features and prepare FLD if needed:
            trajectory_features = self.trajectory_features[None, None, ...]
            
            # Apply FLD if enabled
            if self.use_fld:
                from models.hat.lda import LDA
                
                # Get transfer history
                transfer_hist_embeds = []
                transfer_hist_ids = []
                transfer_hist_scores = []
                
                for id_label in self.trajectory_id_labels[0].tolist():
                    if id_label in self.fld_history_queues:
                        embeds, scores = self.fld_history_queues[id_label].get()
                        transfer_hist_embeds.extend(embeds)
                        transfer_hist_scores.extend(scores)
                        transfer_hist_ids.extend([torch.tensor([id_label]) for _ in range(len(embeds))])
                
                if len(transfer_hist_embeds) > self.fld_factor_thr * len(self.trajectory_id_labels[0]):
                    # Fit LDA model
                    detr_dim = output_embeds.shape[-1]
                    lda_model = LDA(
                        use_shrinkage=True,
                        dtype=self.fld_transfer_dtype,
                        use_standard_scaler=self.fld_use_standard_scaler,
                        direct_inter_class_diff=self.fld_direct_inter_class_diff,
                        use_weighted_class_mean=self.fld_use_weighted_class_mean,
                        device=output_embeds.device,
                    )
                    transfer_hist_embeds = torch.stack(transfer_hist_embeds, dim=0)
                    transfer_hist_scores = torch.stack(transfer_hist_scores, dim=0)
                    transfer_hist_ids = torch.cat(transfer_hist_ids, dim=0)
                    
                    lda_model.fit(transfer_hist_embeds, transfer_hist_ids, score=transfer_hist_scores)
                    
                    # Transform features (LDA dim = num_classes-1); project or pad to detr_dim for trajectory_modeling
                    cur_tf = lda_model.transform(current_features[0])
                    cur_tf = self._fld_project_or_pad(cur_tf, detr_dim)
                    current_features = cur_tf[None, ...]
                    traj_tf = lda_model.transform(self.trajectory_features)
                    traj_tf = self._fld_project_or_pad(traj_tf, detr_dim)
                    trajectory_features = traj_tf[None, None, ...]
            
            # 3. prepare seq_info:
            seq_info = {
                "trajectory_features": trajectory_features,
                "trajectory_boxes": self.trajectory_boxes[None, None, ...],
                "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],
                "trajectory_times": self.trajectory_times[None, None, ...],
                "trajectory_masks": self.trajectory_masks[None, None, ...],
                "unknown_features": current_features[None, None, ...],
                "unknown_boxes": current_boxes[None, None, ...],
                "unknown_masks": current_masks[None, None, ...],
                "unknown_times": current_times[None, None, ...],
            }
            # 4. forward:
            seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")
            # 5. get scores:
            id_logits = id_logits[0, 0, 0]
            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()
            # 6. assign id labels:
            # Different assignment protocols:
            match self.assignment_protocol:
                case "hungarian": id_labels = self._hungarian_assignment(id_scores=id_scores)
                case "object-max": id_labels = self._object_max_assignment(id_scores=id_scores)
                case "id-max": id_labels = self._id_max_assignment(id_scores=id_scores)
                case _: raise NotImplementedError

            id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
            return id_pred_labels

    def _get_id_pred_labels_hat(self, boxes: torch.Tensor, output_embeds: torch.Tensor) -> torch.Tensor:
        """HAT branch: LDA on transfer history + similarity matching + Hungarian assignment. No MOTIP id_decoder."""
        if self.trajectory_features.shape[0] == 0 or self.trajectory_id_labels.shape[1] == 0:
            return self.num_id_vocabulary * torch.ones(
                output_embeds.shape[0], dtype=torch.int64, device=output_embeds.device
            )

        dev = output_embeds.device
        n_objs = output_embeds.shape[0]
        memo_features = self.trajectory_features  # (T, N, C)
        memo_id_labels = self.trajectory_id_labels[0]  # (N,) unique id_labels per track
        num_tracks = memo_id_labels.shape[0]

        # Build transfer history from queues (shared with FLD when use_fld)
        transfer_hist_embeds = []
        transfer_hist_scores = []
        transfer_hist_ids = []
        if hasattr(self, "fld_history_queues"):
            for id_label in memo_id_labels.tolist():
                id_label = int(id_label)
                if id_label in self.fld_history_queues:
                    embeds, scores = self.fld_history_queues[id_label].get()
                    transfer_hist_embeds.extend(embeds)
                    transfer_hist_scores.extend(scores)
                    transfer_hist_ids.extend([id_label] * len(embeds))

        current_embeds = output_embeds
        memo_embeds = memo_features[-1] if memo_features.dim() == 3 else memo_features  # (N, C) latest frame

        if len(transfer_hist_embeds) > self.hat_factor_thr * num_tracks and len(set(transfer_hist_ids)) >= 2:
            from models.hat.lda import LDA
            lda = LDA(
                use_shrinkage=True,
                dtype=self.hat_transfer_dtype,
                use_standard_scaler=self.hat_use_standard_scaler,
                direct_inter_class_diff=self.hat_direct_inter_class_diff,
                use_weighted_class_mean=self.hat_use_weighted_class_mean,
                device=dev,
            )
            transfer_hist_embeds_t = torch.stack(transfer_hist_embeds, dim=0).to(dev)
            transfer_hist_scores_t = torch.stack(
                [s if isinstance(s, torch.Tensor) else torch.tensor(s, device=dev) for s in transfer_hist_scores],
                dim=0,
            ).to(dev)
            transfer_hist_ids_t = torch.tensor(transfer_hist_ids, dtype=torch.long, device=dev)
            lda.fit(transfer_hist_embeds_t, transfer_hist_ids_t, score=transfer_hist_scores_t)
            current_embeds = lda.transform(current_embeds)
            memo_embeds = lda.transform(memo_embeds)

        # Similarity: cosine
        current_embeds_n = F.normalize(current_embeds, p=2, dim=1)
        memo_embeds_n = F.normalize(memo_embeds, p=2, dim=1)
        match_scores = torch.mm(current_embeds_n, memo_embeds_n.t())  # (n_objs, num_tracks)

        # Append newborn column so each detection can be assigned to "new" (num_id_vocabulary)
        newborn_col = match_scores.new_zeros((n_objs, 1))
        match_scores = torch.cat([match_scores, newborn_col], dim=1)

        trajectory_id_labels_set = set(memo_id_labels.tolist())
        cost = -match_scores.cpu().numpy()
        match_rows, match_cols = linear_sum_assignment(cost)

        id_pred_labels = [self.num_id_vocabulary] * n_objs
        for i in range(len(match_rows)):
            obj_idx = match_rows[i]
            col_idx = match_cols[i]
            if col_idx >= num_tracks:
                id_pred_labels[obj_idx] = self.num_id_vocabulary
            else:
                id_label = memo_id_labels[col_idx].item()
                if id_label not in trajectory_id_labels_set:
                    id_pred_labels[obj_idx] = self.num_id_vocabulary
                elif match_scores[obj_idx, col_idx].item() < self.id_thresh:
                    id_pred_labels[obj_idx] = self.num_id_vocabulary
                else:
                    id_pred_labels[obj_idx] = id_label
        return torch.tensor(id_pred_labels, dtype=torch.int64, device=dev)

    def _assign_newborn_id_labels(self, pred_id_labels: torch.Tensor):
        # 1. how many newborn instances?
        n_newborns = (pred_id_labels == self.num_id_vocabulary).sum().item()
        if n_newborns == 0:
            return pred_id_labels
        else:
            # 2. get available id labels from id_queue:
            newborn_id_labels = torch.tensor(
                list(self.id_queue)[:n_newborns], dtype=torch.int64, device=distributed_device(),
            )
            # 3. make sure these id labels are not in trajectory infos:
            trajectory_remove_idxs = torch.zeros(
                self.trajectory_id_labels.shape[1], dtype=torch.bool, device=distributed_device(),
            )
            for _ in range(len(newborn_id_labels)):
                if self.trajectory_id_labels.shape[0] > 0:
                    trajectory_remove_idxs |= (self.trajectory_id_labels[0] == newborn_id_labels[_])
                if newborn_id_labels[_].item() in self.id_label_to_id:
                    self.id_label_to_id.pop(newborn_id_labels[_].item())
                # Initialize FLD history queue for new ID
                if self.use_fld or self.use_hat:
                    from models.hat.fifo_queue import FIFOQueue
                    self.fld_history_queues[newborn_id_labels[_].item()] = FIFOQueue(
                        self.fld_hist_len if self.use_fld else self.hat_hist_len,
                        0.9 if self.use_fld else self.hat_weight_decay,
                        True,
                    )
            # remove from trajectory infos:
            self.trajectory_features = self.trajectory_features[:, ~trajectory_remove_idxs]
            self.trajectory_boxes = self.trajectory_boxes[:, ~trajectory_remove_idxs]
            self.trajectory_id_labels = self.trajectory_id_labels[:, ~trajectory_remove_idxs]
            self.trajectory_times = self.trajectory_times[:, ~trajectory_remove_idxs]
            self.trajectory_masks = self.trajectory_masks[:, ~trajectory_remove_idxs]
            # 4. assign id labels to newborn instances:
            pred_id_labels[pred_id_labels == self.num_id_vocabulary] = newborn_id_labels
            # 5. update id infos:
            for _ in range(len(newborn_id_labels)):
                self.id_label_to_id[newborn_id_labels[_].item()] = self.next_id
                self.next_id += 1

            return pred_id_labels

    def _assign_newborn_id_labels_hat(self, pred_id_labels: torch.Tensor) -> torch.Tensor:
        """Assign newborn IDs for HAT (id_queue_hat / id_label_to_id_hat). Same trajectory/FIFO hygiene as MOTIP newborns."""
        n_newborns = (pred_id_labels == self.num_id_vocabulary).sum().item()
        if n_newborns == 0:
            return pred_id_labels
        newborn_id_labels = torch.tensor(
            list(self.id_queue_hat)[:n_newborns], dtype=torch.int64, device=distributed_device(),
        )
        trajectory_remove_idxs = torch.zeros(
            self.trajectory_id_labels.shape[1], dtype=torch.bool, device=distributed_device(),
        )
        for _ in range(len(newborn_id_labels)):
            if self.trajectory_id_labels.shape[0] > 0:
                trajectory_remove_idxs |= (self.trajectory_id_labels[0] == newborn_id_labels[_])
            if newborn_id_labels[_].item() in self.id_label_to_id_hat:
                self.id_label_to_id_hat.pop(newborn_id_labels[_].item())
            if self.use_fld or self.use_hat:
                from models.hat.fifo_queue import FIFOQueue
                self.fld_history_queues[newborn_id_labels[_].item()] = FIFOQueue(
                    self.fld_hist_len if self.use_fld else self.hat_hist_len,
                    0.9 if self.use_fld else self.hat_weight_decay,
                    True,
                )
        self.trajectory_features = self.trajectory_features[:, ~trajectory_remove_idxs]
        self.trajectory_boxes = self.trajectory_boxes[:, ~trajectory_remove_idxs]
        self.trajectory_id_labels = self.trajectory_id_labels[:, ~trajectory_remove_idxs]
        self.trajectory_times = self.trajectory_times[:, ~trajectory_remove_idxs]
        self.trajectory_masks = self.trajectory_masks[:, ~trajectory_remove_idxs]
        pred_id_labels[pred_id_labels == self.num_id_vocabulary] = newborn_id_labels
        for _ in range(len(newborn_id_labels)):
            self.id_label_to_id_hat[newborn_id_labels[_].item()] = self.next_id_hat
            self.next_id_hat += 1
        return pred_id_labels

    def _update_trajectory_infos(self, boxes: torch.Tensor, output_embeds: torch.Tensor, id_labels: torch.Tensor):
        # 1. cut trajectory infos:
        self.trajectory_features = self.trajectory_features[-self.miss_tolerance + 2:, ...]
        self.trajectory_boxes = self.trajectory_boxes[-self.miss_tolerance + 2:, ...]
        self.trajectory_id_labels = self.trajectory_id_labels[-self.miss_tolerance + 2:, ...]
        self.trajectory_times = self.trajectory_times[-self.miss_tolerance + 2:, ...]
        self.trajectory_masks = self.trajectory_masks[-self.miss_tolerance + 2:, ...]
        # 2. find out all new instances:
        already_id_labels = set(self.trajectory_id_labels[0].tolist() if self.trajectory_id_labels.shape[0] > 0 else [])
        _id_labels = set(id_labels.tolist())
        newborn_id_labels = _id_labels - already_id_labels
        # 3. add newborn instances to trajectory infos:
        if len(newborn_id_labels) > 0:
            newborn_id_labels = torch.tensor(list(newborn_id_labels), dtype=torch.int64, device=distributed_device())
            _T = self.trajectory_id_labels.shape[0]
            _N = len(newborn_id_labels)
            _id_labels = einops.repeat(newborn_id_labels, 'n -> t n', t=_T)
            _boxes = torch.zeros((_T, _N, 4), dtype=self.dtype, device=distributed_device())
            _times = einops.repeat(
                torch.arange(_T, dtype=torch.int64, device=distributed_device()), 't -> t n', n=_N,
            )
            _features = torch.zeros(
                (_T, _N, 256), dtype=self.dtype, device=distributed_device(),
            )
            _masks = torch.ones((_T, _N), dtype=torch.bool, device=distributed_device())
            # 3.1. padding to trajectory infos:
            self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, _id_labels], dim=1)
            self.trajectory_boxes = torch.cat([self.trajectory_boxes, _boxes], dim=1)
            self.trajectory_times = torch.cat([self.trajectory_times, _times], dim=1)
            self.trajectory_features = torch.cat([self.trajectory_features, _features], dim=1)
            self.trajectory_masks = torch.cat([self.trajectory_masks, _masks], dim=1)
        # 4. update trajectory infos:
        _N = self.trajectory_id_labels.shape[1]
        current_id_labels = self.trajectory_id_labels[0] if self.trajectory_id_labels.shape[0] > 0 else id_labels
        current_features = torch.zeros((_N, 256), dtype=self.dtype, device=distributed_device())
        current_boxes = torch.zeros((_N, 4), dtype=self.dtype, device=distributed_device())
        current_times = self.trajectory_id_labels.shape[0] * torch.ones((_N,), dtype=torch.int64, device=distributed_device())
        current_masks = torch.ones((_N,), dtype=torch.bool, device=distributed_device())
        # 4.1. find out the same id labels (matching):
        indices = torch.eq(current_id_labels[:, None], id_labels[None, :]).nonzero(as_tuple=False)
        current_idxs = indices[:, 0]
        idxs = indices[:, 1]
        # 4.2. fill in the infos:
        current_id_labels[current_idxs] = id_labels[idxs]
        current_features[current_idxs] = output_embeds[idxs]
        current_boxes[current_idxs] = boxes[idxs]
        current_masks[current_idxs] = False
        
        # Update FLD/HAT history queues (shared when both are on)
        if self.use_fld or self.use_hat:
            for idx, id_label in enumerate(id_labels):
                id_label = id_label.item()
                if id_label not in self.fld_history_queues:
                    from models.hat.fifo_queue import FIFOQueue
                    hist_len = self.fld_hist_len if self.use_fld else self.hat_hist_len
                    w_decay = 0.9 if self.use_fld else self.hat_weight_decay
                    self.fld_history_queues[id_label] = FIFOQueue(hist_len, w_decay, True)
                self.fld_history_queues[id_label].add(feature=output_embeds[idx], score=torch.tensor(1.0))
        
        # 4.3. cat to trajectory infos:
        self.trajectory_features = torch.cat([self.trajectory_features, current_features[None, ...]], dim=0).contiguous()
        self.trajectory_boxes = torch.cat([self.trajectory_boxes, current_boxes[None, ...]], dim=0).contiguous()
        self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, current_id_labels[None, ...]], dim=0).contiguous()
        self.trajectory_times = torch.cat([self.trajectory_times, current_times[None, ...]], dim=0).contiguous()
        self.trajectory_masks = torch.cat([self.trajectory_masks, current_masks[None, ...]], dim=0).contiguous()
        # 4.4. a hack implementation to fix "times":
        self.trajectory_times = einops.repeat(
            torch.arange(self.trajectory_times.shape[0], dtype=torch.int64, device=distributed_device()),
            't -> t n', n=self.trajectory_times.shape[1],
        ).contiguous().clone()
        return

    def _filter_out_inactive_tracks(self):
        is_active = torch.sum((~self.trajectory_masks).to(torch.int64), dim=0) > 0
        active_id_labels = self.trajectory_id_labels[0][is_active].tolist()
        
        # Filter FLD/HAT history queues
        if self.use_fld or self.use_hat:
            inactive_ids = [id_label for id_label in self.fld_history_queues if id_label not in active_id_labels]
            for id_label in inactive_ids:
                del self.fld_history_queues[id_label]
        
        self.trajectory_features = self.trajectory_features[:, is_active]
        self.trajectory_boxes = self.trajectory_boxes[:, is_active]
        self.trajectory_id_labels = self.trajectory_id_labels[:, is_active]
        self.trajectory_times = self.trajectory_times[:, is_active]
        self.trajectory_masks = self.trajectory_masks[:, is_active]
        return

    def _hungarian_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        if len(id_scores) > 1:
            id_scores_newborn_repeat = id_scores[:, -1:].repeat(1, len(id_scores) - 1)
            id_scores = torch.cat((id_scores, id_scores_newborn_repeat), dim=-1)
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())
        match_rows, match_cols = linear_sum_assignment(1 - id_scores.cpu())
        for _ in range(len(match_rows)):
            _id = match_cols[_]
            if _id not in trajectory_id_labels_set:
                id_labels.append(self.num_id_vocabulary)
            elif _id >= self.num_id_vocabulary:
                id_labels.append(self.num_id_vocabulary)
            elif id_scores[match_rows[_], _id] < self.id_thresh:
                id_labels.append(self.num_id_vocabulary)
            else:
                id_labels.append(_id)
        return id_labels

    def _object_max_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        object_max_confs, object_max_id_labels = torch.max(id_scores, dim=-1)   # get the target ID labels and confs
        # Get the max confs of each ID label:
        id_max_confs = dict()
        for conf, id_label in zip(object_max_confs.tolist(), object_max_id_labels.tolist()):
            if id_label not in id_max_confs:
                id_max_confs[id_label] = conf
            else:
                id_max_confs[id_label] = max(id_max_confs[id_label], conf)
        if self.num_id_vocabulary in id_max_confs:
            id_max_confs[self.num_id_vocabulary] = 0.0  # special token

        # Assign ID labels:
        for _ in range(len(object_max_id_labels)):
            if object_max_id_labels[_].item() not in trajectory_id_labels_set:         # not in tracked IDs -> newborn
                id_labels.append(self.num_id_vocabulary)
            else:
                _id_label = object_max_id_labels[_].item()
                _conf = object_max_confs[_].item()
                if _conf < self.id_thresh or _conf < id_max_confs[_id_label]:  # low conf or not the max conf -> newborn
                    id_labels.append(self.num_id_vocabulary)
                elif _id_label in id_labels:
                    id_labels.append(self.num_id_vocabulary)
                else:                                                          # normal case
                    id_labels.append(_id_label)

        return id_labels

    def _id_max_assignment(self, id_scores: torch.Tensor):
        id_labels = [self.num_id_vocabulary] * len(id_scores)  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        id_max_confs, id_max_obj_idxs = torch.max(id_scores, dim=0)
        # Get the max confs of each object:
        object_max_confs = dict()
        for conf, object_idx in zip(id_max_confs.tolist(), id_max_obj_idxs.tolist()):
            if object_idx not in object_max_confs:
                object_max_confs[object_idx] = conf
            else:
                if conf == object_max_confs[object_idx]:    # a very rare case
                    conf = conf - 0.0001
                object_max_confs[object_idx] = max(object_max_confs[object_idx], conf)

        # Assign ID labels:
        for _ in range(len(id_max_obj_idxs)):
            _obj_idx, _id_label, _conf = id_max_obj_idxs[_].item(), _, id_max_confs[_].item()
            if _conf < self.id_thresh or _conf < object_max_confs[_obj_idx]:
                pass
            elif _id_label not in trajectory_id_labels_set:
                pass
            else:
                id_labels[_obj_idx] = _id_label

        return id_labels