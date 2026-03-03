# Copyright (c) Ruopeng Gao. All Rights Reserved.
# Fisher Linear Discriminant Analysis (LDA) for History-Aware Transformation.
# Adapted from HATReID-MOT: https://github.com/HELLORPG/HATReID-MOT

import torch
import scipy.linalg
import numpy as np
from torch.nn import functional


class LDA:
    """
    Fisher Linear Discriminant Analysis for transforming ReID features
    in the History-Aware Transformation (HAT) framework.

    Given historical trajectory features with their trajectory IDs,
    this class computes a projection matrix W that maximizes inter-trajectory
    differences while minimizing intra-trajectory differences (Fisher criterion).

    The transformed features f' = f @ W live in a discriminative subspace
    of dimension (num_classes - 1), which is optimal for distinguishing
    different trajectories within the same video sequence.
    """

    def __init__(
            self,
            use_shrinkage: bool = True,
            use_weighted_class_mean: bool = True,
            weighted_class_mean_alpha: float = 1.0,
            dtype: torch.dtype = torch.float32,
            device: str = "cuda",
    ):
        """
        Args:
            use_shrinkage: Whether to use Ledoit-Wolf shrinkage estimation
                for the covariance matrix (stabilizes when samples are few).
            use_weighted_class_mean: Whether to use weighted mean for each
                trajectory centroid (enables Temporal-Shifted Centroid from
                the HAT paper, Section 3.3).
            weighted_class_mean_alpha: Exponent for the weight when computing
                weighted class means. Higher values emphasize high-weight samples more.
            dtype: Tensor data type for computation.
            device: Device for computation.
        """
        self.use_shrinkage = use_shrinkage
        self.use_weighted_class_mean = use_weighted_class_mean
        self.weighted_class_mean_alpha = weighted_class_mean_alpha
        self.dtype = dtype
        self.device = device

        # LDA attributes:
        self.classes = None
        self.class_means = None
        self.project_matrix = None

    def clear(self):
        """Reset the LDA model state."""
        self.classes = None
        self.class_means = None
        self.project_matrix = None

    def is_fit(self) -> bool:
        """Check if the model has been fitted."""
        return self.project_matrix is not None

    def fit(self, X: torch.Tensor, y: torch.Tensor, score: torch.Tensor = None):
        """
        Fit the LDA model to compute the projection matrix W.

        This solves the Fisher criterion:
            argmax_W tr{ (W^T S_W W)^{-1} (W^T S_B W) }
        where S_W is the within-class scatter matrix and S_B is the
        between-class scatter matrix.

        Args:
            X: Feature matrix of shape (N, D), where N is the number of
               historical feature samples and D is the feature dimension.
            y: Class labels of shape (N,), indicating which trajectory
               each feature belongs to.
            score: Optional weight/score for each sample of shape (N,).
                   Used for temporal-shifted centroid computation.
        """
        # 1. Pre-process:
        if isinstance(X, torch.Tensor):
            X = X.to(dtype=self.dtype, device=self.device)
        else:
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        if isinstance(y, torch.Tensor):
            y = y.to(dtype=torch.long, device=self.device)
        else:
            y = torch.tensor(y, dtype=torch.long, device=self.device)
        if score is not None:
            if isinstance(score, torch.Tensor):
                score = score.to(dtype=self.dtype, device=self.device)
            else:
                score = torch.tensor(score, dtype=self.dtype, device=self.device)
            score = torch.clamp(score, min=0.0)

        # 2. Get unique classes:
        self.classes = torch.unique(y).tolist()

        # 3. Compute within-class scatter matrix S_W:
        S_w = self._get_class_cov(X, y)

        # 4. Compute between-class scatter matrix S_B:
        self.class_means = self._get_class_means(
            X, y, score=score if self.use_weighted_class_mean else None
        )
        S_b = self._get_inter_class_diff(
            X, y, score=score if self.use_weighted_class_mean else None
        )

        # 5. Solve generalized eigenvalue problem:
        eig_vals, eig_vecs = scipy.linalg.eigh(
            S_b.cpu().numpy(), S_w.cpu().numpy()
        )
        eig_vals = torch.tensor(eig_vals, dtype=self.dtype, device=self.device)
        eig_vecs = torch.tensor(eig_vecs, dtype=self.dtype, device=self.device)
        sorted_indices = torch.argsort(eig_vals, descending=True)
        eig_vecs = eig_vecs[:, sorted_indices]
        # Keep top (C-1) eigenvectors where C is the number of classes:
        self.project_matrix = eig_vecs[:, :len(self.classes) - 1]

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform features using the fitted projection matrix.

        Args:
            X: Feature matrix of shape (N, D) or (..., D).

        Returns:
            Transformed features of shape (N, D') or (..., D')
            where D' = num_classes - 1.
        """
        assert self.project_matrix is not None, "Please fit the model first."
        if isinstance(X, torch.Tensor):
            X = X.to(dtype=self.dtype, device=self.device)
        else:
            X = torch.tensor(X, dtype=self.dtype, device=self.device)
        return X @ self.project_matrix

    def _get_class_means(self, X, y, score=None):
        """Compute the mean feature vector for each class (trajectory)."""
        class_means = []
        for c in self.classes:
            class_means.append(
                self._get_mean(
                    X[y == c], dim=0,
                    weights=score[y == c] if score is not None else None,
                )
            )
        return torch.stack(class_means, dim=0)

    def _get_inter_class_diff(self, X, y, score=None):
        """Compute between-class scatter matrix S_B."""
        S_b = torch.zeros(
            (X.shape[1], X.shape[1]), dtype=self.dtype, device=self.device
        )
        overall_mean = self._get_mean(X, dim=0, weights=score)
        for i, c in enumerate(self.classes):
            if self.use_weighted_class_mean and score is not None:
                n = torch.sum(score[y == c])
            else:
                n = (y == c).sum()
            mean_diff = self.class_means[i] - overall_mean
            S_b += n * mean_diff[:, None] @ mean_diff[None, :]
        return S_b

    def _get_class_cov(self, X, y):
        """Compute within-class scatter matrix S_W."""
        _N, _C = X.shape
        class_cov = torch.zeros(
            (_C, _C), dtype=self.dtype, device=self.device
        )
        for c in self.classes:
            class_X = X[y == c]
            class_cov += (len(class_X) / _N) * self._get_cov(class_X)
        return class_cov

    def _get_cov(self, X):
        """Compute covariance matrix, optionally with shrinkage."""
        if self.use_shrinkage:
            # Standardize first for numerical stability:
            mean = X.mean(dim=0)
            std = X.std(dim=0)
            std = torch.clamp(std, min=1e-12)
            X_scaled = (X - mean) / std

            # Compute shrunk covariance:
            cov = self._get_shrunk_cov(X_scaled)

            # Scale back:
            cov = std[:, None] * cov * std[None, :]
            return cov
        else:
            return torch.cov(X.T, correction=0)

    def _get_shrunk_cov(self, X):
        """Compute Ledoit-Wolf shrinkage covariance."""
        X = X - X.mean(dim=0)
        shrinkage = self._get_shrinkage(X)
        cov = X.T @ X / X.shape[0]
        mu = torch.trace(cov) / cov.shape[0]
        shrunk_cov = (1 - shrinkage) * cov
        diag_indices = torch.arange(cov.shape[0], device=cov.device)
        shrunk_cov[diag_indices, diag_indices] += shrinkage * mu
        return shrunk_cov

    def _get_shrinkage(self, X):
        """Compute the optimal shrinkage coefficient (Ledoit-Wolf)."""
        _N, _C = X.shape
        X2 = X ** 2
        emp_cov_trace = torch.sum(X2, dim=0) / _N
        mu = torch.sum(emp_cov_trace) / _C

        beta_ = torch.sum(torch.matmul(X2.T, X2))
        delta_ = torch.sum(torch.matmul(X.T, X) ** 2)
        delta_ /= _N ** 2

        beta = 1.0 / (_C * _N) * (beta_ / _N - delta_)
        delta = delta_ - 2.0 * mu * emp_cov_trace.sum() + _C * mu ** 2
        delta /= _C

        beta = min(beta, delta)
        shrinkage = 0 if beta == 0 else beta / delta
        return shrinkage

    def _get_mean(self, X, dim, weights=None):
        """Compute mean, optionally weighted."""
        if weights is None:
            return torch.mean(X, dim=dim)
        else:
            assert len(X.shape) == 2 and dim == 0
            assert len(X) == len(weights)
            weights = weights ** self.weighted_class_mean_alpha
            return torch.sum(weights[:, None] * X, dim=dim) / torch.sum(weights)
