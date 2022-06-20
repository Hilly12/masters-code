"""Wrappers for optimizers that specific algorithms use during training. """

from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import (
    _check_processed_flag,
    _get_flat_grad_sample,
    _mark_as_processed,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdaptiveClippingOptimizer(DPOptimizer):
    """
    A wrapper for ``torch.optim.Optimizer`` that adds additional functionality
    to clip per sample gradients and add Gaussian noise.
    Similar to ``opacus.optimizers.DPOptimizer``, but uses adaptive clipping thresholds.

    On a high level ``AdaptiveClippingOptimizer``'s step looks like this:
    1) Aggregate ``p.grad_sample`` over all parameters to calculate per sample norms.
    2) Adaptively clip ``p.grad_sample`` so that the per sample norm is not above
    the computed threshold for each sample.
    3) Aggregate clipped per sample gradients into ``p.grad``.
    4) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
    adaptive clipping thresholds.
    5) Call underlying optimizer to perform optimization step.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        noise_multiplier: float,
        base_clipping_threshold: float,
        n_groups: int,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator: torch.Generator = None,
        secure_mode: bool = False,
        log_thresholds: bool = False
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer):
                The wrapped optimizer.
            noise_multiplier (float):
                The noise multiplier.
            base_clipping_threshold (float):
                The base clipping threshold used to clip the norm of the gradients.
            n_groups (int):
                The number of protected subgroups in the data.
            expected_batch_size (Optional[int]):
                The batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required if ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``.
            loss_reduction (str, optional):
                Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean".
                Defaults to "mean".
            generator (torch.Generator, optional):
                torch.Generator() object used as a source of randomness for the noise.
                Defaults to None.
            secure_mode (bool, optional):
                If ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See ``opacus.optimizers.optimizer._generate_noise`` for details.
                Defaults to False.
            log_thresholds (bool, optional):
                Logs the clipping mean group thresholds at each iteration,
                storing them in class variable ``self.logged_thresholds``.
                Defaults to False.
        """

        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=base_clipping_threshold,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )

        self.base_clip = base_clipping_threshold
        self.n_groups = n_groups

        self.batch_params: Dict[str, Any] = {}
        self.log_thresholds = log_thresholds

        if self.log_thresholds:
            self.logged_thresholds: List[List[float]] = [[] for i in range(n_groups)]

    def clip_and_accumulate(self):
        """Performs gradient clipping and accumulation.
        Stores clipped and aggregated gradients into ``p.summed_grad``.
        """

        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

        # Modified from parent

        C = self.compute_clipping_bounds(per_sample_norms)
        self.max_grad_norm = C.max()

        per_sample_clip_factor = (C / (per_sample_norms + 1e-6)).clamp(max=1.0)

        #

        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = _get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    @abstractmethod
    def compute_clipping_bounds(self, per_sample_norms: torch.Tensor) -> torch.Tensor:
        """Computes the variable clipping bounds for each sample in the batch.

        Args:
            per_sample_norms (torch.Tensor):
                The norms of the gradients for each sample in the batch.
                Should of shape (batch_size,).

        Returns:
            torch.Tensor:
                The clipping bounds of shape (batch_size,).
        """
        pass

    def set_batch_params(self, **kwargs):
        """Sets the batch parameters used by ``compute_clipping_bounds``.

        Args:
            **kwargs:
                By default any key word arguments passed to this function
                will be stored in the ``batch_params`` dictionary and
                passed to ``compute_clipping_bounds``.
        """

        self.batch_params = kwargs

    def _log_clipping_thresholds(self, C, group_labels):
        if not self.log_thresholds:
            return

        for k in range(self.n_groups):
            group_mask = group_labels == k
            if torch.sum(group_mask) == 0:
                self.logged_thresholds[k].append(self.base_clip)
                continue

            threshold = torch.mean(C[group_mask]).detach().cpu().numpy().squeeze()
            self.logged_thresholds[k].append(float(threshold))


class DPSGDFOptimizer(AdaptiveClippingOptimizer):
    """
    Implementation of DPSGD-F from Xu et al. [1].

    References:
        [1] Xu D, Du W, Wu X. Removing disparate impact of differentially private
        stochastic gradient descent on model accuracy.
        arXiv preprint arXiv:200303699. 2020.
    """

    def compute_clipping_bounds(self, per_sample_norms: torch.Tensor) -> torch.Tensor:
        if "group_labels" not in self.batch_params:
            raise ValueError(
                "``group_labels`` has not been set. Pass the batch group labels \
                to ``DPSGDFOptimizer.set_batch_params()`` before calling \
                ``DPSGDFOptimizer.step()``."
            )

        batch_group_labels = self.batch_params["group_labels"]
        assert len(batch_group_labels) == len(per_sample_norms)

        n_samples = len(batch_group_labels)
        C = torch.full((n_samples,), self.base_clip).to(device)

        m_mask = per_sample_norms > self.base_clip
        m_sum = 0
        for k in range(self.n_groups):
            group_mask = batch_group_labels == k
            count = torch.sum(group_mask)
            if count == 0:
                continue

            m = torch.sum(group_mask & m_mask)
            m_ratio = m / count

            C[group_mask] = self.base_clip * (1 + n_samples * m_ratio)

            m_sum += m

        if m_sum > 0:
            C /= m_sum

        self._log_clipping_thresholds(C, batch_group_labels)

        return C
