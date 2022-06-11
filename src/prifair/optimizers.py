"""Wrappers for optimizers that specific algorithms use during training. """

from typing import Callable, List, Optional

import torch
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import (  # _generate_noise,
    _check_processed_flag,
    _get_flat_grad_sample,
    _mark_as_processed,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DPSGDFOptimizer(DPOptimizer):
    """
    Implementation of DPSGD-F from Xu et al. [1].

    A wrapper for ``torch.optim.Optimizer`` that adds additional functionality
    to clip per sample gradients and add Gaussian noise.
    Similar to ``opacus.optimizers.DPOptimizer``, but uses adaptive clipping thresholds.

    Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
    ``DPSGDFOptimizer`` assumes that parameters over which it performs optimization
    belong to GradSampleModule and therefore have the ``grad_sample`` attribute.

    On a high level ``DPSGDOptimizer``'s step looks like this:
    1) Aggregate ``p.grad_sample`` over all parameters to calculate per sample norms.
    2) Adaptively clip ``p.grad_sample`` so that the per sample norm is not above
    the computed threshold for each sample.
    3) Aggregate clipped per sample gradients into ``p.grad``.
    4) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
    adaptive clipping thresholds.
    5) Call underlying optimizer to perform optimization step.

    References:
        [1] Xu D, Du W, Wu X. Removing disparate impact of differentially private
        stochastic gradient descent on model accuracy.
        arXiv preprint arXiv:200303699. 2020.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        noise_multiplier: float,
        base_clipping_threshold: float,
        expected_batch_size: Optional[int],
        n_groups: int,
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
            expected_batch_size (Optional[int]):
                The batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required if ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``.
            n_groups (int):
                The number of sensitive subgroups in the data.
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
                Logs the clipping thresholds at each iteration, storing them in class
                variable ``self.thresholds``. Defaults to False.
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

        self.C0 = base_clipping_threshold
        self.n_groups = n_groups

        self.sample_group_labels = None
        self.log_thresholds = log_thresholds

        if self.log_thresholds:
            self.thresholds: List[List[float]] = [[] for i in range(n_groups)]

    def clip_and_accumulate(self):
        """Performs gradient clipping and accumulation.
        Stores clipped and aggregated gradients into `p.summed_grad```.
        """

        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

        C = self.compute_clipping_bounds(per_sample_norms)
        per_sample_clip_factor = (C / (per_sample_norms + 1e-6)).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = _get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def compute_clipping_bounds(self, per_sample_norms: torch.Tensor) -> torch.Tensor:
        """Computes the variable clipping bounds for each sample.

        Args:
            per_sample_norms (torch.Tensor):
                The norms of the gradients for each sample in the batch.
                Should of shape (batch_size,).

        Returns:
            torch.Tensor:
                The clipping bounds of shape (batch_size,).
        """

        if self.sample_group_labels is None:
            raise ValueError(
                "sample_group_labels has not been cached. Pass the group labels to \
                ``DPSGDFOptimizer.update_cached_params`` before calling \
                DPSGDFOptimizer.step"
            )

        n_samples = len(self.sample_group_labels)
        C = torch.full((n_samples,), self.C0).to(device)

        m_mask = per_sample_norms > self.C0
        m_sum = 0
        for k in range(self.n_groups):
            group_mask = self.sample_group_labels == k
            count = torch.sum(group_mask)
            if count == 0:
                continue

            m = torch.sum(group_mask & m_mask)
            m_ratio = m / count

            C[group_mask] = self.C0 * (1 + n_samples * m_ratio)

            m_sum += m

        C /= m_sum

        self.max_grad_norm = C.max()

        if self.log_thresholds:
            for k in range(self.n_groups):
                group_mask = self.sample_group_labels == k
                if torch.sum(group_mask) == 0:
                    self.thresholds[k].append(self.C0)
                    continue

                threshold = torch.mean(C[group_mask]).detach().cpu().numpy().squeeze()
                self.thresholds[k].append(float(threshold))

        return C

    def step(
        self,
        *,
        group_labels: torch.Tensor,
        closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:

        self.sample_group_labels = group_labels

        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            return self.original_optimizer.step()
        else:
            return None
