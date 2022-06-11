"""Tools for handling data"""

from typing import Sequence, Union

import numpy as np
import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager, BatchSplittingSampler
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

from .utils import _data_loader_with_batch_sampler


class NonUniformPoissonSampler(UniformWithReplacementSampler):
    """
    A non-uniform weighted Poisson batch sampler.
    """

    def __init__(
        self,
        *,
        weights: np.ndarray,
        num_samples: int,
        sample_rate: float,
        generator: torch.Generator = None
    ):
        """
        Args:
            weights (np.ndarray):
                The weights for each sample in the dataset.
            num_samples (int):
                The number of samples to draw.
            sample_rate (float):
                The probability used in sampling.
            generator (torch.Generator, optional):
                The generator used to sample random numbers. Defaults to None.
        """

        assert num_samples > 0
        assert len(weights) == num_samples

        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.weighted_sample_rates = torch.as_tensor(
            sample_rate * weights * num_samples, dtype=torch.double
        )
        self.generator = generator

    def __iter__(self):
        num_batches = int(1 / self.sample_rate)
        while num_batches > 0:
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.weighted_sample_rates
            )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            yield indices

            num_batches -= 1


class IndexCachingBatchMemoryManager(BatchMemoryManager):
    """
    Context manager to manage memory consumption during training.
    Adaptation of ``opacus.utils.batch_memory_manager.BatchMemoryManager``.
    Wraps ``opacus.utils.batch_memory_manager.BatchSplittingSampler`` with
    a sampler that caches the last indices sampled for access.
    See ``opacus.utils.batch_memory_manager.BatchMemoryManager`` for more
    details.
    """

    def __enter__(self):
        sampler = BatchSplittingSampler(
            sampler=self.data_loader.batch_sampler,
            max_batch_size=self.max_physical_batch_size,
            optimizer=self.optimizer,
        )

        return _data_loader_with_batch_sampler(self.data_loader, sampler, wrap=True)


class WeightedDataLoader(torch.utils.data.DataLoader):
    """
    A data loader that uses a weighted sampler.

    Args:
        data_loader (torch.utils.data.DataLoader):
            The data loader to weight.
    """

    def __init__(self, data_loader: torch.utils.data.DataLoader):
        N = len(data_loader.dataset)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(np.ones(N) / N, N)

        super().__init__(
            dataset=data_loader.dataset,
            batch_size=data_loader.batch_size,
            sampler=weighted_sampler,
            shuffle=False,
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            generator=data_loader.generator,
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
        )

    def update_weights(self, weights: Union[Sequence[float], np.ndarray, torch.Tensor]):
        """Updates the weights of the sampler.

        Args:
            weights (Union[Sequence[float], np.ndarray, torch.Tensor]):
                The new weights.
        """

        self.sampler.weights = torch.as_tensor(weights, dtype=torch.double)
