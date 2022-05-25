"""Tools for handling data."""

from typing import List, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Sampler


class NonUniformPoissonSampler(Sampler[List[int]]):
    """A non-uniform weighted Poisson batch sampler."""

    # pylint: disable=W0231
    def __init__(self, weights: np.ndarray, num_samples: int, sample_rate: float):
        assert num_samples > 0
        assert len(weights) == num_samples

        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.weighted_sample_rates = torch.as_tensor(
            sample_rate * weights * num_samples, dtype=torch.double
        )

    def __len__(self):
        return int(1 / self.sample_rate)

    def __iter__(self):
        num_batches = int(1 / self.sample_rate)
        while num_batches > 0:
            mask = torch.rand(self.num_samples) < self.weighted_sample_rates
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            yield indices

            num_batches -= 1


class WeightedDataLoader(torch.utils.data.DataLoader):
    """A data loader that uses a weighted sampler.

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
