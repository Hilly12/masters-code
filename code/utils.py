"""Utilities to supplement the training code. """

from typing import List, Sequence

import torch
from torch.utils.data import Sampler


class NonUniformPoissonSampler(Sampler[List[int]]):
    """A non-uniform weighted Poisson batch sampler."""

    # pylint: disable=W0231
    def __init__(self, weights: Sequence[float], num_samples: int, sample_rate: float):
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


def _shape_safe(x):
    return x.shape if hasattr(x, "shape") else ()


def _wrap_collate_with_empty(data_loader):
    sample_empty_shapes = [[0, *_shape_safe(x)] for x in data_loader.dataset[0]]

    def collate(batch):
        if len(batch) > 0:
            # pylint: disable=W0212
            return torch.utils.data._utils.collate.default_collate(batch)

        return [torch.zeros(x) for x in sample_empty_shapes]

    return collate


def _data_loader_with_sampler(
    data_loader: torch.utils.data.DataLoader, batch_sampler: Sampler[List[int]]
) -> torch.utils.data.DataLoader:

    return torch.utils.data.DataLoader(
        dataset=data_loader.dataset,
        batch_sampler=batch_sampler,
        num_workers=data_loader.num_workers,
        collate_fn=_wrap_collate_with_empty(data_loader),
        pin_memory=data_loader.pin_memory,
        timeout=data_loader.timeout,
        worker_init_fn=data_loader.worker_init_fn,
        multiprocessing_context=data_loader.multiprocessing_context,
        generator=data_loader.generator,
        prefetch_factor=data_loader.prefetch_factor,
        persistent_workers=data_loader.persistent_workers,
    )
