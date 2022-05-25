"""Utilities to supplement the training code. """

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def update_weights(self, weights: Sequence[float]):
        """Updates the weights of the sampler.

        Args:
            weights (Sequence[float]):
                The new weights.
        """

        self.sampler.weights = torch.as_tensor(weights, dtype=torch.double)


def validate_model(model, val_loader, loss_fn):
    losses = []
    accuracies = []

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = loss_fn(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            acc = (preds == labels).mean()

            losses.append(loss.item())
            accuracies.append(acc)

    return losses, accuracies


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    verbose: bool = True,
    return_outputs: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Evaluates the model on the test set.

    Args:
        model (torch.nn.Module):
            The model to evaluate.
        test_loader (torch.utils.data.DataLoader):
            The test data loader.
        verbose (bool, optional):
            Whether to print the results. Defaults to True.
        return_outputs (bool, optional):
            Whether to return the raw outputs. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            The predictions of the model on the test data. The raw outputs are also
            returned if `return_outputs` is True.
    """

    model.eval()

    correct = 0
    pred_list = []
    raw_outputs = []
    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)

            output = model(images).detach().cpu()
            preds = np.argmax(output, axis=1).numpy()
            labels = target.numpy()

            correct += (preds == labels).sum()

            pred_list.append(preds)
            raw_outputs.append(output)

    acc = correct / len(test_loader.dataset)

    if verbose:
        print(f"Test Accuracy: {acc * 100:.6f}")

    if return_outputs:
        return np.concatenate(pred_list), np.concatenate(raw_outputs)

    return np.concatenate(pred_list)


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
