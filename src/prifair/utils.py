"""Utilities"""

from typing import List, Tuple, Type, Union

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_model(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
) -> Tuple[List[int], List[int]]:
    """Validates the model on the validation set.

    Args:
        model (torch.nn.Module):
            The model to validate.
        val_loader (torch.utils.data.DataLoader):
            The validation data loader.
        loss_fn (torch.nn.Module):
            The loss function to use.

    Returns:
        Tuple[List[int], List[int]]:
            The validation losses and the validation accuracies for each iteration
            in the epoch.
    """

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


def dataset_with_indices(cls: Type) -> Type:
    """Modifies the given Dataset class's dunder method __getitem__ to
    return a tuple data, target, index instead of data, target.

    Args:
        cls (Type):
            The dataset class to modify.

    Returns:
        Type:
            The modified Dataset class.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(
        cls.__name__,
        (cls,),
        {
            "__getitem__": __getitem__,
        },
    )


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
    data_loader: torch.utils.data.DataLoader,
    batch_sampler: torch.utils.data.Sampler[List[int]],
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
