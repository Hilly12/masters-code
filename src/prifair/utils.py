"""Utilities"""

from typing import List, Tuple, Union

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

    model.eval()

    losses = []
    accuracies = []

    with torch.no_grad():
        for batch in val_loader:
            data = batch[0].to(device)
            target = batch[1].to(device)

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
        for batch in test_loader:
            images = batch[0].to(device)

            output = model(images).detach().cpu()
            preds = np.argmax(output, axis=1).numpy()
            labels = batch[1].numpy()

            correct += (preds == labels).sum()

            pred_list.append(preds)
            raw_outputs.append(output)

    acc = correct / len(test_loader.dataset)

    if verbose:
        print(f"Test Accuracy: {acc * 100:.6f}")

    if return_outputs:
        return np.concatenate(pred_list), np.concatenate(raw_outputs)

    return np.concatenate(pred_list)


def predict(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()

    outputs = torch.zeros(0, dtype=torch.long)
    for batch in data_loader:
        output = model.to(device)(batch[0].to(device)).detach().cpu()
        probs = torch.argmax(output, dim=1)
        outputs = torch.cat((outputs, probs))

    return outputs


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


def _data_loader_with_batch_sampler(
    data_loader: torch.utils.data.DataLoader,
    batch_sampler: torch.utils.data.Sampler[List[int]],
    wrap: bool = False,
) -> torch.utils.data.DataLoader:

    data_loader_cls = torch.utils.data.DataLoader

    if wrap:
        data_loader_cls = _DataLoaderWrapper
        batch_sampler = _SamplerWrapper(batch_sampler)

    return data_loader_cls(
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


class Logger:
    def __init__(self, **kwargs):
        self.metrics = {
            "epochs": 0,
            "loss": [],
            "acc": [],
            "loss_per_epoch": [],
            "acc_per_epoch": [],
            "val_loss": [],
            "val_acc": [],
            "val_loss_per_epoch": [],
            "val_acc_per_epoch": [],
        }
        for key, value in kwargs.items():
            self.metrics[key] = value

    def record(self, epoch_losses, epoch_accs):
        self.metrics["loss"].extend(epoch_losses)
        self.metrics["acc"].extend(epoch_accs)
        self.metrics["loss_per_epoch"].append(np.mean(epoch_losses))
        self.metrics["acc_per_epoch"].append(np.mean(epoch_accs))
        self.metrics["epochs"] += 1

    def record_val(self, epoch_val_losses, epoch_val_accs):
        self.metrics["val_loss"].extend(epoch_val_losses)
        self.metrics["val_acc"].extend(epoch_val_accs)
        self.metrics["val_loss_per_epoch"].append(np.mean(epoch_val_losses))
        self.metrics["val_acc_per_epoch"].append(np.mean(epoch_val_accs))

    def log(self, epsilon=None, delta=None):
        print(
            f"Epoch: {self.metrics['epochs']}",
            f"Train Loss: {self.metrics['loss_per_epoch'][-1]:.2f}",
            f"Train Acc@1: {self.metrics['acc_per_epoch'][-1] * 100:.2f}",
            end=" ",
        )

        if (
            len(self.metrics["val_loss_per_epoch"]) > 0
            and len(self.metrics["val_acc_per_epoch"]) > 0
        ):
            print(
                f"Val Loss: {self.metrics['val_loss_per_epoch'][-1]:.2f}",
                f"Val Acc@1: {self.metrics['val_acc_per_epoch'][-1] * 100:.2f}",
                end=" ",
            )

        if epsilon is not None and delta is not None:
            print(f"(ε = {epsilon:.2f}, δ = {delta})", end=" ")

        print()

    def set_metric(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key] = value

    def get_metrics(self):
        return self.metrics


class _SamplerWrapper(torch.utils.data.Sampler):
    def __init__(self, sampler: torch.utils.data.Sampler):
        self.sampler = sampler
        self.last_sampled = None

    def __len__(self):
        return self.sampler.__len__()

    def __iter__(self):
        for x in self.sampler.__iter__():
            self.last_sampled = x
            yield x


class _DataLoaderWrapper(torch.utils.data.DataLoader):
    def get_indices(self):
        return self._index_sampler.last_sampled
