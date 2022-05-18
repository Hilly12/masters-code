"""Algorithms to supplement the training code. """

from typing import Sequence

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reweigh(labels: Sequence[int]) -> np.ndarray:
    """Returns the inverse weighting for each sample in the dataset.

    Args:
        labels (Sequence[int]):
            The relevant sensitive group labels of the dataset.

    Returns:
        np.ndarray:
            The inverse weighting for each sample in the dataset.
    """

    num_samples = len(labels)
    labels = np.array(labels)

    sensitive_groups = []
    for value, counts in np.unique(labels, return_counts=True):
        sensitive_groups.append((labels == value, counts))

    n_unique = len(sensitive_groups)
    target_prob = 1 / n_unique

    weights = np.zeros(num_samples)
    for mask, counts in sensitive_groups:
        weights[mask] = target_prob / counts

    return weights


def latent_reweigh(
    train_loader: torch.utils.data.DataLoader,
    vae: torch.nn.Module,
    alpha: float = 0.01,
    k: int = 16,
) -> np.ndarray:
    """Returns the inverse weighting for each sample in the dataset computed
    using the latent distributions.

    Args:
        train_loader (torch.utils.data.DataLoader):
            The relevant training data loader.
        vae (torch.nn.Module):
            The relevant VAE model.
        alpha (float):
            The hyperparameter for the latent space. Defaults to 0.01.
        k (int):
            The number of samples to use for the latent space. Defaults to 16.

    Returns:
        np.ndarray:
            The inverse weighting for each sample in the dataset.
    """

    dataloader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=train_loader.batch_size
    )

    mus = []
    for imgs, _ in dataloader:
        mu, _ = vae.encode(imgs.to(device))
        mus.append(mu.cpu().detach().numpy())

    mu = np.concatenate(mus)

    bin_edges = np.histogram_bin_edges(mu.reshape(-1), bins=k)
    bin_edges[0] = float("-inf")
    bin_edges[-1] = float("inf")

    weights = np.zeros(mu.shape[0])
    latent_dim = mu.shape[1]
    for i in range(latent_dim):
        hist = np.histogram(mu[:, i], density=True, bins=bin_edges)[0]
        bin_idxs = np.digitize(mu[:, i], bin_edges)

        hist += alpha
        hist = hist / np.sum(hist)

        p = 1.0 / (hist[bin_idxs - 1])
        p /= p.sum()

        weights = np.maximum(weights, p)

    weights /= weights.sum()

    return weights
