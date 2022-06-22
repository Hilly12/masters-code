# PriFair

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hilly12/prifair/blob/main/PrivacyFairnessMNIST.ipynb)
[![Documentation Status](https://readthedocs.org/projects/fairlens/badge/?version=latest)](https://prifair.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A light-weight library for training and evaluating fair and privacy-preserving machine learning
models. Built on top of PyTorch and Opacus.

## Installation
```bash
pip install git+https://github.com/Hilly12/prifair.git
```

## Usage

Training a DP-SGD model with reweighing.

```python
import prifair as pf

# Setup

weights = pf.core.reweigh(train_labels)

train_loader, model, optimizer, accountant = pf.core.setup_weighted_dpsgd(
    data_loader=train_loader,
    model=model,
    optimizer=optimizer,
    weights=weights,
    target_epsilon=target_epsilon,
    target_delta=target_delta,
    max_grad_norm=max_grad_norm,
    epochs=epochs,
)

# Standard Opacus Training

for epoch in range(epochs):

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer,
    ) as memory_safe_data_loader:

        for images, target in tqdm(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

epsilon = accountant.get_epsilon(target_delta)
```
