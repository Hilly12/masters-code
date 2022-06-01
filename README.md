# PriFair

A library of tools for training private and fair machine learning models.
Supplementary to my thesis.

## Installation
```bash
pip install git+https://github.com/Hilly12/prifair.git
```

## Usage

Debiased VAE

```python
vae_train_loader = pf.data.WeightedDataLoader(unlabelled_data_loader)

for epoch in range(num_epochs):
    for img, _ in tqdm(vae_train_loader):
        optimizer.zero_grad()

        x = img.to(device)
        recon_x, mu, logvar = vae_model(x)

        loss = vae.loss_function(recon_x, x, mu, logvar, beta)

        loss.backward()
        optimizer.step()

    # Update Weights
    weights = pf.core.latent_reweigh(vae_train_loader, vae_model)
    vae_train_loader.update_weights(weights)
```

Reweighed DPSGD

```python
import prifair as pf

# Setup

# weights = pf.core.reweigh(train_labels)
weights = pf.core.latent_reweigh(train_loader, vae_model)

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
