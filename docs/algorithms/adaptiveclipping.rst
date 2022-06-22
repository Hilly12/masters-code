=================
Adaptive Clipping
=================

Training
--------

.. code-block:: python
    :linenos:

    import prifair as pf
    from prifair.utils import IndexCachingBatchMemoryManager

    n_groups = group_labels.max() + 1
    train_loader, model, optimizer, accountant = pf.core.setup_adaptive_clipped_dpsgd(
        data_loader=train_loader,
        model=model,
        optimizer=optimizer,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        base_clipping_threshold=base_clipping_threshold,
        epochs=epochs,
        n_groups=n_groups,
        log_thresholds=log_thresholds,
    )

    for epoch in range(epochs):

        with IndexCachingBatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_data_loader:

            for images, target in tqdm(memory_safe_data_loader):
                optimizer.zero_grad()

                idxs = memory_safe_data_loader.get_indices()
                batch_groups = group_labels[idxs].to(device)

                images = images.to(device)
                target = target.to(device)

                output = model(images)
                loss = criterion(output, target)

                loss.backward()
                optimizer.set_batch_params(group_labels=batch_groups)
                optimizer.step()

    epsilon = accountant.get_epsilon(target_delta)


Advanced
--------
