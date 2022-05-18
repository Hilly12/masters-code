"""Source code for training models. """

from typing import Mapping, Sequence, Tuple, Optional, Type

import numpy as np
import torch
from opacus import PrivacyEngine, GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.optimizers import DPOptimizer
from opacus.privacy_engine import forbid_accumulation_hook
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm

from .algorithms import reweigh, latent_reweigh
from .utils import NonUniformPoissonSampler, _data_loader_with_sampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_vanilla(
    train_loader: torch.utils.data.DataLoader,
    model_class: Type[torch.nn.Module],
    optim_class: Type[torch.optim.Optimizer],
    loss_fn: torch.nn.Module,
    epochs: int,
    **kwargs,
) -> Tuple[torch.nn.Module, Mapping[str, Sequence[float]]]:
    """Train a model in the given environment.

    Args:
        train_loader (torch.utils.data.DataLoader):
            The training data loader.
        model_class (Type[torch.nn.Module]):
            The class of the model to be used during training.
        optim_class (Type[torch.optim.Optimizer]):
            The class of the optimizer to be used during training.
        loss_fn (torch.nn.Module):
            The loss function.
        epochs (float):
            Number of epochs to train for.
        **kwargs:
            Passed to optim_class constructor.

    Returns:
        Tuple[torch.nn.Module, Mapping[str, Sequence[float]]]:
            The trained model and a dictionary consisting of train-time metrics.
    """

    model = model_class()
    model.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True
    )

    criterion = loss_fn
    optimizer = optim_class(model.parameters(), **kwargs)

    model.train()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []

        for images, target in tqdm(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            acc = (preds == labels).mean()

            epoch_losses.append(loss.item())
            epoch_accuracies.append(acc)

            loss.backward()
            optimizer.step()

        print(
            f"Train Epoch: {epoch + 1} "
            f"Loss: {np.mean(epoch_losses):.6f} "
            f"Acc@1: {np.mean(epoch_accuracies) * 100:.6f} "
        )

        losses.extend(epoch_losses)
        accuracies.extend(epoch_accuracies)

    metrics = {"loss": losses, "accuracy": accuracies}
    return model, metrics


def train_dpsgd(
    train_loader: torch.utils.data.DataLoader,
    model_class: Type[torch.nn.Module],
    optim_class: Type[torch.optim.Optimizer],
    loss_fn: torch.nn.Module,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
    epochs: int,
    max_physical_batch_size: int = 128,
    **kwargs,
) -> Tuple[torch.nn.Module, Mapping[str, Sequence[float]]]:
    """Train a model with DP-SGD in the given environment.

    Args:
        train_loader (torch.utils.data.DataLoader):
            The training data loader.
        model_class (Type[torch.nn.Module]):
            The class of the model to be used during training.
        optim_class (Type[torch.optim.Optimizer]):
            The class of the optimizer to be used during training.
        loss_fn (torch.nn.Module):
            The loss function.
        target_epsilon (float):
            Target epsilon for DP-SGD.
        target_delta (float):
            Target delta for DP-SGD.
        max_grad_norm (float):
            Gradient clipping bound for DP-SGD.
        epochs (float):
            Number of epochs to train for.
        max_physical_batch_size (int, optional):
            Maximum physical batch size for memory manager. Defaults to 128.
        **kwargs:
            Passed to optim_class constructor.

    Returns:
        Tuple[torch.nn.Module, Mapping[str, Sequence[float]]]:
            The trained model and a dictionary consisting of train-time metrics.
    """

    model = model_class()
    model = ModuleValidator.fix(model)
    assert not ModuleValidator.validate(model, strict=False)

    model = model.to(device)

    criterion = loss_fn
    optimizer = optim_class(model.parameters(), **kwargs)
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

    model.train()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_data_loader:

            for images, target in tqdm(memory_safe_data_loader):
                optimizer.zero_grad()
                model.zero_grad()

                images = images.to(device)
                target = target.to(device)

                output = model(images)
                loss = criterion(output, target)

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()

                acc = (preds == labels).mean()

                epoch_losses.append(loss.item())
                epoch_accuracies.append(acc)

                loss.backward()
                optimizer.step()

            epsilon = privacy_engine.get_epsilon(target_delta)

        print(
            f"Train Epoch: {epoch + 1} "
            f"Loss: {np.mean(epoch_losses):.6f} "
            f"Acc@1: {np.mean(epoch_accuracies) * 100:.6f} "
            f"(ε = {epsilon:.2f}, δ = {target_delta})"
        )

        losses.extend(epoch_losses)
        accuracies.extend(epoch_accuracies)

    metrics = {"loss": losses, "accuracy": accuracies}
    return model, metrics


def train_dpsgd_weighted(
    train_loader: torch.utils.data.DataLoader,
    model_class: Type[torch.nn.Module],
    optim_class: Type[torch.optim.Optimizer],
    loss_fn: torch.nn.Module,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
    weighting: str,
    epochs: int,
    max_physical_batch_size: int = 128,
    vae: Optional[torch.nn.Module] = None,
    alpha: Optional[float] = None,
    k: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[torch.nn.Module, Mapping[str, Sequence[float]]]:
    """Train a model with DP-SGD-W in the given environment.

    Args:
        train_loader (torch.utils.data.DataLoader):
            The training data loader.
        model_class (Type[torch.nn.Module]):
            The class of the model to be used during training.
        optim_class (Type[torch.optim.Optimizer]):
            The class of the optimizer to be used during training.
        loss_fn (torch.nn.Module):
            The loss function.
        target_epsilon (float):
            Target epsilon for DP-SGD-W.
        target_delta (float):
            Target delta for DP-SGD-W.
        max_grad_norm (float):
            Gradient clipping bound for DP-SGD-W.
        weighting (str):
            The scheme to use for weighting the data.
        epochs (int):
            The number of epochs to train for.
        max_physical_batch_size (int, optional):
            Maximum physical batch size for memory manager. Defaults to 128.
        weights (np.ndarray, optional):
            The weights to use for the reweighing if weighting is set to "custom".
        alpha (float, optional):
            The weight smoothing parameter for latent reweighing if weighting is set to "latent".
        k (float, optional):
            The number of latent bins for latent reweighing if weighing is set to "latent".
        **kwargs:
            Passed to optim_class constructor.

    Returns:
        Tuple[torch.nn.Module, Mapping[str, Sequence[float]]]:
            The trained model and a dictionary consisting of train-time metrics.
    """

    print("Reweighing...")

    if weighting == "latent":
        kws = {}
        if alpha is not None:
            kws["alpha"] = alpha
        if k is not None:
            kws["k"] = k
        if vae is None:
            raise ValueError("vae cannot be None if weighting is set to 'latent'")

        weights = latent_reweigh(train_loader, vae, **kws)

    elif weighting == "sensitive_attr":
        labels = kwargs.pop("labels")
        weights = reweigh(labels)

    elif weighting != "custom":
        raise ValueError(
            "weighting must be one of ['latent', 'sensitive_attr', 'custom']"
        )

    model = model_class()
    model = ModuleValidator.fix(model)
    assert not ModuleValidator.validate(model, strict=False)

    model.to(device)

    criterion = loss_fn
    optimizer = optim_class(model.parameters(), **kwargs)

    model = GradSampleModule(model)
    model.register_forward_pre_hook(forbid_accumulation_hook)

    sample_rate = 1 / len(train_loader)
    expected_batch_size = int(len(train_loader.dataset) * sample_rate)

    batch_sampler = NonUniformPoissonSampler(
        weights=weights, num_samples=len(train_loader.dataset), sample_rate=sample_rate
    )

    train_loader = _data_loader_with_sampler(train_loader, batch_sampler)

    accountant = RDPAccountant()
    optimizer = DPOptimizer(
        optimizer=optimizer,
        noise_multiplier=get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant=accountant.mechanism(),
        ),
        max_grad_norm=max_grad_norm,
        expected_batch_size=expected_batch_size,
    )

    optimizer.attach_step_hook(
        accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
    )

    model.train()
    losses = []
    accuracies = []

    print("Training Model...")

    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []

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

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()

                acc = (preds == labels).mean()

                epoch_losses.append(loss.item())
                epoch_accuracies.append(acc)

                loss.backward()
                optimizer.step()

            epsilon = accountant.get_epsilon(target_delta)

        print(
            f"Train Epoch: {epoch + 1} "
            f"Loss: {np.mean(epoch_losses):.6f} "
            f"Acc@1: {np.mean(epoch_accuracies) * 100:.6f} "
            f"(ε = {epsilon:.2f}, δ = {target_delta})"
        )

        losses.extend(epoch_losses)
        accuracies.extend(epoch_accuracies)

    metrics = {"loss": losses, "accuracy": accuracies}
    return model, metrics


def train_pate(
    train_loader: torch.utils.data.DataLoader,
    student_loader: torch.utils.data.DataLoader,
    model_class: Type[torch.nn.Module],
    optim_class: Type[torch.optim.Optimizer],
    loss_fn: torch.nn.Module,
    n_teachers: int,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    **kwargs,
):
    """Train a model with PATE in the given environment.

    Args:
        train_loader (torch.utils.data.DataLoader):
            The training dataloader used to train the teacher ensemble model.
        student_loader (torch.utils.data.DataLoader):
            The training dataloader used to train the student model.
        model_class (Type[torch.nn.Module]):
            The class of the model to be used during training.
        optim_class (Type[torch.optim.Optimizer]):
            The class of the optimizer to be used during training.
        loss_fn (torch.nn.Module):
            The loss function.
        n_teachers (int):
            The number of teachers to use in the ensemble.
        target_epsilon (float):
            Target epsilon for DP-SGD-W.
        target_delta (float):
            Target delta for DP-SGD-W.
        epochs (int):
            The number of epochs to train for.
        **kwargs:
            Passed to optim_class constructor.
    """

    teacher_loaders = []
    n_train = len(train_loader.dataset)
    data_size = n_train // n_teachers
    for i in range(n_teachers):
        idxs = list(range(i * data_size, max((i + 1) * data_size, n_train)))
        subset_data = torch.utils.data.Subset(train_loader.dataset, idxs)
        loader = torch.utils.data.DataLoader(
            subset_data, batch_size=train_loader.batch_size, shuffle=True
        )
        teacher_loaders.append(loader)

    criterion = loss_fn
    teachers = []

    print(f"Training {n_teachers} Teacher Models...")

    for i in range(n_teachers):
        model = model_class()
        model.to(device)
        optimizer = optim_class(model.parameters(), **kwargs)

        model.train()

        for epoch in tqdm(range(epochs)):
            epoch_losses = []
            epoch_accuracies = []

            for images, target in teacher_loaders[i]:
                optimizer.zero_grad()
                images = images.to(device)
                target = target.to(device)

                output = model(images)
                loss = criterion(output, target)

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()

                acc = (preds == labels).mean()

                epoch_losses.append(loss.item())
                epoch_accuracies.append(acc)

                loss.backward()
                optimizer.step()

        teachers.append(model.cpu())

        print(
            f"Teacher Model: {i + 1} "
            f"Loss: {np.mean(epoch_losses):.6f} "
            f"Acc@1: {np.mean(epoch_accuracies) * 100:.6f} "
        )

    print("Aggregating Teachers...")

    n_train_student = len(student_loader.dataset)
    preds = torch.zeros((n_teachers, n_train_student), dtype=torch.long)
    for i, model in enumerate(tqdm(teachers)):
        outputs = torch.zeros(0, dtype=torch.long)

        model.eval()
        for images, target in student_loader:
            output = model(images)
            probs = torch.argmax(output, dim=1)
            outputs = torch.cat((outputs, probs))

        preds[i] = outputs

    bins = preds.max() + 1
    label_counts = torch.zeros((n_train_student, bins), dtype=torch.long)
    for col in preds:
        label_counts[np.arange(n_train_student), col] += 1

    beta = 1 / target_epsilon
    label_counts += np.random.laplace(0, beta, 1)
    labels = label_counts.argmax(dim=1)

    def gen_student_loader(student_loader, labels):
        for i, (imgs, _) in enumerate(iter(student_loader)):
            yield imgs, labels[i * len(imgs) : (i + 1) * len(imgs)]

    student_model = model_class()
    student_model.to(device)

    optimizer = optim_class(student_model.parameters(), **kwargs)

    student_model.train()
    losses = []
    accuracies = []

    print("Training Student Model...")

    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []

        generator = gen_student_loader(student_loader, labels)
        for images, target in tqdm(generator, total=len(student_loader)):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            output = student_model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            lbls = target.detach().cpu().numpy()

            acc = (preds == lbls).mean()

            epoch_losses.append(loss.item())
            epoch_accuracies.append(acc)

            loss.backward()
            optimizer.step()

        print(
            f"Train Epoch: {epoch + 1} "
            f"Loss: {np.mean(epoch_losses):.6f} "
            f"Acc@1: {np.mean(epoch_accuracies) * 100:.6f} "
        )

        losses.extend(epoch_losses)
        accuracies.extend(epoch_accuracies)

    metrics = {"student_loss": losses, "student_accuracy": accuracies}
    return student_model, metrics
