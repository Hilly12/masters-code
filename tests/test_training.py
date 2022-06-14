import torch

import prifair as pf

N_SAMPLES = 10000
VAL_SAMPLES = 1000
STUDENT_SAMPLES = 5000
INPUTS = 1000
OUTPUTS = 5
BATCH_SIZE = 256
MAX_PHYSICAL_BATCH_SIZE = 128
EPSILON = 2.0
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
N_TEACHERS = 4
N_GROUPS = 10
EPOCHS = 2


class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=INPUTS, out_features=OUTPUTS),
            torch.nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)


X = torch.randn(N_SAMPLES + VAL_SAMPLES, INPUTS)
Y = torch.randint(0, OUTPUTS, (N_SAMPLES + VAL_SAMPLES,))
student = torch.randn(STUDENT_SAMPLES, INPUTS)
groups = torch.randint(0, N_GROUPS, (N_SAMPLES,))
weights = torch.ones(N_SAMPLES) / N_SAMPLES

train_data = torch.utils.data.TensorDataset(X[:N_SAMPLES], Y[:N_SAMPLES])
val_data = torch.utils.data.TensorDataset(X[N_SAMPLES:], Y[N_SAMPLES:])
student_data = torch.utils.data.TensorDataset(student, torch.zeros(STUDENT_SAMPLES))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)
student_loader = torch.utils.data.DataLoader(student_data, batch_size=BATCH_SIZE)

model_class = MockModel
optim_class = torch.optim.NAdam
criterion = torch.nn.NLLLoss()


def test_vanilla():
    model, metrics = pf.training.train_vanilla(
        train_loader=train_loader,
        val_loader=val_loader,
        model_class=model_class,
        optim_class=optim_class,
        loss_fn=criterion,
        epochs=EPOCHS,
    )
    assert model is not None and metrics is not None


def test_dpsgd():
    model, metrics = pf.training.train_dpsgd(
        train_loader=train_loader,
        val_loader=val_loader,
        model_class=model_class,
        optim_class=optim_class,
        loss_fn=criterion,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
        epochs=EPOCHS,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
    )
    assert model is not None and metrics is not None


def test_dpsgd_weighted():
    model, metrics = pf.training.train_dpsgd_weighted(
        train_loader=train_loader,
        val_loader=val_loader,
        model_class=model_class,
        optim_class=optim_class,
        loss_fn=criterion,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
        epochs=EPOCHS,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        weighting="sensitive_attr",
        labels=groups.numpy(),
    )
    assert model is not None and metrics is not None

    model, metrics = pf.training.train_dpsgd_weighted(
        train_loader=train_loader,
        val_loader=val_loader,
        model_class=model_class,
        optim_class=optim_class,
        loss_fn=criterion,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
        epochs=EPOCHS,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        weighting="custom",
        weights=weights,
    )
    assert model is not None and metrics is not None


def test_dpsgdf():
    model, metrics = pf.training.train_dpsgdf(
        train_loader=train_loader,
        val_loader=val_loader,
        model_class=model_class,
        optim_class=optim_class,
        loss_fn=criterion,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        base_clipping_threshold=MAX_GRAD_NORM,
        epochs=EPOCHS,
        group_labels=groups,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
    )
    assert model is not None and metrics is not None


def test_pate():
    model, metrics = pf.training.train_pate(
        train_loader=train_loader,
        val_loader=val_loader,
        student_loader=student_loader,
        model_class=model_class,
        optim_class=optim_class,
        loss_fn=criterion,
        n_teachers=N_TEACHERS,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        epochs=EPOCHS,
    )
    assert model is not None and metrics is not None


def test_reweighed_sft_pate():
    model, metrics = pf.training.train_reweighed_sftpate(
        train_loader=train_loader,
        val_loader=val_loader,
        student_loader=student_loader,
        model_class=model_class,
        optim_class=optim_class,
        loss_fn=criterion,
        n_teachers=N_TEACHERS,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        epochs=EPOCHS,
        weights=weights,
    )
    assert model is not None and metrics is not None
