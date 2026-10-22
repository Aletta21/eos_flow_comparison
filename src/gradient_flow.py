import torch
import torch.nn.functional as F

from src.utils import set_parameter_vector


def flat_gradient(
    model,
    weights,
    x,
    targets,
):
    set_parameter_vector(
        model,
        weights,
    )

    model.zero_grad()

    output = model(x)

    loss = F.mse_loss(
        output,
        targets,
    )

    params = tuple(
        model.parameters()
    )

    grads = torch.autograd.grad(
        loss,
        params,
    )

    gradient = torch.cat(
        [
            g.reshape(-1)
            for g in grads
        ]
    )

    return (
        loss.item(),
        gradient.detach(),
    )


def gradient_flow_step(
    model,
    weights,
    x,
    targets,
    eta,
    dt,
):
    loss, gradient = flat_gradient(
        model=model,
        weights=weights,
        x=x,
        targets=targets,
    )

    velocity = (
        -eta * gradient
    )

    new_weights = (
        weights
        + dt * velocity
    )

    return (
        new_weights.detach(),
        loss,
    )