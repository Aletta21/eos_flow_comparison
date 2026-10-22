import torch
import torch.nn.functional as F

from src.utils import set_parameter_vector


def flat_gradient(
    model,
    weights,
    x,
    targets,
):
    """
    Compute the full-batch MSE gradient
    with respect to the flattened parameters.
    """

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

    grads = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
    )

    flat_grad = torch.cat(
        [g.reshape(-1) for g in grads]
    )

    return loss.item(), flat_grad.detach()


def hessian_vector_product(
    model,
    weights,
    vector,
    x,
    targets,
):
    """
    Compute H(weights) @ vector
    without constructing the Hessian.
    """

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
        create_graph=True,
    )

    flat_grad = torch.cat(
        [g.reshape(-1) for g in grads]
    )

    grad_dot_v = torch.dot(
        flat_grad,
        vector,
    )

    hv = torch.autograd.grad(
        grad_dot_v,
        params,
    )

    return torch.cat(
        [h.reshape(-1) for h in hv]
    ).detach()


def edge_substep(
    model,
    center,
    direction,
    amplitude,
    x,
    targets,
    eta,
    rho,
    epsilon=1e-8,
):
    """
    One Edge Gradient Descent substep.

    State
    -----
    center:
        Slow center of the oscillation.

    direction:
        Unit oscillation direction.

    amplitude:
        Half-length of the oscillation.

    rho:
        Numerical integration step.
        Typically rho = eta / K.
    """

    # --------------------------------------------------
    # 1. Oscillation endpoints
    # --------------------------------------------------

    w_plus = (
        center
        + amplitude * direction
    )

    w_minus = (
        center
        - amplitude * direction
    )

    # --------------------------------------------------
    # 2. Endpoint gradients
    # --------------------------------------------------

    _, grad_plus = flat_gradient(
        model,
        w_plus,
        x,
        targets,
    )

    _, grad_minus = flat_gradient(
        model,
        w_minus,
        x,
        targets,
    )

    sym_grad = 0.5 * (
        grad_plus
        + grad_minus
    )

    # --------------------------------------------------
    # 3. Hessian-vector product at the center
    # --------------------------------------------------

    hv = hessian_vector_product(
        model=model,
        weights=center,
        vector=direction,
        x=x,
        targets=targets,
    )

    sharpness = torch.norm(
        hv
    ).item()

    # --------------------------------------------------
    # 4. Save old state
    #
    # The three updates use quantities evaluated
    # at the same current Edge Flow state.
    # --------------------------------------------------

    old_center = center
    old_amplitude = amplitude

    # --------------------------------------------------
    # 5. Center update
    # --------------------------------------------------

    center = (
        old_center
        - rho * sym_grad
    )

    # --------------------------------------------------
    # 6. Direction update
    #
    # One power-iteration step:
    #
    #     u <- H u / ||H u||
    # --------------------------------------------------

    hv_norm = torch.norm(hv)

    if hv_norm > 1e-12:
        direction = (
            hv / hv_norm
        )

    # --------------------------------------------------
    # 7. Amplitude update
    #
    # x <- x [1 + rho (S - 2/eta)]
    # --------------------------------------------------

    growth_factor = (
        1.0
        + rho
        * (
            sharpness
            - 2.0 / eta
        )
    )

    amplitude = (
        old_amplitude
        * growth_factor
    )

    # Paper uses a small base oscillation level
    amplitude = torch.clamp(
        amplitude,
        min=epsilon,
    )

    return {
        "center": center.detach(),
        "direction": direction.detach(),
        "amplitude": amplitude.detach(),
        "sharpness": sharpness,
        "growth_factor": growth_factor,
    }