import torch
import torch.nn.functional as F

from src.utils import set_parameter_vector


def loss_and_gradient(
    model,
    weights,
    x,
    targets,
    create_graph=False,
):
    """
    Full-batch MSE loss and flattened gradient.
    """

    set_parameter_vector(model, weights)

    model.zero_grad()

    output = model(x)

    loss = F.mse_loss(
        output,
        targets,
    )

    params = tuple(model.parameters())

    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
    )

    flat_grad = torch.cat(
        [g.reshape(-1) for g in grads]
    )

    return loss, flat_grad


def hessian_vector_product(
    model,
    weights,
    vector,
    x,
    targets,
    create_graph=False,
):
    """
    Compute H @ vector without forming H.
    """

    set_parameter_vector(model, weights)

    model.zero_grad()

    output = model(x)

    loss = F.mse_loss(
        output,
        targets,
    )

    params = tuple(model.parameters())

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
        create_graph=create_graph,
    )

    flat_hv = torch.cat(
        [h.reshape(-1) for h in hv]
    )

    return flat_hv


def top_eigenpair_warmstart(
    model,
    weights,
    x,
    targets,
    v=None,
    num_iters=10,
):
    """
    Power iteration for the top Hessian eigenpair.

    Reuses the previous eigenvector as a warm start.
    """

    if v is None:

        v = torch.randn_like(
            weights
        )

        v = v / torch.norm(v)

    else:

        v = v / torch.norm(v)

    for _ in range(num_iters):

        hv = hessian_vector_product(
            model=model,
            weights=weights,
            vector=v,
            x=x,
            targets=targets,
            create_graph=False,
        ).detach()

        norm = torch.norm(hv)

        if norm < 1e-12:
            break

        v = hv / norm

    hv = hessian_vector_product(
        model=model,
        weights=weights,
        vector=v,
        x=x,
        targets=targets,
        create_graph=False,
    ).detach()

    eigenvalue = torch.dot(
        v,
        hv,
    )

    return (
        eigenvalue.item(),
        v.detach(),
    )


def sharpness_gradient(
    model,
    weights,
    eigenvector,
    x,
    targets,
):
    """
    Compute

        grad_w [u^T H(w) u]

    treating u as fixed.

    This is the third-derivative quantity needed
    by Central Flow.
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

    # First derivative
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
    )

    flat_grad = torch.cat(
        [g.reshape(-1) for g in grads]
    )

    # H u
    grad_dot_u = torch.dot(
        flat_grad,
        eigenvector,
    )

    hv = torch.autograd.grad(
        grad_dot_u,
        params,
        create_graph=True,
    )

    flat_hv = torch.cat(
        [h.reshape(-1) for h in hv]
    )

    # u^T H u
    rayleigh = torch.dot(
        eigenvector,
        flat_hv,
    )

    # grad_w (u^T H u)
    third = torch.autograd.grad(
        rayleigh,
        params,
    )

    q = torch.cat(
        [t.reshape(-1) for t in third]
    )

    return q.detach()


def central_flow_substep(
    model,
    weights,
    x,
    targets,
    eta,
    dt,
    eigenvector=None,
    eig_iters=10,
    critical_factor=1.95,
):
    """
    One rank-1 Central Flow substep.

    If lambda < critical_factor / eta:
        ordinary gradient flow.

    Otherwise:
        activate the oscillation covariance correction.
    """

    loss, gradient = loss_and_gradient(
        model=model,
        weights=weights,
        x=x,
        targets=targets,
        create_graph=False,
    )

    gradient = gradient.detach()

    eigenvalue, eigenvector = (
        top_eigenpair_warmstart(
            model=model,
            weights=weights,
            x=x,
            targets=targets,
            v=eigenvector,
            num_iters=eig_iters,
        )
    )

    critical_threshold = (
        critical_factor / eta
    )

    stability_threshold = (
        2.0 / eta
    )

    # --------------------------------------------------
    # Below critical region:
    # Central Flow reduces to gradient flow
    # --------------------------------------------------

    if eigenvalue <= critical_threshold:

        velocity = (
            -eta * gradient
        )

        sigma = 0.0

        new_weights = (
            weights
            + dt * velocity
        )

        predicted_loss = (
            loss.item()
        )

        return {
            "weights": new_weights.detach(),
            "eigenvector": eigenvector,
            "eigenvalue": eigenvalue,
            "sigma": sigma,
            "loss": loss.item(),
            "predicted_loss": predicted_loss,
        }

    # --------------------------------------------------
    # Critical regime
    # --------------------------------------------------

    q = sharpness_gradient(
        model=model,
        weights=weights,
        eigenvector=eigenvector,
        x=x,
        targets=targets,
    )

    # Stable gradient-flow component
    dw_stable = (
        -eta * gradient
    )

    # How sharpness changes under gradient flow
    d_lambda_stable = torch.dot(
        q,
        dw_stable,
    ).item()

    # Effect of covariance correction on motion
    dw_sigma = (
        -0.5 * eta * q
    )

    # Effect of that correction on sharpness
    d_lambda_sigma = (
        -torch.dot(
            q,
            dw_sigma,
        ).item()
    )

    # Predicted distance from the stability boundary
    residual_stable = (
        stability_threshold
        - (
            eigenvalue
            + dt * d_lambda_stable
        )
    )

    denominator = (
        dt * d_lambda_sigma
    )

    # Rank-1 complementarity solution
    if (
        residual_stable < 0.0
        and denominator > 1e-12
    ):

        sigma = (
            -residual_stable
            / denominator
        )

    else:

        sigma = 0.0

    velocity = (
        dw_stable
        + sigma * dw_sigma
    )

    new_weights = (
        weights
        + dt * velocity
    )

    predicted_loss = (
        loss.item()
        + sigma / eta
    )

    return {
        "weights": new_weights.detach(),
        "eigenvector": eigenvector,
        "eigenvalue": eigenvalue,
        "sigma": sigma,
        "loss": loss.item(),
        "predicted_loss": predicted_loss,
    }