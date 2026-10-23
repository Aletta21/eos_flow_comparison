import torch
import torch.nn.functional as F

from src.utils import set_parameter_vector


def flat_gradient(
    model,
    weights,
    x,
    targets,
    create_graph=False,
):
    """
    Gradient of the full-batch loss with respect to
    the flattened model parameters.
    """

    set_parameter_vector(model, weights)

    model.zero_grad()

    output = model(x)

    loss = F.mse_loss(
        output,
        targets,
    )

    grads = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
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
):
    """
    Compute H(weights) @ vector without
    explicitly forming the Hessian.
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

    grad_dot_vector = torch.dot(
        flat_grad,
        vector,
    )

    hv = torch.autograd.grad(
        grad_dot_vector,
        params,
    )

    return torch.cat(
        [h.reshape(-1) for h in hv]
    ).detach()


def initialize_extent(delta):
    """
    Initial Sigma = delta delta^T.

    Store it in rank-1 eigendecomposition form:

        Sigma = V diag(lambdas) V^T
    """

    norm_delta = torch.norm(delta)

    if norm_delta < 1e-12:
        raise ValueError(
            "Initial rod delta is too small."
        )

    v = delta / norm_delta

    V = v.unsqueeze(1)

    lambdas = torch.tensor(
        [norm_delta ** 2],
        device=delta.device,
        dtype=delta.dtype,
    )

    return V, lambdas


def get_delta(V, lambdas):
    """
    Recover rod half-vector from the
    leading eigenpair of Sigma.
    """

    idx = torch.argmax(lambdas)

    lambda_max = torch.clamp(
        lambdas[idx],
        min=0.0,
    )

    return (
        torch.sqrt(lambda_max)
        * V[:, idx]
    )


def orthogonal_component(
    vector,
    V,
):
    """
    Compute the component of `vector`
    orthogonal to the columns of V.

    Uses two-pass modified Gram-Schmidt
    for better numerical stability.
    """

    if V.shape[1] == 0:
        return vector

    result = vector.clone()

    # Two passes greatly reduce loss of
    # orthogonality in float32.
    for _ in range(2):
        coefficients = (
            V.T @ result
        )

        result = (
            result
            - V @ coefficients
        )

    return result


def add_direction(
    V,
    direction,
    relative_tol=1e-6,
):
    """
    Add a normalized direction only if it
    contains a numerically meaningful component
    outside the current span.

    Assumes columns of V are already orthonormal.
    """

    original_norm = torch.norm(
        direction
    )

    if original_norm <= 1e-12:
        return V

    residual = orthogonal_component(
        direction,
        V,
    )

    residual_norm = torch.norm(
        residual
    )

    # Do not normalize numerical noise.
    if (
        residual_norm
        <= relative_tol * original_norm
    ):
        return V

    new_direction = (
        residual / residual_norm
    ).unsqueeze(1)

    return torch.cat(
        [
            V,
            new_direction,
        ],
        dim=1,
    )

def stabilize_extent(
    V,
    lambdas,
    rank,
):
    """
    Re-orthonormalize the low-rank representation

        Sigma = V diag(lambdas) V^T

    without changing Sigma.
    """

    # V = Q R
    Q, R = torch.linalg.qr(
        V,
        mode="reduced",
    )

    # Transform covariance into Q basis:
    #
    # Sigma
    # = V diag(lambda) V^T
    # = Q [R diag(lambda) R^T] Q^T
    small_cov = (
        R
        @ torch.diag(lambdas)
        @ R.T
    )

    # Symmetrize against numerical drift
    small_cov = 0.5 * (
        small_cov
        + small_cov.T
    )

    eigvals, eigvecs = torch.linalg.eigh(
        small_cov
    )

    order = torch.argsort(
        eigvals,
        descending=True,
    )

    keep = order[:rank]

    eigvals = torch.clamp(
        eigvals[keep],
        min=0.0,
    )

    eigvecs = eigvecs[
        :,
        keep,
    ]

    V_stable = (
        Q @ eigvecs
    )

    

    return (
        V_stable,
        eigvals,
    )

def update_extent(
    V,
    lambdas,
    grad_plus,
    grad_minus,
    eta,
    dt,
    rank=3,
):
    """
    Low-rank update for Sigma following
    Appendix C of the Rod Flow paper.
    """

    # ----------------------------------
    # Step 1: decay
    # ----------------------------------

    lambdas = (
        (1.0 - 2.0 * dt)
        * lambdas
    )

    lambdas = torch.clamp(
        lambdas,
        min=0.0,
    )

    old_rank = V.shape[1]

    # ----------------------------------
    # Steps 2-3:
    # augment basis with gradient
    # directions not already represented
    # ----------------------------------

    V_aug = V

    V_aug = add_direction(
        V_aug,
        grad_plus,
    )

    V_aug = add_direction(
        V_aug,
        grad_minus,
    )

    aug_rank = V_aug.shape[1]

    # ----------------------------------
    # Existing Sigma in augmented basis
    # ----------------------------------

    Sigma_small = torch.zeros(
        aug_rank,
        aug_rank,
        device=V.device,
        dtype=V.dtype,
    )

    Sigma_small[
        :old_rank,
        :old_rank,
    ] = torch.diag(lambdas)

    # ----------------------------------
    # Step 4:
    # gradient outer-product update
    # ----------------------------------

    g_plus_small = (
        V_aug.T @ grad_plus
    )

    g_minus_small = (
        V_aug.T @ grad_minus
    )

    factor = (
        eta ** 2
        * dt
        / 4.0
    )

    Sigma_small += factor * (
        torch.outer(
            g_plus_small,
            g_plus_small,
        )
        +
        torch.outer(
            g_minus_small,
            g_minus_small,
        )
    )

    # ----------------------------------
    # Step 5:
    # eigendecompose and truncate
    # ----------------------------------

    eigvals, eigvecs = torch.linalg.eigh(
        Sigma_small
    )

    order = torch.argsort(
        eigvals,
        descending=True,
    )

    keep = order[:rank]

    eigvals = torch.clamp(
        eigvals[keep],
        min=0.0,
    )

    eigvecs = eigvecs[:, keep]

    V_new = (
        V_aug @ eigvecs
    )

    # --------------------------------------------------
    # Numerical stabilization
    #
    # Preserve Sigma exactly while restoring an
    # orthonormal low-rank basis.
    # --------------------------------------------------

    V_new, eigvals = stabilize_extent(
        V_new,
        eigvals,
        rank=rank,
    )

    return V_new, eigvals


def rod_vector_field(
    model,
    center,
    V,
    lambdas,
    x,
    targets,
    eta,
):
    """
    Compute d(center)/dt plus endpoint
    gradients required for dSigma/dt.
    """

    delta = get_delta(
        V,
        lambdas,
    )

    w_plus = (
        center + delta
    )

    w_minus = (
        center - delta
    )

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

    grad_sum = (
        grad_plus
        + grad_minus
    )

    hv_plus = hessian_vector_product(
        model,
        w_plus,
        grad_sum,
        x,
        targets,
    )

    hv_minus = hessian_vector_product(
        model,
        w_minus,
        grad_sum,
        x,
        targets,
    )

    # Exact Rod Flow center ODE
    center_velocity = (
        -(eta / 2.0)
        * grad_sum
        -
        (eta ** 2 / 8.0)
        * (hv_plus + hv_minus)
    )

    return (
        center_velocity,
        grad_plus,
        grad_minus,
        delta,
    )


def rod_step(
    model,
    center,
    V,
    lambdas,
    x,
    targets,
    eta,
    dt=0.1,
    rank=3,
):
    """
    One Euler integration step of Rod Flow.
    """

    (
        center_velocity,
        grad_plus,
        grad_minus,
        delta,
    ) = rod_vector_field(
        model,
        center,
        V,
        lambdas,
        x,
        targets,
        eta,
    )

    center_new = (
        center
        + dt * center_velocity
    )

    V_new, lambdas_new = update_extent(
        V,
        lambdas,
        grad_plus,
        grad_minus,
        eta,
        dt,
        rank=rank,
    )

    return (
        center_new.detach(),
        V_new.detach(),
        lambdas_new.detach(),
        delta.detach(),
    )