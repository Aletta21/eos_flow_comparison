import torch
import torch.nn.functional as F

from src.utils import set_parameter_vector


# ============================================================
# LOSS + GRADIENT
# ============================================================


def loss_and_gradient(
    model,
    weights,
    x,
    targets,
):
    """
    Full-batch loss and gradient at a flattened
    parameter vector.
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


# ============================================================
# HESSIAN VECTOR PRODUCT
# ============================================================


def hessian_vector_product(
    model,
    weights,
    vector,
    x,
    targets,
):
    """
    Compute H(weights) @ vector without explicitly
    constructing the full Hessian.
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
        [
            g.reshape(-1)
            for g in grads
        ]
    )

    directional_derivative = torch.dot(
        flat_grad,
        vector,
    )

    hv = torch.autograd.grad(
        directional_derivative,
        params,
    )

    flat_hv = torch.cat(
        [
            h.reshape(-1)
            for h in hv
        ]
    )

    return flat_hv.detach()


# ============================================================
# MULTIPLE HESSIAN EIGENVECTORS
# ============================================================


def top_eigenspace(
    model,
    weights,
    x,
    targets,
    n_eig=6,
    warm_U=None,
    num_iters=8,
):
    """
    HVP-based block subspace iteration.

    Tracks several dominant Hessian eigen-directions
    simultaneously.

    warm_U allows the previous eigenspace to be reused
    as initialization.
    """

    n_params = weights.numel()

    # --------------------------------------------------
    # Initialize subspace
    # --------------------------------------------------

    if warm_U is None:

        Q = torch.randn(
            n_params,
            n_eig,
            device=weights.device,
            dtype=weights.dtype,
        )

    else:

        Q = warm_U.to(
            weights.device
        )

        # Add random directions if needed.
        if Q.shape[1] < n_eig:

            extra = torch.randn(
                n_params,
                n_eig - Q.shape[1],
                device=weights.device,
                dtype=weights.dtype,
            )

            Q = torch.cat(
                [
                    Q,
                    extra,
                ],
                dim=1,
            )

        elif Q.shape[1] > n_eig:

            Q = Q[
                :,
                :n_eig,
            ]

    # Initial orthonormalization.
    Q, _ = torch.linalg.qr(
        Q,
        mode="reduced",
    )

    # --------------------------------------------------
    # Block power / subspace iterations
    # --------------------------------------------------

    for _ in range(
        num_iters
    ):

        HQ_columns = []

        for j in range(
            Q.shape[1]
        ):

            hv = hessian_vector_product(
                model=model,
                weights=weights,
                vector=Q[:, j],
                x=x,
                targets=targets,
            )

            HQ_columns.append(
                hv
            )

        HQ = torch.stack(
            HQ_columns,
            dim=1,
        )

        Q, _ = torch.linalg.qr(
            HQ,
            mode="reduced",
        )

    # --------------------------------------------------
    # Rayleigh-Ritz refinement
    # --------------------------------------------------

    HQ_columns = []

    for j in range(
        Q.shape[1]
    ):

        hv = hessian_vector_product(
            model=model,
            weights=weights,
            vector=Q[:, j],
            x=x,
            targets=targets,
        )

        HQ_columns.append(
            hv
        )

    HQ = torch.stack(
        HQ_columns,
        dim=1,
    )

    small_H = (
        Q.T @ HQ
    )

    # Protect against small numerical asymmetries.
    small_H = 0.5 * (
        small_H
        + small_H.T
    )

    eigvals, eigvecs = torch.linalg.eigh(
        small_H
    )

    order = torch.argsort(
        eigvals,
        descending=True,
    )

    eigvals = eigvals[
        order
    ]

    eigvecs = eigvecs[
        :,
        order
    ]

    U = (
        Q @ eigvecs
    )

    return (
        eigvals.detach(),
        U.detach(),
    )


# ============================================================
# THIRD DERIVATIVE
# ============================================================


def third_derivative_pair(
    model,
    weights,
    u,
    v,
    x,
    targets,
):
    """
    Compute

        grad_w [u^T H(w) v]

    while treating u and v as fixed.
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

    # --------------------------------------------------
    # First derivative
    # --------------------------------------------------

    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
    )

    flat_grad = torch.cat(
        [
            g.reshape(-1)
            for g in grads
        ]
    )

    # --------------------------------------------------
    # H v
    # --------------------------------------------------

    grad_v = torch.dot(
        flat_grad,
        v,
    )

    Hv = torch.autograd.grad(
        grad_v,
        params,
        create_graph=True,
    )

    flat_Hv = torch.cat(
        [
            h.reshape(-1)
            for h in Hv
        ]
    )

    # --------------------------------------------------
    # u^T H v
    # --------------------------------------------------

    scalar = torch.dot(
        u,
        flat_Hv,
    )

    # --------------------------------------------------
    # grad_w (u^T H v)
    # --------------------------------------------------

    third = torch.autograd.grad(
        scalar,
        params,
    )

    q = torch.cat(
        [
            t.reshape(-1)
            for t in third
        ]
    )

    return q.detach()


# ============================================================
# SYMMETRIC MATRIX UTILITIES
# ============================================================


def upper_indices(
    k,
    device,
):
    return torch.triu_indices(
        k,
        k,
        device=device,
    )


def symmetric_metric(
    k,
    device,
    dtype,
):
    """
    Entries corresponding to X_ii have weight 1.

    Entries corresponding to X_ij, i != j, have
    weight 2 because the symmetric matrix contains
    both X_ij and X_ji.
    """

    i, j = upper_indices(
        k,
        device,
    )

    metric = torch.where(
        i == j,
        torch.ones_like(
            i,
            dtype=dtype,
        ),
        2.0
        * torch.ones_like(
            i,
            dtype=dtype,
        ),
    )

    return metric


def upper_to_symmetric(
    values,
    k,
):
    """
    Convert upper-triangular coordinates into
    a symmetric k x k matrix.
    """

    device = values.device
    dtype = values.dtype

    i, j = upper_indices(
        k,
        device,
    )

    matrix = torch.zeros(
        k,
        k,
        device=device,
        dtype=dtype,
    )

    matrix[
        i,
        j,
    ] = values

    matrix[
        j,
        i,
    ] = values

    return matrix


def symmetric_to_upper(
    matrix,
):
    """
    Extract upper-triangular coordinates from
    a symmetric matrix.
    """

    k = matrix.shape[0]

    i, j = upper_indices(
        k,
        matrix.device,
    )

    return matrix[
        i,
        j,
    ]


# ============================================================
# CENTRAL FLOW STEP
# ============================================================


def central_flow_substep(
    model,
    weights,
    x,
    targets,
    eta,
    dt,
    sdcp_solver,
    warm_U=None,
    n_eig=6,
    eig_iters=8,
    track_threshold=1.5,
    sdcp_threshold=1.95,
):
    """
    One multi-direction Central Flow substep.

    Two thresholds are kept separate:

        track_threshold / eta

    describes the broader high-curvature region that
    we want the eigensolver to track.

    The actual SDCP critical subspace uses

        sdcp_threshold / eta.

    For eta = 0.025:

        tracking threshold = 1.5 / eta = 60
        SDCP threshold     = 1.95 / eta = 78
        EOS boundary       = 2.0 / eta = 80

    Only eigen-directions above the SDCP threshold enter
    the covariance correction.
    """

    # --------------------------------------------------
    # Gradient
    # --------------------------------------------------

    loss, gradient = loss_and_gradient(
        model=model,
        weights=weights,
        x=x,
        targets=targets,
    )

    # --------------------------------------------------
    # Dominant Hessian eigenspace
    # --------------------------------------------------

    eigvals, U = top_eigenspace(
        model=model,
        weights=weights,
        x=x,
        targets=targets,
        n_eig=n_eig,
        warm_U=warm_U,
        num_iters=eig_iters,
    )

    # --------------------------------------------------
    # Thresholds
    # --------------------------------------------------

    tracking_threshold_raw = (
        track_threshold / eta
    )

    critical_threshold = (
        sdcp_threshold / eta
    )

    # top_eigenspace currently always returns n_eig
    # directions. tracking_threshold_raw is retained
    # explicitly because tracking and SDCP thresholds
    # are conceptually different.
    _ = tracking_threshold_raw

    critical_mask = (
        eigvals
        > critical_threshold
    )

    critical_count = int(
        critical_mask.sum().item()
    )

    # --------------------------------------------------
    # Ordinary gradient flow velocity
    # --------------------------------------------------

    stable_velocity = (
        -eta
        * gradient
    )

    # --------------------------------------------------
    # No critical direction:
    # ordinary gradient flow
    # --------------------------------------------------

    if critical_count == 0:

        new_weights = (
            weights
            + dt
            * stable_velocity
        )

        return {
            "weights": (
                new_weights.detach()
            ),
            "eigvals": (
                eigvals.detach()
            ),
            "U": (
                U.detach()
            ),
            "critical_count": 0,
            "sigma_eigs": (
                torch.zeros_like(
                    eigvals
                )
            ),
            "sigma_trace": 0.0,
            "loss": loss,
        }

    # --------------------------------------------------
    # Critical eigenvectors
    # --------------------------------------------------

    # Eigenvalues are sorted descending, therefore
    # all values above the threshold appear first.
    crit_eigs = eigvals[
        :critical_count
    ]

    crit_U = U[
        :,
        :critical_count
    ]

    k = critical_count

    # --------------------------------------------------
    # All unique (i,j) Hessian-pair derivatives
    # --------------------------------------------------

    pair_i, pair_j = torch.triu_indices(
        k,
        k,
        device=weights.device,
    )

    q_vectors = []

    for a in range(
        pair_i.numel()
    ):

        i = pair_i[a].item()
        j = pair_j[a].item()

        q = third_derivative_pair(
            model=model,
            weights=weights,
            u=crit_U[:, i],
            v=crit_U[:, j],
            x=x,
            targets=targets,
        )

        q_vectors.append(
            q
        )

    # Shape:
    #
    # m x num_parameters
    #
    # where m = k(k+1)/2
    Q = torch.stack(
        q_vectors,
        dim=0,
    )

    metric = symmetric_metric(
        k,
        weights.device,
        weights.dtype,
    )

    # --------------------------------------------------
    # Hessian change caused by ordinary gradient flow
    # --------------------------------------------------

    dH_stable_upper = (
        Q
        @ stable_velocity
    )

    dH_stable = upper_to_symmetric(
        dH_stable_upper,
        k,
    )

    # --------------------------------------------------
    # Residual to EOS boundary
    # --------------------------------------------------

    I = torch.eye(
        k,
        device=weights.device,
        dtype=weights.dtype,
    )

    predicted_H = (
        torch.diag(
            crit_eigs
        )
        + dt
        * dH_stable
    )

    # EOS condition:
    #
    # H <= (2 / eta) I
    A = (
        (2.0 / eta)
        * I
        - predicted_H
    )

    # Symmetrize for numerical robustness.
    A = 0.5 * (
        A
        + A.T
    )

    # --------------------------------------------------
    # Build linear covariance-response operator
    #
    # Correction velocity:
    #
    #   -eta/2 sum_ij X_ij q_ij
    #
    # in upper-triangular coordinates.
    # --------------------------------------------------

    gram = (
        Q @ Q.T
    )

    B_upper = (
        dt
        * eta
        / 2.0
        * gram
        @ torch.diag(
            metric
        )
    )

    # --------------------------------------------------
    # Solve PSD covariance
    # --------------------------------------------------

    X_np = sdcp_solver.solve(
        A.detach()
        .cpu()
        .numpy(),

        B_upper.detach()
        .cpu()
        .numpy(),
    )

    X = torch.tensor(
        X_np,
        device=weights.device,
        dtype=weights.dtype,
    )

    # Numerical symmetrization.
    X = 0.5 * (
        X
        + X.T
    )

    X_upper = symmetric_to_upper(
        X
    )

    # --------------------------------------------------
    # Central Flow covariance correction
    # --------------------------------------------------

    correction_coefficients = (
        metric
        * X_upper
    )

    correction_velocity = (
        -0.5
        * eta
        * (
            Q.T
            @ correction_coefficients
        )
    )

    velocity = (
        stable_velocity
        + correction_velocity
    )

    new_weights = (
        weights
        + dt
        * velocity
    )

    # --------------------------------------------------
    # Covariance eigenvalues
    # --------------------------------------------------

    sigma_eigs_small = torch.linalg.eigvalsh(
        X
    )

    sigma_eigs_small = torch.flip(
        sigma_eigs_small,
        dims=[0],
    )

    sigma_eigs = torch.zeros_like(
        eigvals
    )

    sigma_eigs[
        :k
    ] = sigma_eigs_small

    sigma_trace = torch.trace(
        X
    ).item()

    return {
        "weights": (
            new_weights.detach()
        ),
        "eigvals": (
            eigvals.detach()
        ),
        "U": (
            U.detach()
        ),
        "critical_count": (
            critical_count
        ),
        "sigma_eigs": (
            sigma_eigs.detach()
        ),
        "sigma_trace": (
            sigma_trace
        ),
        "loss": (
            loss
        ),
    }