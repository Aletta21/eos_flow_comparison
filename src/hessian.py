import torch


def hessian_vector_product(loss, params, vector):
    """
    Compute H @ v without explicitly forming the Hessian.

    Parameters
    ----------
    loss : torch.Tensor
        Scalar loss.
    params : list[torch.nn.Parameter]
        Model parameters.
    vector : torch.Tensor
        Flattened vector v.

    Returns
    -------
    hvp : torch.Tensor
        Flattened Hessian-vector product H @ v.
    """

    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
    )

    flat_grads = torch.cat([g.reshape(-1) for g in grads])

    grad_dot_v = torch.dot(flat_grads, vector)

    hv = torch.autograd.grad(
        grad_dot_v,
        params,
        retain_graph=True,
    )

    hv_flat = torch.cat([h.reshape(-1) for h in hv])

    return hv_flat


def top_hessian_eigenpair(
    model,
    loss_fn,
    x,
    y,
    num_iters=30,
    tol=1e-6,
):
    """
    Estimate the largest Hessian eigenvalue/eigenvector
    using power iteration.
    """

    params = [p for p in model.parameters() if p.requires_grad]

    # Total number of parameters
    n_params = sum(p.numel() for p in params)

    # Random initial vector
    v = torch.randn(
        n_params,
        device=x.device,
        dtype=x.dtype,
    )

    v = v / torch.norm(v)

    eigenvalue = None

    for _ in range(num_iters):

        model.zero_grad()

        output = model(x)
        loss = loss_fn(output, y)

        hv = hessian_vector_product(
            loss,
            params,
            v,
        )

        hv_norm = torch.norm(hv)

        if hv_norm < 1e-12:
            break

        v_new = hv / hv_norm

        new_eigenvalue = torch.dot(v_new, hv).item()

        if eigenvalue is not None:
            if abs(new_eigenvalue - eigenvalue) < tol:
                v = v_new
                eigenvalue = new_eigenvalue
                break

        v = v_new
        eigenvalue = new_eigenvalue

    return eigenvalue, v.detach()


if __name__ == "__main__":
    import torch.nn as nn
    import torch.nn.functional as F

    from src.model import MLP

    torch.manual_seed(0)

    model = MLP(
        input_dim=3 * 32 * 32,
        hidden_dim=20,
        num_classes=4,
    )

    # Dummy CIFAR-like batch
    x = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 4, (16,))

    targets = F.one_hot(
        labels,
        num_classes=4,
    ).float()

    loss_fn = nn.MSELoss()

    eigenvalue, eigenvector = top_hessian_eigenpair(
        model,
        loss_fn,
        x,
        targets,
        num_iters=20,
    )

    print("Top Hessian eigenvalue:", eigenvalue)
    print("Eigenvector shape:", eigenvector.shape)