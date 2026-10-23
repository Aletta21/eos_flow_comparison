import torch
import torch.nn.functional as F

from src import config
from src.data import get_cifar10_subset
from src.model import MLP
from src.utils import set_parameter_vector


# --------------------------------------------------
# Settings
# --------------------------------------------------

START_STEP = 5900
END_STEP = 6300
STRIDE = 5

POWER_ITERS = 20


def get_saved_weight(
    gd_data,
    step,
):
    saved_steps = torch.as_tensor(
        gd_data["saved_steps"]
    )

    matches = (
        saved_steps == step
    ).nonzero(
        as_tuple=True
    )[0]

    if len(matches) == 0:
        return None

    index = matches.item()

    return gd_data["trajectory"][index]


def flatten_tensors(tensors):
    return torch.cat(
        [
            tensor.reshape(-1)
            for tensor in tensors
        ]
    )


def hessian_vector_product(
    model,
    x,
    targets,
    vector,
):
    """
    Compute H v using automatic differentiation.
    """

    outputs = model(x)

    loss = F.mse_loss(
        outputs,
        targets,
    )

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=True,
    )

    flat_gradient = flatten_tensors(
        gradients
    )

    hv = torch.autograd.grad(
        flat_gradient,
        parameters,
        grad_outputs=vector,
    )

    return flatten_tensors(
        hv
    ).detach()


def top_hessian_eigenvalue(
    model,
    x,
    targets,
    initial_vector=None,
    num_iterations=20,
):
    """
    Estimate the largest Hessian eigenvalue
    using power iteration.

    Returns:
        eigenvalue
        eigenvector
    """

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if initial_vector is None:

        vector = torch.randn(
            num_parameters,
            device=x.device,
        )

    else:

        vector = initial_vector.clone().to(
            x.device
        )

    vector = vector / torch.norm(
        vector
    )

    # --------------------------------------------------
    # Power iteration
    # --------------------------------------------------

    for _ in range(
        num_iterations
    ):

        hv = hessian_vector_product(
            model=model,
            x=x,
            targets=targets,
            vector=vector,
        )

        hv_norm = torch.norm(
            hv
        )

        if hv_norm < 1e-12:
            break

        vector = (
            hv
            / hv_norm
        )

    # --------------------------------------------------
    # Rayleigh quotient
    # --------------------------------------------------

    hv = hessian_vector_product(
        model=model,
        x=x,
        targets=targets,
        vector=vector,
    )

    eigenvalue = torch.dot(
        vector,
        hv,
    ).item()

    return (
        eigenvalue,
        vector.detach(),
    )


def main():

    torch.manual_seed(
        config.SEED
    )

    device = torch.device(
        config.DEVICE
    )

    # --------------------------------------------------
    # Load training data
    # --------------------------------------------------

    x, labels = get_cifar10_subset(
        num_samples=config.NUM_SAMPLES,
        data_dir=config.DATA_DIR,
    )

    x = x.to(
        device
    ).float()

    labels = labels.to(
        device
    )

    targets = F.one_hot(
        labels,
        num_classes=config.NUM_CLASSES,
    ).float()

    # --------------------------------------------------
    # Load GD trajectory
    # --------------------------------------------------

    gd_data = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = MLP().to(
        device
    )

    # --------------------------------------------------
    # Measure dense sharpness
    # --------------------------------------------------

    steps = []
    sharpness_values = []

    previous_eigenvector = None

    threshold = (
        2.0
        / config.LR
    )

    print()
    print(
        "Dense sharpness measurement"
    )
    print(
        "---------------------------"
    )
    print(
        f"EOS threshold = {threshold:.4f}"
    )
    print()

    for step in range(
        START_STEP,
        END_STEP + 1,
        STRIDE,
    ):

        weights = get_saved_weight(
            gd_data,
            step,
        )

        if weights is None:

            print(
                f"Step {step}: "
                f"weights not stored"
            )

            continue

        weights = weights.to(
            device
        )

        set_parameter_vector(
            model,
            weights,
        )

        eigenvalue, eigenvector = (
            top_hessian_eigenvalue(
                model=model,
                x=x,
                targets=targets,
                initial_vector=previous_eigenvector,
                num_iterations=POWER_ITERS,
            )
        )

        # Warm start next step
        previous_eigenvector = (
            eigenvector
        )

        steps.append(
            step
        )

        sharpness_values.append(
            eigenvalue
        )

        print(
            f"Step {step:5d}"
            f" | sharpness = "
            f"{eigenvalue:8.4f}"
            f" | threshold = "
            f"{threshold:8.4f}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_path = (
        "outputs/trajectories/"
        "sharpness_dense.pt"
    )

    torch.save(
        {
            "steps": torch.tensor(
                steps
            ),
            "sharpness": torch.tensor(
                sharpness_values
            ),
            "threshold": threshold,
            "start_step": START_STEP,
            "end_step": END_STEP,
            "stride": STRIDE,
        },
        output_path,
    )

    print()
    print(
        f"Saved dense sharpness to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()