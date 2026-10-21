import torch
import torch.nn.functional as F

from src import config
from src.data import get_cifar10_subset
from src.model import MLP
from src.utils import (
    get_parameter_vector,
    set_parameter_vector,
)


def get_saved_weight(data, step):
    saved_steps = data["saved_steps"]
    trajectory = data["trajectory"]

    matches = (
        saved_steps == step
    ).nonzero(as_tuple=True)[0]

    if len(matches) == 0:
        raise ValueError(
            f"Step {step} was not saved."
        )

    idx = matches.item()

    return trajectory[idx]


def loss_and_gradient(
    model,
    weights,
    x,
    targets,
):
    """
    Set model parameters to 'weights',
    compute full-batch MSE loss,
    and return flattened gradient.
    """

    set_parameter_vector(
        model,
        weights,
    )

    model.zero_grad()

    outputs = model(x)

    loss = F.mse_loss(
        outputs,
        targets,
    )

    loss.backward()

    gradient = torch.cat([
        p.grad.reshape(-1)
        for p in model.parameters()
    ]).detach().clone()

    return loss.item(), gradient


def main():

    torch.manual_seed(
        config.SEED
    )

    device = torch.device(
        config.DEVICE
    )

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    x, y = get_cifar10_subset(
        data_dir=config.DATA_DIR,
        classes=config.CLASSES,
        samples_per_class=config.SAMPLES_PER_CLASS,
        seed=config.SEED,
    )

    x = x.to(device)
    y = y.to(device)

    targets = F.one_hot(
        y,
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

    w_520 = get_saved_weight(
        gd_data,
        520,
    ).to(device)

    w_521 = get_saved_weight(
        gd_data,
        521,
    ).to(device)

    # --------------------------------------------------
    # Construct rod
    # --------------------------------------------------

    center = 0.5 * (
        w_520 + w_521
    )

    delta = 0.5 * (
        w_521 - w_520
    )

    w_minus = center - delta
    w_plus = center + delta

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = MLP(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
    ).to(device)

    # --------------------------------------------------
    # Endpoint losses + gradients
    # --------------------------------------------------

    loss_minus, grad_minus = loss_and_gradient(
        model,
        w_minus,
        x,
        targets,
    )

    loss_plus, grad_plus = loss_and_gradient(
        model,
        w_plus,
        x,
        targets,
    )

    # Average endpoint gradient
    grad_avg = 0.5 * (
        grad_minus + grad_plus
    )

    # Difference between endpoint gradients
    grad_diff = 0.5 * (
        grad_plus - grad_minus
    )

    # --------------------------------------------------
    # Print diagnostics
    # --------------------------------------------------

    print("Rod endpoint diagnostics")
    print("------------------------")

    print(
        f"Loss at w_minus: "
        f"{loss_minus:.6f}"
    )

    print(
        f"Loss at w_plus : "
        f"{loss_plus:.6f}"
    )

    print()

    print(
        "||grad_minus|| =",
        torch.norm(
            grad_minus
        ).item(),
    )

    print(
        "||grad_plus||  =",
        torch.norm(
            grad_plus
        ).item(),
    )

    print(
        "||grad_avg||   =",
        torch.norm(
            grad_avg
        ).item(),
    )

    print(
        "||grad_diff||  =",
        torch.norm(
            grad_diff
        ).item(),
    )

    print()

    print(
        "||delta||      =",
        torch.norm(
            delta
        ).item(),
    )

    # --------------------------------------------------
    # Leading Rod Flow center velocity
    # --------------------------------------------------

    center_velocity = (
        -config.LR * grad_avg
    )

    print()

    print(
        "||center velocity|| =",
        torch.norm(
            center_velocity
        ).item(),
    )


if __name__ == "__main__":
    main()