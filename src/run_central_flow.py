from pathlib import Path

import torch
import torch.nn.functional as F

from src import config
from src.data import get_cifar10_subset
from src.model import MLP
from src.central_flow import central_flow_substep
from src.utils import set_parameter_vector


START_STEP = 6000

NUM_FLOW_STEPS = 10

# Same convention as the official minimal implementation:
# four substeps per GD-equivalent time unit.
NSUBSTEPS = 4

DT = 1.0 / NSUBSTEPS

EIG_ITERS = 10


def get_saved_weight(
    gd_data,
    step,
):
    saved_steps = (
        gd_data["saved_steps"]
    )

    trajectory = (
        gd_data["trajectory"]
    )

    matches = (
        saved_steps == step
    ).nonzero(as_tuple=True)[0]

    if len(matches) == 0:

        raise ValueError(
            f"GD step {step} "
            "is not stored."
        )

    return trajectory[
        matches.item()
    ]


def empirical_midpoint(
    gd_data,
    step,
):
    w0 = get_saved_weight(
        gd_data,
        step,
    )

    w1 = get_saved_weight(
        gd_data,
        step + 1,
    )

    return 0.5 * (
        w0 + w1
    )


def loss_at_weights(
    model,
    weights,
    x,
    targets,
):
    set_parameter_vector(
        model,
        weights,
    )

    with torch.no_grad():

        output = model(x)

        return F.mse_loss(
            output,
            targets,
        ).item()


def main():

    torch.manual_seed(
        config.SEED
    )

    device = torch.device(
        config.DEVICE
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    x, y = get_cifar10_subset(
        data_dir=config.DATA_DIR,
        num_samples=config.NUM_SAMPLES,
    )

    x = x.to(device)
    y = y.to(device)

    targets = F.one_hot(
        y,
        num_classes=config.NUM_CLASSES,
    ).float()

    # --------------------------------------------------
    # GD reference
    # --------------------------------------------------

    gd_data = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    # Use GD midpoint as Central Flow initial center
    weights = empirical_midpoint(
        gd_data,
        START_STEP,
    ).to(device)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = MLP().to(device)

    eigenvector = None

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------

    centers = [
        weights.detach().cpu()
    ]

    steps = [
        START_STEP
    ]

    distances = []
    eigenvalues = []
    sigmas = []
    losses = []
    predicted_losses = []

    print()
    print(
        "Central Flow initialisation"
    )

    print(
        "---------------------------"
    )

    print(
        f"Start step: {START_STEP}"
    )

    print(
        f"eta: {config.LR}"
    )

    print(
        f"2/eta: {2.0 / config.LR}"
    )

    print(
        f"Substeps: {NSUBSTEPS}"
    )

    print()

    # --------------------------------------------------
    # Central Flow integration
    # --------------------------------------------------

    for flow_step in range(
        1,
        NUM_FLOW_STEPS + 1,
    ):

        latest = None

        for _ in range(
            NSUBSTEPS
        ):

            latest = (
                central_flow_substep(
                    model=model,
                    weights=weights,
                    x=x,
                    targets=targets,
                    eta=config.LR,
                    dt=DT,
                    eigenvector=eigenvector,
                    eig_iters=EIG_ITERS,
                )
            )

            weights = (
                latest["weights"]
            )

            eigenvector = (
                latest["eigenvector"]
            )

        gd_step = (
            START_STEP
            + flow_step
        )

        gd_midpoint = empirical_midpoint(
            gd_data,
            gd_step,
        ).to(device)

        distance = torch.norm(
            weights - gd_midpoint
        ).item()

        center_loss = loss_at_weights(
            model,
            weights,
            x,
            targets,
        )

        centers.append(
            weights.detach().cpu()
        )

        steps.append(
            gd_step
        )

        distances.append(
            distance
        )

        eigenvalues.append(
            latest["eigenvalue"]
        )

        sigmas.append(
            latest["sigma"]
        )

        losses.append(
            center_loss
        )

        predicted_losses.append(
            latest["predicted_loss"]
        )

        print(
            f"Step {gd_step:4d} | "
            f"distance = {distance:.6e} | "
            f"lambda = {latest['eigenvalue']:.4f} | "
            f"sigma = {latest['sigma']:.6e} | "
            f"loss = {center_loss:.6f}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_dir = Path(
        "outputs/trajectories"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "central_flow_test.pt"
    )

    torch.save(
        {
            "steps": torch.tensor(
                steps,
            ),
            "centers": torch.stack(
                centers,
            ),
            "distances_to_gd": torch.tensor(
                distances,
            ),
            "eigenvalues": torch.tensor(
                eigenvalues,
            ),
            "sigmas": torch.tensor(
                sigmas,
            ),
            "center_losses": torch.tensor(
                losses,
            ),
            "predicted_losses": torch.tensor(
                predicted_losses,
            ),
            "start_step": START_STEP,
            "dt": DT,
        },
        output_path,
    )

    print()
    print(
        "Saved Central Flow test to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()