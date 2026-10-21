from pathlib import Path

import torch
import torch.nn.functional as F

from src import config
from src.data import get_cifar10_subset
from src.model import MLP
from src.rod_flow import (
    initialize_extent,
    rod_step,
)
from src.utils import set_parameter_vector


START_STEP = 6000

# Number of GD-equivalent steps to test
NUM_FLOW_STEPS = 50

# Numerical integration step
DT = 0.1

# Low-rank approximation of Sigma
RANK = 3


def get_saved_weight(gd_data, step):
    saved_steps = gd_data["saved_steps"]
    trajectory = gd_data["trajectory"]

    matches = (
        saved_steps == step
    ).nonzero(as_tuple=True)[0]

    if len(matches) == 0:
        raise ValueError(
            f"GD step {step} is not stored."
        )

    idx = matches.item()

    return trajectory[idx]


def empirical_midpoint(gd_data, step):
    """
    Midpoint between consecutive GD iterates:

        (w_step + w_{step+1}) / 2
    """

    w_left = get_saved_weight(
        gd_data,
        step,
    )

    w_right = get_saved_weight(
        gd_data,
        step + 1,
    )

    return 0.5 * (
        w_left + w_right
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

        loss = F.mse_loss(
            output,
            targets,
        )

    return loss.item()


def main():

    torch.manual_seed(
        config.SEED
    )

    device = torch.device(
        config.DEVICE
    )

    # --------------------------------------------------
    # Load data
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
    # Load dense GD trajectory
    # --------------------------------------------------

    gd_data = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    w0 = get_saved_weight(
        gd_data,
        START_STEP,
    ).to(device)

    w1 = get_saved_weight(
        gd_data,
        START_STEP + 1,
    ).to(device)

    # --------------------------------------------------
    # Initial rod
    # --------------------------------------------------

    center = 0.5 * (
        w0 + w1
    )

    delta = 0.5 * (
        w1 - w0
    )

    V, lambdas = initialize_extent(
        delta
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = MLP().to(device)

    # --------------------------------------------------
    # Diagnostics storage
    # --------------------------------------------------

    flow_centers = [
        center.detach().cpu()
    ]

    flow_steps = [
        START_STEP
    ]

    distances = []
    delta_norms = []
    center_losses = []

    initial_midpoint = empirical_midpoint(
        gd_data,
        START_STEP,
    ).to(device)

    initial_distance = torch.norm(
        center - initial_midpoint
    ).item()

    print()
    print("Rod Flow initialisation")
    print("-----------------------")

    print(
        f"Start GD pair: "
        f"{START_STEP}, {START_STEP + 1}"
    )

    print(
        f"Initial ||delta||: "
        f"{torch.norm(delta).item():.8f}"
    )

    print(
        f"Initial midpoint error: "
        f"{initial_distance:.8e}"
    )

    print()
    print("Running Rod Flow...")
    print()

    # --------------------------------------------------
    # Integration
    # --------------------------------------------------

    # For DT=0.1, 10 substeps correspond
    # to one GD-equivalent time unit.
    substeps_per_unit = round(
        1.0 / DT
    )

    for flow_step in range(
        1,
        NUM_FLOW_STEPS + 1,
    ):

        for _ in range(
            substeps_per_unit
        ):

            (
                center,
                V,
                lambdas,
                delta,
            ) = rod_step(
                model=model,
                center=center,
                V=V,
                lambdas=lambdas,
                x=x,
                targets=targets,
                eta=config.LR,
                dt=DT,
                rank=RANK,
            )

        # ----------------------------------------------
        # Compare to corresponding empirical GD midpoint
        # ----------------------------------------------

        gd_step = (
            START_STEP
            + flow_step
        )

        gd_midpoint = empirical_midpoint(
            gd_data,
            gd_step,
        ).to(device)

        distance = torch.norm(
            center - gd_midpoint
        ).item()

        delta_norm = torch.norm(
            delta
        ).item()

        center_loss = loss_at_weights(
            model,
            center,
            x,
            targets,
        )

        distances.append(
            distance
        )

        delta_norms.append(
            delta_norm
        )

        center_losses.append(
            center_loss
        )

        flow_centers.append(
            center.detach().cpu()
        )

        flow_steps.append(
            gd_step
        )

        print(
            f"Step {gd_step:4d} | "
            f"distance to GD midpoint = "
            f"{distance:.6e} | "
            f"||delta|| = "
            f"{delta_norm:.6e} | "
            f"loss(center) = "
            f"{center_loss:.6f}"
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
        / "rod_flow_test.pt"
    )

    torch.save(
        {
            "steps": torch.tensor(
                flow_steps
            ),
            "centers": torch.stack(
                flow_centers
            ),
            "distances_to_gd": torch.tensor(
                distances
            ),
            "delta_norms": torch.tensor(
                delta_norms
            ),
            "center_losses": torch.tensor(
                center_losses
            ),
            "dt": DT,
            "rank": RANK,
            "start_step": START_STEP,
        },
        output_path,
    )

    print()
    print(
        f"Saved Rod Flow test to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()