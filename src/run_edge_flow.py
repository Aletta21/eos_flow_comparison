from pathlib import Path

import torch
import torch.nn.functional as F

from src import config
from src.data import get_cifar10_subset
from src.model import MLP
from src.edge_flow import edge_substep
from src.utils import set_parameter_vector


START_STEP = 6000

NUM_FLOW_STEPS = 1000

# Number of Edge Flow substeps
# per GD-equivalent step
K = 4

# Numerical integration step
RHO = config.LR / K

EPSILON = 1e-8


def get_saved_weight(
    gd_data,
    step,
):
    saved_steps = gd_data["saved_steps"]
    trajectory = gd_data["trajectory"]

    matches = (
        saved_steps == step
    ).nonzero(as_tuple=True)[0]

    if len(matches) == 0:
        raise ValueError(
            f"GD step {step} is not stored."
        )

    return trajectory[
        matches.item()
    ]


def empirical_midpoint(
    gd_data,
    step,
):
    """
    Empirical oscillation center:

        (w_t + w_{t+1}) / 2
    """

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
    # Data
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

    w0 = get_saved_weight(
        gd_data,
        START_STEP,
    ).to(device)

    w1 = get_saved_weight(
        gd_data,
        START_STEP + 1,
    ).to(device)

    # --------------------------------------------------
    # Initialize Edge Flow
    # --------------------------------------------------

    center = 0.5 * (
        w0 + w1
    )

    delta = 0.5 * (
        w1 - w0
    )

    amplitude = torch.norm(
        delta
    )

    direction = (
        delta / amplitude
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = MLP().to(device)

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------

    centers = [
        center.detach().cpu()
    ]

    steps = [
        START_STEP
    ]

    distances = []
    amplitudes = []
    sharpness_values = []
    center_losses = []
    growth_factors = []

    # --------------------------------------------------
    # Initial diagnostics
    # --------------------------------------------------

    initial_midpoint = empirical_midpoint(
        gd_data,
        START_STEP,
    ).to(device)

    initial_error = torch.norm(
        center - initial_midpoint
    ).item()

    print()
    print("Edge Flow initialisation")
    print("------------------------")

    print(
        f"Start GD pair: "
        f"{START_STEP}, {START_STEP + 1}"
    )

    print(
        f"eta: {config.LR}"
    )

    print(
        f"2/eta: {2.0 / config.LR}"
    )

    print(
        f"K: {K}"
    )

    print(
        f"rho: {RHO}"
    )

    print(
        f"Initial amplitude: "
        f"{amplitude.item():.8e}"
    )

    print(
        f"Initial midpoint error: "
        f"{initial_error:.8e}"
    )

    print()
    print("Running Edge Flow...")
    print()

    # --------------------------------------------------
    # Integration
    # --------------------------------------------------

    for flow_step in range(
        1,
        NUM_FLOW_STEPS + 1,
    ):

        latest = None

        # K Edge Flow substeps
        # correspond to one GD-equivalent step
        for _ in range(K):

            latest = edge_substep(
                model=model,
                center=center,
                direction=direction,
                amplitude=amplitude,
                x=x,
                targets=targets,
                eta=config.LR,
                rho=RHO,
                epsilon=EPSILON,
            )

            center = latest["center"]
            direction = latest["direction"]
            amplitude = latest["amplitude"]

        gd_step = (
            START_STEP
            + flow_step
        )

        # Compare Edge Flow center
        # with empirical GD midpoint
        gd_midpoint = empirical_midpoint(
            gd_data,
            gd_step,
        ).to(device)

        distance = torch.norm(
            center - gd_midpoint
        ).item()

        center_loss = loss_at_weights(
            model,
            center,
            x,
            targets,
        )

        # --------------------------------------------------
        # Save diagnostics
        # --------------------------------------------------

        centers.append(
            center.detach().cpu()
        )

        steps.append(
            gd_step
        )

        distances.append(
            distance
        )

        amplitudes.append(
            amplitude.item()
        )

        sharpness_values.append(
            latest["sharpness"]
        )

        growth_factors.append(
            float(
                latest["growth_factor"]
            )
        )

        center_losses.append(
            center_loss
        )

        # --------------------------------------------------
        # Print progress
        # --------------------------------------------------

        print(
            f"Step {gd_step:4d} | "
            f"distance = {distance:.6e} | "
            f"amplitude = {amplitude.item():.6e} | "
            f"sharpness = {latest['sharpness']:.4f} | "
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
        / "edge_flow_test.pt"
    )

    torch.save(
        {
            "steps": torch.tensor(
                steps,
                dtype=torch.long,
            ),

            "centers": torch.stack(
                centers
            ),

            "distances_to_gd": torch.tensor(
                distances
            ),

            "amplitudes": torch.tensor(
                amplitudes
            ),

            "sharpness": torch.tensor(
                sharpness_values
            ),

            "growth_factors": torch.tensor(
                growth_factors
            ),

            "center_losses": torch.tensor(
                center_losses
            ),

            "start_step": START_STEP,
            "K": K,
            "rho": RHO,
        },
        output_path,
    )

    print()
    print(
        "Saved Edge Flow test to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()