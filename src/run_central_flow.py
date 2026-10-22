from pathlib import Path

import torch
import torch.nn.functional as F

from src import config
from src.central_flow import (
    central_flow_substep,
)
from src.data import (
    get_cifar10_subset,
)
from src.model import MLP
from src.sdcp import SDCPSolver
from src.utils import (
    set_parameter_vector,
)


# ============================================================
# SETTINGS
# ============================================================

START_STEP = 6000

# Start small because the full CF is much more expensive.
NUM_FLOW_STEPS = 1000

# Four substeps per GD-equivalent iteration.
NSUBSTEPS = 4

DT = 1.0 / NSUBSTEPS

# Track several Hessian directions.
N_EIG = 6

# HVP block-iteration count.
EIG_ITERS = 8

# Official code tracks eigenvalues above a multiple of 1/eta.
TRACK_THRESHOLD = 1.5


# ============================================================
# GD HELPERS
# ============================================================


def get_saved_weight(
    gd_data,
    step,
):

    saved_steps = (
        gd_data[
            "saved_steps"
        ]
    )

    trajectory = (
        gd_data[
            "trajectory"
        ]
    )

    matches = (
        saved_steps
        == step
    ).nonzero(
        as_tuple=True
    )[0]

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

    return (
        0.5
        * (
            w0 + w1
        )
    )


# ============================================================
# LOSS
# ============================================================


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


# ============================================================
# MAIN
# ============================================================


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
    # GD trajectory
    # --------------------------------------------------

    gd_data = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    # Start CF from the actual GD state at t=6000.
    weights = get_saved_weight(
        gd_data,
        START_STEP,
    ).to(device)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = MLP().to(device)

    # --------------------------------------------------
    # SDCP solver
    # --------------------------------------------------

    sdcp_solver = (
        SDCPSolver()
    )

    # Previous eigenspace will be reused.
    warm_U = None

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

    top_eigenvalues = []

    sigma_traces = []

    critical_counts = []

    center_losses = []

    all_eigenvalues = []

    # --------------------------------------------------
    # Initial information
    # --------------------------------------------------

    print()
    print(
        "Full Central Flow initialisation"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Start step: {START_STEP}"
    )

    print(
        f"eta: {config.LR}"
    )

    print(
        f"2/eta: "
        f"{2.0 / config.LR}"
    )

    print(
        f"Tracked eigenvectors: "
        f"{N_EIG}"
    )

    print(
        f"Critical threshold: "
        f"{TRACK_THRESHOLD}/eta = "
        f"{TRACK_THRESHOLD / config.LR}"
    )

    print(
        f"Substeps per GD step: "
        f"{NSUBSTEPS}"
    )

    print()

    # --------------------------------------------------
    # Integration
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
                    sdcp_solver=sdcp_solver,
                    warm_U=warm_U,
                    n_eig=N_EIG,
                    eig_iters=EIG_ITERS,
                    track_threshold=(
                        TRACK_THRESHOLD
                    ),
                )
            )

            weights = (
                latest[
                    "weights"
                ]
            )

            warm_U = (
                latest[
                    "U"
                ]
            )

        # --------------------------------------------------
        # Corresponding GD step
        # --------------------------------------------------

        gd_step = (
            START_STEP
            + flow_step
        )

        # Central Flow models a smoothed/central path,
        # so compare against the empirical GD midpoint.
        gd_midpoint = empirical_midpoint(
            gd_data,
            gd_step,
        ).to(device)

        distance = torch.norm(
            weights
            - gd_midpoint
        ).item()

        center_loss = (
            loss_at_weights(
                model,
                weights,
                x,
                targets,
            )
        )

        eigvals = (
            latest[
                "eigvals"
            ]
        )

        top_lambda = (
            eigvals[0].item()
        )

        sigma_trace = (
            latest[
                "sigma_trace"
            ]
        )

        critical_count = (
            latest[
                "critical_count"
            ]
        )

        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        centers.append(
            weights.detach().cpu()
        )

        steps.append(
            gd_step
        )

        distances.append(
            distance
        )

        top_eigenvalues.append(
            top_lambda
        )

        sigma_traces.append(
            sigma_trace
        )

        critical_counts.append(
            critical_count
        )

        center_losses.append(
            center_loss
        )

        all_eigenvalues.append(
            eigvals.detach().cpu()
        )

        # --------------------------------------------------
        # Print
        # --------------------------------------------------

        eig_string = ", ".join(
            [
                f"{value:.2f}"
                for value
                in eigvals.tolist()
            ]
        )

        print(
            f"Step {gd_step:4d} | "
            f"distance = "
            f"{distance:.6e} | "
            f"lambda1 = "
            f"{top_lambda:.4f} | "
            f"critical = "
            f"{critical_count} | "
            f"trace(Sigma) = "
            f"{sigma_trace:.6e} | "
            f"loss = "
            f"{center_loss:.6f}"
        )

        print(
            f"           eigs = "
            f"[{eig_string}]"
        )

    # --------------------------------------------------
    # Save results
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
        / "central_flow_full_test.pt"
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

            "distances_to_gd": (
                torch.tensor(
                    distances,
                    dtype=torch.float32,
                )
            ),

            "eigenvalues": (
                torch.tensor(
                    top_eigenvalues,
                    dtype=torch.float32,
                )
            ),

            # Keep this name for compatibility
            # with the old plotting code.
            "sigmas": torch.tensor(
                sigma_traces,
                dtype=torch.float32,
            ),

            "critical_counts": (
                torch.tensor(
                    critical_counts,
                    dtype=torch.long,
                )
            ),

            "center_losses": (
                torch.tensor(
                    center_losses,
                    dtype=torch.float32,
                )
            ),

            "all_eigenvalues": (
                torch.stack(
                    all_eigenvalues
                )
            ),

            "start_step": (
                START_STEP
            ),

            "dt": DT,

            "n_eig": N_EIG,

            "track_threshold": (
                TRACK_THRESHOLD
            ),
        },
        output_path,
    )

    print()
    print(
        "Saved full Central Flow test to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()