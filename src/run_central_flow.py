from pathlib import Path

import torch

from src import config
from src.central_flow import central_flow_substep
from src.data import get_cifar10_subset
from src.model import MLP
from src.sdcp import SDCPSolver
from src.utils import set_parameter_vector


# ============================================================
# Configuration
# ============================================================


START_STEP = 6000

# Run from 6000 -> 7000
NUM_FLOW_STEPS = 1000

# Four Central Flow substeps per GD-equivalent step.
NSUBSTEPS = 4

DT = (
    1.0 / NSUBSTEPS
)

# Number of dominant Hessian directions tracked.
N_EIG = 6

# Number of block power iterations.
EIG_ITERS = 8

# Effective tracking threshold.
#
# Raw Hessian equivalent:
#
#     1.5 / eta
#
TRACK_THRESHOLD = 1.5

# Effective SDCP / critical threshold.
#
# Raw Hessian equivalent:
#
#     1.95 / eta
#
SDCP_THRESHOLD = 1.95


# ============================================================
# Helpers
# ============================================================


def get_saved_weight(
    gd_data,
    step,
):
    """
    Retrieve one stored GD parameter vector.
    """

    saved_steps = gd_data[
        "saved_steps"
    ]

    trajectory = gd_data[
        "trajectory"
    ]

    matches = (
        saved_steps == step
    ).nonzero(
        as_tuple=True
    )[0]

    if len(matches) == 0:

        raise ValueError(
            f"GD step {step} is not stored."
        )

    return trajectory[
        matches.item()
    ]


# ============================================================
# Main
# ============================================================


def main():

    torch.manual_seed(
        config.SEED
    )

    device = torch.device(
        config.DEVICE
    )

    print()
    print(
        "Loading data..."
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

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

    targets = torch.nn.functional.one_hot(
        labels,
        num_classes=config.NUM_CLASSES,
    ).float()

    # --------------------------------------------------------
    # GD trajectory
    # --------------------------------------------------------

    gd_path = Path(
        "outputs/trajectories/gd.pt"
    )

    gd_data = torch.load(
        gd_path,
        map_location="cpu",
        weights_only=False,
    )

    # Central Flow starts at the actual GD iterate w_6000.
    weights = get_saved_weight(
        gd_data,
        START_STEP,
    ).to(
        device
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MLP().to(
        device
    )

    set_parameter_vector(
        model,
        weights,
    )

    # --------------------------------------------------------
    # SDCP solver
    # --------------------------------------------------------

    sdcp_solver = SDCPSolver()

    # --------------------------------------------------------
    # Threshold diagnostics
    # --------------------------------------------------------

    raw_track_threshold = (
        TRACK_THRESHOLD
        / config.LR
    )

    raw_sdcp_threshold = (
        SDCP_THRESHOLD
        / config.LR
    )

    eos_threshold = (
        2.0
        / config.LR
    )

    print()
    print(
        "Central Flow configuration"
    )

    print(
        "--------------------------"
    )

    print(
        f"Start step: {START_STEP}"
    )

    print(
        f"Flow steps: {NUM_FLOW_STEPS}"
    )

    print(
        f"Substeps per GD step: "
        f"{NSUBSTEPS}"
    )

    print(
        f"dt: {DT}"
    )

    print(
        f"Tracked eigenvectors: "
        f"{N_EIG}"
    )

    print()

    print(
        "Thresholds:"
    )

    print(
        f"  tracking       = "
        f"{TRACK_THRESHOLD:.2f} effective "
        f"= {raw_track_threshold:.2f} raw"
    )

    print(
        f"  SDCP critical  = "
        f"{SDCP_THRESHOLD:.2f} effective "
        f"= {raw_sdcp_threshold:.2f} raw"
    )

    print(
        f"  EOS boundary   = "
        f"2.00 effective "
        f"= {eos_threshold:.2f} raw"
    )

    print()

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    centers = [
        weights.detach().cpu()
    ]

    steps = [
        START_STEP
    ]

    center_losses = []

    eigenvalues = []

    critical_counts = []

    sigma_eigenvalues = []

    sigma_traces = []

    # Warm-start eigenspace.
    warm_U = None

    # --------------------------------------------------------
    # Integration
    # --------------------------------------------------------

    print(
        "Running Central Flow..."
    )

    print()

    for flow_step in range(
        1,
        NUM_FLOW_STEPS + 1,
    ):

        latest_result = None

        # ----------------------------------------------------
        # Central Flow substeps
        # ----------------------------------------------------

        for _ in range(
            NSUBSTEPS
        ):

            result = central_flow_substep(
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
                sdcp_threshold=(
                    SDCP_THRESHOLD
                ),
            )

            weights = result[
                "weights"
            ]

            warm_U = result[
                "U"
            ]

            latest_result = result

        # ----------------------------------------------------
        # Store after one GD-equivalent unit
        # ----------------------------------------------------

        gd_step = (
            START_STEP
            + flow_step
        )

        centers.append(
            weights.detach().cpu()
        )

        steps.append(
            gd_step
        )

        center_losses.append(
            latest_result[
                "loss"
            ]
        )

        eigenvalues.append(
            latest_result[
                "eigvals"
            ].detach().cpu()
        )

        critical_counts.append(
            latest_result[
                "critical_count"
            ]
        )

        sigma_eigenvalues.append(
            latest_result[
                "sigma_eigs"
            ].detach().cpu()
        )

        sigma_traces.append(
            latest_result[
                "sigma_trace"
            ]
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        top_eigs = latest_result[
            "eigvals"
        ]

        eig_string = (
            ", ".join(
                f"{value.item():.2f}"
                for value in top_eigs[
                    :min(
                        6,
                        len(top_eigs),
                    )
                ]
            )
        )

        print(
            f"Step {gd_step} "
            f"| eigs [{eig_string}] "
            f"| critical="
            f"{latest_result['critical_count']} "
            f"| trace(Sigma)="
            f"{latest_result['sigma_trace']:.6e} "
            f"| loss="
            f"{latest_result['loss']:.6f}"
        )

    # --------------------------------------------------------
    # Stack results
    # --------------------------------------------------------

    centers = torch.stack(
        centers
    )

    steps = torch.tensor(
        steps,
        dtype=torch.long,
    )

    center_losses = torch.tensor(
        center_losses,
        dtype=torch.float32,
    )

    eigenvalues = torch.stack(
        eigenvalues
    )

    critical_counts = torch.tensor(
        critical_counts,
        dtype=torch.long,
    )

    sigma_eigenvalues = torch.stack(
        sigma_eigenvalues
    )

    sigma_traces = torch.tensor(
        sigma_traces,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
            "steps": steps,
            "centers": centers,
            "center_losses": (
                center_losses
            ),
            "eigenvalues": (
                eigenvalues
            ),
            "critical_counts": (
                critical_counts
            ),
            "sigma_eigenvalues": (
                sigma_eigenvalues
            ),
            "sigma_traces": (
                sigma_traces
            ),
            "start_step": (
                START_STEP
            ),
            "num_flow_steps": (
                NUM_FLOW_STEPS
            ),
            "nsubsteps": (
                NSUBSTEPS
            ),
            "dt": (
                DT
            ),
            "n_eig": (
                N_EIG
            ),
            "eig_iters": (
                EIG_ITERS
            ),
            "track_threshold": (
                TRACK_THRESHOLD
            ),
            "sdcp_threshold": (
                SDCP_THRESHOLD
            ),
            "eta": (
                config.LR
            ),
        },
        output_path,
    )

    print()
    print(
        "Central Flow finished."
    )

    print(
        f"Saved trajectory to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()