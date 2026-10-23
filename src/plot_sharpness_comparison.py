from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src import config


START_STEP = 6000
END_STEP = 7000


# ============================================================
# Helpers
# ============================================================


def as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()

    return torch.tensor(x)


def get_saved_weight(
    gd_data,
    step,
):
    saved_steps = as_tensor(
        gd_data["saved_steps"]
    )

    trajectory = as_tensor(
        gd_data["trajectory"]
    )

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
# GD reference trajectories
# ============================================================


def simple_midpoint(
    gd_data,
    step,
):
    """
    Simple consecutive midpoint:

        (w_t + w_{t+1}) / 2

    Natural reference for Rod Flow
    and our Edge Flow center.
    """

    w = get_saved_weight(
        gd_data,
        step,
    )

    w_next = get_saved_weight(
        gd_data,
        step + 1,
    )

    return 0.5 * (
        w + w_next
    )


def second_order_midpoint(
    gd_data,
    step,
):
    """
    Second-order midpoint used by
    the Central Flows implementation:

        w*_t =
        (w_{t-1} + 2 w_t + w_{t+1}) / 4
    """

    w_prev = get_saved_weight(
        gd_data,
        step - 1,
    )

    w = get_saved_weight(
        gd_data,
        step,
    )

    w_next = get_saved_weight(
        gd_data,
        step + 1,
    )

    return (
        w_prev
        + 2.0 * w
        + w_next
    ) / 4.0


# ============================================================
# Generic helpers
# ============================================================


def crop_by_steps(
    steps,
    values,
    start_step=START_STEP,
    end_step=END_STEP,
):
    steps = as_tensor(
        steps
    ).flatten()

    values = as_tensor(
        values
    )

    n = min(
        len(steps),
        len(values),
    )

    steps = steps[:n]
    values = values[:n]

    mask = (
        (steps >= start_step)
        & (steps <= end_step)
    )

    return (
        steps[mask],
        values[mask],
    )


def get_key(
    data,
    candidates,
    description,
):
    for key in candidates:
        if key in data:
            return key

    raise KeyError(
        f"Could not find {description}.\n"
        f"Available keys: {list(data.keys())}"
    )


def trajectory_distance(
    flow_steps,
    flow_centers,
    reference,
):
    """
    Compute distance between a flow trajectory
    and its chosen GD reference trajectory.
    """

    flow_steps = as_tensor(
        flow_steps
    ).flatten()

    flow_centers = as_tensor(
        flow_centers
    )

    distances = []
    valid_steps = []

    for i, step_tensor in enumerate(
        flow_steps
    ):

        step = int(
            step_tensor.item()
        )

        if (
            step < START_STEP
            or step > END_STEP
        ):
            continue

        if step not in reference:
            continue

        target = reference[
            step
        ]

        distance = torch.norm(
            flow_centers[i]
            - target
        ).item()

        valid_steps.append(
            step
        )

        distances.append(
            distance
        )

    return (
        torch.tensor(
            valid_steps,
            dtype=torch.long,
        ),
        torch.tensor(
            distances,
            dtype=torch.float32,
        ),
    )


# ============================================================
# Main
# ============================================================


def main():

    threshold = (
        2.0 / config.LR
    )

    # --------------------------------------------------------
    # Load trajectories
    # --------------------------------------------------------

    gd = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    gf = torch.load(
        "outputs/trajectories/gradient_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    rod = torch.load(
        "outputs/trajectories/rod_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    cf = torch.load(
        "outputs/trajectories/central_flow_full_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    edge = torch.load(
        "outputs/trajectories/edge_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    gf_sharp = torch.load(
        "outputs/trajectories/gradient_flow_sharpness.pt",
        map_location="cpu",
        weights_only=False,
    )

    # ========================================================
    # Build both GD reference trajectories
    # ========================================================

    gd_simple_midpoint = {}
    gd_second_order_midpoint = {}

    for step in range(
        START_STEP,
        END_STEP + 1,
    ):

        gd_simple_midpoint[
            step
        ] = simple_midpoint(
            gd,
            step,
        )

        gd_second_order_midpoint[
            step
        ] = second_order_midpoint(
            gd,
            step,
        )

    # ========================================================
    # PANEL 1:
    # Method-specific tracking error
    # ========================================================

    # Gradient Flow:
    # compare against the smoother second-order midpoint
    gf_dist_steps, gf_dist = (
        trajectory_distance(
            gf["steps"],
            gf["centers"],
            gd_second_order_midpoint,
        )
    )

    # Rod Flow:
    # compare against simple consecutive midpoint
    rod_dist_steps, rod_dist = (
        trajectory_distance(
            rod["steps"],
            rod["centers"],
            gd_simple_midpoint,
        )
    )

    # Central Flow:
    # compare against official second-order midpoint
    cf_dist_steps, cf_dist = (
        trajectory_distance(
            cf["steps"],
            cf["centers"],
            gd_second_order_midpoint,
        )
    )

    # Edge Flow:
    # compare against simple consecutive midpoint
    edge_dist_steps, edge_dist = (
        trajectory_distance(
            edge["steps"],
            edge["centers"],
            gd_simple_midpoint,
        )
    )

    # ========================================================
    # PANEL 2:
    # Training loss
    # ========================================================

    # Gradient Flow loss
    gf_loss_steps = as_tensor(
        gf["steps"]
    )[1:]

    gf_losses = as_tensor(
        gf["losses"]
    )

    gf_loss_steps, gf_losses = (
        crop_by_steps(
            gf_loss_steps,
            gf_losses,
        )
    )

    # Rod Flow loss
    rod_loss_key = get_key(
        rod,
        [
            "center_losses",
            "losses",
        ],
        "Rod Flow loss",
    )

    rod_loss_steps = as_tensor(
        rod["steps"]
    )[1:]

    rod_loss_steps, rod_losses = (
        crop_by_steps(
            rod_loss_steps,
            rod[rod_loss_key],
        )
    )

    # Central Flow loss
    cf_loss_key = get_key(
        cf,
        [
            "center_losses",
            "losses",
        ],
        "Central Flow loss",
    )

    cf_loss_steps = as_tensor(
        cf["steps"]
    )[1:]

    cf_loss_steps, cf_losses = (
        crop_by_steps(
            cf_loss_steps,
            cf[cf_loss_key],
        )
    )

    # Edge Flow loss
    edge_loss_key = get_key(
        edge,
        [
            "center_losses",
            "losses",
        ],
        "Edge Flow loss",
    )

    edge_loss_steps = as_tensor(
        edge["steps"]
    )[1:]

    edge_loss_steps, edge_losses = (
        crop_by_steps(
            edge_loss_steps,
            edge[edge_loss_key],
        )
    )

    # GD loss
    gd_losses = as_tensor(
        gd["losses"]
    )

    gd_loss_steps = torch.arange(
        1,
        len(gd_losses) + 1,
        dtype=torch.long,
    )

    gd_loss_steps, gd_losses_plot = (
        crop_by_steps(
            gd_loss_steps,
            gd_losses,
        )
    )

    # ========================================================
    # PANEL 3:
    # Sharpness
    # ========================================================

    # Gradient Flow sharpness
    gf_sharp_steps, gf_sharpness = (
        crop_by_steps(
            gf_sharp["steps"],
            gf_sharp["eigenvalues"],
        )
    )

    # Central Flow sharpness
    cf_sharp_key = get_key(
        cf,
        [
            "eigenvalues",
            "sharpness",
            "lambda1",
            "top_eigenvalues",
        ],
        "Central Flow sharpness",
    )

    cf_sharp_steps = as_tensor(
        cf["steps"]
    )[1:]

    cf_sharp_steps, cf_sharpness = (
        crop_by_steps(
            cf_sharp_steps,
            cf[cf_sharp_key],
        )
    )

    # Edge Flow sharpness
    edge_sharp_key = get_key(
        edge,
        [
            "sharpness",
            "eigenvalues",
        ],
        "Edge Flow sharpness",
    )

    edge_sharp_steps = as_tensor(
        edge["steps"]
    )[1:]

    edge_sharp_steps, edge_sharpness = (
        crop_by_steps(
            edge_sharp_steps,
            edge[edge_sharp_key],
        )
    )

    # --------------------------------------------------------
    # Optional GD sharpness
    # --------------------------------------------------------

    gd_sharp_steps = None
    gd_sharpness = None

    possible_gd_sharpness_files = [
        Path(
            "outputs/trajectories/sharpness.pt"
        ),
        Path(
            "outputs/trajectories/gd_sharpness.pt"
        ),
    ]

    for path in possible_gd_sharpness_files:

        if not path.exists():
            continue

        gd_sharp_data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        step_key = get_key(
            gd_sharp_data,
            [
                "steps",
                "measured_steps",
            ],
            "GD sharpness steps",
        )

        value_key = get_key(
            gd_sharp_data,
            [
                "sharpness",
                "eigenvalues",
                "top_eigenvalues",
            ],
            "GD sharpness",
        )

        gd_sharp_steps, gd_sharpness = (
            crop_by_steps(
                gd_sharp_data[step_key],
                gd_sharp_data[value_key],
            )
        )

        break

    # ========================================================
    # Plot
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
    )

    # ========================================================
    # Panel 1:
    # Method-specific tracking error
    # ========================================================

    ax = axes[0]

    ax.plot(
        gf_dist_steps,
        gf_dist,
        label="Gradient Flow",
        linewidth=2,
    )

    ax.plot(
        rod_dist_steps,
        rod_dist,
        label="Rod Flow",
        linewidth=2,
    )

    ax.plot(
        cf_dist_steps,
        cf_dist,
        label="Central Flow",
        linewidth=2,
    )

    ax.plot(
        edge_dist_steps,
        edge_dist,
        label="Edge Flow",
        linewidth=2,
    )

    ax.set_title(
        "Method-specific tracking error"
    )

    ax.set_xlabel(
        "Iteration"
    )

    ax.set_ylabel(
        "Tracking error"
    )

    ax.set_yscale(
        "log"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    # ========================================================
    # Panel 2:
    # Loss
    # ========================================================

    ax = axes[1]

    ax.plot(
        gd_loss_steps,
        gd_losses_plot,
        label="GD",
        linewidth=1.5,
        alpha=0.8,
    )

    ax.plot(
        gf_loss_steps,
        gf_losses,
        label="Gradient Flow",
        linewidth=2,
    )

    ax.plot(
        rod_loss_steps,
        rod_losses,
        label="Rod Flow",
        linewidth=2,
    )

    ax.plot(
        cf_loss_steps,
        cf_losses,
        label="Central Flow",
        linewidth=2,
    )

    ax.plot(
        edge_loss_steps,
        edge_losses,
        label="Edge Flow",
        linewidth=2,
    )

    ax.set_title(
        "Training loss"
    )

    ax.set_xlabel(
        "Iteration"
    )

    ax.set_ylabel(
        "MSE loss"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    # ========================================================
    # Panel 3:
    # Sharpness
    # ========================================================

    ax = axes[2]

    if (
        gd_sharp_steps is not None
        and len(gd_sharp_steps) > 0
    ):

        ax.plot(
            gd_sharp_steps,
            gd_sharpness,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label="GD",
        )

    ax.plot(
        gf_sharp_steps,
        gf_sharpness,
        marker="o",
        markersize=3,
        linewidth=2,
        label="Gradient Flow",
    )

    ax.plot(
        cf_sharp_steps,
        cf_sharpness,
        linewidth=2,
        label="Central Flow",
    )

    ax.plot(
        edge_sharp_steps,
        edge_sharpness,
        linewidth=2,
        label="Edge Flow",
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=r"$2/\eta$",
    )

    ax.set_title(
        "Sharpness"
    )

    ax.set_xlabel(
        "Iteration"
    )

    ax.set_ylabel(
        r"$\lambda_{\max}(H)$"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    # ========================================================
    # Shared formatting
    # ========================================================

    for ax in axes:
        ax.set_xlim(
            START_STEP,
            END_STEP,
        )

    fig.suptitle(
        "Long-horizon EOS dynamics",
        fontsize=18,
    )

    plt.tight_layout()

    # ========================================================
    # Save
    # ========================================================

    output_dir = Path(
        "outputs/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        output_dir
        / "long_horizon_comparison.png"
    )

    pdf_path = (
        output_dir
        / "long_horizon_comparison.pdf"
    )

    plt.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print(
        f"Saved PNG to: {png_path}"
    )

    print(
        f"Saved PDF to: {pdf_path}"
    )


if __name__ == "__main__":
    main()