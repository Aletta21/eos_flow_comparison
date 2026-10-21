from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main():

    data = torch.load(
        "outputs/trajectories/rod_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    steps = data["steps"]

    # First stored step is the initial state,
    # but diagnostics begin after the first evolution step.
    diagnostic_steps = steps[1:]

    distances = data["distances_to_gd"]
    delta_norms = data["delta_norms"]
    center_losses = data["center_losses"]

    output_dir = Path("outputs/figures")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
    )

    # --------------------------------------------------
    # Distance to GD midpoint
    # --------------------------------------------------

    axes[0].plot(
        diagnostic_steps,
        distances,
        marker="o",
        markersize=3,
    )


    axes[0].set_xlabel("GD-equivalent step")
    axes[0].set_ylabel(
        r"$\|\bar{w}_{Rod} - \bar{w}_{GD}\|$"
    )
    axes[0].set_title(
        "Rod Flow tracking error"
    )

    # --------------------------------------------------
    # Rod amplitude
    # --------------------------------------------------

    axes[1].plot(
        diagnostic_steps,
        delta_norms,
        marker="o",
        markersize=3,
    )

    axes[1].set_xlabel("GD-equivalent step")
    axes[1].set_ylabel(
        r"$\|\delta\|$"
    )
    axes[1].set_title(
        "Rod amplitude"
    )

    # --------------------------------------------------
    # Center loss
    # --------------------------------------------------

    axes[2].plot(
        diagnostic_steps,
        center_losses,
        marker="o",
        markersize=3,
    )

    axes[2].set_xlabel("GD-equivalent step")
    axes[2].set_ylabel("MSE loss")
    axes[2].set_title(
        "Rod center loss"
    )

    plt.tight_layout()

    output_path = (
        output_dir / "rod_flow_diagnostics.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved figure to: {output_path}"
    )


if __name__ == "__main__":
    main()