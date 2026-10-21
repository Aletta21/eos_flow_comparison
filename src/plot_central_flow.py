from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main():

    data = torch.load(
        "outputs/trajectories/central_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    steps = data["steps"]

    # Diagnostics start after the initial state
    diagnostic_steps = steps[1:]

    distances = data["distances_to_gd"]
    eigenvalues = data["eigenvalues"]
    sigmas = data["sigmas"]

    threshold = 80.0

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
        r"$\|w_{\mathrm{CF}}-\bar{w}_{\mathrm{GD}}\|$"
    )
    axes[0].set_title(
        "Central Flow tracking error"
    )

    # --------------------------------------------------
    # Sharpness
    # --------------------------------------------------

    axes[1].plot(
        diagnostic_steps,
        eigenvalues,
        marker="o",
        markersize=3,
        label=r"$\lambda_{\max}(H)$",
    )

    axes[1].axhline(
        threshold,
        linestyle="--",
        label=r"$2/\eta$",
    )

    axes[1].set_xlabel("GD-equivalent step")
    axes[1].set_ylabel("Sharpness")
    axes[1].set_title(
        "Sharpness constraint"
    )

    axes[1].legend()

    # --------------------------------------------------
    # Sigma
    # --------------------------------------------------

    axes[2].plot(
        diagnostic_steps,
        sigmas,
        marker="o",
        markersize=3,
    )

    axes[2].set_xlabel("GD-equivalent step")
    axes[2].set_ylabel(r"$\sigma$")
    axes[2].set_title(
        "Central Flow covariance strength"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    plt.tight_layout()

    output_path = (
        output_dir
        / "central_flow_diagnostics.png"
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