from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main():

    data = torch.load(
        "outputs/trajectories/edge_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    steps = data["steps"]
    diagnostic_steps = steps[1:]

    distances = data["distances_to_gd"]
    amplitudes = data["amplitudes"]
    sharpness = data["sharpness"]

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

    # ------------------------------------------
    # Tracking error
    # ------------------------------------------

    axes[0].plot(
        diagnostic_steps,
        distances,
        marker="o",
        markersize=3,
    )

    axes[0].set_xlabel(
        "GD-equivalent step"
    )

    axes[0].set_ylabel(
        r"$\|\bar{w}_{EF} - \bar{w}_{GD}\|$"
    )

    axes[0].set_title(
        "Edge Flow tracking error"
    )

    # ------------------------------------------
    # Amplitude
    # ------------------------------------------

    axes[1].plot(
        diagnostic_steps,
        amplitudes,
        marker="o",
        markersize=3,
    )

    axes[1].set_xlabel(
        "GD-equivalent step"
    )

    axes[1].set_ylabel(
        "Amplitude"
    )

    axes[1].set_title(
        "Edge Flow amplitude"
    )

    # ------------------------------------------
    # Sharpness
    # ------------------------------------------

    axes[2].plot(
        diagnostic_steps,
        sharpness,
        marker="o",
        markersize=3,
        label=r"$\lambda_{\max}(H)$",
    )

    axes[2].axhline(
        threshold,
        linestyle="--",
        label=r"$2/\eta$",
    )

    axes[2].set_xlabel(
        "GD-equivalent step"
    )

    axes[2].set_ylabel(
        "Sharpness"
    )

    axes[2].set_title(
        "Edge Flow sharpness"
    )

    axes[2].legend()

    plt.tight_layout()

    output_path = (
        output_dir
        / "edge_flow_diagnostics.png"
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