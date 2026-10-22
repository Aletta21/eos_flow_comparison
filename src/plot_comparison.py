from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main():

    # --------------------------------------------------
    # Load results
    # --------------------------------------------------

    rod = torch.load(
        "outputs/trajectories/rod_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    central = torch.load(
        "outputs/trajectories/central_flow_full_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    edge = torch.load(
        "outputs/trajectories/edge_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    # --------------------------------------------------
    # Steps
    # --------------------------------------------------

    rod_steps = rod["steps"][1:]
    cf_steps = central["steps"][1:]
    ef_steps = edge["steps"][1:]

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------

    rod_error = rod["distances_to_gd"]
    cf_error = central["distances_to_gd"]
    ef_error = edge["distances_to_gd"]

    rod_amplitude = rod["delta_norms"]
    ef_amplitude = edge["amplitudes"]

    cf_sharpness = central["eigenvalues"]
    ef_sharpness = edge["sharpness"]

    threshold = 80.0

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    output_dir = Path(
        "outputs/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.2),
    )

    # ==================================================
    # 1. Tracking error
    # ==================================================

    axes[0].plot(
        rod_steps,
        rod_error,
        label="Rod Flow",
    )

    axes[0].plot(
        cf_steps,
        cf_error,
        label="Central Flow",
    )

    axes[0].plot(
        ef_steps,
        ef_error,
        label="Edge Flow",
    )

    axes[0].set_xlabel(
        "GD-equivalent step"
    )

    axes[0].set_ylabel(
        "Tracking error"
    )

    axes[0].set_title(
        "Center tracking"
    )

    axes[0].legend()

    # ==================================================
    # 2. Oscillation scale
    # ==================================================

    axes[1].plot(
        rod_steps,
        rod_amplitude,
        label=r"Rod Flow $\|\delta\|$",
    )

    axes[1].plot(
        ef_steps,
        ef_amplitude,
        label="Edge Flow amplitude",
    )

    axes[1].set_xlabel(
        "GD-equivalent step"
    )

    axes[1].set_ylabel(
        "Oscillation scale"
    )

    axes[1].set_title(
        "Oscillation magnitude"
    )

    axes[1].legend()

    # ==================================================
    # 3. Sharpness
    # ==================================================

    axes[2].plot(
        cf_steps,
        cf_sharpness,
        label="Central Flow",
    )

    axes[2].plot(
        ef_steps,
        ef_sharpness,
        label="Edge Flow",
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
        "EOS sharpness behavior"
    )

    axes[2].legend()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    plt.tight_layout()

    output_path = (
        output_dir
        / "flow_comparison.png"
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