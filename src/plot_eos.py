from pathlib import Path

import matplotlib.pyplot as plt
import torch


# --------------------------------------------------
# Settings
# --------------------------------------------------

ZOOM_START = 5900
ZOOM_END = 6300


def main():

    # --------------------------------------------------
    # Load GD results
    # --------------------------------------------------

    gd_data = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    losses = torch.as_tensor(
        gd_data["losses"]
    )

    loss_steps = torch.arange(
        len(losses)
    )

    # --------------------------------------------------
    # Load dense sharpness
    # --------------------------------------------------

    sharpness_data = torch.load(
        "outputs/trajectories/sharpness_dense.pt",
        map_location="cpu",
        weights_only=False,
    )

    sharpness_steps = torch.as_tensor(
        sharpness_data["steps"]
    )

    sharpness = torch.as_tensor(
        sharpness_data["sharpness"]
    )

    threshold = float(
        sharpness_data["threshold"]
    )

    # --------------------------------------------------
# Find first EOS threshold crossing
# --------------------------------------------------

    above_threshold = (
        sharpness > threshold
    )

    crossing_indices = torch.where(
        above_threshold
    )[0]

    if len(crossing_indices) > 0:

        first_crossing_index = (
            crossing_indices[0].item()
        )

        first_crossing_step = int(
            sharpness_steps[
                first_crossing_index
            ]
        )

        if first_crossing_index > 0:

            flow_start = int(
                sharpness_steps[
                    first_crossing_index - 1
                ]
            )

        else:

            flow_start = (
                first_crossing_step
            )

    else:

        raise RuntimeError(
            "Sharpness never exceeds "
            "the EOS threshold."
        )

    print(
        f"First threshold crossing: "
        f"{first_crossing_step}"
    )

    print(
        f"Suggested flow start: "
        f"{flow_start}"
    )

    FLOW_START = flow_start
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
    # Loss zoom mask
    # --------------------------------------------------

    zoom_mask = (
        (loss_steps >= ZOOM_START)
        & (loss_steps <= ZOOM_END)
    )

    zoom_steps = (
        loss_steps[zoom_mask]
    )

    zoom_losses = (
        losses[zoom_mask]
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
    # Panel 1: Full training loss
    # ==================================================

    axes[0].plot(
        loss_steps,
        losses,
        linewidth=1.0,
    )

    axes[0].axvline(
        FLOW_START,
        linestyle="--",
        alpha=0.7,
        label="Flow start: step 5910",
    )

    axes[0].set_xlabel(
        "GD step"
    )

    axes[0].set_ylabel(
        "MSE loss"
    )

    axes[0].set_title(
        "GD training loss"
    )

    axes[0].legend()

    # ==================================================
    # Panel 2: Zoomed loss near EOS
    # ==================================================

    axes[1].plot(
        zoom_steps,
        zoom_losses,
        marker="o",
        markersize=4,
        linewidth=0.8,
    )

    axes[1].axvline(
        FLOW_START,
        linestyle="--",
        alpha=0.7,
        label="Flow start: step 5910",
    )

    axes[1].set_xlabel(
        "GD step"
    )

    axes[1].set_ylabel(
        "MSE loss"
    )

    axes[1].set_title(
        "Loss near EOS"
    )

    axes[1].set_xlim(
        ZOOM_START,
        ZOOM_END,
    )

    axes[1].legend()

    # ==================================================
    # Panel 3: Dense sharpness near EOS
    # ==================================================

    axes[2].plot(
        sharpness_steps,
        sharpness,
        marker="o",
        markersize=3,
        linewidth=1.0,
        label=r"$\lambda_{\max}(H)$",
    )

    axes[2].axhline(
        threshold,
        linestyle="--",
        linewidth=1.2,
        label=r"$2/\eta$",
    )

    axes[2].axvline(
        FLOW_START,
        linestyle=":",
        alpha=0.7,
        label="Flow start: step 5910",
    )

    axes[2].set_xlabel(
        "GD step"
    )

    axes[2].set_ylabel(
        "Sharpness"
    )

    axes[2].set_title(
        "Sharpness near EOS"
    )

    axes[2].set_xlim(
        ZOOM_START,
        ZOOM_END,
    )

    axes[2].legend()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    plt.tight_layout()

    output_path = (
        output_dir
        / "training_eos_diagnostics.png"
    )

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved figure to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()