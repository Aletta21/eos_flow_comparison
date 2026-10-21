from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main():

    # --------------------------------------------------
    # Load GD results
    # --------------------------------------------------

    gd_data = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    losses = gd_data["losses"]

    # --------------------------------------------------
    # Load sharpness results
    # --------------------------------------------------

    sharpness_data = torch.load(
        "outputs/trajectories/sharpness.pt",
        map_location="cpu",
        weights_only=False,
    )

    sharpness_steps = sharpness_data["steps"]
    sharpness = sharpness_data["sharpness"]
    threshold = sharpness_data["threshold"]

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    output_dir = Path("outputs/figures")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4),
    )

    # -------------------------
    # Loss
    # -------------------------

    axes[0].plot(
        range(len(losses)),
        losses,
    )

    axes[0].set_xlabel("GD step")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title("Training loss")

    axes[0].axvline(
        500,
        linestyle="--",
        alpha=0.6,
        label="Approx. EOS onset",
    )

    axes[0].legend()

    # -------------------------
    # Sharpness
    # -------------------------

    axes[1].plot(
        sharpness_steps,
        sharpness,
        marker="o",
        markersize=3,
        label=r"$\lambda_{\max}(H)$",
    )

    axes[1].axhline(
        threshold,
        linestyle="--",
        label=r"$2/\eta$",
    )

    axes[1].set_xlabel("GD step")
    axes[1].set_ylabel("Sharpness")
    axes[1].set_title("Edge of Stability")

    axes[1].legend()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    plt.tight_layout()

    output_path = (
        output_dir / "eos_diagnostics.png"
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