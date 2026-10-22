from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


# ============================================================
# SETTINGS
# ============================================================

FLOW_START = 6000
FLOW_END = 6050

# Define PCA from the discrete GD trajectory only.
PCA_START = 5900
PCA_END = 6300


# ============================================================
# HELPERS
# ============================================================


def get_saved_weight(gd_data, step):
    saved_steps = gd_data["saved_steps"]
    trajectory = gd_data["trajectory"]

    matches = (saved_steps == step).nonzero(as_tuple=True)[0]

    if len(matches) == 0:
        raise ValueError(f"GD step {step} is not stored.")

    return trajectory[matches.item()]


def midpoint(gd_data, step):
    w0 = get_saved_weight(gd_data, step)
    w1 = get_saved_weight(gd_data, step + 1)

    return 0.5 * (w0 + w1)


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


# ============================================================
# MAIN
# ============================================================


def main():

    # --------------------------------------------------------
    # Load trajectories
    # --------------------------------------------------------

    gd = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    gradient = torch.load(
        "outputs/trajectories/gradient_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

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

    # ========================================================
    # PCA BASIS: GD ONLY
    # ========================================================

    pca_gd = torch.stack(
        [
            get_saved_weight(gd, step)
            for step in range(PCA_START, PCA_END + 1)
        ]
    )

    pca = PCA(n_components=2)

    pca.fit(
        as_numpy(pca_gd)
    )

    explained = (
        100.0
        * pca.explained_variance_ratio_
    )

    print()
    print(f"PC1: {explained[0]:.2f}%")
    print(f"PC2: {explained[1]:.2f}%")
    print(f"Total: {explained.sum():.2f}%")

    # ========================================================
    # GD COMPARISON WINDOW
    # ========================================================

    gd_raw = torch.stack(
        [
            get_saved_weight(gd, step)
            for step in range(
                FLOW_START,
                FLOW_END + 2,
            )
        ]
    )

    gd_center = torch.stack(
        [
            midpoint(gd, step)
            for step in range(
                FLOW_START,
                FLOW_END + 1,
            )
        ]
    )

    n = (
        FLOW_END
        - FLOW_START
        + 1
    )

    # ========================================================
    # FLOW TRAJECTORIES
    # ========================================================

    gradient_center = gradient[
        "centers"
    ][:n]

    rod_center = rod[
        "centers"
    ][:n]

    central_center = central[
        "centers"
    ][:n]

    edge_center = edge[
        "centers"
    ][:n]

    # ========================================================
    # PROJECT EVERYTHING
    # ========================================================

    gd_raw_2d = pca.transform(
        as_numpy(gd_raw)
    )

    gd_center_2d = pca.transform(
        as_numpy(gd_center)
    )

    gradient_2d = pca.transform(
        as_numpy(gradient_center)
    )

    rod_2d = pca.transform(
        as_numpy(rod_center)
    )

    central_2d = pca.transform(
        as_numpy(central_center)
    )

    edge_2d = pca.transform(
        as_numpy(edge_center)
    )

    # ========================================================
    # FIGURE
    # ========================================================

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5.5),
        sharex=True,
        sharey=True,
    )

    # --------------------------------------------------------
    # Common panel function
    # --------------------------------------------------------

    def plot_panel(
        ax,
        flow,
        title,
        label,
        color,
    ):

        # GD iterates
        ax.scatter(
            gd_raw_2d[:, 0],
            gd_raw_2d[:, 1],
            s=25,
            color="0.35",
            alpha=0.8,
            edgecolors="none",
            label="GD iterates",
            zorder=2,
        )

        # Simple GD midpoint
        ax.plot(
            gd_center_2d[:, 0],
            gd_center_2d[:, 1],
            color="black",
            linewidth=1.5,
            label=r"GD midpoint",
            zorder=3,
        )

        # Flow
        ax.plot(
            flow[:, 0],
            flow[:, 1],
            color=color,
            linewidth=2.5,
            label=label,
            zorder=4,
        )

        # Start
        ax.scatter(
            flow[0, 0],
            flow[0, 1],
            marker="*",
            s=220,
            color=color,
            edgecolor=color,
            label="Start",
            zorder=10,
        )

        ax.set_title(title)

        ax.set_xlabel("PC1")

        ax.grid(
            alpha=0.25,
            linewidth=0.8,
        )

        ax.legend(
            loc="best",
            framealpha=0.95,
        )

    # --------------------------------------------------------
    # Gradient Flow
    # --------------------------------------------------------

    plot_panel(
        axes[0],
        gradient_2d,
        "Gradient Flow",
        r"Gradient Flow $w$",
        "green",
    )

    # --------------------------------------------------------
    # Rod Flow
    # --------------------------------------------------------

    plot_panel(
        axes[1],
        rod_2d,
        "Rod Flow",
        r"Rod Flow $\bar{w}$",
        "blue",
    )

    # --------------------------------------------------------
    # Central Flow
    # --------------------------------------------------------

    plot_panel(
        axes[2],
        central_2d,
        "Central Flow",
        r"Central Flow $\bar{w}$",
        "red",
    )

    # --------------------------------------------------------
    # Edge Flow
    # --------------------------------------------------------

    plot_panel(
        axes[3],
        edge_2d,
        "Edge Flow",
        r"Edge Flow $\bar{w}$",
        "darkorange",
    )

    axes[0].set_ylabel("PC2")

    # ========================================================
    # COMMON AXIS LIMITS
    # ========================================================

    all_x = np.concatenate(
        [
            gd_raw_2d[:, 0],
            gd_center_2d[:, 0],
            gradient_2d[:, 0],
            rod_2d[:, 0],
            central_2d[:, 0],
            edge_2d[:, 0],
        ]
    )

    all_y = np.concatenate(
        [
            gd_raw_2d[:, 1],
            gd_center_2d[:, 1],
            gradient_2d[:, 1],
            rod_2d[:, 1],
            central_2d[:, 1],
            edge_2d[:, 1],
        ]
    )

    x_range = (
        all_x.max()
        - all_x.min()
    )

    y_range = (
        all_y.max()
        - all_y.min()
    )

    x_pad = 0.04 * x_range
    y_pad = 0.08 * y_range

    for ax in axes:
        ax.set_xlim(
            all_x.min() - x_pad,
            all_x.max() + x_pad,
        )

        ax.set_ylim(
            all_y.min() - y_pad,
            all_y.max() + y_pad,
        )

    # ========================================================
    # TITLE + SAVE
    # ========================================================

    fig.suptitle(
        "Edge-of-Stability Flow Comparison",
        fontsize=22,
        y=1.02,
    )

    plt.tight_layout()

    output_dir = Path(
        "outputs/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "four_flow_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        f"Saved figure to: {output_path}"
    )


if __name__ == "__main__":
    main()