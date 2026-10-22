from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.decomposition import PCA


START_STEP = 6000
END_STEP = 6050


def get_saved_weight(
    gd_data,
    step,
):
    saved_steps = gd_data["saved_steps"]
    trajectory = gd_data["trajectory"]

    matches = (
        saved_steps == step
    ).nonzero(as_tuple=True)[0]

    if len(matches) == 0:
        raise ValueError(
            f"GD step {step} is not stored."
        )

    return trajectory[
        matches.item()
    ]


def empirical_midpoint(
    gd_data,
    step,
):
    """
    GD oscillation center:

        (w_t + w_{t+1}) / 2
    """

    w0 = get_saved_weight(
        gd_data,
        step,
    )

    w1 = get_saved_weight(
        gd_data,
        step + 1,
    )

    return 0.5 * (
        w0 + w1
    )


def to_numpy(x):

    if isinstance(x, torch.Tensor):

        return (
            x.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(x)


def main():

    # --------------------------------------------------
    # Load trajectories
    # --------------------------------------------------

    gd = torch.load(
        "outputs/trajectories/gd.pt",
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

    # --------------------------------------------------
    # Build empirical GD midpoint trajectory
    # --------------------------------------------------

    gd_midpoints = []

    gd_steps = range(
        START_STEP,
        END_STEP + 1,
    )

    for step in gd_steps:

        midpoint = empirical_midpoint(
            gd,
            step,
        )

        gd_midpoints.append(
            midpoint
        )

    gd_midpoints = torch.stack(
        gd_midpoints
    )

    # --------------------------------------------------
    # Flow centers
    # --------------------------------------------------

    rod_centers = rod["centers"]

    central_centers = central["centers"]

    edge_centers = edge["centers"]

    # --------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------

    gd_np = to_numpy(
        gd_midpoints
    )

    rod_np = to_numpy(
        rod_centers
    )

    central_np = to_numpy(
        central_centers
    )

    edge_np = to_numpy(
        edge_centers
    )

    print(
        "GD midpoint trajectory:",
        gd_np.shape,
    )

    print(
        "Rod Flow:",
        rod_np.shape,
    )

    print(
        "Central Flow:",
        central_np.shape,
    )

    print(
        "Edge Flow:",
        edge_np.shape,
    )

    # --------------------------------------------------
    # Fit ONE common PCA basis
    # --------------------------------------------------

    combined = np.concatenate(
        [
            gd_np,
            rod_np,
            central_np,
            edge_np,
        ],
        axis=0,
    )

    print()
    print(
        "Combined PCA data:",
        combined.shape,
    )

    pca = PCA(
        n_components=2
    )

    pca.fit(
        combined
    )

    # --------------------------------------------------
    # Project everything
    # --------------------------------------------------

    gd_2d = pca.transform(
        gd_np
    )

    rod_2d = pca.transform(
        rod_np
    )

    central_2d = pca.transform(
        central_np
    )

    edge_2d = pca.transform(
        edge_np
    )

    # --------------------------------------------------
    # Explained variance
    # --------------------------------------------------

    explained = (
        pca.explained_variance_ratio_
        * 100
    )

    print()

    print(
        f"PC1 explained variance: "
        f"{explained[0]:.2f}%"
    )

    print(
        f"PC2 explained variance: "
        f"{explained[1]:.2f}%"
    )

    print(
        f"Total explained variance: "
        f"{explained.sum():.2f}%"
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    # GD center
    ax.plot(
        gd_2d[:, 0],
        gd_2d[:, 1],
        marker="o",
        markersize=3,
        linewidth=2,
        label="GD midpoint",
    )

    # Rod Flow
    ax.plot(
        rod_2d[:, 0],
        rod_2d[:, 1],
        linewidth=2,
        label="Rod Flow",
    )

    # Central Flow
    ax.plot(
        central_2d[:, 0],
        central_2d[:, 1],
        linewidth=2,
        label="Central Flow",
    )

    # Edge Flow
    ax.plot(
        edge_2d[:, 0],
        edge_2d[:, 1],
        linewidth=2,
        label="Edge Flow",
    )

    # --------------------------------------------------
    # Starting points
    # --------------------------------------------------

    ax.scatter(
        gd_2d[0, 0],
        gd_2d[0, 1],
        s=70,
        marker="o",
        label="Start",
    )

    # --------------------------------------------------
    # Final points
    # --------------------------------------------------

    ax.scatter(
        gd_2d[-1, 0],
        gd_2d[-1, 1],
        s=70,
        marker="x",
    )

    ax.scatter(
        rod_2d[-1, 0],
        rod_2d[-1, 1],
        s=70,
        marker="x",
    )

    ax.scatter(
        central_2d[-1, 0],
        central_2d[-1, 1],
        s=70,
        marker="x",
    )

    ax.scatter(
        edge_2d[-1, 0],
        edge_2d[-1, 1],
        s=70,
        marker="x",
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    plt.tight_layout()

    # --------------------------------------------------
    # Labels
    # --------------------------------------------------

    ax.set_xlabel(
        f"PC1 ({explained[0]:.1f}% variance)"
    )

    ax.set_ylabel(
        f"PC2 ({explained[1]:.1f}% variance)"
    )

    ax.set_title(
        "EOS trajectory comparison"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_dir = Path(
        "outputs/figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "pca_flow_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(
        f"Saved figure to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()