from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


START_STEP = 6000
END_STEP = 6050


def get_saved_weight(gd_data, step):
    saved_steps = gd_data["saved_steps"]
    trajectory = gd_data["trajectory"]

    matches = (saved_steps == step).nonzero(as_tuple=True)[0]

    if len(matches) == 0:
        raise ValueError(f"GD step {step} is not stored.")

    return trajectory[matches.item()]


def build_gd_midpoints(gd_data, start_step, end_step):
    """
    Build the reference GD center:
        \bar w_k = (w_k + w_{k+1}) / 2
    for k = start_step, ..., end_step.
    """
    midpoints = []
    steps = []

    for k in range(start_step, end_step + 1):
        w_k = get_saved_weight(gd_data, k)
        w_k1 = get_saved_weight(gd_data, k + 1)
        midpoint = 0.5 * (w_k + w_k1)
        midpoints.append(midpoint)
        steps.append(k)

    return torch.stack(midpoints), torch.tensor(steps, dtype=torch.long)


def crop_flow(flow_data, key="centers"):
    """
    Extract only the requested window [START_STEP, END_STEP].
    """
    steps = flow_data["steps"]
    values = flow_data[key]

    mask = (steps >= START_STEP) & (steps <= END_STEP)

    return steps[mask], values[mask]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load trajectories
    # --------------------------------------------------
    gd_data = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    rod_data = torch.load(
        "outputs/trajectories/rod_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    # if you saved the improved CF with this filename
    central_path = Path("outputs/trajectories/central_flow_full_test.pt")
    if not central_path.exists():
        central_path = Path("outputs/trajectories/central_flow_test.pt")

    central_data = torch.load(
        central_path,
        map_location="cpu",
        weights_only=False,
    )

    edge_data = torch.load(
        "outputs/trajectories/edge_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    # Optional: gradient flow if you have it
    gf_path = Path("outputs/trajectories/gradient_flow_test.pt")
    has_gf = gf_path.exists()
    if has_gf:
        gf_data = torch.load(
            gf_path,
            map_location="cpu",
            weights_only=False,
        )

    # --------------------------------------------------
    # Reference GD midpoints
    # --------------------------------------------------
    gd_midpoints, gd_steps = build_gd_midpoints(
        gd_data,
        START_STEP,
        END_STEP,
    )

    # --------------------------------------------------
    # Crop all flows to same window
    # --------------------------------------------------
    rod_steps, rod_centers = crop_flow(rod_data, key="centers")
    cf_steps, cf_centers = crop_flow(central_data, key="centers")
    ef_steps, ef_centers = crop_flow(edge_data, key="centers")

    if has_gf:
        gf_steps, gf_centers = crop_flow(gf_data, key="centers")

    # Sanity check
    assert torch.equal(rod_steps, gd_steps), "Rod steps do not match GD steps"
    assert torch.equal(cf_steps, gd_steps), "Central steps do not match GD steps"
    assert torch.equal(ef_steps, gd_steps), "Edge steps do not match GD steps"
    if has_gf:
        assert torch.equal(gf_steps, gd_steps), "GF steps do not match GD steps"

    # --------------------------------------------------
    # Residuals with respect to GD midpoint
    # --------------------------------------------------
    rod_res = rod_centers - gd_midpoints
    cf_res = cf_centers - gd_midpoints
    ef_res = ef_centers - gd_midpoints

    if has_gf:
        gf_res = gf_centers - gd_midpoints

    # --------------------------------------------------
    # Residual norms
    # --------------------------------------------------
    rod_err = torch.norm(rod_res, dim=1).numpy()
    cf_err = torch.norm(cf_res, dim=1).numpy()
    ef_err = torch.norm(ef_res, dim=1).numpy()

    if has_gf:
        gf_err = torch.norm(gf_res, dim=1).numpy()

    # --------------------------------------------------
    # PCA on residuals (THIS is the important part)
    # --------------------------------------------------
    residual_blocks = [
        rod_res,
        cf_res,
        ef_res,
    ]

    labels = [
        "Rod Flow",
        "Central Flow",
        "Edge Flow",
    ]

    if has_gf:
        residual_blocks.append(gf_res)
        labels.append("Gradient Flow")

    residual_matrix = torch.cat(residual_blocks, dim=0).numpy()

    pca = PCA(n_components=2)
    residual_2d = pca.fit_transform(residual_matrix)

    n = len(gd_steps)

    rod_2d = residual_2d[0:n]
    cf_2d = residual_2d[n:2 * n]
    ef_2d = residual_2d[2 * n:3 * n]

    if has_gf:
        gf_2d = residual_2d[3 * n:4 * n]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ------------------------------------------
    # Panel 1: residual norm vs step
    # ------------------------------------------
    ax = axes[0]
    ax.plot(gd_steps.numpy(), rod_err, marker="o", markersize=3, label="Rod Flow")
    ax.plot(gd_steps.numpy(), cf_err, marker="o", markersize=3, label="Central Flow")
    ax.plot(gd_steps.numpy(), ef_err, marker="o", markersize=3, label="Edge Flow")
    if has_gf:
        ax.plot(gd_steps.numpy(), gf_err, marker="o", markersize=3, label="Gradient Flow")

    ax.set_title("Tracking error to GD midpoint")
    ax.set_xlabel("GD-equivalent step")
    ax.set_ylabel(r"$\|w^{\mathrm{flow}} - \bar w^{\mathrm{GD}}\|$")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ------------------------------------------
    # Panel 2: residual PCA
    # ------------------------------------------
    ax = axes[1]
    ax.plot(rod_2d[:, 0], rod_2d[:, 1], marker="o", markersize=3, label="Rod Flow")
    ax.plot(cf_2d[:, 0], cf_2d[:, 1], marker="o", markersize=3, label="Central Flow")
    ax.plot(ef_2d[:, 0], ef_2d[:, 1], marker="o", markersize=3, label="Edge Flow")
    if has_gf:
        ax.plot(gf_2d[:, 0], gf_2d[:, 1], marker="o", markersize=3, label="Gradient Flow")

    # start markers
    ax.scatter(rod_2d[0, 0], rod_2d[0, 1], s=120, marker="*", label="Start")
    ax.set_title("Residual PCA")
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}% var)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ------------------------------------------
    # Panel 3: signed residual coordinate vs step
    # ------------------------------------------
    ax = axes[2]
    ax.plot(gd_steps.numpy(), rod_2d[:, 0], label="Rod Flow")
    ax.plot(gd_steps.numpy(), cf_2d[:, 0], label="Central Flow")
    ax.plot(gd_steps.numpy(), ef_2d[:, 0], label="Edge Flow")
    if has_gf:
        ax.plot(gd_steps.numpy(), gf_2d[:, 0], label="Gradient Flow")

    ax.set_title("Residual PC1 vs step")
    ax.set_xlabel("GD-equivalent step")
    ax.set_ylabel("Residual PC1 coordinate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    save_path = output_dir / "flow_differences.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    main()