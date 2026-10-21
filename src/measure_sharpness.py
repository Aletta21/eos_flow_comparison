from pathlib import Path

import torch
import torch.nn.functional as F

from src import config
from src.data import get_cifar10_subset
from src.model import MLP
from src.utils import set_parameter_vector
from src.hessian import top_hessian_eigenpair


def main():

    torch.manual_seed(config.SEED)

    device = torch.device(config.DEVICE)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    x, y = get_cifar10_subset(
        data_dir=config.DATA_DIR,
        num_samples=config.NUM_SAMPLES,
    )

    x = x.to(device)
    y = y.to(device)

    targets = F.one_hot(
        y,
        num_classes=config.NUM_CLASSES,
    ).float()

    # --------------------------------------------------
    # Load GD trajectory
    # --------------------------------------------------

    gd_data = torch.load(
        "outputs/trajectories/gd.pt",
        map_location=device,
        weights_only=False,
    )

    trajectory = gd_data["trajectory"]
    saved_steps = gd_data["saved_steps"]

    print("Trajectory:", trajectory.shape)
    print("Saved points:", len(saved_steps))

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    model = MLP().to(device)

    # --------------------------------------------------
    # Steps we would like to inspect
    # --------------------------------------------------

    steps_to_measure = [
        5800,
        5900,
        5950,
        6000,
        6050,
        6100,
        6150,
        6200,
        6250,
        6300,
        6500,
    ]

    measured_steps = []
    sharpness_values = []

    loss_fn = torch.nn.MSELoss()

    # --------------------------------------------------
    # Measure sharpness
    # --------------------------------------------------

    for target_step in steps_to_measure:

        matches = (
            saved_steps == target_step
        ).nonzero(as_tuple=True)[0]

        if len(matches) == 0:

            print(
                f"Step {target_step} "
                "not found in trajectory."
            )

            continue

        idx = matches.item()

        weights = trajectory[idx].to(device)

        set_parameter_vector(
            model,
            weights,
        )

        # ----------------------------------------------
        # Loss at this checkpoint
        # ----------------------------------------------

        with torch.no_grad():

            output = model(x)

            loss = loss_fn(
                output,
                targets,
            )

        print()
        print(
            f"Step {target_step}"
        )

        print(
            f"Loss: {loss.item():.6f}"
        )

        print(
            "Computing top Hessian eigenvalue..."
        )

        # ----------------------------------------------
        # Top Hessian eigenvalue
        # ----------------------------------------------

        eigenvalue, _ = top_hessian_eigenpair(
            model=model,
            loss_fn=loss_fn,
            x=x,
            y=targets,
            num_iters=20,
            tol=1e-4,
        )

        measured_steps.append(
            target_step
        )

        sharpness_values.append(
            eigenvalue
        )

        print(
            f"Sharpness: {eigenvalue:.6f}"
        )

    # --------------------------------------------------
    # EOS threshold
    # --------------------------------------------------

    threshold = (
        2.0 / config.LR
    )

    print()
    print(
        "--------------------------------"
    )

    print(
        f"EOS threshold 2/eta: "
        f"{threshold:.6f}"
    )

    print(
        "--------------------------------"
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    for step, sharpness in zip(
        measured_steps,
        sharpness_values,
    ):

        ratio = (
            sharpness / threshold
        )

        print(
            f"Step {step:5d} | "
            f"lambda_max = "
            f"{sharpness:10.4f} | "
            f"lambda/(2/eta) = "
            f"{ratio:.4f}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_dir = Path(
        "outputs/trajectories"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "sharpness.pt"
    )

    torch.save(
        {
            "steps": torch.tensor(
                measured_steps,
                dtype=torch.long,
            ),
            "sharpness": torch.tensor(
                sharpness_values,
                dtype=torch.float32,
            ),
            "threshold": threshold,
            "lr": config.LR,
        },
        output_path,
    )

    print()
    print(
        "Saved sharpness results to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()