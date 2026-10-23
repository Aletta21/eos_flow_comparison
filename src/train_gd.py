from pathlib import Path

import torch
import torch.nn.functional as F

from src import config
from src.data import get_cifar10_subset
from src.model import MLP
from src.utils import get_parameter_vector


def run_gd(
    model,
    x,
    y,
    lr,
    num_steps,
):
    """
    Run full-batch gradient descent.

    Saving strategy:
    - before 21000: save every 500 steps
    - 21000 to 21999: save every 100 steps
    - 22000 to 22500: save every step
    - after 22500: save every 100 steps
    """

    model.train()

    targets = F.one_hot(
        y,
        num_classes=config.NUM_CLASSES,
    ).float()

    trajectory = []
    losses = []
    saved_steps = []

    # Save initial parameters
    trajectory.append(
        get_parameter_vector(model).cpu()
    )
    saved_steps.append(0)

    for step in range(num_steps):

        # -------------------------
        # Forward + backward
        # -------------------------

        model.zero_grad()

        outputs = model(x)

        loss = F.mse_loss(
            outputs,
            targets,
        )

        loss.backward()

        losses.append(
            loss.item()
        )

        # -------------------------
        # Full-batch GD update
        # -------------------------

        with torch.no_grad():

            for param in model.parameters():

                param -= lr * param.grad

        current_step = step + 1

        # -------------------------
        # Decide whether to save
        # -------------------------

        should_save = (
            (current_step < 5900 and current_step % 500 == 0)
            or (5900 <= current_step <= 7001)
            or (current_step > 7001 and current_step % 500 == 0)
        )

        if should_save:

            trajectory.append(
                get_parameter_vector(model).cpu()
            )

            saved_steps.append(
                current_step
            )

        # -------------------------
        # Console progress
        # -------------------------

        if current_step % 500 == 0:

            print(
                f"Step {current_step:5d} | "
                f"Loss = {loss.item():.6f}"
            )

    # -------------------------
    # Ensure final point saved
    # -------------------------

    if saved_steps[-1] != num_steps:

        trajectory.append(
            get_parameter_vector(model).cpu()
        )

        saved_steps.append(
            num_steps
        )

    trajectory = torch.stack(
        trajectory
    )

    losses = torch.tensor(
        losses,
        dtype=torch.float32,
    )

    saved_steps = torch.tensor(
        saved_steps,
        dtype=torch.long,
    )

    return (
        trajectory,
        losses,
        saved_steps,
    )


def main():

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    torch.manual_seed(
        config.SEED
    )

    device = torch.device(
        config.DEVICE
    )

    # --------------------------------------------------
    # Load first 1000 CIFAR-10 samples
    # --------------------------------------------------

    x, y = get_cifar10_subset(
        data_dir=config.DATA_DIR,
        num_samples=config.NUM_SAMPLES,
    )

    x = x.to(device)
    y = y.to(device)

    print(
        "Dataset:",
        x.shape,
        y.shape,
    )

    # --------------------------------------------------
    # Build paper-style MLP
    # --------------------------------------------------

    model = MLP().to(device)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Number of parameters:",
        num_params,
    )

    print(
        "Learning rate:",
        config.LR,
    )

    print(
        "EOS threshold 2/eta:",
        2.0 / config.LR,
    )

    print(
        "GD steps:",
        config.GD_STEPS,
    )

    print()

    # --------------------------------------------------
    # Run GD
    # --------------------------------------------------

    trajectory, losses, saved_steps = run_gd(
        model=model,
        x=x,
        y=y,
        lr=config.LR,
        num_steps=config.GD_STEPS,
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output_dir = Path(
        "outputs/trajectories"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "gd.pt"
    )

    torch.save(
        {
            "trajectory": trajectory,
            "losses": losses,
            "saved_steps": saved_steps,
            "lr": config.LR,
            "num_steps": config.GD_STEPS,
        },
        output_path,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print(
        "Saved GD trajectory to:"
    )

    print(
        output_path
    )

    print()

    print(
        "Trajectory shape:",
        trajectory.shape,
    )

    print(
        "Number of saved checkpoints:",
        len(saved_steps),
    )

    print()

    print(
        "First saved steps:",
        saved_steps[:20].tolist(),
    )

    print(
        "Last saved steps:",
        saved_steps[-20:].tolist(),
    )

    print()

    print(
        "Final loss:",
        losses[-1].item(),
    )


if __name__ == "__main__":
    main()