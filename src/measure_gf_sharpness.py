from pathlib import Path

import torch
import torch.nn.functional as F

from src import config
from src.data import get_cifar10_subset
from src.hessian import top_hessian_eigenpair
from src.model import MLP
from src.utils import set_parameter_vector


EVERY = 25


def main():

    device = torch.device(config.DEVICE)

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

    gf = torch.load(
        "outputs/trajectories/gradient_flow_test.pt",
        map_location="cpu",
        weights_only=False,
    )

    steps = gf["steps"]
    centers = gf["centers"]

    model = MLP().to(device)

    measured_steps = []
    eigenvalues = []

    print()
    print("Gradient Flow sharpness")
    print("-----------------------")
    print(f"EOS threshold = {2.0 / config.LR:.2f}")
    print()

    loss_fn = torch.nn.MSELoss()

    for i in range(
        0,
        len(steps),
        EVERY,
    ):

        step = int(
            steps[i].item()
        )

        weights = centers[i].to(device)

        set_parameter_vector(
            model,
            weights,
        )

        eigval, _ = top_hessian_eigenpair(
            model=model,
            loss_fn=loss_fn,
            x=x,
            y=targets,
            num_iters=20,
            tol=1e-4,
        )

        eigval = float(
            eigval
        )

        measured_steps.append(
            step
        )

        eigenvalues.append(
            eigval
        )

        print(
            f"Step {step:4d} | "
            f"lambda_max = {eigval:.4f} | "
            f"ratio = {eigval / (2.0 / config.LR):.4f}"
        )

    output_path = Path(
        "outputs/trajectories/"
        "gradient_flow_sharpness.pt"
    )

    torch.save(
        {
            "steps": torch.tensor(
                measured_steps,
                dtype=torch.long,
            ),
            "eigenvalues": torch.tensor(
                eigenvalues,
                dtype=torch.float32,
            ),
        },
        output_path,
    )

    print()
    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()