from pathlib import Path

import torch
import torch.nn.functional as F

from src import config
from src.data import get_cifar10_subset
from src.gradient_flow import gradient_flow_step
from src.model import MLP
from src.utils import set_parameter_vector


START_STEP = 6000
NUM_FLOW_STEPS = 1000

NSUBSTEPS = 10
DT = 1.0 / NSUBSTEPS


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


def loss_at_weights(
    model,
    weights,
    x,
    targets,
):
    set_parameter_vector(
        model,
        weights,
    )

    with torch.no_grad():
        output = model(x)

        loss = F.mse_loss(
            output,
            targets,
        )

    return loss.item()


def main():

    torch.manual_seed(
        config.SEED
    )

    device = torch.device(
        config.DEVICE
    )

    # --------------------------------------------------
    # Data
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
    # Load GD
    # --------------------------------------------------

    gd = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    # Important:
    # ordinary Gradient Flow starts from the actual
    # GD state, not from the Rod midpoint.
    weights = get_saved_weight(
        gd,
        START_STEP,
    ).to(device)

    model = MLP().to(device)

    centers = [
        weights.detach().cpu()
    ]

    steps = [
        START_STEP
    ]

    losses = []

    print()
    print("Gradient Flow")
    print("----------------")
    print(f"Start step: {START_STEP}")
    print(f"eta: {config.LR}")
    print(f"substeps: {NSUBSTEPS}")
    print()

    # --------------------------------------------------
    # Integrate
    # --------------------------------------------------

    for flow_step in range(
        1,
        NUM_FLOW_STEPS + 1,
    ):

        latest_loss = None

        for _ in range(
            NSUBSTEPS
        ):

            weights, latest_loss = (
                gradient_flow_step(
                    model=model,
                    weights=weights,
                    x=x,
                    targets=targets,
                    eta=config.LR,
                    dt=DT,
                )
            )

        gd_step = (
            START_STEP
            + flow_step
        )

        center_loss = loss_at_weights(
            model,
            weights,
            x,
            targets,
        )

        centers.append(
            weights.detach().cpu()
        )

        steps.append(
            gd_step
        )

        losses.append(
            center_loss
        )

        print(
            f"Step {gd_step:4d} | "
            f"loss = {center_loss:.6f}"
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
        output_dir
        / "gradient_flow_test.pt"
    )

    torch.save(
        {
            "steps": torch.tensor(
                steps,
                dtype=torch.long,
            ),
            "centers": torch.stack(
                centers
            ),
            "losses": torch.tensor(
                losses,
                dtype=torch.float32,
            ),
            "start_step": START_STEP,
            "dt": DT,
        },
        output_path,
    )

    print()
    print(
        f"Saved Gradient Flow to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()