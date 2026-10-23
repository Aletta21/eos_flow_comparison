import torch
from datasets import load_dataset
from torchvision import transforms

from src import config
from src.model import MLP
from src.utils import set_parameter_vector


STEPS_TO_CHECK = [
    0,
    5000,
    5500,
    5800,
    5900,
    6000,
    6100,
    6500,
    7000,
]


def get_saved_weight(
    gd_data,
    step,
):
    saved_steps = gd_data[
        "saved_steps"
    ]

    trajectory = gd_data[
        "trajectory"
    ]

    matches = (
        saved_steps == step
    ).nonzero(
        as_tuple=True
    )[0]

    if len(matches) == 0:
        raise ValueError(
            f"Step {step} is not stored in gd.pt"
        )

    return trajectory[
        matches.item()
    ]


def load_cifar10_test():
    """
    Load the official CIFAR-10 test split
    using Hugging Face, with the same image
    representation used for training.
    """

    dataset = load_dataset(
        "uoft-cs/cifar10",
        split="test",
        cache_dir="./data/hf_cache",
    )

    to_tensor = transforms.ToTensor()

    images = []
    labels = []

    for example in dataset:

        image = to_tensor(
            example["img"]
        )

        images.append(
            image
        )

        labels.append(
            example["label"]
        )

    x = torch.stack(
        images
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long,
    )

    return x, labels


def accuracy_at_weights(
    model,
    weights,
    x,
    labels,
):
    set_parameter_vector(
        model,
        weights,
    )

    model.eval()

    with torch.no_grad():

        outputs = model(x)

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        correct = (
            predictions == labels
        ).sum().item()

        accuracy = (
            correct
            / len(labels)
        )

    return accuracy


def main():

    device = torch.device(
        config.DEVICE
    )

    # --------------------------------------------------
    # Load CIFAR-10 test set
    # --------------------------------------------------

    print()
    print(
        "Loading CIFAR-10 test set..."
    )

    x_test, labels_test = (
        load_cifar10_test()
    )

    x_test = x_test.to(
        device
    ).float()

    labels_test = labels_test.to(
        device
    )

    print(
        f"Test examples: "
        f"{len(labels_test)}"
    )

    # --------------------------------------------------
    # Load GD trajectory
    # --------------------------------------------------

    gd_data = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = MLP().to(
        device
    )

    # --------------------------------------------------
    # Test accuracy
    # --------------------------------------------------

    print()
    print(
        "Test accuracy"
    )
    print(
        "-------------"
    )

    for step in STEPS_TO_CHECK:

        try:
            weights = get_saved_weight(
                gd_data,
                step,
            ).to(
                device
            )

        except ValueError:

            print(
                f"Step {step}: not stored"
            )

            continue

        accuracy = accuracy_at_weights(
            model=model,
            weights=weights,
            x=x_test,
            labels=labels_test,
        )

        print(
            f"Step {step:5d} "
            f"| accuracy = "
            f"{100.0 * accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()