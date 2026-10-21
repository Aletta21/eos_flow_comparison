from pathlib import Path

import torch
from datasets import load_dataset
from torchvision import transforms


def get_cifar10_subset(
    data_dir="./data",
    num_samples=1000,
):
    """
    Load the first num_samples CIFAR-10
    training examples.
    """

    data_dir = Path(data_dir)

    dataset = load_dataset(
        "uoft-cs/cifar10",
        split="train",
        cache_dir=str(data_dir / "hf_cache"),
    )

    transform = transforms.ToTensor()

    images = []
    labels = []

    for i in range(num_samples):

        sample = dataset[i]

        image = transform(
            sample["img"]
        )

        label = sample["label"]

        images.append(image)
        labels.append(label)

    x = torch.stack(images)

    y = torch.tensor(
        labels,
        dtype=torch.long,
    )

    return x, y


if __name__ == "__main__":

    x, y = get_cifar10_subset()

    print("Images:", x.shape)
    print("Labels:", y.shape)
    print("Classes:", torch.unique(y))

    for cls in torch.unique(y):

        count = (y == cls).sum().item()

        print(
            f"Class {cls.item()}: {count}"
        )