import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 50),
            nn.SiLU(),
            nn.Linear(50, 50),
            nn.SiLU(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = MLP()

    print(model)

    print(
        "Number of parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )