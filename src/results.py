from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Trajectory:
    weights: torch.Tensor
    losses: Optional[torch.Tensor] = None
    sharpness: Optional[torch.Tensor] = None
    directions: Optional[torch.Tensor] = None
    amplitudes: Optional[torch.Tensor] = None


def save_trajectory(traj, path):
    torch.save(traj, path)


def load_trajectory(path):
    return torch.load(path, weights_only=False)