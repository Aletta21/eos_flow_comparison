import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def get_parameter_vector(model):
    """
    Flatten all model parameters into a single vector.
    """
    return parameters_to_vector(model.parameters()).detach().clone()


def set_parameter_vector(model, vector):
    """
    Replace model parameters using a flattened parameter vector.
    """
    vector_to_parameters(vector, model.parameters())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from src.model import MLP

    model = MLP()

    w = get_parameter_vector(model)

    print("Parameter vector shape:", w.shape)
    print("Number of parameters:", count_parameters(model))