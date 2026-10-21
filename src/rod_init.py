import torch


def get_saved_weight(data, step):
    """
    Extract the parameter vector corresponding
    to a particular saved GD step.
    """

    saved_steps = data["saved_steps"]
    trajectory = data["trajectory"]

    matches = (
        saved_steps == step
    ).nonzero(as_tuple=True)[0]

    if len(matches) == 0:
        raise ValueError(
            f"Step {step} was not saved."
        )

    idx = matches.item()

    return trajectory[idx]


def initialize_rod(
    w_left,
    w_right,
):
    """
    Construct Rod Flow variables from
    two consecutive GD iterates.

        center = (w_t + w_{t+1}) / 2

        delta  = (w_{t+1} - w_t) / 2
    """

    center = 0.5 * (
        w_left + w_right
    )

    delta = 0.5 * (
        w_right - w_left
    )

    return center, delta


def main():

    gd_data = torch.load(
        "outputs/trajectories/gd.pt",
        map_location="cpu",
        weights_only=False,
    )

    w_520 = get_saved_weight(
        gd_data,
        520,
    )

    w_521 = get_saved_weight(
        gd_data,
        521,
    )

    center, delta = initialize_rod(
        w_520,
        w_521,
    )

    print("w_520 shape:", w_520.shape)
    print("w_521 shape:", w_521.shape)

    print()
    print("Rod center shape:", center.shape)
    print("Rod delta shape:", delta.shape)

    print()
    print(
        "||w_521 - w_520|| =",
        torch.norm(
            w_521 - w_520
        ).item(),
    )

    print(
        "||delta|| =",
        torch.norm(delta).item(),
    )

    # Verify reconstruction

    reconstructed_left = (
        center - delta
    )

    reconstructed_right = (
        center + delta
    )

    left_error = torch.norm(
        reconstructed_left - w_520
    )

    right_error = torch.norm(
        reconstructed_right - w_521
    )

    print()
    print(
        "Left reconstruction error:",
        left_error.item(),
    )

    print(
        "Right reconstruction error:",
        right_error.item(),
    )


if __name__ == "__main__":
    main()