import torch
import matplotlib.pyplot as plt


def eos_threshold(lr):
    """
    Gradient Descent Edge-of-Stability threshold:
        lambda_max(H) ~ 2 / lr
    """
    return 2.0 / lr


def save_sharpness_plot(
    sharpness,
    lr,
    output_path="outputs/figures/sharpness.png",
):
    """
    Plot top Hessian eigenvalue over training
    together with the EOS threshold 2/lr.
    """

    sharpness = torch.as_tensor(sharpness).cpu().numpy()

    threshold = eos_threshold(lr)

    plt.figure(figsize=(6, 4))

    plt.plot(sharpness, label=r"$\lambda_{\max}(H)$")

    plt.axhline(
        threshold,
        linestyle="--",
        label=r"$2/\eta$",
    )

    plt.xlabel("GD step")
    plt.ylabel("Sharpness")
    plt.title("Edge of Stability")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def eos_ratio(sharpness, lr):
    """
    Ratio:
        lambda_max(H) / (2/lr)

    Values close to 1 indicate the EOS regime.
    """
    threshold = eos_threshold(lr)

    return torch.as_tensor(sharpness) / threshold


if __name__ == "__main__":
    # Fake example just to test the plotting code

    lr = 0.02

    fake_sharpness = torch.linspace(
        20,
        105,
        200,
    )

    print("EOS threshold:", eos_threshold(lr))

    ratio = eos_ratio(fake_sharpness, lr)

    print("Final EOS ratio:", ratio[-1].item())

    save_sharpness_plot(
        fake_sharpness,
        lr,
        "outputs/figures/test_sharpness.png",
    )