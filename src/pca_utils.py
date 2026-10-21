import numpy as np
from sklearn.decomposition import PCA


def fit_common_pca(*trajectories, n_components=2):
    """
    Fit one PCA basis to multiple trajectories.

    Each trajectory should have shape:
        [num_steps, num_parameters]
    """

    arrays = []

    for traj in trajectories:
        if hasattr(traj, "detach"):
            traj = traj.detach().cpu().numpy()

        arrays.append(traj)

    combined = np.concatenate(arrays, axis=0)

    pca = PCA(n_components=n_components)
    pca.fit(combined)

    projected = [
        pca.transform(arr)
        for arr in arrays
    ]

    return pca, projected