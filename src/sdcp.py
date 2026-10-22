import cvxpy as cp
import numpy as np


class SDCPSolver:
    """
    Solve the small PSD covariance problem used by Central Flow.

    We solve

        min_X  <A, X> + 1/2 <X, B[X]>

        subject to X >= 0

    where X is a small symmetric covariance matrix in the
    critical Hessian eigenspace.

    The first-order optimality conditions give the
    semidefinite complementarity condition used by
    Central Flow.
    """

    def __init__(self):
        pass


    @staticmethod
    def _upper_indices(k):
        rows = []
        cols = []

        for i in range(k):
            for j in range(i, k):
                rows.append(i)
                cols.append(j)

        return rows, cols


    @staticmethod
    def _metric(k):
        """
        Full symmetric matrix inner-product weights.

        Diagonal entries count once.
        Off-diagonal entries count twice.
        """

        values = []

        for i in range(k):
            for j in range(i, k):

                if i == j:
                    values.append(1.0)
                else:
                    values.append(2.0)

        return np.asarray(
            values,
            dtype=np.float64,
        )


    def solve(
        self,
        A,
        B,
    ):
        """
        Parameters
        ----------
        A : ndarray [k, k]
            Stable sharpness residual.

        B : ndarray [m, m]
            Linear operator in upper-triangular coordinates,
            where m = k(k+1)/2.

        Returns
        -------
        X : ndarray [k, k]
            PSD oscillation covariance.
        """

        A = np.asarray(
            A,
            dtype=np.float64,
        )

        B = np.asarray(
            B,
            dtype=np.float64,
        )

        k = A.shape[0]

        if k == 0:
            return np.zeros(
                (0, 0),
                dtype=np.float64,
            )

        # Symmetrize A for numerical safety.
        A = 0.5 * (
            A + A.T
        )

        # If A is already PSD, no covariance correction
        # is required.
        min_eig = np.linalg.eigvalsh(
            A
        ).min()

        if min_eig >= -1e-10:

            return np.zeros_like(
                A
            )

        rows, cols = (
            self._upper_indices(k)
        )

        metric = self._metric(k)

        # Convert A to upper-triangular coordinates.
        a_upper = np.asarray(
            [
                A[i, j]
                for i, j
                in zip(rows, cols)
            ],
            dtype=np.float64,
        )

        # Full symmetric inner product:
        #
        # <A, X>
        #
        # means off-diagonal entries count twice.
        c = (
            metric
            * a_upper
        )

        # The quadratic form also uses the symmetric
        # matrix metric.
        P = (
            np.diag(metric)
            @ B
        )

        # Numerical symmetrization.
        P = 0.5 * (
            P + P.T
        )

        # Project tiny negative numerical eigenvalues away.
        eigvals, eigvecs = np.linalg.eigh(
            P
        )

        eigvals = np.maximum(
            eigvals,
            0.0,
        )

        P = (
            eigvecs
            @ np.diag(eigvals)
            @ eigvecs.T
        )

        # Small numerical regularization.
        P += (
            1e-12
            * np.eye(
                P.shape[0]
            )
        )

        # --------------------------------------------------
        # CVXPY variable
        # --------------------------------------------------

        X = cp.Variable(
            (k, k),
            symmetric=True,
        )

        x_upper = cp.hstack(
            [
                X[i, j]
                for i, j
                in zip(rows, cols)
            ]
        )

        objective = cp.Minimize(
            0.5
            * cp.quad_form(
                x_upper,
                cp.psd_wrap(P),
            )
            + c @ x_upper
        )

        constraints = [
            X >> 0
        ]

        problem = cp.Problem(
            objective,
            constraints,
        )

        installed = (
            cp.installed_solvers()
        )

        if "CLARABEL" in installed:

            solver = "CLARABEL"

        elif "SCS" in installed:

            solver = "SCS"

        else:

            raise RuntimeError(
                "No suitable CVXPY solver found."
            )

        problem.solve(
            solver=solver,
            verbose=False,
        )

        if X.value is None:

            raise RuntimeError(
                "Central Flow SDCP solver failed."
            )

        result = np.asarray(
            X.value,
            dtype=np.float64,
        )

        # Numerical symmetrization.
        result = 0.5 * (
            result + result.T
        )

        return result