from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance


def linear_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
) -> np.ndarray:
    return X @ X_prime.T


def exponential_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    l: float
) -> np.ndarray:
    d = distance.cdist(X, X_prime, metric='minkovski', p=1.0)
    return A * np.exp(- d / l)


def rbf_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    ls: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_prime:
        Data matrix
    A:
        Output variance
    ls:
        Kernel lengthscale

    Returns
    -------
    kernel matrix

    Notes
    -------
    Alternative parametrization (e.g. en sklearn)
    gamma = 0.5 / ls**2

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_process_regression as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> X_prime = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> kernel_matrix = gp.rbf_kernel(X, X_prime, A, l)
    >>> print(kernel_matrix)
    """
    d = distance.cdist(X, X_prime, metric='euclidean')
    return A * np.exp(-0.5 * (d / ls)**2)


def compute_hat(
    K: np.ndarray,
    ) -> np.ndarray :
    """
    Auxiliary function to kernel_pca. Computes Gram matrix of the centered kernel given
    the Gram matrix of the kernel.
    """
    N = K.shape[0]
    ones = np.ones(K.shape)
    return K - 1/N*(K@ones) - 1/N*(ones@K) + 1/(N**2)*(ones@K@ones)

def kernel_pca(
    X: np.ndarray,
    X_test: np.ndarray,
    kernel: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_test:
        data matrix
    A:
        output variance
    ls:
        kernel lengthscale

    Returns
    -------
    X_test_hat:
        Projection of X_test on the principal components
    lambda_eigenvals:
        Eigenvalues of the centered kernel
    alpha_eigenvecs:
        Principal components. These are the eigenvectors
        of the centered kernel with the RKHS normalization

    Notes
    -------
    In the corresponding method of sklearn the eigenvectors
    are normalized in l2.

    """

    # Build Gram Matrix
    K = kernel(X, X)

    # Gram matrix of centered kernel
    K_hat = compute_hat(K)

    # Compute eigen vectors and values
    lambda_eigenvals, alpha_eigenvecs = np.linalg.eig(K_hat)

    # Handle complex values
    if np.iscomplexobj(lambda_eigenvals):
        threshold_imaginary_part = 1.0e-6
        max_imaginary_part = np.max(
            np.abs(np.imag(lambda_eigenvals))
        )
        if max_imaginary_part > threshold_imaginary_part:
            warnings.warn(
                'Maximum imaginary part is {}'.format(max_imaginary_part)
            )

        lambda_eigenvals = np.real(lambda_eigenvals)

    K_test = kernel(X_test, X_test)
    K_hat_test = compute_hat(K_test)
    X_test_hat = K_hat_test@alpha_eigenvecs

    return X_test_hat, lambda_eigenvals, alpha_eigenvecs
