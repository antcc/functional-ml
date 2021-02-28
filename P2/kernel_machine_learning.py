import warnings
from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance


def linear_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
) -> np.ndarray:
    return X@X_prime.T


def exponential_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    ls: float,
) -> np.ndarray:
    d = distance.cdist(X, X_prime, metric='minkowski', p=1.0)
    return A*np.exp(-d/ls)


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
    Alternative parametrization (e.g. in sklearn)
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
    return A*np.exp(-0.5*(d/ls)**2)


def compute_centered_gram_matrix(
    K1: np.ndarray,
    K2: np.ndarray = None,
) -> np.ndarray:
    """
    Compute Gram matrix of centered kernel.

    Parameters
    ----------
    K1:
        An NxN kernel Gram matrix from training data.
    K2:
        An LxN kernel Gram matrix from test data.

    Notes
    -------
    If K2=None, it computes the Gram matrix of the
    centered kernel for training data. Otherwise it computes the
    kernel Gram matrix for test data.
    """
    N = K1.shape[0]
    L = K2.shape[0] if K2 is not None else N
    K2 = K1 if K2 is None else K2
    ones = np.ones((N, N))
    ones_prime = ones if K2 is None else np.ones((L, N))

    return K2 - 1/N*(K2@ones) - 1/N*(ones_prime@K1) \
        + 1/(N**2)*(ones_prime@K1@ones)


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
    kernel:
        kernel function

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
    # Gram matrix of kernel
    K = kernel(X, X)

    # Gram matrix of centered kernel
    K_hat = compute_centered_gram_matrix(K)

    # Compute eigenvectors and eigenvalues (in ascending order)
    lambda_eigenvals, alpha_eigenvecs = np.linalg.eigh(K_hat)
    tol = 1.0e-10
    lambda_eigenvals[lambda_eigenvals < tol] = 0.0

    # Order eigenvalues and eigenvectors in descending order
    lambda_eigenvals = lambda_eigenvals[::-1]
    alpha_eigenvecs = alpha_eigenvecs[:, ::-1]

    # Compute (centered) projection matrix
    K_test = kernel(X_test, X)
    K_test_hat = compute_centered_gram_matrix(K, K_test)

    """
    NOTE: To follow sklearn's criterion and impose a deterministic
    output when it comes to choosing the sign of the eigenvectors,
    we would do the following:

        from sklearn.utils.extmath import svd_flip
        alpha_eigenvecs, _ = svd_flip(alpha_eigenvecs,
            np.zeros_like(alpha_eigenvecs).T)
    """

    # RKHS normalization of eigenvectors, ignoring null components
    non_zero = np.flatnonzero(lambda_eigenvals)
    alpha_eigenvecs[:, non_zero] = (alpha_eigenvecs[:, non_zero]
                                    / np.sqrt(lambda_eigenvals[non_zero]))

    # Project principal components of non-zero eigenvalues
    X_test_hat = K_test_hat@alpha_eigenvecs[:, non_zero]

    return X_test_hat, lambda_eigenvals, alpha_eigenvecs
