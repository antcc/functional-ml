"""
Implementación de DiffusionMaps extraída de la Hoja de ejercicios 3.
"""

import warnings

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import (
    rbf_kernel, laplacian_kernel, pairwise_distances
)


def deterministic_vector_sign_flip(u):
    """Modify the sign of vectors for reproducibility.

    Utility function that flips the sign of elements
    of all the vectors (rows of u) such that the absolute
    maximum element of each vector is positive.

    Available in sklearn/utils/exmath.py.

    Parameters
    ----------
    u : ndarray of shape (N, M)
        Array with vectors as its rows.

    Returns
    -------
    u_flipped : ndarray of shape (N, M)
        Array with the sign flipped vectors as its rows.
    """
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u


def diagonalize(M):
    """Compute the eigenvalues and eigenvectors of a given square matrix.

    The matrix is not assumed to be symmetric, and the eigenvalues
    are returned in decreasing order of magnitude. The eigenvalue
    at position i corresponds with the eigenvector at position i.

    Parameters
    ----------
    M : ndarray of shape (N, N)
        Matrix to diagonalize.
        
    Returns
    -------
    eigvals : ndarray of shape (N,)
        Array with the eigenvalues
    eigvecs : ndarray of shape (N, N)
        Array with the eigenvectors as its columns.
    """
    # Get eigenvalues and right eigenvectors
    eigvals, eigvecs = np.linalg.eig(M)

    # Check that complex part is negligible and ignore it
    if np.iscomplexobj(eigvals):
        threshold_imaginary_part = 1e-6
        max_imaginary_part = np.max(
            np.abs(np.imag(eigvals))
        )
        if max_imaginary_part > threshold_imaginary_part:
            warnings.warn(
                "Maximum imaginary part is {}".format(
                    max_imaginary_part)
            )
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

    # Sort in descending order
    idx = eigvals.argsort()[::-1]

    return eigvals[idx], eigvecs[:, idx]


class DiffusionMaps(BaseEstimator):
    """Diffusion Maps algorithm for manifold learning [1].

    The data is used to define a weighted graph based on
    kernel similarity, then a random walk over the graph
    is defined, and finally the transition probabilities are
    used to find an embedding to a lower-dimensional space.

    Parameters
    ----------
    n_components : int or float
        Dimension of the embedded space. If it is a float,
        it represents the desired relative precision in the
        (Euclidean) distance approximation.
    step : int
        Number of steps to advance in the underlying Markov chain.
    alpha : float
        Normalization parameter that controls the density influence.
    affinity : {'rbf', 'laplacian', 'precomputed'} or callable
        How to construct the affinity matrix. If 'precomputed', the data matrix
        itself is interpreted as an affinity matrix.
        If a callable, it should have the signature 'affinity(X, Y=None)',
        with the understanding that when Y=None the affinity between X and
        itself is computed.
    sigma : float or {'percentile', 'maximum', 'auto'}
        Kernel width parameter for rbf and laplacian kernels. The meaning of
        the possible string values is the following:
            - 'percentile': a percentile of the pairwise distances of the
             data, specified by `self.p`.
            - 'maximum': the maximum of the pairwise distances of the data.
            - 'auto': it has the same meaning as in Sklearn.
     p : float
        Percentile for calculating sigma. Ignored if `self.sigma` is not
        'percentile'.

    Attributes
    ----------
    n_components_ : int
        Actual dimension of the embedding.
    sigma_ : float
        Sigma parameter effectively used. Only available if `self.affinity` is
        'rbf' or 'laplacian'.
    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Affinity matrix constructed from samples or precomputed.
    embedding_ : ndarray of shape (n_samples, `self.n_components_`)
        Spectral embedding of the training matrix.
        
    References
    ----------
    [1] Coifman, R. R., & Lafon, S. (2006). Diffusion maps.
        Applied and computational harmonic analysis, 21(1), 5-30.
    """

    def __init__(
        self,
        n_components=2,
        step=1,
        alpha=1.0,
        affinity='rbf',
        sigma='percentile',
        p=50,
    ):
        """Construct a DiffusionMaps object."""
        self.n_components = n_components
        self.step = step
        self.alpha = alpha
        self.affinity = affinity
        self.sigma = sigma
        self.p = p

    def _choose_sigma(self):
        """Handle the initialization of sigma."""
        if self.sigma == 'percentile':
            self.sigma_ = np.percentile(pairwise_distances(self._data), self.p)
        elif self.sigma == 'maximum':
            self.sigma_ = np.max(pairwise_distances(self._data))
        elif self.sigma == 'auto':
            self.sigma_ = np.sqrt(self._data.shape[1]/2.)
        elif isinstance(self.sigma, (int, float)):
            if self.sigma <= 0:
                raise ValueError(
                    ("sigma must be positive. Got %s") % self.sigma)
            self.sigma_ = self.sigma
        else:
            raise ValueError(("%s is not a valid sigma parameter. "
                              "Expected 'percentile', 'maximum', 'auto' "
                              "or a float.") % self.sigma)

    def _choose_n_components(self, eigvals):
        r"""Handle the initialization of the embedding dimension.

        If a float value is specified in `self.n_components`, the
        following formula is used:
        ```
            n\_components = \max_{l} |\lambda_l|^T > \delta |\lambda_1|^T.
        ```
        """
        if isinstance(self.n_components, int):
            self.n_components_ = self.n_components
        elif isinstance(self.n_components, float):
            if self.step == 0:
                self.n_components_ = 1
            else:
                self.n_components_ = np.argmin(
                    np.abs(eigvals)**self.step >
                    self.n_components*np.abs(eigvals[0])**self.step)
        else:
            raise ValueError(("%s is not a valid n_components parameter. "
                              "Expected int of float.") % self.n_components)

    def _get_affinity_matrix(self, X, Y=None):
        """Compute an affinity_matrix from samples."""
        if not hasattr(self, "sigma_"):
            self._choose_sigma()

        if self.affinity == 'rbf':
            return rbf_kernel(X, Y, gamma=1/(2*self.sigma_**2))
        if self.affinity == 'laplacian':
            return laplacian_kernel(X, Y, gamma=1/(2*self.sigma_**2))
        if self.affinity == 'precomputed':
            return X
        return self.affinity(X, Y)

    def fit(self, X, y=None):
        """Compute the embedding vectors for training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training set.

            If `self.affinity` is 'precomputed'
            X : ndarray of shape (n_samples, n_samples)
            Interpret X as an adjacency graph computed from samples.
        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._data = X

        if isinstance(self.affinity, str):
            if self.affinity not in {"rbf", "laplacian", "precomputed"}:
                raise ValueError(("%s is not a valid kernel. Expected "
                                  "'precomputed', 'rbf', 'laplacian' "
                                  "or a callable.") % self.affinity)
        elif not callable(self.affinity):
            raise ValueError(("'affinity' is expected to be a kernel "
                              "name or a callable. Got: %s") % self.affinity)

        # Compute affinity matrix
        self.affinity_matrix_ = self._get_affinity_matrix(X)

        # Degree diagonal matrix (add 1e-9 to avoid division by zero)
        degree_v = (np.sum(
            self.affinity_matrix_, axis=1) + 1e-9)**(-self.alpha)
        Dv = np.diag(degree_v)

        # Density normalization
        K_alpha = Dv@self.affinity_matrix_@Dv
        D_alpha = np.sum(K_alpha, axis=1).reshape(-1, 1) + 1e-9

        # Transition probability matrix
        P = K_alpha/D_alpha

        # Eigendecomposition of P
        eigvals, eigvecs = diagonalize(P)
        self._choose_n_components(eigvals[1:])
        self._eigvals = eigvals[1:self.n_components_ + 1]
        eigvecs = eigvecs[:, 1:self.n_components_ + 1]
        self._eigvecs = deterministic_vector_sign_flip(eigvecs.T).T

        # Embedding
        self.embedding_ = (self._eigvals**self.step)*self._eigvecs

        return self

    def diffusion_distance(self):
        """Compute the diffusion distance approximation matrix D(xi, xj)."""
        if not hasattr(self, "embedding_"):
            raise NotFittedError("The object was not fitted.")

        return pairwise_distances(self.embedding_)

    def fit_transform(self, X, y=None):
        """Compute the embedding vectors for data X and transform X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training set.

            If `self.affinity` is 'precomputed'
            X : ndarray of shape (n_samples, n_samples)
            Interpret X as an adjacency graph computed from samples.
        y : Ignored

        Returns
        -------
        X_red : ndarray of shape (n_samples, `self.n_components_`)
            The transformed training data.
        """
        self.fit(X)
        return self.embedding_

    def transform(self, X):
        """Transform X using the Nyström formula for out-of-sample embedding.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to be transformed.

        Returns
        -------
        X_red : ndarray of shape (n_samples, `self.n_components_`)
            The transformed data.
        """
        if not hasattr(self, "embedding_"):
            raise NotFittedError("The object was not fitted.")

        if np.array_equal(self._data, X):
            return self.embedding_

        K = self._get_affinity_matrix(X, self._data)
        Dv_left = np.diag((np.sum(K, axis=1) + 1e-9)**(-self.alpha))
        Dv_right = np.diag((np.sum(K, axis=0) + 1e-9)**(-self.alpha))
        K_alpha = Dv_left@K@Dv_right
        D_alpha = np.sum(K_alpha, axis=1).reshape(-1, 1) + 1e-9
        P = K_alpha/D_alpha

        return (1./self._eigvals)*(P@self._eigvecs)
