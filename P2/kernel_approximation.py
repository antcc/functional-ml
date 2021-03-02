# /usr/bin/python
# -*- coding: utf-8 -*-

"""
Kernel matrix approximation methods.

Authors: <alberto.suarez@uam.es>
         Luis Antonio Ortega Andrés
         Antonio Coín Castro
"""

from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


class RandomFeaturesSampler(ABC):
    """ Base class for random feature samplers. """

    def __init__(self, n_random_features: int) -> None:
        self.n_random_features = n_random_features
        self.w = None

    @abstractmethod
    def fit(self, n_features: int) -> None:
        """
        Initialize w's for the random features.

        This should be implemented for each kernel.

        Parameters
        ----------
        n_features:
            Number of original features of the data.
        """
        pass

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y = None
    ) -> np.ndarray:
        """
        Initialize  w's (fit) and compute random features (transform).

        Parameters
        ----------
        n_random_features:
            Number of random features to sample.
        X:
            Data matrix of shape (n_instances, n_features).

        Returns
        -------
        Array of shape (n_instances, self.n_random_features).
        """
        n_features = np.shape(X)[1]
        self.fit(n_features)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the random features.

        Assumes that the vector of w's has been initialized.

        Parameters
        ----------
        X:
            Data matrix of shape (n_instances, n_features).

        Returns
        -------
        random_features:
            Array of shape (n_instances, self.n_random_features).
        """
        if self.w is None:
            raise ValueError('Use fit_transform to initialize w.')

        n_instances, n_features = np.shape(X)

        if np.shape(self.w)[1] != n_features:
            raise ValueError('Different # of features for X and w.')

        # Monte Carlo approximation
        random_features = np.empty((n_instances, self.n_random_features))
        random_features[:, ::2] = np.cos(X@self.w.T)
        random_features[:, 1::2] = np.sin(X@self.w.T)

        # Normalize features
        norm_factor = np.sqrt(self.n_random_features//2)
        random_features = random_features/norm_factor

        return random_features


class RandomFeaturesSamplerRBF(RandomFeaturesSampler):
    """ Random Fourier Features for the RBF kernel. """

    def __init__(self, sigma_kernel: float = 1, n_random_features: int = 1) -> None:
        self.sigma = 1.0/sigma_kernel
        super().__init__(n_random_features)

    def fit(self, n_features: int) -> RandomFeaturesSamplerRBF:
        """
        Compute w's for the random RBF features.

        In this case, the RBF kernel is the characteristic function
        of a certain multivariate normal distribution.
        """
        w_mean = np.zeros(n_features)
        w_cov_matrix = self.sigma**2*np.identity(n_features)

        # Sample from multivariate normal distribution
        rng = np.random.default_rng()
        self.w = rng.multivariate_normal(
            w_mean,
            w_cov_matrix,
            self.n_random_features//2,
        )

        return self


class RandomFeaturesSamplerMatern(RandomFeaturesSampler):
    """ Random Fourier Features for the Matérn kernel. """

    def __init__(self, length_scale: float = 1, nu: float = 1, n_random_features: int = 1) -> None:
        self.scale = 1.0/length_scale
        self.nu = 2.0*nu
        super().__init__(n_random_features)

    def fit(self, n_features: int) -> RandomFeaturesSamplerMatern:
        """
        Compute w's for the random Matérn features.

        The Fourier transform of the Matérn kernel is a Student's t
        distribution with twice the degrees of freedom.

            (Ref.) Chapter 4 of Carl Edward Rasmussen and Christopher K. I.
            Williams. 2005. Gaussian Processes for Machine Learning
            (Adaptive Computation and Machine Learning). The MIT Press.

        [There is probably a mistake with the scale factor.]
        """
        # Scale of the Fourier transform of the kernel
        w_mean = np.zeros(n_features)
        w_cov_matrix = self.scale**2*np.identity(n_features)

        # Sample from multivariate student t distribution
        self.w = self._random_multivariate_student_t(
            w_mean,
            w_cov_matrix,
            self.nu,
            self.n_random_features//2,
        )

        return self

    def _random_multivariate_student_t(
        self,
        mean: np.ndarray,
        cov_matrix: np.ndarray,
        df: float,
        n_samples: int,
    ) -> np.ndarray:
        """
        Generate samples from a multivariate Student's t distribution.

            (Ref.) https://en.wikipedia.org/wiki/Multivariate_t-distribution

        Parameters
        ----------
        mean:
            Mean vector of the distribution.
        cov_matrix:
            Covariance matrix of the distribution.
        df:
            Degrees of freedom.
        n_samples:
            Number of samples to generate.

        Returns
        -------
        X:
            Array of shape (n_samples, len(mean)) with the generated samples.
        """
        # Dimensions of multivariate Student's t distribution.
        D = len(mean)

        # Formula for generating samples of a Student's t
        rng = np.random.default_rng()
        x = rng.chisquare(df, n_samples)/df
        Z = rng.multivariate_normal(
            np.zeros(D),
            cov_matrix,
            n_samples,
        )
        X = mean + Z/np.sqrt(x)[:, np.newaxis]

        return X


class NystroemFeaturesSampler():
    """ Sample features following the Nyström method. """

    def __init__(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        self.kernel = kernel
        self.component_indices = None
        self.X_reduced = None
        self.reduced_kernel_matrix = None
        self.sqrtm_pinv_reduced_kernel_matrix = None

    def fit(
        self,
        X: np.ndarray,
        n_features_sampled: int
    ) -> NystroemFeaturesSampler:
        """
        Precompute auxiliary quantities for Nyström features.

        Parameters
        ----------
        X:
            Data matrix.
        n_features_sampled:
            Number of features to sample.

        Returns
        -------
        self:
            The instance itself.
        """
        # Sample subset of training instances
        n_instances = len(X)
        rng = np.random.default_rng()
        self.component_indices = rng.choice(
            range(n_instances),
            size=n_features_sampled,
            replace=False,
        )
        self.X_reduced = X[self.component_indices, :]

        # Compute reduced kernel matrix
        self.reduced_kernel_matrix = self.kernel(
            self.X_reduced,
            self.X_reduced
        )

        # Enforce symmetry of kernel matrix
        self.reduced_kernel_matrix = (
            self.reduced_kernel_matrix + self.reduced_kernel_matrix.T
        )/2.0

        # Compute the matrix (W+)^1/2
        self.sqrtm_pinv_reduced_kernel_matrix = sp.linalg.sqrtm(
            np.linalg.pinv(
                self.reduced_kernel_matrix,
                rcond=1.0e-6,
                hermitian=True
            )
        )

        # Check that complex part is negligible and eliminate it
        if np.iscomplexobj(self.sqrtm_pinv_reduced_kernel_matrix):
            threshold_imaginary_part = 1.0e-6
            max_imaginary_part = np.max(
                np.abs(np.imag(self.sqrtm_pinv_reduced_kernel_matrix))
            )

            if max_imaginary_part > threshold_imaginary_part:
                warnings.warn(
                    'Maximum imaginary part is {}'.format(max_imaginary_part)
                )

            self.sqrtm_pinv_reduced_kernel_matrix = np.real(
                self.sqrtm_pinv_reduced_kernel_matrix
            )

        return self

    def approximate_kernel_matrix(
        self,
        X: np.ndarray,
        n_features_sampled: int,
    ) -> np.ndarray:
        """
        Approximate the kernel matrix k(X, X) using Nyström features.

        Parameters
        ----------
        X:
            Data matrix.
        n_features_sampled:
            Number of features to sample.

        Returns
        -------
        Approximated kernel matrix.
        """
        X_features = self.fit_transform(n_features_sampled, X)
        return X_features@X_features.T

    def fit_transform(
        self,
        n_features_sampled: int,
        X: np.ndarray,
        X_prime: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fit model and compute Nyström features.

        Use X as training data and compute features of X_prime.

        Parameters
        ----------
        n_features_sampled:
            Number of features to sample.
        X:
            Training data matrix.
        X_prime:
            Data matrix to compute Nyström features.
        """
        self.fit(X, n_features_sampled)
        if X_prime is None:
            X_prime = X
        X_prime_nystroem = self.transform(X_prime)

        return X_prime_nystroem

    def transform(self, X_prime: np.ndarray) -> np.ndarray:
        """Compute Nyström features using fitted quantities."""
        J = self.kernel(X_prime, self.X_reduced)
        X_prime_nystroem = J@self.sqrtm_pinv_reduced_kernel_matrix
        return X_prime_nystroem


def demo_kernel_approximation_features(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    features_sampler: Union[RandomFeaturesSampler, NystroemFeaturesSampler],
    n_random_features: np.array,
) -> None:
    """
    Kernel approximation using Random Fourier features (RFF) or Nyström method.

    It shows a graph of each approximated kernel and also the mean and max
    absolute error of the approximation.

    Parameters
    ----------
    X:
        Data matrix.
    kernel:
        Kernel function that represents the kernel matrix to approximate.
    features_sampler:
        Object representing the sampling strategy.
    n_random_features:
        Array with a collection of numbers of random features to sample.
    """
    # Set plot options
    n_plots = len(n_random_features) + 1
    fig, axes = plt.subplots(1, n_plots)
    fig.set_size_inches(15, 4)
    font = {'fontname': 'arial', 'fontsize': 18}

    # Plot original kernel
    kernel_matrix = kernel(X, X)
    axes[0].imshow(kernel_matrix, cmap=plt.cm.Blues)
    axes[0].set_title('Exact kernel', **font)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot kernel approximations
    for n, ax in zip(n_random_features, axes[1:]):
        # print('[Kernel approximation] # of features = ', n)

        # Get kernel matrix approximation
        X_features = features_sampler.fit_transform(n, X)
        kernel_matrix_approx = X_features@X_features.T

        # Plot approximation
        ax.imshow(kernel_matrix_approx, cmap=plt.cm.Blues)

        # Compute and plot approximation errors
        err_approx = kernel_matrix - kernel_matrix_approx
        err_mean = np.mean(np.abs(err_approx))
        err_max = np.max(np.abs(err_approx))
        ax.set_xlabel('err (mean) = {:.4f} \n err (max) = {:.4f}'.format(
            err_mean,
            err_max
        ), **font)

        ax.set_title('{} features'.format(n), **font)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    """
    from sklearn import datasets, svm
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.metrics.pairwise import rbf_kernel

    # A not so simple 2-D problem
    X, Y = datasets.make_moons(n_samples=100, noise=0.3, random_state=0)

    # Compute grid of points for plotting the decision regions
    grid_x, grid_y = np.meshgrid(
        np.linspace(-3, 3, 50),
        np.linspace(-3, 3, 50)
    )
    grid_X = np.c_[grid_x.ravel(), grid_y.ravel()]

    # Kernel matrix
    gamma = 0.5
    def kernel(X, Y):
        return rbf_kernel(X, Y, gamma=gamma)

    # Nyström features
    n_nystroem_features = 20
    nystroem_sampler = NystroemFeaturesSampler(kernel)
    nystroem_features = nystroem_sampler.fit_transform(n_nystroem_features, X)
    nystroem_features_grid = nystroem_sampler.transform(grid_X)

    # Classifier
    clf = svm.SVC(kernel='linear')
    # clf = svm.NuSVC(gamma='auto')
    clf.fit(nystroem_features, Y)
    """

    from sklearn import datasets
    from sklearn.metrics.pairwise import rbf_kernel

    # 3-D data
    n_instances = 1000
    X, t = datasets.make_s_curve(n_instances, noise=0.1)
    X = X[np.argsort(t)]

    # 2-D data
    # X, y = datasets.make_moons(n_samples=400, noise=.05)
    # X = X[np.argsort(y)]

    # Reshape if necessary
    if (X.ndim == 1):
        X = X[:, np.newaxis]

    # Kernel parameters
    sigma = 1.0
    gamma = 1.0 / (2.0 * sigma**2)

    # Kernel function
    def kernel(X, Y):
        return rbf_kernel(X, Y, gamma=gamma)

    # Nyström features
    n_nystroem_features = [10, 100, 1000]
    nystroem_features = NystroemFeaturesSampler(kernel)

    demo_kernel_approximation_features(
        X,
        kernel,
        nystroem_features,
        n_nystroem_features,
    )
