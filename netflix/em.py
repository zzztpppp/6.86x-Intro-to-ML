"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

    def __multi_norm_pdf(x: np.ndarray, mus: np.ndarray, var: float) -> float:
        """
        Calculate density of a uncorrated multivariate normal distribution sharing the
        same variance given its mean vector and covariance vector

        Will remove 0s in x, for specific use of this function
        """
        x = x[x != 0]
        mus = mus[x != 0]

        pi = 3.14159265
        d = len(x)
        cov_matrix = np.identity(d) * var

        diff = np.expand_dims(x - mus, 0) @ np.linalg.inv(cov_matrix) @ np.expand_dims(x - mus, 1)
        return (1 / np.power(2 * pi, d / 2)) * (1 / np.sqrt(np.linalg.det(cov_matrix))) * (np.exp(-diff / 2)).item()

    the_mus = mixture.mu
    the_vars = mixture.var
    the_ps = mixture.p

    num_points, n_d = X.shape
    num_k = the_mus[0]

    # To avoid iteration, use matrix calculation
    # for each point-cluster pairs
    x_exp = np.tile(X, (num_k, 1))
    mus_exp = np.repeat(the_mus, num_points, axis=0)
    var_exp = the_vars.repeat(num_points)

    # P(i | j)
    x_mus_exp = np.hstack([x_exp, mus_exp, np.expand_dims(var_exp, axis=1)])
    likelihood = np.apply_along_axis(lambda x: __multi_norm_pdf(x[:n_d], x[n_d:-1], x[-1]))\
        .reshape((num_points,num_k), order='F')

    # P(i, j)
    joint_i_j = np.multiply(likelihood, np.expand_dims(the_ps, 0))

    # P(i)
    p_i = np.sum(joint_i_j, axis=1)

    # Posterior P(j | i)
    posterior = joint_i_j / np.expand_dims(p_i, 1)

    # Log-likelihood
    l_p = np.sum(np.log(p_i))
    return posterior, l_p


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
