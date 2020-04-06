"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    def __multi_norm_pdf(x: np.ndarray, mus: np.ndarray, vars: np.ndarray) -> float:
        """
        Calculate density of a uncorrated multivariate normal distribution,
        given its mean vector and covariance vector
        """
        pi = 3.14159265
        d = len(x)
        cov_matrix = np.diag(vars)
        diff = np.expand_dims(x - mus, 0) @ np.linalg.inv(cov_matrix) @ np.expand_dims(x - mus, 1)
        return (1 / np.power(2 * pi, d / 2)) * (1 / np.sqrt(np.linalg.det(cov_matrix))) * (np.exp(-diff / 2)).item()

    the_mus = mixture.mu
    the_vars = mixture.var
    the_ps = mixture.p

    # P(i| k)
    likelihood = []
    for k in range(len(the_ps)):
        likelihood.append(np.apply_along_axis(lambda x: __multi_norm_pdf(x, the_mus[k, :], the_vars), 1, X))

    likelihood = np.array(likelihood).T

    # P(i, k)
    p_i_k = np.multiply(likelihood, np.expand_dims(the_ps, 1))

    # P(i)
    p_i = np.sum(p_i_k, axis=1)

    # P(k | i)
    posterior = p_i_k / np.expand_dims(p_i, 1)

    # Log-likelihood
    l_p = np.sum(np.log(p_i))

    return posterior, l_p


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    X_exp = np.tile(X, (post.shape[1], 1))

    new_mus = []
    for k in post.shape[1]:
        count = np.sum(np.multiply(X, np.expand_dims(post[:, k], 1)), axis=0)
        new_mus.append(count / np.sum(post[:, k]))



    new_mus = np.array(new_mus)

    new_vars = []


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
