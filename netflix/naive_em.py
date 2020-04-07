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

    def __multi_norm_pdf(x: np.ndarray, mus: np.ndarray, var: float) -> float:
        """
        Calculate density of a uncorrated multivariate normal distribution sharing the
        same variance given its mean vector and covariance vector
        """
        pi = 3.14159265
        d = len(x)
        cov_matrix = np.identity(d) * var

        diff = np.expand_dims(x - mus, 0) @ np.linalg.inv(cov_matrix) @ np.expand_dims(x - mus, 1)
        return (1 / np.power(2 * pi, d / 2)) * (1 / np.sqrt(np.linalg.det(cov_matrix))) * (np.exp(-diff / 2)).item()

    the_mus = mixture.mu
    the_vars = mixture.var
    the_ps = mixture.p

    # P(i| k)
    likelihood = []
    for k in range(len(the_ps)):
        likelihood.append(np.apply_along_axis(lambda x: __multi_norm_pdf(x, the_mus[k, :], the_vars[k]), 1, X))

    likelihood = np.array(likelihood).T

    # P(i, k)
    p_i_k = np.multiply(likelihood, np.expand_dims(the_ps, 0))

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

    num_points = X.shape[0]
    num_k = post.shape[1]

    # Update expectation
    new_mus = []
    for k in range(num_k):
        count = np.sum(np.multiply(X, np.expand_dims(post[:, k], 1)), axis=0)
        new_mus.append(count / np.sum(post[:, k]))

    new_mus = np.array(new_mus)

    # Update variance
    new_vars = []
    X_exp = np.tile(X, (num_k, 1))
    new_mus_exp = np.repeat(new_mus, num_points, axis=0)
    se = (X_exp - new_mus_exp)**2
    post_flatten = post.reshape(num_points*num_k, order='F')
    weighted_vars = np.multiply(se, np.expand_dims(post_flatten, 1))
    for k in range(num_k):
        weighted_sum_vars = np.sum(weighted_vars[k*num_points: k*num_points + num_points, :], axis=0)
        new_vars.append(
            weighted_sum_vars / np.sum(post_flatten[k*num_points: k*num_points + num_points])
        )
    new_vars = np.mean(new_vars, axis=1)

    # Update prior
    new_ps = np.mean(post, axis=0)

    return GaussianMixture(new_mus, new_vars, new_ps)


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

    # Arbitrary initialization
    lp = -np.nan
    while True:
        old_lp = lp
        post, lp = estep(X, mixture)
        mixture = mstep(X, post)
        if lp - old_lp <= 10e-7 * np.abs(lp):
            break
    return mixture, post, lp

