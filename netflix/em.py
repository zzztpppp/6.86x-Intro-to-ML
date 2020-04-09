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
        x_obs = x[x != 0]
        mus = mus[x != 0]

        pi = 3.14159265
        d = len(x_obs)
        cov_matrix = np.identity(d) * var

        diff = np.expand_dims(x_obs - mus, 0) @ np.linalg.inv(cov_matrix) @ np.expand_dims(x_obs - mus, 1)
        return (1 / np.power(2 * pi, d / 2)) * (1 / np.sqrt(np.linalg.det(cov_matrix))) * (np.exp(-diff / 2)).item()

    the_mus = mixture.mu
    the_vars = mixture.var
    the_ps = mixture.p

    num_points, n_d = X.shape
    num_k = the_mus.shape[0]

    # To avoid iteration, use matrix calculation
    # for each point-cluster pairs
    x_exp = np.tile(X, (num_k, 1))
    mus_exp = np.repeat(the_mus, num_points, axis=0)
    var_exp = the_vars.repeat(num_points)

    # P(i | j)
    x_mus_exp = np.hstack([x_exp, mus_exp, np.expand_dims(var_exp, axis=1)])
    likelihood = np.apply_along_axis(lambda x: __multi_norm_pdf(x[:n_d], x[n_d:-1], x[-1]), 1, x_mus_exp)\
        .reshape((num_points,num_k), order='F')

    # log(P(i, j))
    log_joint_i_j = np.log(likelihood) + np.log(np.expand_dims(the_ps + 1e-16, 0))

    # P(i)
    p_i = np.sum(np.exp(log_joint_i_j), axis=1)

    # log Posterior P(j | i)
    log_posterior = log_joint_i_j - np.expand_dims(logsumexp(log_joint_i_j, 1), 1)

    # Log-likelihood
    l_p = np.sum(np.log(p_i))
    return np.exp(log_posterior), l_p


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

    is_missing = (X == 0)

    num_points, num_k = post.shape
    num_d = X.shape[1]
    old_mus = mixture.mu

    # Update expectation
    new_mus = []
    for k in range(num_k):
        post_k_d = np.tile(np.expand_dims(post[:, k], 1), (1, num_d))
        post_k_d[is_missing] = 0
        soft_count = np.sum(np.multiply(X, post_k_d), axis=0)
        new_mus.append(soft_count / np.sum(post_k_d, axis=0))
    new_mus = np.array(new_mus)

    # Don't update the mus when its soft count < 1
    soft_count_k_d = []
    for k in range(num_k):
        soft_count_k = np.sum(~is_missing * np.expand_dims(post[:, k], 1), axis=0)
        soft_count_k_d.append(soft_count_k)
    soft_count_k_d = np.array(soft_count_k_d) < 1
    new_mus = old_mus * soft_count_k_d + new_mus * (~soft_count_k_d)

    # Update variance
    new_vars = []
    X_exp = np.tile(X, (num_k, 1))
    new_mus_exp = np.repeat(new_mus, num_points, axis=0)
    is_missing_exp = np.tile(is_missing, (num_k, 1))
    observed_length = np.sum(~is_missing_exp, axis=1)

    new_mus_exp[is_missing_exp] = 0

    se = (X_exp - new_mus_exp) ** 2
    post_flatten_exp = np.tile(np.expand_dims(post.reshape(num_points * num_k, order='F'), 1), (1, num_d))
    weighted_vars = np.multiply(se, post_flatten_exp)
    for k in range(num_k):
        weighted_sum_vars = np.sum(weighted_vars[k * num_points: k * num_points + num_points, :])
        weights = np.sum(
            post_flatten_exp[k * num_points: k * num_points + num_points, 0] * observed_length[k * num_points: k * num_points + num_points],
        )

        new_vars.append(
            weighted_sum_vars / weights
        )

    new_vars = np.array(new_vars)
    new_vars[new_vars < min_variance] = min_variance
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
        mixture = mstep(X, post, mixture)
        print(post)
        print("mu is")
        print(mixture.mu)
        print("var is")
        print(mixture.var)
        print("p is")
        print(mixture.p)
        print('-ll')
        print(lp)
        print()


        if lp - old_lp <= 10e-7 * np.abs(lp):
            break
    return mixture, post, lp


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
