"""Matrix factorization model for matrix completion"""

import numpy as np
import common
from scipy.sparse.linalg import svds

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")


def altering_minimize(X, us, vs, lamda):
    """
    Fill missing values in X by matrix factorization with
    altering minimizing method.

    Factorize X into the form of UV

    We first keep vs fixed, to evaluate the us. Then evaluate us and vs in such an
    approach alternatively.

    Args:
        X: Ratings with missing values
        us: User factors
        vs: Move factors
        lamda: Normalization term

    Returns:
        The us and vs that minimize out cost
    """
    # X_filled = X
    rmse_fill = np.inf

    # Start by fixing vss
    new_vs = vs
    new_us = us

    print(common.rmse(us @ vs.T, X), loss(X, us, vs, lamda))
    # Iterate until X_filled has no detective change
    count = 0
    while True:
        count += 1
        rmse_old = rmse_fill

        new_us = update_us(X, new_vs, new_us, lamda)
        rmse_fill = common.rmse(new_us @ new_vs.T, X)
        ls = loss(X, new_us, new_vs, lamda)
        print(rmse_fill, ls)

        new_vs = update_vs(X, new_vs, new_us, lamda)
        rmse_fill = common.rmse(new_us @ new_vs.T, X)
        ls = loss(X, new_us, new_vs, lamda)
        print(rmse_fill, ls)

        print()
        if rmse_old - rmse_fill < 0.001:
            break

    return new_us, new_vs


def update_us(X, vs, us, l):
    new_us = us.copy()
    num_u, num_f = us.shape
    if num_f == 1:
        return (X @ vs) / ((X != 0) @ (vs ** 2) + l)
    else:
        for i in range(num_u):
            x_i = X[i, :]
            x_observed = (x_i != 0)
            vs_rated = vs.T[:, x_observed]
            matrix_a = (vs_rated @ vs_rated.T + np.sum(x_observed)*l*np.identity(num_f))
            matrix_v = vs_rated @ np.expand_dims(x_i[x_observed], 1)
            new_us[i, :] = np.squeeze(np.linalg.inv(matrix_a) @ matrix_v)
    return np.array(new_us)


def update_vs(X, vs, us, l):
    new_vs = vs.copy()
    num_v, num_f = vs.shape
    if num_f == 1:
        return (X.T @ us) / ((X != 0).T @ (us ** 2) + l)
    else:
        for j in range(num_v):
            x_j = X[:, j]
            x_obeserved = (x_j != 0)
            us_rated = us.T[:, x_obeserved]
            matrix_a = (us_rated @ us_rated.T + np.sum(x_obeserved)*l*np.identity(num_f))
            matrix_v = us_rated @ np.expand_dims(x_j[x_obeserved], 1)
            new_vs[j, :] = np.squeeze(np.linalg.inv(matrix_a) @ matrix_v)

    return np.array(new_vs)


def loss(X, us, vs, l):
    not_missing = X != 0
    se = np.sum((X - (us @ vs.T) * not_missing)**2)/2
    regularization = np.sum(us ** 2) + np.sum(vs**2)
    regularization = regularization*l/2

    return se + regularization


if __name__ == "__main__":
    test_X = X
    num_u,num_i = test_X.shape
    is_missing = test_X == 0
    # us, s, vs = svds(test_X)
    # us, vs = altering_minimize(test_X, us, vs, 1)
    rmse = []
    for k in [1,2,3,4,5,6,7,8,9,10]:
        us = np.random.randint(1, 6, (num_u, k)).astype('float')
        vs = np.random.randint(1, 6, (num_i, k)).astype('float')
        for l in [0]:
            us, vs = altering_minimize(test_X, us, vs, l)
            x_pre_raw = (us @ vs.T)
            x_pred = x_pre_raw * is_missing + (test_X *(~is_missing))
            rmse.append(common.rmse(x_pred,X_gold))

    print(rmse)
