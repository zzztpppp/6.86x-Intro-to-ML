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
    X_filled = X
    not_missing = (X != 0)

    # Start by fixing vss
    new_vs = vs

    # Iterate until X_filled has no detective change
    count = 0
    while True:
        count += 1
        print(count)
        X_old = X_filled
        new_us = (X @ new_vs) / (not_missing @ (new_vs ** 2) + lamda)
        new_vs = (X.T @ new_us) / (not_missing.T @ (new_us**2) + lamda)

        # us and vs only have one rank
        X_filled = new_us @ new_vs.T
        if common.rmse(X_filled, X_old) < 0.001:
            break

    return new_us, new_vs

def fact_with_surprise(X):
    from surprise import NMF
    from surprise import trainset
    model = NMF(n_factors=1, init_low=1, init_high=5)
    model.fit(X)
    return


if __name__ == "__main__":
    test_X = X
    num_u,num_i = test_X.shape
    is_missing = test_X == 0
    us = np.random.randint(1, 6, (num_u, 1))
    vs = np.random.randint(1, 6, (num_i, 1))
    # us, vs = altering_minimize(test_X, us, vs, 1)
    rmse = []
    fact_with_surprise(test_X)
    for l in [0]:
        us, vs = altering_minimize(test_X, us, vs, l)
        x_pre_raw = (us @ vs.T)
        x_pred = x_pre_raw * is_missing + (test_X *(~is_missing))
        rmse.append(common.rmse(x_pre_raw,test_X))

    print(rmse)
