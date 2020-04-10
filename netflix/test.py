import numpy as np
import em
import common

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

K = 4
n, d = X.shape
seed = 0


def test_em():
    init_mixture, post = common.init(X, K, seed)
    mixture, post, c = em.run(X, init_mixture, post)

    prediction = em.fill_matrix(X, mixture)
    print(c)
    print(common.rmse(prediction, X_gold))


def test_incomplete_em():
    for k_s in [1, 12]:
        lps = []
        for s in [0, 1, 2, 3, 4]:
            print(k_s, s)
            init_mixture, post = common.init(X, k_s, s)
            model = em.run(X, init_mixture, post)
            lps.append(model)
        best = max(lps, key=lambda x: x[-1])
        print(best[-1])


def test_k12():
    lls = []
    for s in [0, 1, 2, 3, 4]:
        print(s)
        init_mixture, post = common.init(X, 12, s)
        model = em.run(X, init_mixture, post)
        lls.append(model)
    m, p, l = max(lls, key=lambda x: x[-1])
    prediction = em.fill_matrix(X, m)
    return common.rmse(prediction, X_gold)


if __name__ == "__main__":
    print(test_k12())
