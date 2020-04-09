import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0


def test_em():
    init_mixture, post = common.init(X, K, seed)
    mixture, post, c = em.run(X, init_mixture, post)

    print(em.fill_matrix(X, mixture))


def test_incomplete_em():
    for k_s in [1, 12]:
        lps = []
        for s in [0, 1, 2, 3, 4,5]:
            init_mixture, post = common.init(X, k_s, s)
            _, _, lp = em.run(X, init_mixture, post)
            lps.append(lp)
        print(max(lps))


def test_k12():
    lls = []
    for s in [0, 1, 2, 3, 4, 5]:
        init_mixture, post = common.init(X, 12, s)
        lls.append(em.run(X, init_mixture,post))
    m, p, l = max(lls, key=lambda x: x[-1])
    prediction = em.fill_matrix(X, m)
    print(common.rmse(prediction, X_gold))


if __name__ == "__main__":
    print(test_k12())
