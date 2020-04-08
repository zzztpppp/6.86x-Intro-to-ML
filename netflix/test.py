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
    em.run(X, init_mixture, post)


if __name__ == "__main__":
    test_em()
