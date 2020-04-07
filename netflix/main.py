import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")


# TODO: Your code here
def test_kmeans():
    for k in [1, 2, 3, 4]:
        cost_list = []
        for seed in [0,1, 2, 3, 4]:
            gm, post = common.init(X, k, seed)
            mixture, p, cost = kmeans.run(X, gm,post)
            cost_list.append(cost)

        print(min(cost_list))


def test_naive_em():
    for k in [3]:
        cost_list = []
        for seed in [0]:
            gm, post = common.init(X, k, seed)
            mixture, p, cost = naive_em.run(X, gm, post)
            cost_list.append(cost)
    print(cost_list)
    return mixture, p


if __name__ == "__main__":
    m, p = test_naive_em()
    common.plot(X, m, p, 'EM on toy data')
