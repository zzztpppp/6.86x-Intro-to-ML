import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")


# TODO: Your code here
def test_kmeans():
    for k in [1, 2, 3, 4]:
        para_list = []
        for seed in [0,1, 2, 3, 4]:
            gm, post = common.init(X, k, seed)
            mixture, p, cost = kmeans.run(X, gm,post)
            para_list.append((mixture, p, cost))
        max_para = max(para_list, key=lambda x: x[2])
        common.plot(X, max_para[0], max_para[1], 'Kmeans on toy data with {k}'.format(k=k))
    return max_para[0], max_para[1]


def test_naive_em():
    for k in [1, 2, 3, 4]:
        para_list = []
        for seed in [0, 1, 2, 3, 4]:
            gm, post = common.init(X, k, seed)
            mixture, p, cost = naive_em.run(X, gm, post)
            para_list.append((mixture, p, cost))
        max_para = max(para_list, key=lambda x: x[2])
        common.plot(X, max_para[0], max_para[1], 'EM on toy data with {k}'.format(k=k))
    return max_para[0], max_para[1]


if __name__ == "__main__":
    mk, pk = test_kmeans()
    m, p = test_naive_em()
