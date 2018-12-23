from Pocket import PocketAlgorithm 
import numpy as np
import matplotlib.pyplot as plt

def main():
    mu1 = [-1, -1]
    cov1 = np.eye(2)

    mu2 = [2,3]
    cov2 = np.eye(2) * 3

    C1 = np.random.multivariate_normal(mu1, cov1, 50)
    C2 = np.random.multivariate_normal(mu2, cov2, 50)

    plt.plot(C1[:, 0], C1[:, 1], 'or')
    plt.plot(C2[:, 0], C2[:, 1], 'xb')

    plt.xlim([-3, 6])
    plt.ylim([-3, 7])

    X = np.vstack((C1, C2))
    N = X.shape[0]
    T = np.ones(N)
    T[:50] *= -1 

    pocket = PocketAlgorithm(X)
    pocket.train(X,T)
    xt = np.array([-2, 5])
    yt = -pocket.w_pocket[0] * xt / pocket.w_pocket[1]
    plt.plot(xt,yt)