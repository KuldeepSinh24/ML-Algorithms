from  BaseClass import Classifier
import numpy as np
class QDA(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.mu = []
        self.Sigma = []
        self.prior = []
        
    def discriminant(self,X, mu, sigma,prior):
        sigma_inv = np.linalg.inv(sigma)
        diffv = X - mu
        return -0.5  * np.log(np.linalg.det(sigma)) \
            - 0.5 * np.sum(diffv @ sigma_inv * diffv, axis=1) \
            + np.log(prior)
 
    def train(self, X, T):
        Xs = self.normalize(X)
        N = Xs.shape[0]
        for c in np.unique(T):
            c = T==c
            mu = np.mean(Xs[c,:],0)
            Sigma = np.cov(Xs[c].T)
            prior = np.sum(c) / N
            
            self.mu.append(mu)
            self.Sigma.append(Sigma)
            self.prior.append(prior)
            
    def use(self, X):
        Xs = (X-self.meanX)/self.stdX
        d = []
        for i in range(len(self.mu)):
            print("XS:"+str(Xs.shape))
            print("mu:"+str(self.mu[i]))
            print("Sigma:"+str(self.Sigma[i]))
            print("Prior:"+str(self.prior[i]))

            d.append(self.discriminant(Xs, self.mu[i], self.Sigma[i], self.prior[i]))
        return d