from BaseClass import Classifier
import numpy as np

class LDA(Classifier):
    
    def __init__(self):
        Classifier.__init__(self)
        self.mu = []
        self.Sigma = []
        self.prior = []        

    def discriminant(self, X, mu, sigma, prior):
        sigInv = np.linalg.inv(sigma)
        return X@sigInv@mu - .5*mu.T@sigInv@mu + np.log(prior)
                    
    def train(self, X, T):
        Xs = self.normalize(X)
        N = Xs.shape[0]
        self.Sigma = np.cov(Xs.T)
        for c in np.unique(T):
            c = T==c
            mu = np.mean(Xs[c,:],0)
            prior = np.sum(c) / N
            
            self.mu.append(mu)
            self.prior.append(prior)
        
            
    def use(self, X):
        Xs = (X-self.meanX)/self.stdX
        d = []
        for i in range(len(self.mu)):
            d.append(self.discriminant(Xs, self.mu[i], self.Sigma, self.prior[i]))
        return d