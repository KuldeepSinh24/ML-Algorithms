from BaseClass import Classifier
import numpy as np
from copy import deepcopy as copy

class PocketAlgorithm(Classifier):
    """
        class for Pocket Algorithm 
    """
    def __init__(self,X,maxiter=500,alpha=0.1):
        Classifier.__init__(self)
        self.w = np.zeros((X.shape[1]))
        self.w_pocket = copy(self.w)
        self.maxiter = maxiter
        self.alpha = alpha


    def compare(self,X,T,w,wp):
        y = np.sign(X @ w)
        yp = np.sign(X @ wp)
        return 1 if np.sum(y==T) >= np.sum(yp == T) else -1
    
    def train(self,X,T):
        Xs = self.normalize(X)
        for i in range(self.maxiter):
            converged = True
            for k in np.random.permutation(Xs.shape[0]):
                y = self.w @ Xs[k]
                np.sign(y)
                np.sign(T[k])
                if np.sign(y) != np.sign(T[k]):
                    self.w += self.alpha * T[k] * Xs[k]
                    converged = False
                    if self.compare(Xs,T,self.w,self.w_pocket) > 0:
                        self.w_pocket[:] = self.w[:]
            if converged:
                print("converged at ",i)
                break
        print("End of training: ",i)
        
    def use(self,X):
        Xs = self.normalize(X)
        #plt.plot(Xs@self.w_pocket)
        return Xs@self.w_pocket