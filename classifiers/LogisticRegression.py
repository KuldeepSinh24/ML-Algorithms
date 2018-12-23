from BaseClass import Classifier
import numpy as np

# LogisticRegression Class
from copy import deepcopy as copy
class LogRegression(Classifier): 
    """ 
        Logistic Regression class 
        
        attributes
        ===========
        
    """
    def __init__(self,alpha):
        Classifier.__init__(self)
        self.w = None
        self.alpha = alpha
    
    def softmax(self,z):
        if not isinstance(z, np.ndarray):
            z = np.asarray(z)
        f = np.exp(z) 
        return f / (np.sum(f, axis=1, keepdims=True) if len(z.shape) == 2 else np.sum(f))

    def g(self,X, w):
        return self.softmax(X @ w) 
        
    def train(self, X, T):
        X = self.normalize(X)
       
        N = X.shape[0]
        D = X.shape[1]
        K = len(np.unique(T))
        
        self.w = np.random.rand(D+1, K)
        niter = 10
        X = self.add_ones(X)
        T = T.as_matrix()
        T = T.reshape(-1,1)
        T = self.add_ones(T)
        
        for step in range(niter):
            for i in range(N):
                ys = self.g(X,self.w)
                self.w += self.alpha * X.T @ (T - ys)  
              
    
    # apply the learned model to data X
    def use(self, X):
        Xs = (X - self.meanX)/self.stdX
        Xs = self.add_ones(Xs)
        return self.g(Xs, self.w)