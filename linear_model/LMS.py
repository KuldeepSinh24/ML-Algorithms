import collections 
from LinearModel import LinearModel
import numpy as np
# LMS class 
class LMS(LinearModel):
    """
        Lease Mean Squares. online learning algorithm
    
        attributes
        ==========
        w        nd.array
                 weight matrix
        alpha    float
                 learning rate
    """
    def __init__(self, alpha):
        LinearModel.__init__(self)
        self.alpha = alpha
        self.w = None
    def train(self, X, T):
        """
            performs batch training by using train_step function
        """
        N = X.shape[0]
        self.w = np.zeros(X.shape[1]+1)
        for i in range(N):
            self.train_step(X[i], T[i])

    def train_step(self, x, t):
        """
            train LMS model one step here the x is 1d vector
        """
        x1 = self.add_ones(x.reshape((1,-1)))
        # print("\nY  = x1 shape ({0}) @ w.T shape ({1}) :".format(x1.shape,self.w.T.shape))
        y = x1 @ self.w.T
        self.w = self.w - self.alpha * (y - t)*x1
        # print("\nY shape ({0}) = x1 shape ({1}) @ w.T shape ({2}) :".format(y.shape,x1.shape,self.w.T.shape))
        # print("\nW Value :"+str(self.w))

    
    def use(self, X):
        """
            apply the current model to data X
        """
        X1 = self.add_ones(X)
        y = X1 @ self.w.T
        return y

        