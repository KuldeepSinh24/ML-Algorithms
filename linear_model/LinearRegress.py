from LinearModel import LinearModel
import numpy as np
class LinearRegress(LinearModel): 
    """ 
        LinearRegress class 
        
        attributes
        ===========
        w    nd.array  (column vector/matrix)
             weights
    """
    def __init__(self):
        LinearModel.__init__(self)

    def add_ones(self, X):
        """
            add's a column basis to X input matrix
        """
        return LinearModel.add_ones(self,X)
    
    # train lease-squares model
    def train(self, X, T):
        """
            train's the weights using normal equation
        """
        X1 = self.add_ones(X)
        self.w =  np.linalg.inv(X1.T @ X1) @ X1.T @ T
        
    
    # apply the learned model to data X
    def use(self, X):
        """
            uses the weights to predict the output
        """
        X1 = self.add_ones(X)
        y = X1 @ self.w
        return y