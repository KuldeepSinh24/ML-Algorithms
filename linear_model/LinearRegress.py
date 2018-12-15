import LinearModel

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
            add a column basis to X input matrix
        """
        return LinearModel.add_ones(self,X)
    
    # train lease-squares model
    def train(self, X, T):

        ## TODO: replace this with your codes
        X1 = self.add_ones(X)
        self.w =  np.linalg.inv(X1.T @ X1) @ X1.T @ T
        
    
    # apply the learned model to data X
    def use(self, X):
        ## TODO: replace this with your codes
        X1 = self.add_ones(X)
        y = X1 @ self.w
        return y