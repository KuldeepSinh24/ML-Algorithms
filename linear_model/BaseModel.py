import numpy as np 
from abc import ABC, abstractmethod

# Super class for machine learning models 
class BaseModel(ABC):
    """ Super class for ITCS Machine Learning Class"""
    
    @abstractmethod
    def train(self, X, T):
        pass

    @abstractmethod
    def use(self, X):
        pass
