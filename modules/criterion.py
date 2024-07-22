from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def forward(self, y, y_pred):
        pass

    @abstractmethod
    def backward(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MSE(Loss):
    def forward(self, y, y_pred):
        self.y_pred = y_pred.clip(min=1e-8)
        self.y_one_hot = np.eye(self.y_pred.shape[1])[y]   
        return np.mean((self.y_pred - self.y_one_hot) ** 2)
    
    def backward(self): 
        return 2 * (self.y_pred - self.y_one_hot) / self.y_one_hot.shape[0]
    

class CrossEntropy(Loss):
    def forward(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred.clip(min=1e-8)
        self.y_one_hot = np.eye(self.y_pred.shape[1])[y]
        return -np.sum(self.y_one_hot * np.log(self.y_pred)) / self.y.shape[0]

    def backward(self):
        return (self.y_pred - self.y_one_hot) / self.y.shape[0]