from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Layer):
    def __init__(self, in_dim, out_dim, init=None):
        if init:
            self.W, self.b = init(in_dim, out_dim)
        else:
            self.W = np.random.randn(out_dim, in_dim)
            self.b = np.random.randn(out_dim)
    
    def forward(self, X):  
        self.X = X
        return self.X @ self.W.T + self.b
    
    def backward(self, grad):
        self.dW = grad.T @ self.X
        self.db = grad.sum(0)
        return grad @ self.W


class Conv(Layer):
    pass