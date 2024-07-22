import numpy as np

from .layer import Layer


class ReLU(Layer):
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, grad):
        grad[self.X <= 0] = 0
        return grad
    

class Tanh(Layer):
    def forward(self, X):
        self.output = np.tanh(X)
        return self.output

    def backward(self, grad):
        return grad * (1 - self.output ** 2)


class SoftMax(Layer):
    def forward(self, X):
        shift_X = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(shift_X)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def backward(self, grad):
        return grad