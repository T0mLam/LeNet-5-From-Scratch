from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from nptyping import Number, NDArray, Shape
from scipy import signal

from .init import Initialization


class Layer(ABC):
    @abstractmethod
    def forward(self, X: NDArray) -> NDArray:
        pass

    @abstractmethod
    def backward(self, grad: NDArray) -> NDArray:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Layer):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        init: Optional[Initialization]=None
    ) -> None:
        if init:
            self.W, self.b = init(in_dim, out_dim)
        else:
            self.W = np.random.randn(out_dim, in_dim)
            self.b = np.random.randn(out_dim)
    
    def forward(
            self,
            X: NDArray[Shape["*, *"], Number]
        ) -> NDArray[Shape["*, *"], Number]:  
        self.X = X
        return self.X @ self.W.T + self.b
    
    def backward(
            self,
            grad: NDArray[Shape["*, *"], Number]
        ) -> NDArray[Shape["*, *"], Number]:
        self.dW = grad.T @ self.X
        self.db = grad.sum(0)
        return grad @ self.W


class Conv(Layer):
    def __init__(
        self,
        in_channel: int,
        out_channel: int, 
        kernel_size: int
    ) -> None:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.kernel_shape = (
            out_channel,
            in_channel, 
            kernel_size,
            kernel_size
        )
        self.K = np.random.randn(*self.kernel_shape)
    
    def forward(
        self, 
        X: NDArray[Shape["*, *, *, *"], Number]
    ) -> NDArray[Shape["*, *, *, *"], Number]:
        if len(X.shape) == 3:
            X = X.reshape((
                X.shape[0], 1, X.shape[1], X.shape[2]
            ))
                
        self.in_shape = X.shape
        self.batch_size = self.in_shape[0]

        self.out_shape = (
            self.batch_size, # batch size
            self.out_channel,
            self.in_shape[2] - self.kernel_size + 1, # height
            self.in_shape[3] - self.kernel_size + 1 # width
        )

        self.b = np.random.randn(*self.out_shape)

        self.X = X
        self.Y = np.copy(self.b)

        for b in range(self.batch_size):
            for i in range(self.out_channel):
                for j in range(self.in_channel):
     