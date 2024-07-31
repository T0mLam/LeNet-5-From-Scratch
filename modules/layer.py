from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from nptyping import Number, NDArray, Shape
from scipy import signal


class Layer(ABC):
    @abstractmethod
    def forward(self, X: NDArray, **kwargs) -> NDArray:
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
        self.in_dim = in_dim
        self.out_dim = out_dim

        if init:
            self.W, self.b = init(self)
        else:
            self.W = np.random.randn(out_dim, in_dim)
            self.b = np.random.randn(out_dim)
    
    def forward(
        self,
        X: NDArray[Shape["*, *"], Number],
        **kwargs
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
        shape: Tuple[int, int],
        batch_size: int, 
        in_channel: int,
        out_channel: int, 
        kernel_size: int,
        mapping: Optional[NDArray[Shape['*, *'], Number]]=None,
        init: Optional[Initialization]=None
    ) -> None:
        self.shape = shape
        self.batch_size = batch_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.kernel_shape = (
            out_channel,
            in_channel, 
            kernel_size,
            kernel_size
        )
        self.out_shape = (
            self.batch_size, # batch size
            self.out_channel,
            self.shape[0] - self.kernel_size + 1, # height
            self.shape[1] - self.kernel_size + 1 # width
        )

        self.b = np.random.randn(*self.out_shape)
        
        if init:
            self.K = init(self)
        else:
            self.K = np.random.randn(*self.kernel_shape)

        if mapping:
            self.mapping = mapping 
        else:
            self.mapping = None
    
    def forward(
        self, 
        X: NDArray[Shape["*, *, *, *"], Number],
        **kwargs
    ) -> NDArray[Shape["*, *, *, *"], Number]:
        if len(X.shape) == 3:
            X = X.reshape((
                X.shape[0], 1, X.shape[1], X.shape[2]
            ))
        
        self.X = X
        self.in_shape = X.shape
        self.Y = np.copy(self.b)

        for b in range(self.batch_size):
            for i in range(self.out_channel):
                if self.mapping:
                    in_channels = self.mapping[i]
                else:
                    in_channels = range(self.in_channel)

                for j in in_channels:
                    self.Y[b, i] += signal.correlate2d(
                        self.X[b, j], self.K[i, j], mode='valid'
                    )
                
        return self.Y

    def backward(
        self, 
        grad: NDArray[Shape["*, *, *, *"], Number],
    ) -> NDArray[Shape["*, *, *, *"], Number]:
        self.dK = np.random.randn(*self.kernel_shape)
        self.db = np.copy(grad)
        self.out_grad = np.zeros(self.in_shape)

        for b in range(self.batch_size):
            for i in range(self.out_channel):
                if self.mapping:
                    in_channels = self.mapping[i]
                else:
                    in_channels = range(self.in_channel)

                for j in in_channels:
                    self.dK[i, j] = signal.correlate2d(
                        self.X[b, j], grad[b, i], mode='valid'
                    )
                    self.out_grad[j] += signal.convolve2d(
                        grad[b, i], self.K[i, j], mode='full'
                    )

        return self.out_grad


class Flatten(Layer):
    def forward(
        self,
        X: NDArray[Shape['*, *, ...'], Number],
        **kwargs
    ) -> NDArray[Shape['*, *'], Number]:
        self.X_shape = X.shape
        self.batch_size, *self.feature_dim = self.X_shape
        return X.reshape((self.batch_size, np.prod(self.feature_dim)))

    def backward(
        self, 
        grad: NDArray[Shape['*, *'], Number]
    ) -> NDArray[Shape['*, *, ...'], Number]:
        return grad.reshape(self.X_shape)
    

class RBF(Layer):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        W: NDArray[Shape["*, *, *"], Number]
    ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = W

    def forward(
        self,
        X: NDArray[Shape["*, *"], Number],
        y: Optional[NDArray[Shape["*, *"], Number]]=None,
        train: bool=True,
        **kwargs
    ) -> float:  
        batch_size = self.X.shape[0]
        self.Y = np.zeros(batch_size)

        if y:
            self.y = y
        
        for b in range(batch_size):
            if train:
                bitmap = self.W[y[b]]
                diff = X[b] - bitmap
                self.Y[b] = np.sum(np.power(diff, 2))
            else:
                z = np.zeros(len(self.W))
                for i, bitmap in enumerate(self.W):
                    diff = X[b] - bitmap
                    z[i] = np.sum(np.power(diff, 2))
                self.Y[b] = z.argmin()
    
        return self.Y

    def backward(
        self,
        grad: NDArray[Shape["*, *"], Number]
    ) -> NDArray[Shape["*, *"], Number]:
        pass