from typing import Any, Optional
from abc import ABC, abstractmethod

import numpy as np
from nptyping import NDArray, Number, Shape

from .layer import Layer


class Dropout(Layer):
    """
    Reference:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting,
        https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    """

    def __init__(self, rate: float) -> None:
        self.rate = rate 
    
    def forward(
        self,
        X: NDArray[Any, Number],
        train: bool=True
    ) -> NDArray[Any, Number]:
        if train:
            self.mask = np.random.binomial(1, 1 - self.rate, size=(X.shape))
            return X * self.mask
        return X
    
    def backward(
        self,
        grad: NDArray[Any, Number]
    ) -> NDArray[Any, Number]:
        return grad * self.mask
    

class L2(Layer):
    pass