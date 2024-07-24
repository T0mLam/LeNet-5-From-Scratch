from typing import Any

import numpy as np
from nptyping import NDArray, Shape, Number

from .layer import Layer


class ReLU(Layer):
    def forward(
        self,
        X: NDArray[Any, Number]
    ) -> NDArray[Any, Number]:
        self.X = X
        return np.maximum(0, X)

    def backward(
        self, 
        grad: NDArray[Any, Number]
    ) -> NDArray[Any, Number]:
        grad[self.X <= 0] = 0
        return grad
    

class Tanh(Layer):
    def forward(
        self,
        X: NDArray[Any, Number]
    ) -> NDArray[Any, Number]:
        self.output = np.tanh(X)
        return self.output

    def backward(
        self, 
        grad: NDArray[Any, Number]
    ) ->