from typing import Any, Optional
from abc import ABC, abstractmethod

import numpy as np
from nptyping import NDArray, Number, Shape

from .layer import Layer


class BatchNorm(Layer):
    def forward(
        self, 
        X: NDArray[Any, Number],
        train: bool=True
    ) -> NDArray:
        pass

    def backward(
        self,
        grad: NDArray[Any, Number]
    ) -> NDArray[Any, Number]:
        pass