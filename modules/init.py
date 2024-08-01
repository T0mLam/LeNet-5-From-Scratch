from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from nptyping import NDArray, Shape, Float

from .layer import Linear, Conv


class Initialization(ABC):
    @abstractmethod
    def __call__(self, in_dim: int, out_dim: int):
        pass


class Kaiming(Initialization):
    def __call__(
        self,
        layer: Union[Linear, Conv]
    ) -> Union[
        Tuple[NDArray[Shape["*, *"], Float], NDArray[Shape["*"], Float]],
        NDArray[Shape["*, *, *, *"], Float]
    ]:
        if isinstance(layer, Linear):
            var = 2 / layer.in_dim
            W = np.random.randn(layer.out_dim, layer.in_dim) * np.sqrt(var)
            b = np.zeros(layer.out_dim)
            return W, b
        
        if isinstance(layer, Conv):
            var = 2 / np.prod(layer.kernel_shape)
            K = np.random.randn(layer.kernel_shape) * np.sqrt(var)
            return K
        

class Xavier(Initialization):
    def __call__(
        self,
        layer: Union[Linear, Conv]
    ) -> Union[
        Tuple[NDArray[Shape["*, *"], Float], NDArray[Shape["*"], Float]],
        NDArray[Shape["*, *, *, *"], Float]
    ]:
        if isinstance(layer, Linear):
            var = np.sqrt(6 / (layer.in_dim + layer.out_dim))
            W = np.random.uniform(-var, var, size=(layer.out_dim, layer.in_dim))
            b = np.zeros(layer.out_dim)
            return W, b
        
        if isinstance(layer, Conv):
            kernel_size = layer.kernel_size ** 2
            var = np.sqrt(6 / (layer.in_channel * kernel_size + layer.out_channel * kernel_size))
            K = np.random.uniform(-var, var, size=layer.kernel_shape)
            return K
        
    
class LeCun(Initialization):
    def __call__(
        self,
        layer: Union[Linear, Conv]
    ) -> Union[
        Tuple[NDArray[Shape["*, *"], Float], NDArray[Shape["*"], Float]],
        NDArray[Shape["*, *, *, *"], Float]
    ]:
        if isinstance(layer, Linear):
            var = 2.4 / layer.in_dim
            W = np.random.uniform(-var, var, size=(layer.out_dim, layer.in_dim))
            b = np.zeros(layer.out_dim)
            return W, b
        
        if isinstance(layer, Conv):
            fan_in = layer.kernel_size ** 2
            var = np.sqrt(2.4 / fan_in)
            K = np.random.uniform(-var, var, size=layer.kernel_shape)
            return K