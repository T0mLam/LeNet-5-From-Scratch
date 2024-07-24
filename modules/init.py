from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from nptyping import NDArray, Shape, Float


class Initialization(ABC):
    @abstractmethod
    def __call__(self, in_dim: int, out_dim: int):
        pass


class Kaiming(Initialization):
    def __call__(
            self,
            in_dim: int,
            out_dim: int
        ) -> Tuple[
            NDArray[Shape["*, *"], Float],
            NDArray[Shape["*"], Float]
        ]:
        var = 2 / in_dim
        W = np.random.randn(out_dim, in_dim) * np.sqrt(var)
        b = np.zeros(out_dim)
        return W, b
    

class Xavier(Initialization):
    pass
